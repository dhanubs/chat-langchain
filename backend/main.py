from dotenv import load_dotenv

# Load environment variables before importing other modules
load_dotenv()

from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form, Query, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from backend.chat_workflow import ChatWorkflow
from backend.models import ChatInput, ChatRequest, ThreadCreatePayload, ThreadHistoryPayload, ThreadSearchPayload, ThreadUpdatePayload
from backend.thread_manager import ThreadManager
from pathlib import Path
import os
import json
import logging
import traceback
from typing import List, Optional, Dict, Any
from backend.config import settings
from backend.document_processor import DocumentProcessor, DocumentProcessingError

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
thread_manager = ThreadManager(settings.mongodb_uri)
chat_workflow = ChatWorkflow(
    provider=settings.llm_provider,
    model=settings.llm_model,
    thread_manager=thread_manager
)

logger = logging.getLogger("api")

# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with a more user-friendly response"""
    errors = []
    for error in exc.errors():
        error_msg = {
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        }
        errors.append(error_msg)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error in request data",
            "errors": errors
        }
    )

@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request: Request, exc: DocumentProcessingError):
    """Handle document processing errors with detailed information"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.message,
            "error_type": exc.error_type,
            "file_path": exc.file_path
        }
    )

# Thread management endpoints
@app.post("/threads")
async def create_thread(
    payload: ThreadCreatePayload = Body(...)
):
    """Create a new thread"""
    try:
        return await thread_manager.create(
            metadata=payload.metadata,
            thread_id=payload.thread_id,
            if_exists=payload.if_exists
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get a thread by ID"""
    thread = await thread_manager.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread

@app.post("/threads/search")
async def search_threads(
    payload: ThreadSearchPayload = Body(...)
):
    """Search threads with pagination and filters"""
    return await thread_manager.search(
        metadata=payload.metadata,
        limit=payload.limit,
        offset=payload.offset,
        status=payload.status
    )

@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread"""
    success = await thread_manager.delete(thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"status": "success"}

@app.post("/threads/{thread_id}/history")
async def get_thread_history(
    thread_id: str,
    payload: ThreadHistoryPayload = Body(...)
):
    """Get thread history"""
    return await thread_manager.get_history(
        thread_id,
        limit=payload.limit,
        before=payload.before,
        checkpoint=payload.checkpoint,
        metadata=payload.metadata
    )

@app.post("/threads/{thread_id}/copy")
async def copy_thread(thread_id: str):
    """Copy an existing thread"""
    try:
        return await thread_manager.copy(thread_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.patch("/threads/{thread_id}")  # Note: PATCH method
async def update_thread(
    thread_id: str,
    payload: ThreadUpdatePayload = Body(...)
):
    """Update a thread"""
    try:
        return await thread_manager.update(thread_id, payload.metadata)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/chat/{thread_id}")
async def chat(thread_id: str, message: str):
    """Regular chat endpoint for non-streaming responses"""
    # Save the user message first
    thread = await thread_manager.add_message(thread_id, "human", message)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Generate complete response
    response = await chat_workflow.generate_response(message, thread_id)
    # Save the response
    thread = await thread_manager.add_message(thread_id, "ai", response)
    
    return {
        "thread_id": thread_id,
        "response": response
    }

@app.post("/chat/{thread_id}/stream")
async def chat_stream(
    thread_id: str,
    payload: ChatInput = Body(...)
):
    """Stream chat responses with optional conversation history"""
    # Get thread and save the new message
    thread = await thread_manager.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Save the user message first
    await thread_manager.add_message(thread_id, "human", payload.input)
    
    # Get chat history if requested
    chat_history = []
    if payload.include_history:
        chat_history = [
            {"role": msg["type"], "content": msg["content"]} 
            for msg in thread.messages
            if msg["type"] in ["human", "ai"]
        ]
    
    async def generate_events():
        full_response = ""
        try:
            async for chunk in chat_workflow.stream_response(
                input=payload.input,
                thread_id=thread_id,
                chat_history=chat_history if payload.include_history else None,
                config=payload.config  # Use config from payload instead of thread.values
            ):
                full_response += chunk
                yield f"data: {json.dumps({'event': 'message', 'data': {'content': chunk}})}\n\n"
            
            # Save the complete response
            await thread_manager.add_message(thread_id, "ai", full_response)
            yield f"data: {json.dumps({'event': 'end', 'data': None})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'data': {'error': str(e)}})}\n\n"
    
    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream"
    )

@app.get("/models")
async def list_models():
    """Return list of available models and their endpoints"""
    return {
        "models": [
            {"name": "azure-gpt4", "provider": "azure", "model": "gpt-4o"},
            {"name": "azure-gpt35", "provider": "azure", "model": "gpt-35-turbo-16k"},
            {"name": "openai-gpt4o-mini", "provider": "openai", "model": "gpt-4o-mini"},
            {"name": "openai-gpt35", "provider": "openai", "model": "gpt-3.5-turbo"},
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat/{thread_id}/playground")
async def chat_playground(
    thread_id: str,
    request: ChatRequest = Body(
        ...,
        description="Chat request with optional model configuration",
        examples=[{
            "message": "What is LangChain?",
            "config": {
                "model_name": "gpt-4",
                "temperature": 0.7
            },
            "stream": True
        }]
    )
):
    """
    Test chat completion with different models and configurations.
    """
    # Get thread - await the coroutine
    thread = await thread_manager.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Save the user message
    await thread_manager.add_message(thread_id, "human", request.message)
    
    # Get updated thread with new message
    thread = await thread_manager.get(thread_id)  # Add this line to get fresh thread state
    
    if request.stream:
        async def generate_events():
            full_response = ""
            try:
                async for chunk in chat_workflow.stream_response(
                    input=request.message,
                    thread_id=thread_id,
                    chat_history=thread.messages,
                    config=request.config
                ):
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                # Save the complete response
                await thread_manager.add_message(thread_id, "ai", full_response)
                yield f"data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Playground error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_events(),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming response
        try:
            response = await chat_workflow.generate_response(
                message=request.message,
                thread_id=thread_id,
                chat_history=thread.messages,
                config=request.config
            )
            await thread_manager.add_message(thread_id, "ai", response)
            return {"response": response}
        except Exception as e:
            logger.error(f"Playground error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/documents")
async def ingest_document_folder(
    folder_path: str = Query(..., description="Path to folder containing documents"),
    recursive: bool = Query(True, description="Whether to search subdirectories"),
    chunk_size: int = Query(1000, description="Size of text chunks for splitting", gt=0, le=10000),
    chunk_overlap: int = Query(200, description="Overlap between chunks", ge=0, lt=5000),
    file_extensions: Optional[List[str]] = Query(None, description="List of file extensions to process"),
    parallel: bool = Query(True, description="Whether to process files in parallel"),
    max_concurrency: int = Query(5, description="Maximum number of files to process concurrently", ge=1, le=20)
):
    """
    Process all supported documents in a directory and add them to the vector store.
    
    Args:
        folder_path: Path to the directory containing documents
        recursive: Whether to search subdirectories
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        file_extensions: List of file extensions to process (if None, process all supported types)
        parallel: Whether to process files in parallel
        max_concurrency: Maximum number of files to process concurrently
        
    Returns:
        Dict: Processing statistics
    """
    folder = Path(folder_path)
    
    # Validate folder path
    if not folder.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Directory not found: {folder_path}"
        )
    if not folder.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path is not a directory: {folder_path}"
        )
    
    # Validate chunk parameters
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be less than chunk_size"
        )
    
    try:
        # Initialize document processor
        processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_concurrency=max_concurrency
        )
        
        # Process the directory
        stats = await processor.process_directory(
            directory_path=folder_path,
            recursive=recursive,
            file_extensions=file_extensions,
            parallel=parallel
        )
        
        if stats["total_files"] == 0:
            return {
                "status": "warning",
                "message": stats["warning"],
                "stats": stats
            }
        
        if stats["failed_files"] == stats["total_files"]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="All files failed to process",
                headers={"X-Error-Details": str(stats["error_types"])}
            )
        
        if stats["failed_files"] > 0:
            return {
                "status": "partial_success",
                "message": f"Processed {stats['processed_files']} files successfully, {stats['failed_files']} failed",
                "stats": stats
            }
        
        return {
            "status": "success",
            "message": f"Successfully processed {stats['processed_files']} files",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error processing directory {folder_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing directory: {str(e)}"
        )