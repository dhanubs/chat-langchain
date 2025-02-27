from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from backend.chat_workflow import ChatWorkflow
from backend.models import ChatInput, ChatRequest, ThreadCreatePayload, ThreadHistoryPayload, ThreadSearchPayload, ThreadUpdatePayload
from backend.thread_manager import ThreadManager
from pathlib import Path
from backend.ingest import ingest_pdfs, ingest_documents, process_uploaded_document
import os
import json
import logging
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.document_processor import DocumentProcessor

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

@app.post("/ingest/pdfs")
async def ingest_pdf_documents(
    folder_path: str,
    recursive: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Ingest PDF documents from a specified folder into Azure AI Search.
    
    Args:
        folder_path: Path to folder containing PDFs
        recursive: Whether to search subdirectories
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
    """
    try:
        await ingest_pdfs(
            folder_path,
            recursive=recursive,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return {"status": "success", "message": "PDFs ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/documents")
async def ingest_document_folder(
    folder_path: str,
    recursive: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    file_extensions: Optional[List[str]] = None,
    parallel: bool = True,
    max_concurrency: int = 5
):
    """
    Ingest multiple document types from a specified folder into Azure AI Search.
    
    Args:
        folder_path: Path to folder containing documents
        recursive: Whether to search subdirectories
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        file_extensions: List of file extensions to process (if None, process all supported types)
        parallel: Whether to process files in parallel
        max_concurrency: Maximum number of files to process concurrently
    """
    try:
        # Initialize document processor with concurrency settings
        processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_concurrency=max_concurrency
        )
        
        # Process all documents in the directory
        stats = await processor.process_directory(
            directory_path=folder_path,
            recursive=recursive,
            file_extensions=file_extensions,
            parallel=parallel
        )
        
        return {
            "status": "success", 
            "message": "Documents ingested successfully",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error ingesting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """
    Upload and process a single document file.
    
    Args:
        file: The uploaded file
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Process the uploaded file
        stats = await process_uploaded_document(
            file_content=file_content,
            filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if not stats.get("success", False):
            raise HTTPException(
                status_code=422, 
                detail=f"Failed to process file: {stats.get('error', 'Unknown error')}"
            )
        
        return {
            "status": "success",
            "message": f"Document processed successfully: {stats.get('chunks', 0)} chunks created",
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/upload-batch")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    parallel: bool = Form(True),
    max_concurrency: int = Form(5)
):
    """
    Upload and process multiple document files.
    
    Args:
        files: List of uploaded files
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        parallel: Whether to process files in parallel
        max_concurrency: Maximum number of files to process concurrently
    """
    try:
        # Initialize document processor with concurrency settings
        processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_concurrency=max_concurrency
        )
        
        # Prepare file data
        file_data_list = []
        for file in files:
            # Read file content
            file_content = await file.read()
            file_data_list.append({
                'content': file_content,
                'filename': file.filename
            })
        
        if parallel and len(files) > 1:
            # Process files in parallel
            results = await processor.process_uploaded_files_parallel(file_data_list)
        else:
            # Process files sequentially
            results = []
            for file_data in file_data_list:
                result = await processor.process_uploaded_file(
                    file_content=file_data['content'],
                    filename=file_data['filename']
                )
                results.append(result)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "filename": result.get("filename", ""),
                "success": result.get("success", False),
                "chunks": result.get("chunks", 0),
                "error": result.get("error", None)
            })
        
        # Check if any files were processed successfully
        if not any(result["success"] for result in formatted_results):
            raise HTTPException(
                status_code=422,
                detail="Failed to process any of the uploaded files"
            )
        
        return {
            "status": "success",
            "message": f"Processed {sum(1 for r in formatted_results if r['success'])} of {len(formatted_results)} files successfully",
            "parallel_processing": parallel and len(files) > 1,
            "max_concurrency": max_concurrency if parallel and len(files) > 1 else 1,
            "results": formatted_results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))