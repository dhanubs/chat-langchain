from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from backend.chat_workflow import ChatWorkflow
from backend.models import ChatInput, ChatRequest, ThreadCreatePayload, ThreadHistoryPayload, ThreadSearchPayload, ThreadUpdatePayload
from backend.thread_manager import ThreadManager
from pathlib import Path
from backend.ingest import ingest_pdfs, ingest_documents, process_uploaded_document
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

@app.post("/ingest/pdfs")
async def ingest_pdf_documents(
    folder_path: str = Query(..., description="Path to folder containing PDFs"),
    recursive: bool = Query(True, description="Whether to search subdirectories"),
    chunk_size: int = Query(1000, description="Size of text chunks for splitting", gt=0, le=10000),
    chunk_overlap: int = Query(200, description="Overlap between chunks", ge=0, lt=5000)
):
    """
    Ingest PDF documents from a specified folder into Azure AI Search.
    
    Args:
        folder_path: Path to folder containing PDFs
        recursive: Whether to search subdirectories
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
    """
    # Validate folder path
    folder = Path(folder_path)
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
        await ingest_pdfs(
            folder_path,
            recursive=recursive,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return {
            "status": "success", 
            "message": "PDFs ingested successfully",
            "folder": str(folder),
            "recursive": recursive,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
    except Exception as e:
        logger.error(f"Error ingesting PDFs: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Provide more specific error messages based on the exception
        if "permission" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied accessing directory: {folder_path}"
            )
        elif "not found" in str(e).lower() or "no such" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Directory or files not found: {folder_path}"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

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
    # Validate folder path
    folder = Path(folder_path)
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
    
    # Validate file extensions
    if file_extensions:
        invalid_extensions = [ext for ext in file_extensions if not ext.startswith(".")]
        if invalid_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file extensions (must start with '.'): {invalid_extensions}"
            )
    
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
        
        # Check if any files were processed
        if stats.get("total_files", 0) == 0:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "warning",
                    "message": f"No matching files found in {folder_path}",
                    "stats": stats
                }
            )
        
        # Check if all files failed
        if stats.get("failed_files", 0) == stats.get("total_files", 0) and stats.get("total_files", 0) > 0:
            return JSONResponse(
                status_code=status.HTTP_207_MULTI_STATUS,
                content={
                    "status": "error",
                    "message": "All files failed to process",
                    "stats": stats
                }
            )
        
        # Check if some files failed
        if stats.get("failed_files", 0) > 0:
            return JSONResponse(
                status_code=status.HTTP_207_MULTI_STATUS,
                content={
                    "status": "partial_success",
                    "message": f"Processed {stats.get('processed_files', 0)} files successfully, {stats.get('failed_files', 0)} files failed",
                    "stats": stats
                }
            )
        
        # All files processed successfully
        return {
            "status": "success", 
            "message": f"Successfully processed {stats.get('processed_files', 0)} documents with {stats.get('total_chunks', 0)} chunks",
            "stats": stats
        }
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error ingesting documents: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Provide more specific error messages based on the exception
        if "permission" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied accessing directory: {folder_path}"
            )
        elif "not found" in str(e).lower() or "no such" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Directory or files not found: {folder_path}"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )

@app.post("/ingest/upload")
async def upload_document(
    file: UploadFile = File(..., description="The document file to upload"),
    chunk_size: int = Form(1000, description="Size of text chunks for splitting", gt=0, le=10000),
    chunk_overlap: int = Form(200, description="Overlap between chunks", ge=0, lt=5000)
):
    """
    Upload and process a single document file.
    
    Args:
        file: The uploaded file
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
    """
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing filename"
        )
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    from backend.document_processor import SUPPORTED_EXTENSIONS
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format: {file_ext}. Supported formats: {list(SUPPORTED_EXTENSIONS.keys())}"
        )
    
    # Validate chunk parameters
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be less than chunk_size"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Check if file is empty
        if not file_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file content"
            )
        
        # Process the uploaded file
        stats = await process_uploaded_document(
            file_content=file_content,
            filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if not stats.get("success", False):
            error_type = stats.get("error_type", "unknown_error")
            error_msg = stats.get("error", "Unknown error")
            
            # Return appropriate status code based on error type
            if error_type == "unsupported_format":
                status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            elif error_type in ["empty_file", "empty_content", "missing_filename"]:
                status_code = status.HTTP_400_BAD_REQUEST
            elif error_type == "file_access_error":
                status_code = status.HTTP_403_FORBIDDEN
            elif error_type == "vector_store_error":
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            else:
                status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
                
            raise HTTPException(
                status_code=status_code,
                detail=f"Failed to process file: {error_msg}"
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
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error processing document: {str(e)}"
        )

@app.post("/ingest/upload-batch")
async def upload_multiple_documents(
    files: List[UploadFile] = File(..., description="The document files to upload"),
    chunk_size: int = Form(1000, description="Size of text chunks for splitting", gt=0, le=10000),
    chunk_overlap: int = Form(200, description="Overlap between chunks", ge=0, lt=5000),
    parallel: bool = Form(True, description="Whether to process files in parallel"),
    max_concurrency: int = Form(5, description="Maximum number of files to process concurrently", ge=1, le=20)
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
    # Validate files
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    # Validate chunk parameters
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be less than chunk_size"
        )
    
    # Check file extensions
    from backend.document_processor import SUPPORTED_EXTENSIONS
    unsupported_files = []
    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One or more files missing filename"
            )
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            unsupported_files.append(file.filename)
    
    if unsupported_files:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format(s): {unsupported_files}. Supported formats: {list(SUPPORTED_EXTENSIONS.keys())}"
        )
    
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
            
            # Check if file is empty
            if not file_content:
                logger.warning(f"Empty file content: {file.filename}")
                continue
                
            file_data_list.append({
                'content': file_content,
                'filename': file.filename
            })
        
        # Check if any valid files remain after filtering
        if not file_data_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid files to process (all files were empty)"
            )
        
        if parallel and len(file_data_list) > 1:
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
                "error": result.get("error", None),
                "error_type": result.get("error_type", None)
            })
        
        # Calculate success and failure counts
        success_count = sum(1 for r in formatted_results if r["success"])
        failure_count = len(formatted_results) - success_count
        
        # Check if any files were processed successfully
        if success_count == 0:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "status": "error",
                    "message": "Failed to process any of the uploaded files",
                    "results": formatted_results
                }
            )
        
        # If some files failed, return partial success
        if failure_count > 0:
            return JSONResponse(
                status_code=status.HTTP_207_MULTI_STATUS,
                content={
                    "status": "partial_success",
                    "message": f"Processed {success_count} of {len(formatted_results)} files successfully",
                    "parallel_processing": parallel and len(file_data_list) > 1,
                    "max_concurrency": max_concurrency if parallel and len(file_data_list) > 1 else 1,
                    "results": formatted_results
                }
            )
        
        # All files processed successfully
        return {
            "status": "success",
            "message": f"Successfully processed all {len(formatted_results)} files",
            "parallel_processing": parallel and len(file_data_list) > 1,
            "max_concurrency": max_concurrency if parallel and len(file_data_list) > 1 else 1,
            "results": formatted_results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded documents: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error processing documents: {str(e)}"
        )