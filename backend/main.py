from fastapi import FastAPI, HTTPException, Body, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from backend.chat_workflow import ChatWorkflow
from backend.models import ChatInput, ChatRequest, ThreadCreatePayload, ThreadHistoryPayload, ThreadSearchPayload, ThreadStatePayload, ThreadStateUpdatePayload, ThreadStatus, Thread, ThreadState, Checkpoint, ThreadUpdatePayload
from backend.thread_manager import ThreadManager
from typing import Optional, Dict, List
import os
import json
from pydantic import BaseModel, Field
import logging
from uuid import uuid4
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

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
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
thread_manager = ThreadManager(MONGO_URI)
chat_workflow = ChatWorkflow(
    provider=os.getenv("LLM_PROVIDER", "azure"),
    model=os.getenv("LLM_MODEL", "gpt-35-turbo-16k"),
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
            if msg["type"] in ["human", "assistant"]
        ]
    
    async def generate_events():
        full_response = ""
        try:
            async for chunk in chat_workflow.stream_response(
                input=payload.input,
                thread_id=thread_id,
                chat_history=chat_history if payload.include_history else None,
                config=thread.values.get("config")  # Pass any config from thread values
            ):
                full_response += chunk
                yield f"data: {json.dumps({'event': 'message', 'data': {'content': chunk}})}\n\n"
            
            # Save the complete response
            await thread_manager.add_message(thread_id, "assistant", full_response)
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