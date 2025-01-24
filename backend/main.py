from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import StreamingResponse
from backend.chat_workflow import ChatWorkflow
from backend.models import ChatInput, ChatRequest, ThreadCreatePayload, ThreadHistoryPayload, ThreadSearchPayload, ThreadStatePayload, ThreadStateUpdatePayload, ThreadStatus, Thread, ThreadState, Checkpoint, ThreadUpdatePayload
from backend.thread_manager import ThreadManager
from typing import Optional, Dict, List
import os
import json
from pydantic import BaseModel, Field
import logging

app = FastAPI()

# Initialize managers
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
thread_manager = ThreadManager(MONGO_URI)
chat_workflow = ChatWorkflow(
    provider=os.getenv("LLM_PROVIDER", "azure"),
    model=os.getenv("LLM_MODEL", "gpt-35-turbo-16k")
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

@app.get("/threads/{thread_id}/state")
async def get_thread_state(
    thread_id: str,
    subgraphs: bool = False
):
    """Get thread state"""
    try:
        return await thread_manager.get_state(thread_id, subgraphs=subgraphs)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/threads/{thread_id}/state/checkpoint")
async def get_thread_state_checkpoint(
    thread_id: str,
    payload: ThreadStatePayload = Body(...)
):
    """Get thread state with checkpoint"""
    try:
        return await thread_manager.get_state(
            thread_id,
            checkpoint=payload.checkpoint,
            subgraphs=payload.subgraphs
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/threads/{thread_id}/state")
async def update_thread_state(
    thread_id: str,
    payload: ThreadStateUpdatePayload = Body(...)
):
    """Update thread state"""
    return await thread_manager.update_state(
        thread_id,
        values=payload.values,
        checkpoint_id=payload.checkpoint_id,
        checkpoint=payload.checkpoint,
        as_node=payload.as_node
    )

@app.patch("/threads/{thread_id}/state")
async def patch_thread_state(
    thread_id: str,
    metadata: Dict = Body(...)
):
    """Patch thread state metadata"""
    await thread_manager.patch_state(thread_id, metadata)
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
async def chat(thread_id: str, message: str, stream: bool = False):
    """Chat endpoint with optional streaming"""
    # Save the user message first
    thread = await thread_manager.add_message(thread_id, "human", message)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    if stream:
        # Return streaming response
        async def stream_chat():
            full_response = ""
            async for chunk in chat_workflow.stream_response(message):
                full_response += chunk
                # Send chunk as SSE
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Save the complete response after streaming
            await thread_manager.add_message(thread_id, "assistant", full_response)
            # Send end marker
            yield f"data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_chat(),
            media_type="text/event-stream"
        )
    else:
        # Generate complete response
        response = await chat_workflow.generate_response(message)
        # Save the response
        thread = await thread_manager.add_message(thread_id, "assistant", response)
        
        return {
            "thread_id": thread_id,
            "response": response
        }


@app.post("/chat/{thread_id}/stream")
async def chat_stream(thread_id: str, message: str):
    """Dedicated streaming endpoint"""
    return await chat(thread_id, message, stream=True)

# @app.post("/threads/{thread_id}/messages")
# async def add_message(thread_id: str, role: str, content: str):
#     """Add a message to a thread"""
#     thread = await thread_manager.add_message(thread_id, role, content)
#     if not thread:
#         raise HTTPException(status_code=404, detail="Thread not found")
#     return thread

# @app.get("/threads/{thread_id}/messages")
# async def get_messages(thread_id: str):
#     """Get all messages for a thread"""
#     messages = await thread_manager.get_messages(thread_id)
#     return messages

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
        description="Chat request with optional streaming",
        examples=[{
            "message": "What is LangChain?",
            "stream": False
        }]
    )
):
    """
    Test chat completion with optional streaming.
    Try different messages and see how the model responds.
    """
    return await chat(thread_id, request.message, request.stream) 