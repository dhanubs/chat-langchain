from fastapi import FastAPI, HTTPException
from langserve import add_routes
from backend.chain import create_chain, get_retriever
from backend.thread_manager import ThreadManager
from typing import Optional
import os

app = FastAPI()

# Initialize managers
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
thread_manager = ThreadManager(MONGO_URI)
retriever = get_retriever()
chain = create_chain(retriever)

# Add LangServe routes for the chain
add_routes(
    app,
    chain,
    path="/chat",
    enable_feedback_endpoint=True,
)

# Thread management endpoints
@app.post("/threads")
async def create_thread(metadata: Optional[dict] = None):
    """Create a new thread"""
    thread = await thread_manager.create(metadata)
    return thread

@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get a thread by ID"""
    thread = await thread_manager.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread

@app.get("/threads/search")
async def search_threads(metadata: Optional[dict] = None):
    """Search threads by metadata"""
    return await thread_manager.search(metadata)

@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete a thread"""
    if not await thread_manager.delete(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"status": "success"}

@app.post("/threads/{thread_id}/messages")
async def add_message(thread_id: str, role: str, content: str):
    """Add a message to a thread"""
    thread = await thread_manager.add_message(thread_id, role, content)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread

@app.get("/threads/{thread_id}/messages")
async def get_messages(thread_id: str):
    """Get all messages for a thread"""
    messages = await thread_manager.get_messages(thread_id)
    return messages

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