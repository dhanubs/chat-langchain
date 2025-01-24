from typing import Dict, List, Optional, Any
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from datetime import datetime
import uuid
from enum import Enum
from backend.models import Thread, Message, ThreadState, Checkpoint
from backend.enums import OnConflictBehavior, ThreadStatus
import anyio
import sniffio
import asyncio

class ThreadManager:
    def __init__(self, mongo_uri: str, database_name: str = "chatapp"):
        """Initialize ThreadManager with both async and sync clients.
        
        We need both clients because:
        1. The main FastAPI application uses async operations for better performance
        2. However, LangChain's RunnableWithMessageHistory expects a synchronous 
           get_session_history function and cannot handle async operations
        3. Attempting to run async code synchronously inside an async context 
           (like FastAPI) leads to event loop conflicts
        4. Using pymongo's synchronous client is the cleanest solution to avoid 
           these event loop issues while maintaining compatibility with LangChain
        """
        # Async client for main operations
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[database_name]
        self.threads = self.db.threads
        self.states = self.db.thread_states
        
        # Sync client specifically for LangChain compatibility
        self.sync_client = MongoClient(mongo_uri)
        self.sync_db = self.sync_client[database_name]
    
    def get_sync(self, thread_id: str) -> Optional[Thread]:
        """Synchronous version of get method using pymongo"""
        result = self.sync_db.threads.find_one({"thread_id": thread_id})
        return Thread(**result) if result else None

    async def create(
        self,
        metadata: Optional[Dict] = None,
        thread_id: Optional[str] = None,
        if_exists: Optional[OnConflictBehavior] = None
    ) -> Thread:
        """Create a new thread with conflict handling"""
        if thread_id and if_exists:
            existing = await self.get(thread_id)
            if existing:
                if if_exists == OnConflictBehavior.ERROR:
                    raise ValueError(f"Thread {thread_id} already exists")
                elif if_exists == OnConflictBehavior.REUSE:
                    return existing
                # else UPDATE - continue with creation/update
        
        thread = Thread(
            thread_id=thread_id or str(uuid.uuid4()),
            metadata=metadata or {},
            created_at=datetime.utcnow()
        )
        await self.threads.update_one(
            {"thread_id": thread.thread_id},
            {"$set": thread.dict()},
            upsert=True
        )
        return thread

    async def copy(self, thread_id: str) -> Thread:
        """Copy an existing thread"""
        source = await self.get(thread_id)
        if not source:
            raise ValueError(f"Thread {thread_id} not found")
        
        new_thread = Thread(
            thread_id=str(uuid.uuid4()),
            metadata=source.metadata.copy(),
            messages=source.messages.copy(),
            created_at=datetime.utcnow()
        )
        await self.threads.insert_one(new_thread.dict())
        return new_thread

    async def update(self, thread_id: str, metadata: Optional[Dict] = None) -> Thread:
        """Update a thread's metadata"""
        result = await self.threads.update_one(
            {"thread_id": thread_id},
            {
                "$set": {
                    "metadata": metadata or {},
                    "updated_at": datetime.utcnow()
                }
            }
        )
        if result.modified_count == 0:
            raise ValueError(f"Thread {thread_id} not found")
        return await self.get(thread_id)

    async def get_state(
        self,
        thread_id: str,
        checkpoint: Optional[str] = None,
        include_subgraphs: bool = False
    ) -> ThreadState:
        """Get thread state"""
        query = {"thread_id": thread_id}
        if checkpoint:
            query["checkpoint"] = checkpoint
        
        state = await self.states.find_one(query)
        if not state:
            raise ValueError(f"Thread state not found for {thread_id}")
        return ThreadState(**state)

    async def update_state(
        self,
        thread_id: str,
        values: Dict,
        checkpoint: Optional[Dict] = None,
        checkpoint_id: Optional[str] = None,
        as_node: Optional[str] = None
    ) -> Dict:
        """Update thread state"""
        state_doc = {
            "thread_id": thread_id,
            "values": values,
            "updated_at": datetime.utcnow()
        }
        if checkpoint:
            state_doc["checkpoint"] = checkpoint
        if checkpoint_id:
            state_doc["checkpoint_id"] = checkpoint_id
        if as_node:
            state_doc["node"] = as_node
            
        await self.states.update_one(
            {"thread_id": thread_id},
            {"$set": state_doc},
            upsert=True
        )
        return {"configurable": {}}  # Return empty config as per TypeScript definition

    async def patch_state(self, thread_id: str, metadata: Dict) -> None:
        """Patch thread state metadata"""
        await self.states.update_one(
            {"thread_id": thread_id},
            {"$set": {"metadata": metadata}}
        )

    async def get_history(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        before: Optional[Dict] = None,
        checkpoint: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> List[ThreadState]:
        """Get thread state history"""
        query = {"thread_id": thread_id}
        if before:
            query["created_at"] = {"$lt": before.get("created_at")}
        if checkpoint:
            query.update({f"checkpoint.{k}": v for k, v in checkpoint.items()})
        if metadata:
            query.update({f"metadata.{k}": v for k, v in metadata.items()})
            
        cursor = self.states.find(query).sort("created_at", -1)
        if limit:
            cursor = cursor.limit(limit)
            
        states = await cursor.to_list(length=None)
        return [ThreadState(**state) for state in states]

    async def get(self, thread_id: str) -> Optional[Thread]:
        thread_dict = await self.threads.find_one({"thread_id": thread_id})
        return Thread(**thread_dict) if thread_dict else None
    
    async def search(
        self,
        metadata: Optional[Dict] = None,
        limit: int = 10,
        offset: int = 0,
        status: Optional[ThreadStatus] = None
    ) -> List[Thread]:
        """
        Search threads based on filters.
        
        Args:
            metadata: Optional metadata filter
            limit: Maximum number of threads to return (default: 10)
            offset: Number of threads to skip (default: 0)
            status: Optional thread status filter
        
        Returns:
            List of matching Thread objects
        """
        # Build the query
        query = {}
        if metadata:
            query.update({f"metadata.{k}": v for k, v in metadata.items()})
        if status:
            query["status"] = status
        
        # Execute query with pagination and sorting
        cursor = self.threads.find(query) \
            .sort("created_at", -1) \
            .skip(offset) \
            .limit(limit)
        
        # Convert to list of Thread objects
        threads = await cursor.to_list(length=None)
        return [Thread(**thread) for thread in threads]

    async def get_thread_state(self, thread_id: str) -> Dict:
        """
        Get the current state of a thread.
        
        Args:
            thread_id: ID of the thread
            
        Returns:
            Dict containing thread state
        
        Raises:
            KeyError: If thread_id doesn't exist
        """
        thread_dict = await self.threads.find_one({"thread_id": thread_id})
        if not thread_dict:
            raise KeyError(f"Thread {thread_id} not found")
        
        thread = Thread(**thread_dict)
        return {
            "id": thread.thread_id,
            "status": getattr(thread, 'status', None),
            "metadata": thread.metadata,
            "created_at": thread.created_at,
            "messages": [msg.dict() for msg in thread.messages],
            "error": getattr(thread, 'error', None)
        }

    async def delete(self, thread_id: str) -> bool: 
        """
        Delete a thread and its associated state.
        
        Args:
            thread_id: ID of the thread to delete
            
        Returns:
            bool: True if thread was deleted, False if thread wasn't found
        """
        # Delete thread state first
        await self.states.delete_many({"thread_id": thread_id})
        
        # Delete the thread
        result = await self.threads.delete_one({"thread_id": thread_id})
        
        return result.deleted_count > 0

    async def add_message(self, thread_id: str, role: str, content: str) -> Optional[Thread]:
        """
        Add a message to a thread.
        
        Args:
            thread_id: ID of the thread
            role: Role of the message sender ('human' or 'assistant')
            content: Content of the message
            
        Returns:
            Updated Thread object if successful, None if thread not found
        """
        # Find the thread
        thread_dict = await self.threads.find_one({"thread_id": thread_id})
        if not thread_dict:
            return None
        
        # Create new message
        message = {
            "role": role,
            "content": content,
            "created_at": datetime.utcnow()
        }
        
        # Add message to thread
        result = await self.threads.update_one(
            {"thread_id": thread_id},
            {
                "$push": {"messages": message},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        if result.modified_count > 0:
            # Get updated thread
            thread_dict = await self.threads.find_one({"thread_id": thread_id})
            return Thread(**thread_dict)
        
        return None