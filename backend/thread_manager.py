from typing import Dict, List, Optional, Any
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from datetime import datetime
import uuid
from backend.models import Thread
from backend.enums import OnConflictBehavior, ThreadStatus
from urllib.parse import urlparse

class ThreadManager:
    def __init__(self, mongo_uri: str):
        """Initialize ThreadManager with both async and sync clients.
        Database name is extracted from URI if present, otherwise defaults to 'chatapp'
        """
        # Create temporary client to parse URI and get database name
        temp_client = MongoClient(mongo_uri)
        try:
            # Get database name from URI or use default
            database_name = temp_client.get_database().name
        except Exception:
            database_name = "chatapp"
        finally:
            temp_client.close()
        
        # We need both clients because:
        # 1. The main FastAPI application uses async operations for better performance
        # 2. However, LangChain's RunnableWithMessageHistory expects a synchronous 
        #    get_session_history function and cannot handle async operations
        # 3. Attempting to run async code synchronously inside an async context 
        #    (like FastAPI) leads to event loop conflicts
        # 4. Using pymongo's synchronous client is the cleanest solution to avoid 
        #    these event loop issues while maintaining compatibility with LangChain
           
        # Async client for main operations
        self.client = AsyncIOMotorClient(mongo_uri)
        self.threads = self.client[database_name].threads
        
        # Sync client specifically for LangChain compatibility
        self.sync_client = MongoClient(mongo_uri)
        self.sync_db = self.sync_client[database_name]
    
    def get_sync(self, thread_id: str) -> Optional[Thread]:
        """Synchronous version of get method using pymongo"""
        try:
            result = self.sync_db.threads.find_one({"thread_id": thread_id})
            if not result:
                return None
            return Thread(**result)
        except Exception:
            return None

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
        
        # Initialize with empty values
        thread = Thread(
            thread_id=thread_id or str(uuid.uuid4()),
            metadata=metadata or {},
            values={"messages": []},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        await self.threads.update_one(
            {"thread_id": thread.thread_id},
            {"$set": thread.model_dump()},
            upsert=True
        )
        return thread

    async def copy(self, thread_id: str) -> Thread:
        """Copy an existing thread"""
        source = await self.get(thread_id)
        if not source:
            raise ValueError(f"Thread {thread_id} not found")
        
        # Safely copy values and metadata
        new_values = source.values.copy() if source.values else None
        new_metadata = source.metadata.copy() if source.metadata else {}
        
        new_thread = Thread(
            thread_id=str(uuid.uuid4()),
            metadata=new_metadata,
            values=new_values,
            created_at=datetime.utcnow()
        )
        await self.threads.insert_one(new_thread.dict())
        return new_thread

    async def update(self, thread_id: str, metadata: Optional[Dict] = None) -> Thread:
        """Update a thread's metadata"""
        # Get current thread to preserve values
        current_thread = await self.get(thread_id)
        if not current_thread:
            raise ValueError(f"Thread {thread_id} not found")
            
        update_dict = {
            "updated_at": datetime.utcnow()
        }
        
        # Only update metadata if provided
        if metadata is not None:
            update_dict["metadata"] = metadata
        
        result = await self.threads.update_one(
            {"thread_id": thread_id},
            {"$set": update_dict}
        )
        
        if result.modified_count == 0:
            raise ValueError(f"Thread {thread_id} not found")
            
        return await self.get(thread_id)
    
    async def get(self, thread_id: str) -> Optional[Thread]:
        """Get a thread by ID"""
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
            # Only include non-None values in metadata query
            metadata_query = {
                f"metadata.{k}": v 
                for k, v in metadata.items() 
                if v is not None
            }
            if metadata_query:
                query.update(metadata_query)
                
        if status is not None:
            query["status"] = status
        
        # Execute query with pagination and sorting
        cursor = self.threads.find(query) \
            .sort("created_at", -1) \
            .skip(offset) \
            .limit(limit)
        
        # Convert to list of Thread objects
        threads = await cursor.to_list(length=None)
        thread_objects = []
        
        for thread_dict in threads:
            try:
                thread_objects.append(Thread(**thread_dict))
            except Exception:
                # Skip invalid thread documents
                continue
                
        return thread_objects
    
    async def delete(self, thread_id: str) -> bool: 
        """
        Delete a thread and its associated state.
        
        Args:
            thread_id: ID of the thread to delete
            
        Returns:
            bool: True if thread was deleted, False if thread wasn't found
        """
        # Delete the thread
        result = await self.threads.delete_one({"thread_id": thread_id})
        
        return result.deleted_count > 0

    async def add_message(self, thread_id: str, role: str, content: str) -> Optional[Thread]:
        """
        Add a message to a thread.
        
        Args:
            thread_id: ID of the thread
            role: Role of the message sender ('human' or 'ai')
            content: Content of the message
            
        Returns:
            Updated Thread object if successful, None if thread not found
        """
        # Find the thread
        thread_dict = await self.threads.find_one({"thread_id": thread_id})
        if not thread_dict:
            return None
        
        current_time = datetime.utcnow()
        
        # Get current values or initialize empty
        current_values = thread_dict.get("values") or {}
        current_messages = current_values.get("messages", []) if current_values else []
        
        # Add the new message to values
        current_messages.append({
            "type": role,
            "content": content,
            "updated_at": current_time.isoformat()
        })
        
        # Update thread with new values
        result = await self.threads.update_one(
            {"thread_id": thread_id},
            {
                "$set": {
                    "updated_at": current_time,
                    "values": {
                        **current_values,
                        "messages": current_messages
                    }
                }
            }
        )
        
        if result.modified_count > 0:
            # Get updated thread
            return await self.get(thread_id)
        
        return None

    async def get_history(
        self,
        thread_id: str,
        limit: int = 20,
        before: Optional[str] = None,
        checkpoint: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Get message history for a thread with filtering options.
        
        Args:
            thread_id: ID of the thread
            limit: Maximum number of messages to return
            before: Return messages before this timestamp
            checkpoint: Return messages before this checkpoint
            metadata: Additional metadata filters
            
        Returns:
            List of message dictionaries
        """
        # Match thread and metadata if provided
        match_query = {"thread_id": thread_id}
        if metadata:
            for key, value in metadata.items():
                match_query[f"metadata.{key}"] = value

        pipeline = [
            {"$match": match_query},
            {"$project": {
                "messages": "$values.messages",
            }},
            {"$unwind": "$messages"},
        ]

        # Add filters if provided
        if before:
            pipeline.append({
                "$match": {"messages.updated_at": {"$lt": before}}
            })
        
        if checkpoint:
            pipeline.append({
                "$match": {"messages.updated_at": {"$lt": checkpoint}}
            })

        # Sort and limit results
        pipeline.extend([
            {"$sort": {"messages.updated_at": -1}},
            {"$limit": limit}
        ])

        # Get only message content
        pipeline.append({
            "$replaceRoot": {"newRoot": "$messages"}
        })

        cursor = self.threads.aggregate(pipeline)
        messages = await cursor.to_list(length=None)
        return messages