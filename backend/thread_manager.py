from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import uuid
from backend.models import Thread, Message

class ThreadManager:
    def __init__(self, mongo_uri: str, database_name: str = "chatapp"):
        self.client = AsyncIOMotorClient(mongo_uri)
        self.db = self.client[database_name]
        self.threads = self.db.threads
    
    async def create(self, metadata: Optional[Dict] = None) -> Thread:
        thread = Thread(
            thread_id=str(uuid.uuid4()),
            metadata=metadata or {},
            created_at=datetime.utcnow()
        )
        await self.threads.insert_one(thread.dict())
        return thread
    
    async def get(self, thread_id: str) -> Optional[Thread]:
        thread_dict = await self.threads.find_one({"thread_id": thread_id})
        return Thread(**thread_dict) if thread_dict else None
    
    async def search(self, metadata: Optional[Dict] = None) -> List[Thread]:
        query = {}
        if metadata:
            query.update({f"metadata.{k}": v for k, v in metadata.items()})
        
        cursor = self.threads.find(query)
        threads = await cursor.to_list(length=None)
        return [Thread(**thread) for thread in threads]
    
    async def delete(self, thread_id: str) -> bool:
        result = await self.threads.delete_one({"thread_id": thread_id})
        return result.deleted_count > 0
    
    async def add_message(self, thread_id: str, role: str, content: str) -> Optional[Thread]:
        """Add a message to a thread's history"""
        message = Message(role=role, content=content)
        result = await self.threads.update_one(
            {"thread_id": thread_id},
            {
                "$push": {"messages": message.dict()},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        if result.modified_count:
            return await self.get(thread_id)
        return None
    
    async def get_messages(self, thread_id: str) -> List[Message]:
        """Get all messages for a thread"""
        thread = await self.get(thread_id)
        return thread.messages if thread else [] 