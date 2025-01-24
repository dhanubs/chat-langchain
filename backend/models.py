from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from backend.enums import ThreadStatus, OnConflictBehavior
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
import asyncio

class Message(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
class ChatInput(BaseModel):
    input: str
    include_history: bool = False

class ChatRequest(BaseModel):
    message: str
    config: Optional[Dict] = None
    stream: bool = True

class Thread(BaseModel):
    thread_id: str
    metadata: Dict[str, Any]
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    status: ThreadStatus = Field(default=ThreadStatus.IDLE)
    error: Optional[str] = None

class Checkpoint(BaseModel):
    thread_id: str
    node: Optional[str] = None
    checkpoint_id: Optional[str] = None
    metadata: Optional[Dict] = None

class ThreadState(BaseModel):
    thread_id: str
    values: Dict[str, Any]
    checkpoint: Optional[Checkpoint] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict] = None 


class ThreadCreatePayload(BaseModel):
    metadata: Optional[Dict] = None
    thread_id: Optional[str] = None
    if_exists: Optional[OnConflictBehavior] = None


class ThreadSearchPayload(BaseModel):
    metadata: Optional[Dict] = None
    limit: int = 10
    offset: int = 0
    status: Optional[ThreadStatus] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {"user_id": "123", "source": "web"},
                "limit": 10,
                "offset": 0,
                "status": "idle"
            }
        }

class ThreadUpdatePayload(BaseModel):
    metadata: Optional[Dict] = None


class ThreadStatePayload(BaseModel):
    checkpoint: Optional[Dict] = None
    subgraphs: Optional[bool] = None

class ThreadStateUpdatePayload(BaseModel):
    values: Dict
    checkpoint_id: Optional[str] = None
    checkpoint: Optional[Dict] = None
    as_node: Optional[str] = None

class ThreadHistoryPayload(BaseModel):
    limit: int = Field(default=10)
    before: Optional[Dict] = None
    metadata: Optional[Dict] = None
    checkpoint: Optional[Dict] = None

class ThreadChatMessageHistory(BaseChatMessageHistory):
    """Adapter class to make Thread compatible with LangChain's BaseChatMessageHistory"""
    
    def __init__(self, thread: Thread):
        self.thread = thread
        self._messages = None

    @property
    def messages(self):
        """Return messages in LangChain format (sync)"""
        if self._messages is None and self.thread and hasattr(self.thread, 'messages'):
            self._messages = [
                AIMessage(content=msg.content) if msg.role == "assistant" 
                else HumanMessage(content=msg.content)
                for msg in self.thread.messages
            ]
        return self._messages or []

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add messages to the store"""
        if self._messages is None:
            self._messages = []
        self._messages.extend(messages)

    def clear(self) -> None:
        """Clear messages from the store"""
        self._messages = None