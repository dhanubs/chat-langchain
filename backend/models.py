from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ThreadStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    INTERRUPTED = "interrupted"
    ERROR = "error"

class Message(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
class ChatInput(BaseModel):
    messages: List[Message]
    config: Optional[Dict] = None
    stream: bool = False

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


class ChatRequest(BaseModel):
    message: str
    stream: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is LangChain?",
                "stream": False
            }
        }


class ThreadCreatePayload(BaseModel):
    metadata: Optional[Dict] = None
    thread_id: Optional[str] = None
    if_exists: Optional[str] = None


class ThreadSearchPayload(BaseModel):
    metadata: Optional[Dict] = None
    limit: int = 10
    offset: int = 0
    status: Optional[ThreadStatus] = None

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