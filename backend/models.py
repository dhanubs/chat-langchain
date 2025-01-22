from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any

class Message(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Thread(BaseModel):
    thread_id: str
    metadata: Dict
    created_at: datetime = Field(default_factory=datetime.utcnow)
    messages: List[Message] = Field(default_factory=list)
    values: Dict[str, Any] = Field(default_factory=dict) 