from enum import Enum

class ThreadStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    INTERRUPTED = "interrupted"
    ERROR = "error"

class OnConflictBehavior(str, Enum):
    ERROR = "error"
    UPDATE = "update"
    REUSE = "reuse" 