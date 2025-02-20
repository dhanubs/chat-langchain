#!/usr/bin/env python
from dotenv import load_dotenv
import uvicorn
from backend.config import settings

# Load environment variables before importing app
load_dotenv()

if __name__ == "__main__":
    # Development mode with reload when running directly
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.environment != "production" else False        
    )