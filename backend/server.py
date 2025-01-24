#!/usr/bin/env python
from dotenv import load_dotenv
import uvicorn
import os

# Load environment variables before importing app
load_dotenv()

if __name__ == "__main__":
    # Development mode with reload when running directly
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if os.getenv("ENVIRONMENT") != "production" else False        
    )