"""
Test script for the document processor module.
"""

import os
import asyncio
import tempfile
from pathlib import Path

from backend.document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS

# Sample text content for test files
SAMPLE_TEXT = """
This is a sample document for testing the document processor.
It contains multiple paragraphs and some formatting.

The document processor should be able to extract text from various file formats
including PDF, DOCX, PPTX, and more using Docling.

This is another paragraph with some more text.
"""

def create_test_files():
    """Create test files for different formats."""
    test_dir = Path(tempfile.mkdtemp())
    
    # Create a simple text file
    txt_path = test_dir / "sample.txt"
    with open(txt_path, "w") as f:
        f.write(SAMPLE_TEXT)
    
    print(f"Created test files in {test_dir}")
    return test_dir

async def test_document_processor():
    """Test the document processor with different file types."""
    # Create test files
    test_dir = create_test_files()
    
    try:
        # Initialize document processor
        processor = DocumentProcessor(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Test processing a text file
        txt_file = test_dir / "sample.txt"
        if txt_file.exists():
            print(f"Processing {txt_file}...")
            documents = processor.process_file(txt_file)
            print(f"Extracted {len(documents)} chunks from {txt_file.name}")
            
            # Print the first document content
            if documents:
                print("\nFirst chunk content:")
                print("-" * 40)
                print(documents[0].page_content[:200] + "...")
                print("-" * 40)
                print("\nMetadata:")
                for key, value in documents[0].metadata.items():
                    print(f"  {key}: {value}")
        
        # Test supported extensions
        print("\nSupported file extensions:")
        for ext, desc in SUPPORTED_EXTENSIONS.items():
            print(f"  {ext}: {desc}")
    
    finally:
        # Clean up test files
        for file in test_dir.glob("*"):
            file.unlink()
        test_dir.rmdir()

if __name__ == "__main__":
    asyncio.run(test_document_processor()) 