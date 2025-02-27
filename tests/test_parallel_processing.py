"""
Test script to demonstrate parallel processing for document ingestion.
This script compares sequential vs parallel processing performance.
"""

import os
import time
import asyncio
import tempfile
import random
from pathlib import Path

from backend.document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS

# Sample text content for test files
SAMPLE_TEXT = """
This is a sample document for testing the document processor with parallel processing.
It contains multiple paragraphs and some formatting.

The document processor should be able to extract text from various file formats
including PDF, DOCX, PPTX, and more using Docling.

This is another paragraph with some more text to ensure we have enough content
to demonstrate the chunking capabilities of the document processor.

Parallel processing should significantly improve performance when dealing with
large numbers of files by processing multiple files concurrently.

This allows better utilization of system resources and reduces the overall
processing time for large document collections.
"""

def create_test_files(num_files=20):
    """Create test files for different formats."""
    test_dir = Path(tempfile.mkdtemp())
    
    # Create multiple text files
    file_paths = []
    for i in range(num_files):
        # Add some random content to make files different
        random_text = f"\nThis is file number {i} with some random content: {random.randint(1000, 9999)}"
        content = SAMPLE_TEXT + random_text
        
        txt_path = test_dir / f"sample_{i}.txt"
        with open(txt_path, "w") as f:
            f.write(content)
        file_paths.append(txt_path)
    
    print(f"Created {num_files} test files in {test_dir}")
    return test_dir, file_paths

async def test_sequential_vs_parallel():
    """Test and compare sequential vs parallel processing performance."""
    # Create test files
    num_files = 20  # Adjust based on your system's capabilities
    test_dir, file_paths = create_test_files(num_files)
    
    try:
        print(f"\nTesting with {num_files} files...")
        
        # Test sequential processing
        print("\n--- Sequential Processing ---")
        sequential_processor = DocumentProcessor(
            chunk_size=500,
            chunk_overlap=50,
            max_concurrency=1  # Not used in sequential mode
        )
        
        start_time = time.time()
        sequential_stats = await sequential_processor.process_directory(
            directory_path=test_dir,
            recursive=False,
            parallel=False
        )
        sequential_time = time.time() - start_time
        
        print(f"Sequential processing time: {sequential_time:.2f} seconds")
        print(f"Processed {sequential_stats['processed_files']} files with {sequential_stats['total_chunks']} chunks")
        
        # Test parallel processing with different concurrency levels
        concurrency_levels = [2, 5, 10]
        
        for concurrency in concurrency_levels:
            print(f"\n--- Parallel Processing (concurrency={concurrency}) ---")
            parallel_processor = DocumentProcessor(
                chunk_size=500,
                chunk_overlap=50,
                max_concurrency=concurrency
            )
            
            start_time = time.time()
            parallel_stats = await parallel_processor.process_directory(
                directory_path=test_dir,
                recursive=False,
                parallel=True
            )
            parallel_time = time.time() - start_time
            
            print(f"Parallel processing time: {parallel_time:.2f} seconds")
            print(f"Processed {parallel_stats['processed_files']} files with {parallel_stats['total_chunks']} chunks")
            print(f"Speedup factor: {sequential_time / parallel_time:.2f}x")
    
    finally:
        # Clean up test files
        for file in test_dir.glob("*"):
            file.unlink()
        test_dir.rmdir()

if __name__ == "__main__":
    asyncio.run(test_sequential_vs_parallel()) 