"""
Test script to verify error handling in document processing.
This script tests various error scenarios to ensure proper error handling.
"""

import os
import asyncio
import tempfile
import pytest
from pathlib import Path

from backend.document_processor import (
    DocumentProcessor, 
    UnsupportedFormatError,
    FileAccessError,
    EmptyContentError,
    ParsingError,
    VectorStoreError,
    DocumentProcessingError,
    SUPPORTED_EXTENSIONS
)

# Create a test directory
@pytest.fixture
def test_dir():
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Clean up
    for file in temp_dir.glob("*"):
        try:
            file.unlink()
        except:
            pass
    temp_dir.rmdir()

# Test unsupported file format
async def test_unsupported_format(test_dir):
    # Create an unsupported file
    unsupported_file = test_dir / "test.xyz"
    with open(unsupported_file, "w") as f:
        f.write("Test content")
    
    processor = DocumentProcessor()
    documents, error = processor.process_file(unsupported_file)
    
    assert len(documents) == 0
    assert isinstance(error, UnsupportedFormatError)
    assert error.error_type == "unsupported_format"
    assert str(unsupported_file) in error.file_path
    
    # Test through async method
    result = await processor.process_file_async(unsupported_file)
    assert result["success"] is False
    assert result["error_type"] == "unsupported_format"
    assert "Unsupported file format" in result["error"]

# Test empty content
async def test_empty_content(test_dir):
    # Create an empty text file
    empty_file = test_dir / "empty.txt"
    with open(empty_file, "w") as f:
        f.write("")
    
    processor = DocumentProcessor()
    documents, error = processor.process_file(empty_file)
    
    assert len(documents) == 0
    assert isinstance(error, EmptyContentError)
    assert error.error_type == "empty_content"
    
    # Test through async method
    result = await processor.process_file_async(empty_file)
    assert result["success"] is False
    assert result["error_type"] == "empty_content"
    assert "No content" in result["error"]

# Test file access error
async def test_file_access_error(test_dir):
    # Reference a non-existent file
    nonexistent_file = test_dir / "nonexistent.txt"
    
    processor = DocumentProcessor()
    documents, error = processor.process_file(nonexistent_file)
    
    assert len(documents) == 0
    assert isinstance(error, DocumentProcessingError)
    
    # Test through async method
    result = await processor.process_file_async(nonexistent_file)
    assert result["success"] is False
    assert "error_type" in result
    assert "error" in result

# Test directory validation
async def test_directory_validation():
    processor = DocumentProcessor()
    
    # Test non-existent directory
    with pytest.raises(ValueError) as excinfo:
        await processor.process_directory("/nonexistent/directory")
    assert "does not exist" in str(excinfo.value)
    
    # Test file as directory
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    try:
        with pytest.raises(ValueError) as excinfo:
            await processor.process_directory(temp_file.name)
        assert "not a directory" in str(excinfo.value)
    finally:
        os.unlink(temp_file.name)
    
    # Test invalid file extensions
    with pytest.raises(ValueError) as excinfo:
        await processor.process_directory(
            tempfile.gettempdir(),
            file_extensions=["pdf", "docx"]  # Missing dots
        )
    assert "Invalid file extensions" in str(excinfo.value)

# Test empty directory
async def test_empty_directory(test_dir):
    processor = DocumentProcessor()
    
    # Process an empty directory
    stats = await processor.process_directory(test_dir)
    
    assert stats["total_files"] == 0
    assert "warning" in stats
    assert "No matching files found" in stats["warning"]

# Test batch processing with mixed results
async def test_mixed_results(test_dir):
    # Create a valid file
    valid_file = test_dir / "valid.txt"
    with open(valid_file, "w") as f:
        f.write("This is a valid test file with content.")
    
    # Create an empty file
    empty_file = test_dir / "empty.txt"
    with open(empty_file, "w") as f:
        f.write("")
    
    # Create an unsupported file
    unsupported_file = test_dir / "unsupported.xyz"
    with open(unsupported_file, "w") as f:
        f.write("Unsupported file content")
    
    processor = DocumentProcessor()
    
    # Process the directory
    stats = await processor.process_directory(test_dir)
    
    assert stats["total_files"] == 1  # Only the valid txt file should be counted
    assert stats["processed_files"] == 1
    assert stats["failed_files"] == 0
    
    # Test with explicit file extensions to include unsupported format
    stats = await processor.process_directory(
        test_dir,
        file_extensions=[".txt", ".xyz"]
    )
    
    assert stats["total_files"] == 2  # txt and xyz files
    assert stats["processed_files"] == 1  # Only txt processed successfully
    assert stats["failed_files"] == 1  # xyz file failed
    assert "error_types" in stats
    assert "unsupported_format" in stats["error_types"]
    assert "failed_files_details" in stats
    assert len(stats["failed_files_details"]) == 1

# Test uploaded file processing
async def test_uploaded_file_processing(test_dir):
    processor = DocumentProcessor()
    
    # Test with empty content
    result = await processor.process_uploaded_file(b"", "test.txt")
    assert result["success"] is False
    assert result["error_type"] == "empty_file"
    
    # Test with missing filename
    result = await processor.process_uploaded_file(b"content", "")
    assert result["success"] is False
    assert result["error_type"] == "missing_filename"
    
    # Test with unsupported format
    result = await processor.process_uploaded_file(b"content", "test.xyz")
    assert result["success"] is False
    assert result["error_type"] == "unsupported_format"
    
    # Test with valid content
    valid_content = b"This is valid test content."
    result = await processor.process_uploaded_file(valid_content, "test.txt")
    assert result["success"] is True
    assert result["chunks"] > 0

# Test batch file upload processing
async def test_batch_upload_processing():
    processor = DocumentProcessor()
    
    # Test with empty list
    results = await processor.process_uploaded_files_parallel([])
    assert len(results) == 1
    assert results[0]["error_type"] == "no_files"
    
    # Test with invalid entries
    results = await processor.process_uploaded_files_parallel([
        "not a dictionary",
        {"missing_content": True, "filename": "test.txt"},
        {"content": b"content", "missing_filename": True},
        {"content": b"", "filename": "empty.txt"},
        {"content": b"valid content", "filename": "valid.txt"}
    ])
    
    assert len(results) == 5
    assert results[0]["error_type"] == "invalid_format"
    assert results[1]["error_type"] == "empty_content"
    assert results[2]["error_type"] == "missing_filename"
    assert results[3]["error_type"] == "empty_file"
    assert results[4]["success"] is True

# Test error classification
def test_error_classification():
    processor = DocumentProcessor()
    
    # Test file access error
    error = processor.classify_error(
        "test.txt", 
        Exception("Permission denied")
    )
    assert error.error_type == "file_access_error"
    
    # Test parsing error
    error = processor.classify_error(
        "test.txt", 
        Exception("Error parsing document syntax")
    )
    assert error.error_type == "parsing_error"
    
    # Test vector store error
    error = processor.classify_error(
        "test.txt", 
        Exception("Failed to add to vector index")
    )
    assert error.error_type == "vector_store_error"
    
    # Test unknown error
    error = processor.classify_error(
        "test.txt", 
        Exception("Some other error")
    )
    assert error.error_type == "unknown_error"

if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_unsupported_format(Path(tempfile.mkdtemp())))
    print("✅ Unsupported format test passed")
    
    asyncio.run(test_empty_content(Path(tempfile.mkdtemp())))
    print("✅ Empty content test passed")
    
    asyncio.run(test_file_access_error(Path(tempfile.mkdtemp())))
    print("✅ File access error test passed")
    
    asyncio.run(test_directory_validation())
    print("✅ Directory validation test passed")
    
    asyncio.run(test_empty_directory(Path(tempfile.mkdtemp())))
    print("✅ Empty directory test passed")
    
    asyncio.run(test_mixed_results(Path(tempfile.mkdtemp())))
    print("✅ Mixed results test passed")
    
    asyncio.run(test_uploaded_file_processing(Path(tempfile.mkdtemp())))
    print("✅ Uploaded file processing test passed")
    
    asyncio.run(test_batch_upload_processing())
    print("✅ Batch upload processing test passed")
    
    test_error_classification()
    print("✅ Error classification test passed")
    
    print("\n🎉 All error handling tests passed!") 