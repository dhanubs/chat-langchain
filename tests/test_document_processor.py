"""
Test script for the document processor module.
"""

import os
import asyncio
import tempfile
import requests
from pathlib import Path

from backend.document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS

def download_sample_pdf():
    """Download a sample PDF file from a reliable source."""
    # Using a small, reliable PDF from Mozilla's PDF.js test files
    url = "https://raw.githubusercontent.com/mozilla/pdf.js/master/test/pdfs/basicapi.pdf"
    test_dir = Path(tempfile.mkdtemp())
    pdf_path = test_dir / "sample.pdf"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        
        print(f"Downloaded sample PDF to {pdf_path}")
        return test_dir, pdf_path
    except Exception as e:
        print(f"Error downloading PDF: {str(e)}")
        # Create a simple PDF using reportlab as fallback
        from reportlab.pdfgen import canvas
        
        pdf_path = test_dir / "sample.pdf"
        c = canvas.Canvas(str(pdf_path))
        c.drawString(100, 750, "This is a sample PDF document for testing.")
        c.drawString(100, 700, "It contains some basic text content.")
        c.drawString(100, 650, "The document processor should be able to extract this text.")
        c.save()
        
        print(f"Created fallback PDF at {pdf_path}")
        return test_dir, pdf_path

async def test_document_processor():
    """Test the document processor with a PDF file."""
    # Get sample PDF
    test_dir, pdf_path = download_sample_pdf()
    
    try:
        # Initialize document processor
        processor = DocumentProcessor(
            chunk_size=1000,
            chunk_overlap=200,
            max_concurrency=5
        )
        
        # Test processing a PDF file
        if pdf_path.exists():
            print(f"\nProcessing {pdf_path}...")
            documents, error = processor.process_file(pdf_path)
            
            if error:
                print(f"Error processing PDF: {error.message}")
                print(f"Error type: {error.error_type}")
            else:
                print(f"Successfully extracted {len(documents)} chunks from {pdf_path.name}")
                
                # Print the first document content
                if documents:
                    print("\nFirst chunk content:")
                    print("-" * 40)
                    if hasattr(documents[0], 'page_content'):
                        print(documents[0].page_content[:200] + "...")
                    else:
                        print("No page_content attribute found")
                    
                    print("\nMetadata:")
                    if hasattr(documents[0], 'metadata'):
                        for key, value in documents[0].metadata.items():
                            print(f"  {key}: {value}")
                    else:
                        print("No metadata attribute found")
        
        # Test supported extensions
        print("\nSupported file extensions:")
        for ext, desc in SUPPORTED_EXTENSIONS.items():
            print(f"  {ext}: {desc}")
    
    finally:
        # Clean up test files
        for file in test_dir.glob("*"):
            try:
                file.unlink()
            except Exception as e:
                print(f"Error deleting {file}: {str(e)}")
        try:
            test_dir.rmdir()
        except Exception as e:
            print(f"Error removing test directory: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_document_processor()) 