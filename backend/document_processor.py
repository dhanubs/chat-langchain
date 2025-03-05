"""
Document processor module for handling various document types using Docling.
Supports PDF, DOCX, PPTX, and other document formats.
"""

import os
import logging
import asyncio
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple

# Set environment variables for offline mode
os.environ["DOCLING_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Suppress docling deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="docling_core")

from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

from backend.config import settings

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF Document",
    ".docx": "Word Document",
    ".doc": "Word Document (Legacy)",
    ".pptx": "PowerPoint Presentation",
    ".ppt": "PowerPoint Presentation (Legacy)",
    ".xlsx": "Excel Spreadsheet",
    ".xls": "Excel Spreadsheet (Legacy)",
    ".rtf": "Rich Text Format",
}

# Error categories for better error handling
class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    def __init__(self, message: str, error_type: str, file_path: str, original_error: Optional[Exception] = None):
        self.message = message
        self.error_type = error_type
        self.file_path = file_path
        self.original_error = original_error
        super().__init__(self.message)

class UnsupportedFormatError(DocumentProcessingError):
    """Exception raised for unsupported file formats."""
    def __init__(self, file_path: str):
        super().__init__(
            f"Unsupported file format: {Path(file_path).suffix}",
            "unsupported_format",
            file_path
        )

class FileAccessError(DocumentProcessingError):
    """Exception raised for file access issues."""
    def __init__(self, file_path: str, original_error: Exception):
        super().__init__(
            f"Cannot access file: {original_error}",
            "file_access_error",
            file_path,
            original_error
        )

class ParsingError(DocumentProcessingError):
    """Exception raised for document parsing issues."""
    def __init__(self, file_path: str, original_error: Exception):
        super().__init__(
            f"Error parsing document: {original_error}",
            "parsing_error",
            file_path,
            original_error
        )

class EmptyContentError(DocumentProcessingError):
    """Exception raised when no content could be extracted."""
    def __init__(self, file_path: str):
        super().__init__(
            "No content could be extracted from the document",
            "empty_content",
            file_path
        )

class VectorStoreError(DocumentProcessingError):
    """Exception raised for vector store issues."""
    def __init__(self, file_path: str, original_error: Exception):
        super().__init__(
            f"Error adding document to vector store: {original_error}",
            "vector_store_error",
            file_path,
            original_error
        )

class DocumentProcessor:
    """
    Document processor class for handling various document types.
    Uses Docling for document parsing and extraction in offline mode.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
        max_concurrency: int = 5,
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            separators: Custom separators for text splitting
            max_concurrency: Maximum number of files to process concurrently
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.max_concurrency = max_concurrency
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len
        )
        
        try:
            # Initialize document converter (will use offline mode due to environment variables)
            self.converter = DocumentConverter()
        except Exception as e:
            logger.error("Failed to initialize DocumentConverter. Make sure required models are installed.")
            logger.error(f"Error: {str(e)}")
            raise RuntimeError("DocumentConverter initialization failed. Please ensure models are properly installed in the HuggingFace cache directory.") from e
    
    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if the file type is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if the file type is supported, False otherwise
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    
    def process_file(self, file_path: Union[str, Path]) -> Tuple[List[Document], Optional[DocumentProcessingError]]:
        """
        Process a single file and convert it to LangChain documents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple[List[Document], Optional[DocumentProcessingError]]: 
                A tuple containing the list of documents and an optional error
        """
        file_path = Path(file_path)
        
        if not self.is_supported_file(file_path):
            error = UnsupportedFormatError(str(file_path))
            logger.warning(f"{error.error_type}: {error.message}")
            return [], error
        
        try:
            # Process the document
            docling_doc = self.converter.convert(str(file_path))
            
            # Extract content using Docling's structured parsing
            content_parts = []
            
            # Get main text content
            if docling_doc.text:
                content_parts.append(docling_doc.text)
            
            # Extract text from sections if available
            if hasattr(docling_doc, 'sections'):
                for section in docling_doc.sections:
                    if section.text:
                        content_parts.append(f"\nSection: {section.title if section.title else 'Untitled'}\n{section.text}")
            
            # Extract tables if available
            if hasattr(docling_doc, 'tables'):
                for i, table in enumerate(docling_doc.tables):
                    if table.text:
                        content_parts.append(f"\nTable {i+1}:\n{table.text}")
            
            # Combine all content
            content = "\n\n".join(content_parts)
            
            # Check if content is empty
            if not content or content.isspace():
                error = EmptyContentError(str(file_path))
                logger.warning(f"{error.error_type}: {error.message} for {file_path.name}")
                return [], error
            
            # Extract metadata
            metadata = {
                "source": str(file_path.name),
                "file_path": str(file_path),
                "file_type": SUPPORTED_EXTENSIONS[file_path.suffix.lower()],
                "title": file_path.stem,
                "created_at": datetime.utcnow().isoformat(),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "processor_version": "2.0.0",
            }
            
            # Add Docling metadata
            if hasattr(docling_doc, 'metadata'):
                # Add basic metadata
                for key, value in docling_doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[f"docling_{key}"] = str(value)
                
                # Add language if detected
                if hasattr(docling_doc.metadata, 'language'):
                    metadata['language'] = str(docling_doc.metadata.language)
                
                # Add page count if available
                if hasattr(docling_doc.metadata, 'pages'):
                    metadata['page_count'] = str(len(docling_doc.metadata.pages))
                
                # Add OCR confidence if available
                if hasattr(docling_doc.metadata, 'ocr_confidence'):
                    metadata['ocr_confidence'] = str(docling_doc.metadata.ocr_confidence)
            
            # Create a single document
            doc = Document(page_content=content, metadata=metadata)
            
            # Split the document
            documents = self.text_splitter.split_documents([doc])
            
            if not documents:
                error = EmptyContentError(str(file_path))
                logger.warning(f"{error.error_type}: No chunks created for {file_path.name}")
                return [], error
                
            return documents, None
            
        except Exception as e:
            error = self.classify_error(file_path, e)
            logger.error(f"{error.error_type} for {file_path.name}: {error.message}")
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            return [], error
    
    async def process_file_async(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single file asynchronously and return processing results.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict: Processing results including documents and statistics
        """
        try:
            # Process the file
            documents, error = self.process_file(file_path)
            
            # Handle processing errors
            if error:
                return {
                    "file_path": str(file_path),
                    "success": False,
                    "documents": [],
                    "error": error.message,
                    "error_type": error.error_type,
                    "file_name": Path(file_path).name
                }
            
            # Add to vector store
            try:
                await self.vector_store.aadd_documents(documents)
            except Exception as e:
                error = VectorStoreError(str(file_path), e)
                logger.error(f"{error.error_type} for {Path(file_path).name}: {error.message}")
                return {
                    "file_path": str(file_path),
                    "success": False,
                    "documents": [],
                    "error": f"Failed to add to vector store: {str(e)}",
                    "error_type": "vector_store_error",
                    "file_name": Path(file_path).name
                }
            
            # Get file type
            file_type = SUPPORTED_EXTENSIONS[Path(file_path).suffix.lower()]
            
            logger.info(f"Processed {Path(file_path).name}: {len(documents)} chunks")
            
            return {
                "file_path": str(file_path),
                "success": True,
                "documents": documents,
                "chunks": len(documents),
                "file_type": file_type,
                "file_name": Path(file_path).name
            }
        
        except Exception as e:
            logger.error(f"Unexpected error processing {Path(file_path).name}: {str(e)}")
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            return {
                "file_path": str(file_path),
                "success": False,
                "documents": [],
                "error": f"Unexpected error: {str(e)}",
                "error_type": "unexpected_error",
                "file_name": Path(file_path).name
            }
    
    async def process_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            file_extensions: List of file extensions to process (if None, process all supported types)
            parallel: Whether to process files in parallel
            
        Returns:
            Dict: Processing statistics
        """
        directory_path = Path(directory_path)
        
        # Validate directory path
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        # Validate file extensions
        if file_extensions:
            invalid_extensions = [ext for ext in file_extensions if not ext.startswith(".")]
            if invalid_extensions:
                raise ValueError(f"Invalid file extensions (must start with '.'): {invalid_extensions}")
        
        # Filter extensions if provided
        extensions = file_extensions or list(SUPPORTED_EXTENSIONS.keys())
        
        # Find all matching files
        all_files = []
        try:
            if recursive:
                for ext in extensions:
                    all_files.extend(directory_path.rglob(f"*{ext}"))
            else:
                for ext in extensions:
                    all_files.extend(directory_path.glob(f"*{ext}"))
        except Exception as e:
            raise ValueError(f"Error searching for files: {str(e)}")
        
        # Check if any files were found
        if not all_files:
            return {
                "total_files": 0,
                "processed_files": 0,
                "failed_files": 0,
                "total_chunks": 0,
                "file_types": {},
                "parallel_processing": parallel,
                "max_concurrency": self.max_concurrency if parallel else 1,
                "warning": f"No matching files found in {directory_path}" + 
                          (f" with extensions {extensions}" if file_extensions else "")
            }
        
        # Process statistics
        stats = {
            "total_files": len(all_files),
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "file_types": {},
            "error_types": {},
            "parallel_processing": parallel,
            "max_concurrency": self.max_concurrency if parallel else 1,
            "start_time": datetime.utcnow().isoformat(),
            "failed_files_details": []
        }
        
        try:
            if not parallel:
                # Process files sequentially
                for file_path in all_files:
                    result = await self.process_file_async(file_path)
                    self._update_stats(stats, result)
            else:
                # Process files in parallel with controlled concurrency
                # Split files into batches to control memory usage
                batches = [all_files[i:i + self.max_concurrency] for i in range(0, len(all_files), self.max_concurrency)]
                
                for batch in batches:
                    # Process batch concurrently
                    results = await asyncio.gather(*[self.process_file_async(file_path) for file_path in batch])
                    
                    # Update statistics
                    for result in results:
                        self._update_stats(stats, result)
        except Exception as e:
            logger.error(f"Error during batch processing: {str(e)}")
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            stats["batch_error"] = str(e)
        
        # Add end time
        stats["end_time"] = datetime.utcnow().isoformat()
        
        # Limit the number of failed file details to avoid huge responses
        if len(stats["failed_files_details"]) > 50:
            stats["failed_files_details"] = stats["failed_files_details"][:50]
            stats["failed_files_details_truncated"] = True
        
        return stats
    
    def _update_stats(self, stats: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Update processing statistics with file processing result.
        
        Args:
            stats: Statistics dictionary to update
            result: File processing result
        """
        if result.get("success", False):
            stats["processed_files"] += 1
            stats["total_chunks"] += result.get("chunks", 0)
            
            # Update file type statistics
            file_type = result.get("file_type", "Unknown")
            if file_type not in stats["file_types"]:
                stats["file_types"][file_type] = 0
            stats["file_types"][file_type] += 1
        else:
            stats["failed_files"] += 1
            
            # Track error types
            error_type = result.get("error_type", "unknown_error")
            if error_type not in stats["error_types"]:
                stats["error_types"][error_type] = 0
            stats["error_types"][error_type] += 1
            
            # Add to failed files details
            stats["failed_files_details"].append({
                "file_name": result.get("file_name", "Unknown"),
                "file_path": result.get("file_path", "Unknown"),
                "error": result.get("error", "Unknown error"),
                "error_type": error_type
            }) 