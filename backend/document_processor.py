"""
Document processor module for handling various document types using Docling.
Supports PDF, DOCX, PPTX, TXT, and other document formats.
"""

import os
import logging
import asyncio
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple

from docling import Document as DoclingDocument
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
    ".txt": "Text Document",
    ".md": "Markdown Document",
    ".csv": "CSV Document",
    ".xlsx": "Excel Spreadsheet",
    ".xls": "Excel Spreadsheet (Legacy)",
    ".html": "HTML Document",
    ".htm": "HTML Document",
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
    Uses Docling for document parsing and extraction.
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
        
        # Initialize Azure OpenAI embeddings
        try:
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=settings.azure_openai_embedding_deployment,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version
            )
            
            # Initialize Azure AI Search
            self.vector_store = AzureSearch(
                azure_search_endpoint=settings.azure_search_service_endpoint,
                azure_search_key=settings.azure_search_admin_key,
                index_name=settings.azure_search_index_name,
                embedding_function=self.embeddings,
            )
        except Exception as e:
            logger.error(f"Error initializing document processor: {str(e)}")
            raise ValueError(f"Failed to initialize document processor: {str(e)}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len
        )
    
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
    
    def classify_error(self, file_path: Union[str, Path], error: Exception) -> DocumentProcessingError:
        """
        Classify an error that occurred during document processing.
        
        Args:
            file_path: Path to the file
            error: The original exception
            
        Returns:
            DocumentProcessingError: A classified error
        """
        file_path_str = str(file_path)
        error_message = str(error).lower()
        
        # Check for file access errors
        if any(msg in error_message for msg in ["permission denied", "no such file", "access", "not found"]):
            return FileAccessError(file_path_str, error)
        
        # Check for parsing errors
        if any(msg in error_message for msg in ["parse", "decode", "syntax", "invalid", "corrupt"]):
            return ParsingError(file_path_str, error)
        
        # Check for vector store errors
        if any(msg in error_message for msg in ["vector", "embedding", "index", "azure", "search"]):
            return VectorStoreError(file_path_str, error)
        
        # Default to a generic document processing error
        return DocumentProcessingError(
            f"Error processing document: {error}",
            "unknown_error",
            file_path_str,
            error
        )
    
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
            # Use Docling to extract text and metadata
            docling_doc = DoclingDocument.from_file(str(file_path))
            
            # Create a LangChain document
            content = docling_doc.text
            
            # Check if content is empty
            if not content or content.isspace():
                error = EmptyContentError(str(file_path))
                logger.warning(f"{error.error_type}: {error.message} for {file_path.name}")
                return [], error
            
            # Extract metadata
            metadata = {
                "source": str(file_path.name),
                "file_path": str(file_path),
                "file_type": SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "Unknown"),
                "title": file_path.stem,
                "created_at": datetime.utcnow().isoformat(),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "processor_version": "2.0.0",  # Add version for tracking
            }
            
            # Add additional metadata from Docling if available
            if hasattr(docling_doc, "metadata") and docling_doc.metadata:
                for key, value in docling_doc.metadata.items():
                    # Convert non-string metadata to strings to ensure compatibility
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)
            
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
            file_type = SUPPORTED_EXTENSIONS.get(Path(file_path).suffix.lower(), "Unknown")
            
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
    
    async def process_uploaded_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process an uploaded file from memory.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Dict: Processing statistics
        """
        # Validate inputs
        if not file_content:
            return {
                "filename": filename,
                "error": "Empty file content",
                "error_type": "empty_file",
                "success": False
            }
        
        if not filename:
            return {
                "filename": "unknown",
                "error": "Missing filename",
                "error_type": "missing_filename",
                "success": False
            }
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            return {
                "filename": filename,
                "error": f"Unsupported file format: {file_ext}",
                "error_type": "unsupported_format",
                "success": False
            }
        
        # Create a temporary file
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / filename
        
        try:
            # Write the file content
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Process the file
            documents, error = self.process_file(file_path)
            
            stats = {
                "filename": filename,
                "chunks": len(documents),
                "success": False,
                "file_type": SUPPORTED_EXTENSIONS.get(file_ext, "Unknown")
            }
            
            if error:
                stats["error"] = error.message
                stats["error_type"] = error.error_type
                logger.warning(f"Error processing uploaded file {filename}: {error.message}")
                return stats
            
            if documents:
                try:
                    # Add to vector store
                    await self.vector_store.aadd_documents(documents)
                    stats["success"] = True
                    logger.info(f"Processed uploaded file {filename}: {len(documents)} chunks")
                except Exception as e:
                    stats["error"] = f"Failed to add to vector store: {str(e)}"
                    stats["error_type"] = "vector_store_error"
                    logger.error(f"Vector store error for {filename}: {str(e)}")
            else:
                stats["error"] = "No content extracted"
                stats["error_type"] = "empty_content"
                logger.warning(f"No content extracted from uploaded file {filename}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to process uploaded file {filename}: {str(e)}")
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            return {
                "filename": filename,
                "error": str(e),
                "error_type": "processing_error",
                "success": False
            }
        finally:
            # Clean up the temporary file
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")
    
    async def process_uploaded_files_parallel(
        self, 
        files: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple uploaded files in parallel.
        
        Args:
            files: List of dictionaries containing file content and filename
                  Each dict should have 'content' and 'filename' keys
            
        Returns:
            List[Dict]: Processing results for each file
        """
        # Validate input
        if not files:
            return [{
                "error": "No files provided",
                "error_type": "no_files",
                "success": False
            }]
        
        # Validate each file entry
        valid_files = []
        invalid_results = []
        
        for i, file_data in enumerate(files):
            if not isinstance(file_data, dict):
                invalid_results.append({
                    "index": i,
                    "error": "Invalid file data format",
                    "error_type": "invalid_format",
                    "success": False
                })
                continue
                
            if 'content' not in file_data or not file_data['content']:
                invalid_results.append({
                    "index": i,
                    "error": "Missing or empty file content",
                    "error_type": "empty_content",
                    "success": False
                })
                continue
                
            if 'filename' not in file_data or not file_data['filename']:
                invalid_results.append({
                    "index": i,
                    "error": "Missing filename",
                    "error_type": "missing_filename",
                    "success": False
                })
                continue
                
            valid_files.append(file_data)
        
        if not valid_files:
            return invalid_results
        
        # Process files in parallel with controlled concurrency
        # Split files into batches to control memory usage
        batches = [valid_files[i:i + self.max_concurrency] for i in range(0, len(valid_files), self.max_concurrency)]
        
        all_results = []
        all_results.extend(invalid_results)  # Add invalid file results
        
        try:
            for batch in batches:
                # Process batch concurrently
                batch_results = await asyncio.gather(*[
                    self.process_uploaded_file(
                        file_content=file_data['content'],
                        filename=file_data['filename']
                    ) 
                    for file_data in batch
                ], return_exceptions=True)
                
                # Handle any exceptions from gather
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error in batch processing: {str(result)}")
                        all_results.append({
                            "filename": batch[i].get('filename', 'unknown'),
                            "error": f"Unexpected error: {str(result)}",
                            "error_type": "batch_processing_error",
                            "success": False
                        })
                    else:
                        all_results.append(result)
        except Exception as e:
            logger.error(f"Critical error in parallel processing: {str(e)}")
            logger.debug(f"Detailed error: {traceback.format_exc()}")
            all_results.append({
                "error": f"Critical error in parallel processing: {str(e)}",
                "error_type": "critical_error",
                "success": False
            })
        
        return all_results 