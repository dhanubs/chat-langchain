"""
Document processor module for handling various document types using Docling.
Supports PDF, DOCX, PPTX, TXT, and other document formats.
"""

import os
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

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
    
    def process_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Process a single file and convert it to LangChain documents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List[Document]: List of LangChain documents
        """
        file_path = Path(file_path)
        
        if not self.is_supported_file(file_path):
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []
        
        try:
            # Use Docling to extract text and metadata
            docling_doc = DoclingDocument.from_file(str(file_path))
            
            # Create a LangChain document
            content = docling_doc.text
            
            # Extract metadata
            metadata = {
                "source": str(file_path.name),
                "file_path": str(file_path),
                "file_type": SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "Unknown"),
                "title": file_path.stem,
                "created_at": datetime.utcnow().isoformat(),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
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
            return self.text_splitter.split_documents([doc])
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    async def process_file_async(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single file asynchronously and return processing results.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict: Processing results including documents and statistics
        """
        try:
            # Skip unsupported files
            if not self.is_supported_file(file_path):
                return {
                    "file_path": str(file_path),
                    "success": False,
                    "documents": [],
                    "error": "Unsupported file type"
                }
            
            # Process the file
            documents = self.process_file(file_path)
            
            if documents:
                # Add to vector store
                await self.vector_store.aadd_documents(documents)
                
                # Get file type
                file_type = SUPPORTED_EXTENSIONS.get(Path(file_path).suffix.lower(), "Unknown")
                
                logger.info(f"Processed {Path(file_path).name}: {len(documents)} chunks")
                
                return {
                    "file_path": str(file_path),
                    "success": True,
                    "documents": documents,
                    "chunks": len(documents),
                    "file_type": file_type
                }
            else:
                logger.warning(f"No content extracted from {Path(file_path).name}")
                return {
                    "file_path": str(file_path),
                    "success": False,
                    "documents": [],
                    "error": "No content extracted"
                }
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to process {Path(file_path).name}: {error_msg}")
            return {
                "file_path": str(file_path),
                "success": False,
                "documents": [],
                "error": error_msg
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
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Filter extensions if provided
        extensions = file_extensions or list(SUPPORTED_EXTENSIONS.keys())
        
        # Find all matching files
        all_files = []
        if recursive:
            for ext in extensions:
                all_files.extend(directory_path.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                all_files.extend(directory_path.glob(f"*{ext}"))
        
        # Process statistics
        stats = {
            "total_files": len(all_files),
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "file_types": {},
            "parallel_processing": parallel,
            "max_concurrency": self.max_concurrency if parallel else 1,
        }
        
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
    
    async def process_uploaded_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Process an uploaded file from memory.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Dict: Processing statistics
        """
        # Create a temporary file
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / filename
        
        try:
            # Write the file content
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Process the file
            documents = self.process_file(file_path)
            
            stats = {
                "filename": filename,
                "chunks": len(documents),
                "success": False
            }
            
            if documents:
                # Add to vector store
                await self.vector_store.aadd_documents(documents)
                stats["success"] = True
                logger.info(f"Processed uploaded file {filename}: {len(documents)} chunks")
            else:
                logger.warning(f"No content extracted from uploaded file {filename}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to process uploaded file {filename}: {str(e)}")
            return {
                "filename": filename,
                "error": str(e),
                "success": False
            }
        finally:
            # Clean up the temporary file
            if file_path.exists():
                file_path.unlink()
    
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
        # Process files in parallel with controlled concurrency
        # Split files into batches to control memory usage
        batches = [files[i:i + self.max_concurrency] for i in range(0, len(files), self.max_concurrency)]
        
        all_results = []
        
        for batch in batches:
            # Process batch concurrently
            batch_results = await asyncio.gather(*[
                self.process_uploaded_file(
                    file_content=file_data['content'],
                    filename=file_data['filename']
                ) 
                for file_data in batch
            ])
            
            all_results.extend(batch_results)
        
        return all_results 