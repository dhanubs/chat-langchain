"""
Document processor module for handling various document types using Docling.
Supports PDF, DOCX, PPTX, TXT, and other document formats.
"""

import os
import logging
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
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            separators: Custom separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
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
    
    async def process_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            file_extensions: List of file extensions to process (if None, process all supported types)
            
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
        }
        
        # Process each file
        for file_path in all_files:
            try:
                # Skip unsupported files
                if not self.is_supported_file(file_path):
                    continue
                
                # Process the file
                documents = self.process_file(file_path)
                
                if documents:
                    # Add to vector store
                    await self.vector_store.aadd_documents(documents)
                    
                    # Update statistics
                    stats["processed_files"] += 1
                    stats["total_chunks"] += len(documents)
                    
                    # Update file type statistics
                    file_type = SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "Unknown")
                    if file_type not in stats["file_types"]:
                        stats["file_types"][file_type] = 0
                    stats["file_types"][file_type] += 1
                    
                    logger.info(f"Processed {file_path.name}: {len(documents)} chunks")
                else:
                    stats["failed_files"] += 1
                    logger.warning(f"No content extracted from {file_path.name}")
            
            except Exception as e:
                stats["failed_files"] += 1
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
        
        return stats
    
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