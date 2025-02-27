# Document Ingestion with Docling

This document provides instructions for using the document ingestion functionality in the chat-langchain application. The system uses [Docling](https://github.com/doclingjs/docling) to handle multiple document types including PDF, DOCX, PPTX, and more.

## Supported Document Types

The following document types are supported:

- PDF Documents (`.pdf`)
- Word Documents (`.docx`, `.doc`)
- PowerPoint Presentations (`.pptx`, `.ppt`)
- Text Documents (`.txt`)
- Markdown Documents (`.md`)
- CSV Documents (`.csv`)
- Excel Spreadsheets (`.xlsx`, `.xls`)
- HTML Documents (`.html`, `.htm`)
- Rich Text Format (`.rtf`)

## API Endpoints

### 1. Ingest Documents from a Directory

```http
POST /ingest/documents
```

Parameters:
- `folder_path` (string, required): Path to the directory containing documents
- `recursive` (boolean, optional, default: true): Whether to search subdirectories
- `chunk_size` (integer, optional, default: 1000): Size of text chunks for splitting
- `chunk_overlap` (integer, optional, default: 200): Overlap between chunks
- `file_extensions` (array of strings, optional): List of file extensions to process (if not provided, all supported types will be processed)
- `parallel` (boolean, optional, default: true): Whether to process files in parallel
- `max_concurrency` (integer, optional, default: 5): Maximum number of files to process concurrently

Example request:
```json
{
  "folder_path": "/path/to/documents",
  "recursive": true,
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "file_extensions": [".pdf", ".docx"],
  "parallel": true,
  "max_concurrency": 5
}
```

### 2. Upload a Single Document

```http
POST /ingest/upload
```

This endpoint accepts multipart form data:
- `file` (file, required): The document file to upload
- `chunk_size` (integer, optional, default: 1000): Size of text chunks for splitting
- `chunk_overlap` (integer, optional, default: 200): Overlap between chunks

### 3. Upload Multiple Documents

```http
POST /ingest/upload-batch
```

This endpoint accepts multipart form data:
- `files` (array of files, required): The document files to upload
- `chunk_size` (integer, optional, default: 1000): Size of text chunks for splitting
- `chunk_overlap` (integer, optional, default: 200): Overlap between chunks
- `parallel` (boolean, optional, default: true): Whether to process files in parallel
- `max_concurrency` (integer, optional, default: 5): Maximum number of files to process concurrently

## Usage Examples

### Using cURL

#### Ingest documents from a directory:

```bash
curl -X POST "http://localhost:8000/ingest/documents" \
  -H "Content-Type: application/json" \
  -d '{"folder_path": "/path/to/documents", "recursive": true, "parallel": true, "max_concurrency": 5}'
```

#### Upload a single document:

```bash
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "file=@/path/to/document.pdf" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200"
```

#### Upload multiple documents:

```bash
curl -X POST "http://localhost:8000/ingest/upload-batch" \
  -F "files=@/path/to/document1.pdf" \
  -F "files=@/path/to/document2.docx" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=200" \
  -F "parallel=true" \
  -F "max_concurrency=5"
```

### Using Python Requests

#### Ingest documents from a directory:

```python
import requests

url = "http://localhost:8000/ingest/documents"
payload = {
    "folder_path": "/path/to/documents",
    "recursive": True,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "parallel": True,
    "max_concurrency": 5
}
response = requests.post(url, json=payload)
print(response.json())
```

#### Upload a single document:

```python
import requests

url = "http://localhost:8000/ingest/upload"
files = {"file": open("/path/to/document.pdf", "rb")}
data = {"chunk_size": 1000, "chunk_overlap": 200}
response = requests.post(url, files=files, data=data)
print(response.json())
```

#### Upload multiple documents:

```python
import requests

url = "http://localhost:8000/ingest/upload-batch"
files = [
    ("files", open("/path/to/document1.pdf", "rb")),
    ("files", open("/path/to/document2.docx", "rb"))
]
data = {
    "chunk_size": 1000, 
    "chunk_overlap": 200,
    "parallel": True,
    "max_concurrency": 5
}
response = requests.post(url, files=files, data=data)
print(response.json())
```

## Processing Flow

1. When a document is ingested, it is processed using Docling to extract text and metadata.
2. The extracted text is split into chunks using RecursiveCharacterTextSplitter.
3. Each chunk is stored in Azure AI Search with appropriate metadata.
4. The chunks can then be retrieved during chat sessions using vector search.

## Parallel Processing

The document ingestion system supports parallel processing to improve performance when dealing with large numbers of files:

- **Directory Ingestion**: When processing a directory, files can be processed in parallel using the `parallel` and `max_concurrency` parameters.
- **Batch Upload**: When uploading multiple files, they can be processed in parallel using the same parameters.

Benefits of parallel processing:
- Significantly faster processing for large document collections
- Better utilization of system resources
- Controlled concurrency to prevent overwhelming the system

To optimize performance:
- For small numbers of files (< 5), parallel processing may not provide significant benefits
- For large numbers of files, adjust `max_concurrency` based on your system's capabilities
- A good starting point is 5-10 concurrent files, but this can be increased for systems with more resources

## Customization

You can customize the document processing by modifying the following:

- Chunk size and overlap: Adjust these parameters to control how documents are split.
- Text splitter separators: Modify the separators used for splitting text.
- Supported file extensions: Add or remove file extensions in the `SUPPORTED_EXTENSIONS` dictionary in `document_processor.py`.
- Concurrency settings: Adjust the `max_concurrency` parameter to control parallel processing.

## Troubleshooting

If you encounter issues with document ingestion:

1. Check that the document format is supported.
2. Ensure the document is not corrupted or password-protected.
3. Check the application logs for detailed error messages.
4. Verify that Azure AI Search is properly configured and accessible.
5. If parallel processing causes issues, try disabling it by setting `parallel=false`.

For more information, refer to the [Docling documentation](https://github.com/doclingjs/docling). 