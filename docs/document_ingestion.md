# Document Ingestion with Docling

This document provides instructions for using the document ingestion functionality in the chat-langchain application. The system uses [Docling](https://github.com/doclingjs/docling) to handle multiple document types including PDF, DOCX, PPTX, and more.

## Supported Document Types

The following document types are supported:

- PDF Documents (`.pdf`)
- Word Documents (`.docx`, `.doc`)
- PowerPoint Presentations (`.pptx`, `.ppt`)
- Excel Spreadsheets (`.xlsx`, `.xls`)
- Rich Text Format (`.rtf`)

> Note: Text Documents (`.txt`), Markdown Documents (`.md`), CSV Documents (`.csv`), and HTML Documents (`.html`, `.htm`) are not currently supported by the implementation.

## API Endpoints

### 1. Ingest Documents from a Directory

```http
POST /ingest/documents
```

Parameters:
- `folder_path` (string, required): Path to the directory containing documents
- `recursive` (boolean, optional, default: true): Whether to search subdirectories
- `chunk_size` (integer, optional, default: 1000, max: 10000): Size of text chunks for splitting
- `chunk_overlap` (integer, optional, default: 200, max: 5000): Overlap between chunks
- `file_extensions` (array of strings, optional): List of file extensions to process (if not provided, all supported types will be processed)
- `parallel` (boolean, optional, default: true): Whether to process files in parallel
- `max_concurrency` (integer, optional, default: 5, max: 20): Maximum number of files to process concurrently

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

Response codes:
- `200 OK`: Documents processed successfully
- `207 Multi-Status`: Some documents processed successfully, some failed
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Directory not found
- `415 Unsupported Media Type`: Unsupported file format
- `500 Internal Server Error`: Server error

### 2. Upload a Single Document

```http
POST /ingest/upload
```

This endpoint accepts multipart form data:
- `file` (file, required): The document file to upload
- `chunk_size` (integer, optional, default: 1000, max: 10000): Size of text chunks for splitting
- `chunk_overlap` (integer, optional, default: 200, max: 5000): Overlap between chunks

Response codes:
- `200 OK`: Document processed successfully
- `400 Bad Request`: Invalid parameters or empty file
- `415 Unsupported Media Type`: Unsupported file format
- `422 Unprocessable Entity`: Document processing failed
- `500 Internal Server Error`: Server error

### 3. Upload Multiple Documents

```http
POST /ingest/upload-batch
```

This endpoint accepts multipart form data:
- `files` (array of files, required): The document files to upload
- `chunk_size` (integer, optional, default: 1000, max: 10000): Size of text chunks for splitting
- `chunk_overlap` (integer, optional, default: 200, max: 5000): Overlap between chunks
- `parallel` (boolean, optional, default: true): Whether to process files in parallel
- `max_concurrency` (integer, optional, default: 5, max: 20): Maximum number of files to process concurrently

Response codes:
- `200 OK`: All documents processed successfully
- `207 Multi-Status`: Some documents processed successfully, some failed
- `400 Bad Request`: Invalid parameters or empty files
- `415 Unsupported Media Type`: Unsupported file format
- `422 Unprocessable Entity`: All document processing failed
- `500 Internal Server Error`: Server error

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

## Error Handling

The document ingestion system provides detailed error information to help diagnose and resolve issues:

### Error Types

The system classifies errors into specific categories:

- `unsupported_format`: The file format is not supported
- `file_access_error`: Cannot access the file (permissions, not found)
- `parsing_error`: Error parsing the document content
- `empty_content`: No content could be extracted from the document
- `vector_store_error`: Error adding document to the vector store
- `empty_file`: The file is empty
- `missing_filename`: No filename provided
- `batch_processing_error`: Error during batch processing
- `critical_error`: Critical system error

### Response Format

When errors occur, the API returns detailed information:

```json
{
  "status": "error",
  "message": "Failed to process file: [error message]",
  "error_type": "parsing_error"
}
```

For batch processing, the response includes details for each file:

```json
{
  "status": "partial_success",
  "message": "Processed 3 of 5 files successfully",
  "results": [
    {
      "filename": "document1.pdf",
      "success": true,
      "chunks": 15
    },
    {
      "filename": "document2.docx",
      "success": false,
      "error": "Error parsing document",
      "error_type": "parsing_error"
    },
    ...
  ]
}
```

### HTTP Status Codes

The API uses appropriate HTTP status codes to indicate different error conditions:

- `400 Bad Request`: Invalid parameters or input
- `404 Not Found`: Directory or file not found
- `415 Unsupported Media Type`: Unsupported file format
- `422 Unprocessable Entity`: Document processing failed
- `207 Multi-Status`: Partial success (some files processed, some failed)
- `500 Internal Server Error`: Server-side error

## Input Validation

The API performs validation on all inputs:

- **File Extensions**: Must start with a period (e.g., `.pdf`)
- **Chunk Size**: Must be between 1 and 10,000
- **Chunk Overlap**: Must be between 0 and 5,000 and less than chunk size
- **Max Concurrency**: Must be between 1 and 20
- **File Content**: Files must not be empty
- **Directory Path**: Must exist and be a directory

## Customization

You can customize the document processing by modifying the following:

- Chunk size and overlap: Adjust these parameters to control how documents are split.
- Text splitter separators: Modify the separators used for splitting text.
- Supported file extensions: Add or remove file extensions in the `SUPPORTED_EXTENSIONS` dictionary in `document_processor.py`.
- Concurrency settings: Adjust the `max_concurrency` parameter to control parallel processing.

## Troubleshooting

If you encounter issues with document ingestion:

1. **Check API Response**: The API provides detailed error information to help diagnose issues.
2. **Verify File Format**: Ensure the document format is supported and the file is not corrupted.
3. **Check File Permissions**: Ensure the application has permission to access the files.
4. **Empty Content**: If no content is extracted, check if the document is password-protected or contains only images.
5. **Vector Store Errors**: Verify that Azure AI Search is properly configured and accessible.
6. **Parallel Processing Issues**: If parallel processing causes problems, try disabling it by setting `parallel=false`.
7. **Memory Issues**: For large documents, reduce `max_concurrency` to prevent memory exhaustion.
8. **Check Logs**: Application logs contain detailed error information, including stack traces.

### Common Error Messages and Solutions

| Error Type | Possible Cause | Solution |
|------------|----------------|----------|
| `unsupported_format` | File extension not supported | Use a supported file format |
| `file_access_error` | Permission denied or file not found | Check file permissions and path |
| `parsing_error` | Corrupted or invalid document | Verify the document can be opened normally |
| `empty_content` | No text in document or password-protected | Check if document is protected or contains only images |
| `vector_store_error` | Azure AI Search configuration issue | Verify Azure credentials and connectivity |

For more information, refer to the [Docling documentation](https://github.com/doclingjs/docling). 