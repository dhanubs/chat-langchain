"""
Script to download Docling models for offline use.
This script downloads all necessary models to process PDF, Word, PowerPoint, Excel, and RTF documents.

According to the official documentation:
- For processing PDF documents, Docling requires model weights from https://huggingface.co/ds4sd/docling-models
- Models are stored in $HOME/.cache/docling/models by default
- The script mimics the 'docling-tools models download' utility

Additional models included:
- sentence-transformers/all-MiniLM-L6-v2: Sentence transformer model for document embeddings
"""

import os
import requests
import sys
import time
from pathlib import Path

def get_token():
    """Get Hugging Face token from environment variable or user input."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nYou need a Hugging Face access token to download the models.")
        print("1. Create an account at https://huggingface.co/ if you don't have one")
        print("2. Go to https://huggingface.co/settings/tokens")
        print("3. Create a new token and paste it below\n")
        token = input("Enter your Hugging Face token: ").strip()
        # Validate that token is not empty
        if not token:
            print("Error: Token cannot be empty. Please run the script again and provide a valid token.")
            sys.exit(1)
    return token

# Models required for document processing
MODELS = {
    # Layout model for document layout analysis
    "layout": {
        "base_url": "https://huggingface.co/ds4sd/docling-models/resolve/main/model_artifacts",
        "files": [
            "layout/config.json",
            "layout/pytorch_model.bin",
            "layout/preprocessor_config.json"
        ],
        "target_dir": "layout"
    },
    # TableFormer model for table extraction
    "tableformer": {
        "base_url": "https://huggingface.co/ds4sd/docling-models/resolve/main/model_artifacts",
        "files": [
            "tableformer/config.json",
            "tableformer/pytorch_model.bin",
            "tableformer/preprocessor_config.json"
        ],
        "target_dir": "tableformer"
    },
    # Picture classifier model
    "picture_classifier": {
        "base_url": "https://huggingface.co/ds4sd/docling-models/resolve/main/model_artifacts",
        "files": [
            "picture_classifier/config.json",
            "picture_classifier/pytorch_model.bin",
            "picture_classifier/preprocessor_config.json"
        ],
        "target_dir": "picture_classifier"
    },
    # Code formula model
    "code_formula": {
        "base_url": "https://huggingface.co/ds4sd/docling-models/resolve/main/model_artifacts",
        "files": [
            "code_formula/config.json",
            "code_formula/pytorch_model.bin",
            "code_formula/preprocessor_config.json"
        ],
        "target_dir": "code_formula"
    },
    # Sentence Transformer for embeddings (additional model)
    "sentence_transformer": {
        "base_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main",
        "files": [
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt",
            "modules.json",
            "1_Pooling/config.json",
            "0_Transformer/config.json",
            "README.md",
            "sentence_bert_config.json"
        ],
        "target_dir": "models--sentence-transformers--all-MiniLM-L6-v2/snapshots/latest"
    }
}

def download_file(url: str, target_path: Path, description: str, token: str):
    """Download a file with progress indication."""
    print(f"Downloading {description}...")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Add retry logic
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Ensure parent directories exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            with open(target_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for data in response.iter_content(chunk_size=4096):
                        downloaded += len(data)
                        f.write(data)
                        done = int(50 * downloaded / total_size)
                        percent = int(100 * downloaded / total_size)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {percent}%")
                        sys.stdout.flush()
            
            print()  # New line after progress bar
            
            # If we get here, download was successful
            return
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\nAttempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"\nFailed after {max_retries} attempts: {str(e)}")
                raise

def main():
    # Get authentication token
    token = get_token()
    
    # Get user's home directory in a cross-platform way
    home_dir = Path.home()
    # Use the standard Docling cache directory
    cache_dir = home_dir / ".cache" / "docling" / "models"
    
    print(f"Creating Docling models directory at: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Track overall progress
    total_models = len(MODELS)
    completed_models = 0
    failed_models = []
    
    # Download each model
    for model_name, model_info in MODELS.items():
        print(f"\n[{completed_models + 1}/{total_models}] Downloading {model_name} model...")
        
        if model_name == "sentence_transformer":
            # For sentence transformer, use the huggingface hub structure
            hf_cache_dir = home_dir / ".cache" / "huggingface" / "hub"
            model_dir = hf_cache_dir / model_info["target_dir"]
        else:
            # For Docling models, use the Docling cache structure
            model_dir = cache_dir / model_info["target_dir"]
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Track files successfully downloaded for this model
        files_total = len(model_info["files"])
        files_downloaded = 0
        
        # Download each file for this model
        for file_path in model_info["files"]:
            url = f"{model_info['base_url']}/{file_path}"
            
            if model_name == "sentence_transformer":
                target_path = model_dir / Path(file_path).name
            else:
                # For Docling models, strip the model name from the path
                file_name = Path(file_path).name
                target_path = model_dir / file_name
            
            # Ensure parent directories exist for nested paths
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                download_file(url, target_path, f"{model_name}/{file_path}", token)
                files_downloaded += 1
            except Exception as e:
                print(f"Error downloading {file_path}: {str(e)}")
                print("Continuing with other files...")
                continue
        
        # For sentence transformer, create the refs directory
        if model_name == "sentence_transformer":
            model_base_dir = model_dir.parent.parent
            refs_dir = model_base_dir / "refs"
            refs_dir.mkdir(exist_ok=True)
            
            with open(refs_dir / "main", "w") as f:
                f.write("latest")
        
        # Check if we downloaded at least some files for this model
        if files_downloaded > 0:
            print(f"Downloaded {files_downloaded}/{files_total} files for {model_name} model.")
            completed_models += 1
        else:
            print(f"Failed to download any files for {model_name} model.")
            failed_models.append(model_name)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Download summary: {completed_models}/{total_models} models downloaded successfully.")
    
    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")
    
    # Print instructions for using the downloaded models
    print("\nTo use these models with Docling in offline mode:")
    print("1. Set the following environment variables:")
    print("   - HF_HUB_OFFLINE=1")
    print("   - DOCLING_OFFLINE=1")
    print("2. When initializing DocumentConverter, specify the artifacts path:")
    print(f"   pipeline_options = PdfPipelineOptions(artifacts_path=\"{cache_dir}\")")
    print("   converter = DocumentConverter(format_options={")
    print("       InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)")
    print("   })")
    
    print("\nModels are stored in:")
    print(f"- Docling models: {cache_dir}")
    print(f"- Sentence transformer: {home_dir / '.cache' / 'huggingface' / 'hub' / 'models--sentence-transformers--all-MiniLM-L6-v2'}")

if __name__ == "__main__":
    main() 