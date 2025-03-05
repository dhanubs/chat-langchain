"""
Script to download Docling models for offline use.
This script downloads all necessary models to process PDF, Word, PowerPoint, Excel, and RTF documents.
"""

import os
import requests
from pathlib import Path
import json
import sys
import time

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
    # LayoutLM model for document layout analysis (primarily for PDFs)
    "layoutlm": {
        "base_url": "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main",
        "files": [
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt"
        ],
        "target_dir": "models--microsoft--layoutlm-base-uncased/snapshots/latest"
    },
    # OCR model for text extraction from images in PDFs
    "ocr": {
        "base_url": "https://huggingface.co/microsoft/trocr-base-printed/resolve/main",
        "files": [
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "generation_config.json"
        ],
        "target_dir": "models--microsoft--trocr-base-printed/snapshots/latest"
    },
    # Alternative OCR model (Tesseract-based)
    "alt_ocr": {
        "base_url": "https://huggingface.co/microsoft/dit-base/resolve/main",
        "files": [
            "config.json",
            "pytorch_model.bin",
            "preprocessor_config.json"
        ],
        "target_dir": "models--microsoft--dit-base/snapshots/latest"
    },
    # Table extraction model for Excel, Word tables, and PDF tables
    "table": {
        "base_url": "https://huggingface.co/microsoft/table-transformer-detection/resolve/main",
        "files": [
            "config.json",
            "pytorch_model.bin",
            "preprocessor_config.json"
        ],
        "target_dir": "models--microsoft--table-transformer-detection/snapshots/latest"
    },
    # Document understanding model for Word, PowerPoint, and RTF
    "donut": {
        "base_url": "https://huggingface.co/naver-clova-ix/donut-base/resolve/main",
        "files": [
            "config.json",
            "pytorch_model.bin",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json"
        ],
        "target_dir": "models--naver-clova-ix--donut-base/snapshots/latest"
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
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            # Calculate progress
                            progress = int(50 * downloaded / total_size)
                            print(f"\r[{'=' * progress}{' ' * (50-progress)}] {downloaded}/{total_size} bytes", end='')
            print(f"\nSaved to {target_path}")
            
            # Verify file was downloaded correctly
            if not target_path.exists():
                raise FileNotFoundError(f"Failed to save file to {target_path}")
            
            # Verify file size if we know the expected size
            if total_size > 0 and target_path.stat().st_size != total_size:
                raise ValueError(f"Downloaded file size ({target_path.stat().st_size} bytes) doesn't match expected size ({total_size} bytes)")
                
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
    cache_dir = home_dir / ".cache" / "huggingface" / "hub"
    
    print(f"Creating base cache directory at: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Track overall progress
    total_models = len(MODELS)
    completed_models = 0
    failed_models = []
    
    # Download each model
    for model_name, model_info in MODELS.items():
        print(f"\n[{completed_models + 1}/{total_models}] Downloading {model_name} model...")
        
        model_dir = cache_dir / model_info["target_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Track files successfully downloaded for this model
        files_total = len(model_info["files"])
        files_downloaded = 0
        
        # Download each file for this model
        for file_path in model_info["files"]:
            url = f"{model_info['base_url']}/{file_path}"
            target_path = model_dir / file_path
            try:
                download_file(url, target_path, f"{model_name}/{file_path}", token)
                files_downloaded += 1
            except Exception as e:
                print(f"Error downloading {file_path}: {str(e)}")
                print("Continuing with other files...")
                continue
        
        # Create the refs directory and save the version reference for this model
        model_base_dir = model_dir.parent.parent
        refs_dir = model_base_dir / "refs"
        refs_dir.mkdir(exist_ok=True)
        
        with open(refs_dir / "main", "w") as f:
            f.write("latest")
        
        # Check if we downloaded at least some files for this model
        if files_downloaded > 0:
            completed_models += 1
            print(f"Completed downloading {model_name} model ({files_downloaded}/{files_total} files)")
            
            # If we didn't get all files, add to partial failures
            if files_downloaded < files_total:
                failed_models.append(f"{model_name} (partial: {files_downloaded}/{files_total} files)")
        else:
            failed_models.append(f"{model_name} (complete failure)")
            print(f"Failed to download any files for {model_name} model")
    
    print("\n=== Download Summary ===")
    print(f"Successfully downloaded {completed_models}/{total_models} models")
    
    if failed_models:
        print("\nThe following models had issues:")
        for model in failed_models:
            print(f"- {model}")
        print("\nYou may need to manually download these models or try again later.")
    
    print("\nDownload complete! Models are ready for offline use.")
    print(f"Models are stored in: {cache_dir}")
    print("\nTo use Docling in offline mode, set these environment variables:")
    print("set DOCLING_OFFLINE=1")
    print("set HF_HUB_OFFLINE=1")

if __name__ == "__main__":
    main() 