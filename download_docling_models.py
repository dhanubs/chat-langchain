"""
Script to download Docling models for offline use.
"""

import os
import requests
from pathlib import Path
import json

def get_token():
    """Get Hugging Face token from environment variable or user input."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nYou need a Hugging Face access token to download the models.")
        print("1. Create an account at https://huggingface.co/ if you don't have one")
        print("2. Go to https://huggingface.co/settings/tokens")
        print("3. Create a new token and paste it below\n")
        token = input("Enter your Hugging Face token: ").strip()
    return token

# Base URL for the Docling models on Hugging Face
BASE_URL = "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main"

# Model files to download
MODEL_FILES = [
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt"
]

def download_file(url: str, target_path: Path, description: str, token: str):
    """Download a file with progress indication."""
    print(f"Downloading {description}...")
    headers = {"Authorization": f"Bearer {token}"}
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

def main():
    # Get authentication token
    token = get_token()
    
    # Get user's home directory in a cross-platform way
    home_dir = Path.home()
    cache_dir = home_dir / ".cache" / "huggingface" / "hub" / "models--microsoft--layoutlm-base-uncased" / "snapshots" / "latest"
    
    print(f"Creating directories at: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each model file
    for file_path in MODEL_FILES:
        url = f"{BASE_URL}/{file_path}"
        target_path = cache_dir / file_path
        try:
            download_file(url, target_path, file_path, token)
        except Exception as e:
            print(f"Error downloading {file_path}: {str(e)}")
            continue
    
    # Create the refs directory and save the version reference
    refs_dir = cache_dir.parent.parent / "refs"
    refs_dir.mkdir(exist_ok=True)
    
    with open(refs_dir / "main", "w") as f:
        f.write("latest")
    
    print("\nDownload complete! Models are ready for offline use.")
    print(f"Models are stored in: {cache_dir}")
    print("\nTo use Docling in offline mode, set these environment variables:")
    print("set DOCLING_OFFLINE=1")
    print("set HF_HUB_OFFLINE=1")

if __name__ == "__main__":
    main() 