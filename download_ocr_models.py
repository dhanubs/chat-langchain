"""
Script to download EasyOCR models for offline use.
This script downloads all necessary models to process text from images in PDFs.
Models will be downloaded to a specified directory that can be copied to the offline server.
"""

import os
import sys
from pathlib import Path
import easyocr
import shutil

def download_models(target_dir: str = None):
    """
    Download EasyOCR models to a specified directory.
    
    Args:
        target_dir: Directory to store the models. If None, uses default ~/.EasyOCR/
    """
    print("Downloading EasyOCR models...")
    
    # Initialize EasyOCR to trigger model download
    if target_dir:
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=target_dir)
        print(f"Models downloaded to custom directory: {target_dir}")
    else:
        # Download to default location (~/.EasyOCR/)
        reader = easyocr.Reader(['en'], gpu=False)
        default_dir = Path.home() / '.EasyOCR'
        print(f"Models downloaded to default directory: {default_dir}")

def main():
    # Get target directory from command line argument
    target_dir = None
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    
    try:
        download_models(target_dir)
        print("\nModel download complete!")
        print("\nTo use these models in offline mode:")
        print("1. Copy the entire model directory to your offline server")
        print("2. When initializing DocumentProcessor, specify the model path:")
        print("   processor = DocumentProcessor(")
        print("       enable_ocr=True,")
        print("       ocr_model_path='/path/to/models'")
        print("   )")
        
    except Exception as e:
        print(f"Error downloading models: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 