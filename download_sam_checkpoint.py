#!/usr/bin/env python3
"""
Quick SAM Checkpoint Downloader
Downloads SAM pretrained checkpoints using Python urllib

Usage: python download_sam_checkpoint.py [vit_b|vit_l|vit_h]
"""

import os
import sys
import urllib.request
from pathlib import Path

def download_file(url, filename):
    """Download file with progress"""
    print(f"Downloading: {url}")
    print(f"Saving as: {filename}")
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, block_num * block_size * 100 / total_size)
            print(f"\rProgress: {percent:.1f}% ({block_num * block_size / 1024 / 1024:.1f} MB)", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✓ Download completed: {filename}")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

def main():
    # Checkpoint URLs
    checkpoint_urls = {
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    }
    
    # Model sizes
    model_sizes = {
        'vit_b': '358 MB',
        'vit_l': '1.2 GB', 
        'vit_h': '2.4 GB'
    }
    
    # Parse arguments
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
        if model_type not in checkpoint_urls:
            print(f"Error: Invalid model type '{model_type}'")
            print("Available options: vit_b, vit_l, vit_h")
            sys.exit(1)
        models_to_download = [model_type]
    else:
        # Default: download vit_b
        models_to_download = ['vit_b']
        print("No model specified, downloading vit_b by default")
        print("Usage: python download_sam_checkpoint.py [vit_b|vit_l|vit_h]")
        print()
    
    # Download checkpoints
    for model_type in models_to_download:
        filename = f"sam_{model_type}.pth"
        
        if os.path.exists(filename):
            print(f"✓ {filename} already exists, skipping download")
            continue
        
        print(f"Downloading SAM {model_type.upper()} checkpoint ({model_sizes[model_type]})...")
        
        success = download_file(checkpoint_urls[model_type], filename)
        
        if success:
            # Verify file size
            file_size = os.path.getsize(filename) / 1024 / 1024  # MB
            print(f"File size: {file_size:.1f} MB")
            
            if file_size < 10:  # If file is too small, probably failed
                print("⚠ Warning: Downloaded file seems too small, might be corrupted")
                os.remove(filename)
            else:
                print(f"✓ Successfully downloaded {filename}")
        else:
            print(f"✗ Failed to download {filename}")
    
    print("\nDownload completed!")
    print("\nYou can now run:")
    print("python train_custom_sam.py")

if __name__ == "__main__":
    main()
