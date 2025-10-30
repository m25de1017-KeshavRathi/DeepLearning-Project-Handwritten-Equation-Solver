"""
Upload trained model to Hugging Face Hub.
"""

import os
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub import login
import getpass

def upload_model(repo_name, username=None, token=None):
    """
    Upload model to Hugging Face Hub.
    
    Args:
        repo_name: Name of the repository (e.g., "handwritten-equation-solver")
        username: Hugging Face username (optional, will prompt if not provided)
        token: Hugging Face token (optional, will prompt if not provided)
    """
    print("=" * 80)
    print("HUGGING FACE MODEL UPLOAD")
    print("=" * 80)
    
    # Login to Hugging Face
    if token is None:
        print("\nPlease provide your Hugging Face token.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        token = getpass.getpass("Hugging Face Token: ")
    
    try:
        login(token=token)
        print("✓ Successfully logged in to Hugging Face!")
    except Exception as e:
        print(f"✗ Login failed: {e}")
        return False
    
    # Get username if not provided
    if username is None:
        username = input("\nEnter your Hugging Face username: ")
    
    # Create repository ID
    repo_id = f"{username}/{repo_name}"
    
    # Initialize API
    api = HfApi()
    
    # Create repository
    print(f"\nCreating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✓ Repository created: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"✗ Failed to create repository: {e}")
        return False
    
    # Upload folder
    print("\nUploading model files...")
    upload_dir = "huggingface_upload"
    
    if not os.path.exists(upload_dir):
        print(f"✗ Upload directory not found: {upload_dir}")
        return False
    
    # List files to upload
    files = os.listdir(upload_dir)
    print(f"\nFiles to upload:")
    for f in files:
        file_path = os.path.join(upload_dir, f)
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  - {f} ({size:.1f} MB)")
    
    try:
        upload_folder(
            folder_path=upload_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial model upload - CROHME2019 trained CNN-Transformer"
        )
        print("\n" + "=" * 80)
        print("✓ MODEL SUCCESSFULLY UPLOADED!")
        print("=" * 80)
        print(f"\nYour model is now available at:")
        print(f"🤗 https://huggingface.co/{repo_id}")
        print("\nYou can download it with:")
        print(f"```python")
        print(f"from huggingface_hub import hf_hub_download")
        print(f"model_path = hf_hub_download(")
        print(f"    repo_id='{repo_id}',")
        print(f"    filename='best_model.keras'")
        print(f")")
        print(f"```")
        return True
    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--repo_name", 
        type=str, 
        default="handwritten-equation-solver",
        help="Repository name (default: handwritten-equation-solver)"
    )
    parser.add_argument(
        "--username", 
        type=str, 
        default=None,
        help="Hugging Face username (will prompt if not provided)"
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="Hugging Face token (will prompt if not provided)"
    )
    
    args = parser.parse_args()
    
    success = upload_model(
        repo_name=args.repo_name,
        username=args.username,
        token=args.token
    )
    
    if success:
        print("\n✓ Upload complete!")
    else:
        print("\n✗ Upload failed. Please check the errors above.")

