import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi

DEFAULT_REPO_ID = "TinoBruno/moecher-qwen-3.8-27b-vision-13g"
DEFAULT_MODEL_DIR = "F:/moecher/models/qwen3.8-27B-Vision-13G"

FILES_TO_UPLOAD = [
    "README.md",
    "moecher_manifest.json",
    "tokenizer.json",
    "attention_dense_layers.bin",
    "draft_lm_head_int8_bf16.bin",
    "draft_lm_head_int8.bin",
    "draft_vocab_ids.bin",
    "draft_vocab_ids.json",
    "draft_vocab_ids_mapping.json",
    "draft_vocab_ids_reverse.bin"
]

def upload(repo_id: str, model_dir: Path, token: str = None, private: bool = False):
    api = HfApi(token=token)
    
    print("Checking Hugging Face authentication...")
    user_info = api.whoami()
    print(f"Authenticated as: {user_info.get('name')} ({user_info.get('fullname')})")
    
    print(f"\nEnsuring repository exists: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    
    print(f"\nUploading model files from {model_dir} to {repo_id}...")
    for filename in FILES_TO_UPLOAD:
        local_path = model_dir / filename
        if not local_path.exists():
            print(f"[WARN] Skipping {filename} (not found at {local_path})")
            continue
        
        file_size_gb = local_path.stat().st_size / (1024 ** 3)
        if file_size_gb >= 1.0:
            print(f"Uploading {filename} ({file_size_gb:.2f} GB)...")
        else:
            file_size_mb = local_path.stat().st_size / (1024 ** 2)
            print(f"Uploading {filename} ({file_size_mb:.2f} MB)...")
            
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {filename} for {repo_id}"
        )
        print(f"  [OK] Successfully uploaded {filename}")
        
    print("\nAll files uploaded successfully to Hugging Face!")
    print(f"Model URL: https://huggingface.co/{repo_id}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload qwen3.8-27B-Vision-13G to Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"Hugging Face repo ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help=f"Local model directory (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--token", default=None, help="Hugging Face write token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create repository as private")
    
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        try:
            import huggingface_hub
            token = huggingface_hub.get_token()
        except Exception:
            pass
            
    if not token:
        print("Error: No Hugging Face token provided!")
        print("Please provide your token via:")
        print("  - Command line: python tools/upload_qwen_13g_to_hf.py --token <hf_token>")
        print("  - Or environment variable: set HF_TOKEN=<hf_token>")
        print("  - Or run: huggingface-cli login")
        sys.exit(1)
        
    try:
        success = upload(args.repo_id, Path(args.model_dir), token=token, private=args.private)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\nUpload failed: {e}")
        sys.exit(1)
