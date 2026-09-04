import os
import sys
from huggingface_hub import HfApi

REPO_ID = "TinoBruno/moecher-qwen-3.8-27b-q4"
MODEL_DIR = "f:/moecher/models/qwen3_8_27b_q4"

FILES_TO_UPLOAD = [
    "draft_vocab_ids.bin",
    "draft_lm_head_int8_bf16.bin"
]

def upload(token=None):
    api = HfApi(token=token)
    print(f"Checking authentication...")
    user_info = api.whoami()
    print(f"Authenticated as: {user_info.get('name')} ({user_info.get('fullname')})")
    
    print(f"\nUploading MTP draft files to {REPO_ID}...")
    for filename in FILES_TO_UPLOAD:
        local_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(local_path):
            print(f"Error: {local_path} does not exist!")
            return False
        
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Uploading {filename} ({file_size_mb:.2f} MB)...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Add MTP draft vocabulary and compact draft lm_head ({filename})"
        )
        print(f"  [OK] Successfully uploaded {filename}")
        
    print("\nAll MTP draft files uploaded successfully!")
    return True

if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")
    if len(sys.argv) > 1:
        token = sys.argv[1]
    
    try:
        success = upload(token=token)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\nUpload failed: {e}")
        print("\nTip: Make sure you have provided a Hugging Face Write Token via:")
        print("  - Command line: python tools/upload_to_hf.py <your_hf_token>")
        print("  - Or set environment variable: set HF_TOKEN=<your_hf_token>")
        print("  - Or run: huggingface-cli login")
        sys.exit(1)
