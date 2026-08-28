#!/usr/bin/env python3
"""
Moecher Hugging Face Upload Utility
Uploads Qwen 3.8 27B INT4 and DeepSeek V4 Flash IQ2 models to Hugging Face Hub.
"""

import os
import sys
import argparse

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import HfApi, login

DEFAULT_USERNAME = "TinoBruno"

REPOS = {
    "qwen": {
        "name": "Qwen 3.8 27B INT4 (~18.7 GB)",
        "repo_id": f"{DEFAULT_USERNAME}/moecher-qwen-3.8-27b-q4",
        "folder_path": os.path.join("models", "qwen3_8_27b_q4"),
        "license": "apache-2.0"
    },
    "deepseek": {
        "name": "DeepSeek V4 Flash IQ2 (~81.4 GB)",
        "repo_id": f"{DEFAULT_USERNAME}/moecher-deepseek-v4-flash-iq2",
        "folder_path": os.path.join("models", "deepseek_v4_flash_iq2"),
        "license": "mit"
    },
    "deepseek_q4": {
        "name": "DeepSeek V4 Flash Q4 - 8GB GPU Mode (~78.8 GB)",
        "repo_id": f"{DEFAULT_USERNAME}/moecher-deepseek-v4-flash-q4",
        "folder_path": os.path.join("models", "deepseek_v4_flash_q4"),
        "license": "mit"
    }
}

def upload_model(target, token=None):
    if target not in REPOS:
        print(f"[ERROR] Unknown target '{target}'. Available: {list(REPOS.keys())} or 'all'")
        return

    info = REPOS[target]
    repo_id = info["repo_id"]
    folder_path = os.path.abspath(info["folder_path"])

    if not os.path.exists(folder_path):
        print(f"[ERROR] Source folder does not exist: {folder_path}")
        return

    print("=" * 64)
    print(f"  Uploading {info['name']}")
    print(f"  Target: https://huggingface.co/{repo_id}")
    print(f"  Source: {folder_path}")
    print("=" * 64)

    api = HfApi(token=token)

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"[OK] Repository {repo_id} ready.")
    except Exception as e:
        print(f"[NOTE] Repository check: {e}")

    print("\nUploading files to Hugging Face (chunked & resumable)...")
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {target} model weights and manifest for Moecher"
        )
        print(f"\n[SUCCESS] Upload complete for {repo_id}!")
        print(f"Repository URL: https://huggingface.co/{repo_id}\n")
    except Exception as e:
        print(f"\n[ERROR] Upload encountered an error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload Moecher models to Hugging Face Hub")
    parser.add_argument("model", choices=["qwen", "deepseek", "deepseek_q4", "all"], default=None, nargs="?",
                        help="Which model repository to upload")
    parser.add_argument("--token", default=None, help="Hugging Face Write Access Token (starts with hf_...)")
    parser.add_argument("--user", default=DEFAULT_USERNAME, help=f"Hugging Face username (default: {DEFAULT_USERNAME})")

    args = parser.parse_args()

    print("=" * 64)
    print("  Moecher Inference Engine - Hugging Face Model Uploader")
    print(f"  Account: {args.user}")
    print("=" * 64)

    selected_model = args.model
    if not selected_model:
        print("\nSelect what to upload:")
        print("  [1] Qwen 3.8 27B INT4 (~18.7 GB)")
        print("  [2] DeepSeek V4 Flash IQ2 - Standard (~81.4 GB)")
        print("  [3] DeepSeek V4 Flash Q4 - 8GB GPU Mode (~78.8 GB)")
        print("  [4] All Models")
        choice = input("\nEnter choice (1, 2, 3, or 4) [default: 3]: ").strip()
        if choice == "1":
            selected_model = "qwen"
        elif choice == "2":
            selected_model = "deepseek"
        elif choice == "4":
            selected_model = "all"
        else:
            selected_model = "deepseek_q4"

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("\nPaste your Hugging Face Write Access Token")
        print("(from https://huggingface.co/settings/tokens)")
        token = input("Token (hf_...): ").strip()

    if not token:
        print("\n[ERROR] A valid Hugging Face token is required.")
        sys.exit(1)

    try:
        login(token=token, add_to_git_credential=True)
    except Exception as e:
        print(f"[WARN] Login notification: {e}")

    if args.user != DEFAULT_USERNAME:
        for k in REPOS:
            repo_name = REPOS[k]["repo_id"].split("/")[-1]
            REPOS[k]["repo_id"] = f"{args.user}/{repo_name}"

    if selected_model == "all":
        for k in REPOS:
            upload_model(k, token=token)
    else:
        upload_model(selected_model, token=token)

if __name__ == "__main__":
    main()
