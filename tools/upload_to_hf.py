#!/usr/bin/env python3
"""
Moecher Hugging Face Upload Utility
Uploads Qwen 3.8 27B INT4 and DeepSeek V4 Flash IQ2 models to Hugging Face Hub.
"""

import os
import sys
import argparse

try:
    from huggingface_hub import HfApi, login, create_repo
except ImportError:
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import HfApi, login, create_repo

DEFAULT_USERNAME = "TinoBruno"

REPOS = {
    "qwen": {
        "repo_id": f"{DEFAULT_USERNAME}/moecher-qwen-3.8-27b-q4",
        "folder_path": os.path.join("models", "qwen3_8_27b_q4"),
        "license": "apache-2.0"
    },
    "deepseek": {
        "repo_id": f"{DEFAULT_USERNAME}/moecher-deepseek-v4-flash-iq2",
        "folder_path": os.path.join("models", "deepseek_v4_flash_iq2"),
        "license": "mit"
    }
}

def upload_model(target, token=None):
    if target not in REPOS:
        print(f"Unknown target '{target}'. Available: {list(REPOS.keys())} or 'all'")
        return

    info = REPOS[target]
    repo_id = info["repo_id"]
    folder_path = os.path.abspath(info["folder_path"])

    if not os.path.exists(folder_path):
        print(f"[ERROR] Source folder does not exist: {folder_path}")
        return

    print("=" * 64)
    print(f"  Uploading {target.upper()} to https://huggingface.co/{repo_id}")
    print(f"  Source: {folder_path}")
    print("=" * 64)

    api = HfApi(token=token)

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"[OK] Repository {repo_id} verified/created.")
    except Exception as e:
        print(f"[WARN] Repository check: {e}")

    print("Uploading folder contents with progress tracking (resumable)...")
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {target} model weights and manifest for Moecher"
        )
        print(f"\n[SUCCESS] Upload complete for {repo_id}!")
        print(f"URL: https://huggingface.co/{repo_id}\n")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload Moecher models to Hugging Face Hub")
    parser.add_argument("model", choices=["qwen", "deepseek", "all"], default="all", nargs="?",
                        help="Which model repository to upload (default: all)")
    parser.add_argument("--token", default=None, help="Hugging Face Write Access Token (starts with hf_...)")
    parser.add_argument("--user", default=DEFAULT_USERNAME, help=f"Hugging Face username (default: {DEFAULT_USERNAME})")

    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face Write Access Token (hf_...): ").strip()

    if not token:
        print("[ERROR] A Hugging Face token is required to upload models.")
        sys.exit(1)

    login(token=token, add_to_git_credential=True)

    if args.user != DEFAULT_USERNAME:
        for k in REPOS:
            repo_name = REPOS[k]["repo_id"].split("/")[-1]
            REPOS[k]["repo_id"] = f"{args.user}/{repo_name}"

    if args.model == "all":
        upload_model("qwen", token=token)
        upload_model("deepseek", token=token)
    else:
        upload_model(args.model, token=token)

if __name__ == "__main__":
    main()
