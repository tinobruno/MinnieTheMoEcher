import os
import sys
import time
import urllib.request

FILES = [
    ("mmproj-Qwen3.8-27B-bf16.gguf", "https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/main/mmproj-Qwen3.8-27B-bf16.gguf"),
    ("Qwen3.8-27B-Q3_K_M.gguf", "https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q3_K_M.gguf"),
]

TARGET_DIR = r"E:\moecher\models\gguf_qwen3_8_27b"
os.makedirs(TARGET_DIR, exist_ok=True)

def download_file(filename, url):
    dest_path = os.path.join(TARGET_DIR, filename)
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
    except Exception as e:
        print(f"Error fetching header for {filename}: {e}")
        return False

    existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    if total_size > 0 and existing_size == total_size:
        print(f"[SKIP] {filename} already downloaded ({existing_size / (1024**3):.2f} GB)")
        return True

    print(f"[DOWNLOADING] {filename} ({total_size / (1024**3):.2f} GB)...")
    part_path = dest_path + ".part"
    downloaded = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    headers = {"User-Agent": "MoecherBenchmark/1.0"}
    if downloaded > 0 and downloaded < total_size:
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"
        print(f"  Resuming from {downloaded / (1024**3):.2f} GB...")
    else:
        downloaded = 0
        mode = "wb"

    req = urllib.request.Request(url, headers=headers)
    start_time = time.time()
    last_log = start_time

    with urllib.request.urlopen(req) as resp, open(part_path, mode) as f:
        chunk_size = 8 * 1024 * 1024 # 8 MB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_log >= 5.0:
                elapsed = now - start_time
                mb_per_sec = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                print(f"  [{filename}] {downloaded / (1024**3):.2f} / {total_size / (1024**3):.2f} GB ({pct:.1f}%) @ {mb_per_sec:.1f} MB/s")
                last_log = now

    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(part_path, dest_path)
    print(f"[DONE] {filename} downloaded successfully ({downloaded / (1024**3):.2f} GB).")
    return True

if __name__ == "__main__":
    for fname, url in FILES:
        if not download_file(fname, url):
            print(f"Failed to download {fname}")
            sys.exit(1)
    print("All benchmark GGUF files downloaded successfully.")
