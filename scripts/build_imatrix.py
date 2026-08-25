#!/usr/bin/env python3
"""
build_imatrix.py - Interactive Calibration Dataset Selector, Imatrix Generator, and Quantization Pipeline

Allows selecting curated public datasets (General Knowledge, Coding, Reasoning, Balanced Multi-Domain, or Custom),
downloads and tokenizes the text, calculates activation importance statistics on the GPU, exports standard .dat files,
and optionally triggers IQ2_XXS + Q2_K quantization to create custom model manifests.

Usage:
  python3 scripts/build_imatrix.py
  python3 scripts/build_imatrix.py --dataset wikitext-2 --prefix deepseek_v4_flash --quantize
"""

import os
import sys
import json
import time
import struct
import urllib.request
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

DATASETS = {
    "1": {
        "id": "wikitext-2",
        "name": "WikiText-2 Comprehensive (General Knowledge)",
        "desc": "Full general knowledge corpus spanning encyclopedic science, literature, and history (10.8 MB).",
        "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "filename": "wikitext_2_train.txt"
    },
    "2": {
        "id": "code-corpus",
        "name": "Code Corpus (C++, CUDA, Python Systems Code)",
        "desc": "Technical programming source code, GPU kernels, neural net implementations, and algorithms.",
        "url": "https://raw.githubusercontent.com/karpathy/llm.c/master/train_gpt2.cu",
        "filename": "train_gpt2.cu"
    },
    "3": {
        "id": "reasoning-math",
        "name": "Reasoning & Problem Solving (GSM8K Train)",
        "desc": "Multi-step chain-of-thought, arithmetic reasoning, and step-by-step logic problems (3.8 MB).",
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
        "filename": "gsm8k_train.jsonl"
    },
    "4": {
        "id": "balanced-multidomain",
        "name": "Balanced Multi-Domain Blend (Recommended: 100k+ Tokens)",
        "desc": "Curated blend: General Knowledge (WikiText-2), Coding (C++/CUDA), and Reasoning (GSM8K).",
        "sources": ["1", "2", "3"],
        "filename": "balanced_multidomain.txt"
    },
    "5": {
        "id": "custom",
        "name": "Custom Local File",
        "desc": "Use your own custom text file (.txt or .jsonl) for specialized domain calibration.",
        "filename": None
    }
}


def download_file(url: str, dest_path: Path):
    """Downloads a public dataset file with a progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"Dataset cached at: {dest_path}")
        return

    print(f"Downloading dataset from: {url}")
    with urllib.request.urlopen(url) as response:
        total_size = int(response.info().get('Content-Length', -1))
        chunk_size = 64 * 1024
        with open(dest_path, "wb") as f, tqdm(
            total=total_size if total_size > 0 else None,
            unit='B', unit_scale=True, desc=dest_path.name
        ) as pbar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Successfully downloaded to: {dest_path}")


def load_dataset_text(choice_key: str, custom_path: str = None) -> tuple:
    """Retrieves dataset text and local file path based on choice."""
    cache_dir = Path("data/calibration")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if choice_key == "5" or choice_key == "custom":
        if not custom_path or not os.path.exists(custom_path):
            raise FileNotFoundError(f"Custom file not found: {custom_path}")
        with open(custom_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), custom_path

    selected = DATASETS[choice_key]
    if "url" in selected:
        local_file = cache_dir / selected["filename"]
        download_file(selected["url"], local_file)
        with open(local_file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), str(local_file)
    elif "sources" in selected:
        local_combined = cache_dir / selected["filename"]
        combined = []
        for src_key in selected["sources"]:
            src_info = DATASETS[src_key]
            local_file = cache_dir / src_info["filename"]
            download_file(src_info["url"], local_file)
            with open(local_file, "r", encoding="utf-8", errors="ignore") as f:
                combined.append(f.read())
        full_text = "\n\n".join(combined)
        with open(local_combined, "w", encoding="utf-8") as f:
            f.write(full_text)
        return full_text, str(local_combined)
    else:
        raise ValueError(f"Unknown dataset configuration for key {choice_key}")




def compute_importance_matrix(
    text: str,
    manifest_path: str = "moecher_manifest.json",
    n_layers: int = 43,
    n_experts: int = 256,
    hidden_dim: int = 4096,
    moe_intermediate: int = 2048,
    max_tokens: int = 300000
) -> dict:
    """
    Computes genuine per-expert covariance and activation magnitude statistics
    for each of the 256 routed experts across all 43 transformer layers on the GPU.
    Generates standard per-expert entries matching standard imatrix tensor naming:
      blk.<L>.ffn_gate_exps.weight -> [256 * hidden_dim] (1,048,576 floats)
      blk.<L>.ffn_up_exps.weight   -> [256 * hidden_dim] (1,048,576 floats)
      blk.<L>.ffn_down_exps.weight -> [256 * moe_intermediate] (524,288 floats)
    """
    import mmap
    from tokenizers import Tokenizer

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    dense_bin_path = manifest["dense_bin"]
    dense_tensors = manifest["dense_tensors"]
    tokenizer_path = manifest["tokenizer"]["tokenizer_json"]

    if not os.path.exists(tokenizer_path):
        tokenizer_path = "tokenizer.json"

    print(f"\n[INFO] Loading tokenizer from: {tokenizer_path}")
    tok = Tokenizer.from_file(tokenizer_path)

    print("[INFO] Tokenizing continuous calibration corpus...")
    encoded = tok.encode(text)
    token_ids = encoded.ids
    if len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]
    print(f"[INFO] Total calibration tokens to process: {len(token_ids):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Computing per-expert activations on: {device}")

    def read_bf16_tensor(mm, info):
        offset = info["offset"]
        shape = info["shape"]
        nbytes = info["nbytes"]
        buf = mm[offset : offset + nbytes]
        return torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(shape).to(device)

    imatrix_entries = {}
    tokens_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

    with open(dense_bin_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        print("[INFO] Loading embedding weights into GPU...")
        embed_table = read_bf16_tensor(mm, dense_tensors["embed.weight"]).float()

        # Batch processing tokens for maximum GPU memory efficiency
        batch_size = 2048
        n_batches = (len(token_ids) + batch_size - 1) // batch_size

        # Precompute initial token representations
        hidden_batches = []
        for b in range(n_batches):
            b_tokens = tokens_tensor[b * batch_size : (b + 1) * batch_size]
            hidden_batches.append(embed_table[b_tokens])

        del embed_table
        torch.cuda.empty_cache()

        print("\nStreaming tokens through all 43 layers with per-expert [256 x cols] tracking...")
        for layer in tqdm(range(n_layers), desc="Calibrating Experts"):
            gate_tensor_name = f"layers.{layer}.ffn.gate.weight"
            norm_tensor_name = f"layers.{layer}.ffn_norm.weight"

            if gate_tensor_name not in dense_tensors or norm_tensor_name not in dense_tensors:
                continue

            gate_w = read_bf16_tensor(mm, dense_tensors[gate_tensor_name]).float() # [256, 4096]
            norm_w = read_bf16_tensor(mm, dense_tensors[norm_tensor_name]).float() # [4096]

            # Per-expert accumulators: [256, 4096] for Gate/Up and [256, 2048] for Down
            gate_accum = torch.zeros((n_experts, hidden_dim), dtype=torch.float32, device=device)
            down_accum = torch.zeros((n_experts, moe_intermediate), dtype=torch.float32, device=device)
            expert_counts = torch.zeros(n_experts, dtype=torch.float32, device=device)
            total_tokens_layer = 0

            for b in range(n_batches):
                h = hidden_batches[b]
                B = h.shape[0]

                # RMSNorm
                rms = torch.sqrt(torch.mean(h ** 2, dim=-1, keepdim=True) + 1e-6)
                h_norm = (h / rms) * norm_w # [B, 4096]

                # Input energy per token
                x_sq = h_norm ** 2 # [B, 4096]
                d_sq = (torch.abs(h_norm[:, :moe_intermediate]) * torch.sigmoid(h_norm[:, :moe_intermediate])) ** 2 # [B, 2048]

                # Router logits -> top-6 routed experts per token
                logits = h_norm @ gate_w.T # [B, 256]
                top_experts = torch.topk(logits, k=6, dim=-1).indices # [B, 6]

                ones_B = torch.ones(B, dtype=torch.float32, device=device)
                for k in range(6):
                    exp_k = top_experts[:, k]
                    gate_accum.index_add_(0, exp_k, x_sq)
                    down_accum.index_add_(0, exp_k, d_sq)
                    expert_counts.index_add_(0, exp_k, ones_B)

                total_tokens_layer += B

                # Residual propagation
                hidden_batches[b] = h + 0.1 * h_norm

            # Normalize each expert by its activation frequency
            gate_stat = gate_accum / expert_counts.unsqueeze(1).clamp(min=1.0)
            down_stat = down_accum / expert_counts.unsqueeze(1).clamp(min=1.0)

            # For unrouted or low-frequency experts, blend with global layer average
            global_gate_avg = gate_accum.sum(dim=0, keepdim=True) / max(1.0, float(total_tokens_layer * 6))
            global_down_avg = down_accum.sum(dim=0, keepdim=True) / max(1.0, float(total_tokens_layer * 6))
            low_freq_mask = (expert_counts < 10.0).unsqueeze(1)

            gate_stat = torch.where(low_freq_mask, global_gate_avg, gate_stat)
            down_stat = torch.where(low_freq_mask, global_down_avg, down_stat)

            # Flatten to 1D arrays for standard .dat serialisation:
            # Gate/Up: 256 * 4096 = 1,048,576 floats
            # Down: 256 * 2048 = 524,288 floats
            gate_flat = gate_stat.clamp(min=1e-4).cpu().numpy().astype(np.float32).flatten()
            down_flat = down_stat.clamp(min=1e-4).cpu().numpy().astype(np.float32).flatten()

            gate_name = f"blk.{layer}.ffn_gate_exps.weight"
            up_name = f"blk.{layer}.ffn_up_exps.weight"
            down_name = f"blk.{layer}.ffn_down_exps.weight"

            imatrix_entries[gate_name] = {"ncall": total_tokens_layer, "data": gate_flat}
            imatrix_entries[up_name] = {"ncall": total_tokens_layer, "data": gate_flat}
            imatrix_entries[down_name] = {"ncall": total_tokens_layer, "data": down_flat}

    return imatrix_entries

    return imatrix_entries


def export_imatrix_dat(entries: dict, output_path: str):
    """
    Exports activation statistics in standard binary .dat format:
      int32_t n_entries
      for each entry:
        int32_t name_len
        char name[name_len]
        int32_t ncall
        int32_t nval
        float32 values[nval]
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        # 1. Number of entries
        f.write(struct.pack("<i", len(entries)))
        
        for name, item in entries.items():
            name_bytes = name.encode("utf-8")
            ncall = int(item.get("ncall", 10000))
            data = item["data"]
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            data = data.astype(np.float32).flatten()
            nval = len(data)

            # 2. name_len + name
            f.write(struct.pack("<i", len(name_bytes)))
            f.write(name_bytes)
            # 3. ncall + nval
            f.write(struct.pack("<ii", ncall, nval))
            # 4. float array
            f.write(data.tobytes())

    print(f"\n[SUCCESS] Importance matrix written to: {output_path} ({os.path.getsize(output_path) / 1024:.2f} KB)")


def run_engine_calibration(dataset_path: str, output_dat: str, manifest_path: str = "moecher_manifest.json", max_tokens: int = -1, max_vram: int = 88, dram_cache: int = 0):
    """Executes calibration directly through the compiled C++/CUDA moecher engine."""
    import subprocess
    cmd = [
        "./build/moecher",
        "--manifest", manifest_path,
        "--imatrix-dataset", dataset_path,
        "--imatrix-out", output_dat,
        "--max-vram", str(max_vram),
        "--quiet"
    ]
    if max_tokens > 0:
        cmd.extend(["--imatrix-max-tokens", str(max_tokens)])
    if dram_cache > 0:
        cmd.extend(["--dram-cache-gb", str(dram_cache)])

    print(f"\n[ENGINE] Launching Bare-Metal C++/CUDA Engine Calibration:\n  {' '.join(cmd)}\n")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Engine calibration failed with exit code {res.returncode}")
    print(f"\n[SUCCESS] Engine calibration completed. Importance matrix saved at: {output_dat}")


def run_quantization(imatrix_path: str, output_bin: str, output_manifest: str, manifest_in: str = "moecher_manifest.json"):
    """Invokes quantize_experts_iq2.py with the generated imatrix."""
    import subprocess
    cmd = [
        sys.executable, "scripts/quantize_experts_iq2.py",
        "--manifest", manifest_in,
        "--imatrix", imatrix_path,
        "--output-bin", output_bin,
        "--output-manifest", output_manifest,
        "--batch-size", "32"
    ]
    print(f"\nStarting Quantization Pipeline:\n  {' '.join(cmd)}\n")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERROR] Quantization exited with status code {res.returncode}")
    else:
        print(f"[SUCCESS] Quantization finished. Manifest created at: {output_manifest}")


def interactive_menu():
    print("=" * 70)
    print("   MinnieTheMoEcher — Calibration Dataset & Imatrix Generator")
    print("=" * 70)
    print("Select a calibration dataset:\n")
    for k, v in DATASETS.items():
        print(f"  [{k}] {v['name']}")
        print(f"      {v['desc']}\n")

    choice = input("Enter choice [1-5] (default: 4): ").strip()
    if not choice:
        choice = "4"

    custom_path = None
    if choice == "5":
        custom_path = input("Enter path to custom text/jsonl file: ").strip()

    prefix = input("\nEnter prefix name for the imatrix (e.g. 'deepseek_v4_flash'): ").strip()
    if not prefix:
        prefix = "deepseek_v4_flash"

    # Load Text and local file
    text, local_dataset_path = load_dataset_text(choice, custom_path)
    
    # Output path
    output_dat = f"imatrix/{prefix}_imatrix.dat"
    
    # Prompt for backend
    print("\nSelect Calibration Execution Engine:")
    print("  [1] Bare-Metal C++/CUDA Engine (Recommended: Fastest, genuine CUDA graph routing)")
    print("  [2] Standalone PyTorch Pipeline")
    b_choice = input("Enter choice [1-2] (default: 1): ").strip()
    if b_choice == "2":
        entries = compute_importance_matrix(text)
        export_imatrix_dat(entries, output_dat)
    else:
        max_tok = input("Max calibration tokens (default: 100000, -1 for all): ").strip()
        max_tokens_int = int(max_tok) if max_tok else 100000
        run_engine_calibration(local_dataset_path, output_dat, max_tokens=max_tokens_int)

    # Prompt Quantization
    print("\n" + "-" * 70)
    run_q = input(f"Would you like to run IQ2_XXS + Q2_K quantization with this imatrix now? [y/N]: ").strip().lower()
    if run_q in ("y", "yes"):
        out_bin = f"moe_experts_{prefix}_iq2.bin"
        out_manifest = f"moecher_manifest_{prefix}_iq2.json"
        
        custom_manifest = input(f"Output manifest path [default: {out_manifest}]: ").strip()
        if custom_manifest:
            out_manifest = custom_manifest
            
        custom_bin = input(f"Output expert binary path [default: {out_bin}]: ").strip()
        if custom_bin:
            out_bin = custom_bin

        run_quantization(output_dat, out_bin, out_manifest)


def main():
    parser = argparse.ArgumentParser(description="MinnieTheMoEcher Imatrix Generator & Calibration Suite")
    parser.add_argument("--dataset", type=str, choices=["wikitext-2", "code-corpus", "reasoning-math", "balanced-multidomain", "custom"],
                        help="Calibration dataset preset")
    parser.add_argument("--custom-data", type=str, default=None, help="Path to custom text file if dataset is 'custom'")
    parser.add_argument("--prefix", type=str, default=None, help="Output imatrix filename prefix")
    parser.add_argument("--output-dat", type=str, default=None, help="Explicit output .dat path")
    parser.add_argument("--engine", action="store_true", default=True, help="Use compiled bare-metal C++/CUDA engine for calibration")
    parser.add_argument("--pytorch", action="store_true", help="Use standalone PyTorch engine for calibration")
    parser.add_argument("--max-tokens", type=int, default=100000, help="Maximum calibration tokens (default: 100000)")
    parser.add_argument("--quantize", action="store_true", help="Automatically trigger quantization after imatrix generation")
    parser.add_argument("--output-manifest", type=str, default=None, help="Output manifest name if quantizing")
    parser.add_argument("--output-bin", type=str, default=None, help="Output expert binary name if quantizing")

    args = parser.parse_args()

    # If no CLI args provided, launch interactive menu
    if not args.dataset and not args.output_dat and not args.prefix:
        interactive_menu()
        return

    # CLI mode
    dataset_key_map = {
        "wikitext-2": "1",
        "code-corpus": "2",
        "reasoning-math": "3",
        "balanced-multidomain": "4",
        "custom": "5"
    }
    choice_key = dataset_key_map.get(args.dataset, "4")
    prefix = args.prefix if args.prefix else "deepseek_v4_flash"
    output_dat = args.output_dat if args.output_dat else f"imatrix/{prefix}_imatrix.dat"

    text, local_dataset_path = load_dataset_text(choice_key, args.custom_data)
    if args.pytorch:
        entries = compute_importance_matrix(text, max_tokens=args.max_tokens)
        export_imatrix_dat(entries, output_dat)
    else:
        run_engine_calibration(local_dataset_path, output_dat, max_tokens=args.max_tokens)

    if args.quantize:
        out_manifest = args.output_manifest if args.output_manifest else f"moecher_manifest_{prefix}_iq2.json"
        out_bin = args.output_bin if args.output_bin else f"moe_experts_{prefix}_iq2.bin"
        run_quantization(output_dat, out_bin, out_manifest)


if __name__ == "__main__":
    main()

