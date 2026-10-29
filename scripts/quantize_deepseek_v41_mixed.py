#!/usr/bin/env python3
"""
quantize_deepseek_v41_mixed.py - GPU-Accelerated Mixed Quantizer for DeepSeek-V4.1-Flash
Hardware Target: NVIDIA RTX PRO 6000 96GB + 64GB RAM

Produces:
  1. moe_experts_hot.bin (18,800,640 bytes/expert, FP4 + E8M0) for hot experts
  2. moe_experts_iq2.bin (9,953,280 bytes/expert, IQ2_XXS + Q2_K) for cold experts
"""

import os
import sys
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# Insert scripts dir for IQ2 helper tables
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize_experts_iq2 import dequantize_fp4_gpu, quantize_iq2_xxs_gpu, quantize_q2_k_gpu, ImatrixLoader

def main():
    expert_bin_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/expert.bin")
    map_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/mixed_expert_map.bin")
    imatrix_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/deepseek_v41_imatrix.dat")
    hot_bin_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/moe_experts_hot.bin")
    cold_bin_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/moe_experts_iq2.bin")
    
    n_layers = 40
    n_experts = 384
    total_experts = n_layers * n_experts
    
    src_block_size = 18800640
    hot_block_size = 18800640
    cold_block_size = 9953280
    
    w1_iq2_sz = 3041280
    w3_iq2_sz = 3041280
    w2_q2_sz = 3870720
    assert w1_iq2_sz + w3_iq2_sz + w2_q2_sz == cold_block_size

    print("=" * 78)
    print("  DeepSeek-V4.1-Flash Mixed Quantization Engine (Approach A)")
    print(f"  Source Expert Binary: {expert_bin_path}")
    print(f"  Hot Expert Binary   : {hot_bin_path} ({hot_block_size} bytes/expert)")
    print(f"  Cold Expert Binary  : {cold_bin_path} ({cold_block_size} bytes/expert)")
    print("=" * 78, flush=True)

    if not expert_bin_path.exists():
        print(f"ERROR: {expert_bin_path} not found!")
        sys.exit(1)
        
    with open(map_path, "rb") as f_map:
        expert_map = np.frombuffer(f_map.read(), dtype=np.uint8)
    assert len(expert_map) == total_experts

    imatrix_loader = None
    if imatrix_path.exists():
        print(f"Loading importance matrix from: {imatrix_path}")
        imatrix_loader = ImatrixLoader(str(imatrix_path))
    else:
        print("Notice: No imatrix found at default path, running unweighted quantization.")

    total_hot = int(np.sum(expert_map == 1))
    total_cold = int(np.sum(expert_map == 0))
    print(f"Loaded Expert Map: {total_hot} Hot Experts (NVFP4), {total_cold} Cold Experts (IQ2)")

    # Offsets in src block
    w1_w_off = 0
    w1_s_off = 5898240
    w3_w_off = 6266880
    w3_s_off = 12165120
    w2_w_off = 12533760
    w2_s_off = 18432000

    batch_size = 16
    t_start = time.time()

    # Pre-allocate output files to exact sparse sizes
    print("Pre-allocating output binary files on NVMe SSD...", flush=True)
    with open(hot_bin_path, "wb") as f_hot:
        f_hot.seek(total_experts * hot_block_size - 1)
        f_hot.write(b'\0')
        
    with open(cold_bin_path, "wb") as f_cold:
        f_cold.seek(total_experts * cold_block_size - 1)
        f_cold.write(b'\0')

    print("Starting dual-stream extraction and GPU quantization...", flush=True)
    
    with open(expert_bin_path, "rb") as fin, open(hot_bin_path, "r+b") as f_hot, open(cold_bin_path, "r+b") as f_cold:
        cold_batch_indices = []
        cold_batch_raw = []

        def flush_cold_batch(indices, raw_blocks):
            if not indices:
                return
            b_cnt = len(indices)
            w1_p_list, w1_s_list = [], []
            w3_p_list, w3_s_list = [], []
            w2_p_list, w2_s_list = [], []

            for raw in raw_blocks:
                w1_p = torch.frombuffer(bytearray(raw[w1_w_off : w1_s_off]), dtype=torch.uint8).view(2304, 2560)
                w1_s = torch.frombuffer(bytearray(raw[w1_s_off : w3_w_off]), dtype=torch.uint8).view(2304, 160)
                w1_p_list.append(w1_p)
                w1_s_list.append(w1_s)

                w3_p = torch.frombuffer(bytearray(raw[w3_w_off : w3_s_off]), dtype=torch.uint8).view(2304, 2560)
                w3_s = torch.frombuffer(bytearray(raw[w3_s_off : w2_w_off]), dtype=torch.uint8).view(2304, 160)
                w3_p_list.append(w3_p)
                w3_s_list.append(w3_s)

                w2_p = torch.frombuffer(bytearray(raw[w2_w_off : w2_s_off]), dtype=torch.uint8).view(5120, 1152)
                w2_s = torch.frombuffer(bytearray(raw[w2_s_off : src_block_size]), dtype=torch.uint8).view(5120, 72)
                w2_p_list.append(w2_p)
                w2_s_list.append(w2_s)

            w1_p_gpu = torch.stack(w1_p_list).to("cuda")
            w1_s_gpu = torch.stack(w1_s_list).to("cuda")
            w1_f32 = dequantize_fp4_gpu(w1_p_gpu.view(-1, 2560), w1_s_gpu.view(-1, 160), b_cnt * 2304, 5120, 160)

            w3_p_gpu = torch.stack(w3_p_list).to("cuda")
            w3_s_gpu = torch.stack(w3_s_list).to("cuda")
            w3_f32 = dequantize_fp4_gpu(w3_p_gpu.view(-1, 2560), w3_s_gpu.view(-1, 160), b_cnt * 2304, 5120, 160)

            w2_p_gpu = torch.stack(w2_p_list).to("cuda")
            w2_s_gpu = torch.stack(w2_s_list).to("cuda")
            w2_f32 = dequantize_fp4_gpu(w2_p_gpu.view(-1, 1152), w2_s_gpu.view(-1, 72), b_cnt * 5120, 2304, 72)

            if imatrix_loader:
                w1_cw_list, w3_cw_list, w2_cw_list = [], [], []
                for glob_idx in indices:
                    l_idx = glob_idx // n_experts
                    e_idx = glob_idx % n_experts
                    w1_cw = imatrix_loader.get_expert_weights(l_idx, e_idx, "w1", 5120, device="cuda")
                    w3_cw = imatrix_loader.get_expert_weights(l_idx, e_idx, "w3", 5120, device="cuda")
                    w2_cw = imatrix_loader.get_expert_weights(l_idx, e_idx, "w2", 2304, device="cuda")
                    w1_cw_list.append(w1_cw if w1_cw is not None else torch.ones(5120, device="cuda"))
                    w3_cw_list.append(w3_cw if w3_cw is not None else torch.ones(5120, device="cuda"))
                    w2_cw_list.append(w2_cw if w2_cw is not None else torch.ones(2304, device="cuda"))

                w1_cw_b = torch.stack(w1_cw_list).view(b_cnt, 1, 5120).expand(b_cnt, 2304, 5120).contiguous()
                w3_cw_b = torch.stack(w3_cw_list).view(b_cnt, 1, 5120).expand(b_cnt, 2304, 5120).contiguous()
                w2_cw_b = torch.stack(w2_cw_list).view(b_cnt, 1, 2304).expand(b_cnt, 5120, 2304).contiguous()
            else:
                w1_cw_b, w3_cw_b, w2_cw_b = None, None, None

            w1_iq2_bytes = quantize_iq2_xxs_gpu(w1_f32, w1_cw_b)
            w3_iq2_bytes = quantize_iq2_xxs_gpu(w3_f32, w3_cw_b)
            w2_q2_bytes = quantize_q2_k_gpu(w2_f32, w2_cw_b)
            torch.cuda.synchronize()

            for idx, glob_idx in enumerate(indices):
                w1_part = w1_iq2_bytes[idx * w1_iq2_sz : (idx + 1) * w1_iq2_sz]
                w3_part = w3_iq2_bytes[idx * w3_iq2_sz : (idx + 1) * w3_iq2_sz]
                w2_part = w2_q2_bytes[idx * w2_q2_sz : (idx + 1) * w2_q2_sz]

                f_cold.seek(glob_idx * cold_block_size)
                f_cold.write(w1_part)
                f_cold.write(w3_part)
                f_cold.write(w2_part)

        chunk_experts = 64
        pbar = tqdm(total=total_experts, desc="Processing MoE Experts")

        for c_start in range(0, total_experts, chunk_experts):
            c_cnt = min(chunk_experts, total_experts - c_start)
            raw_chunk = fin.read(c_cnt * src_block_size)
            if not raw_chunk:
                break

            for i in range(c_cnt):
                glob_idx = c_start + i
                block_data = raw_chunk[i * src_block_size : (i + 1) * src_block_size]

                if expert_map[glob_idx] == 1:
                    # Hot expert: write directly to hot bin
                    f_hot.seek(glob_idx * hot_block_size)
                    f_hot.write(block_data)
                else:
                    # Cold expert: queue for GPU quantization
                    cold_batch_indices.append(glob_idx)
                    cold_batch_raw.append(block_data)

                    if len(cold_batch_indices) >= batch_size:
                        flush_cold_batch(cold_batch_indices, cold_batch_raw)
                        cold_batch_indices = []
                        cold_batch_raw = []

            pbar.update(c_cnt)

        # Flush remaining cold experts
        if cold_batch_indices:
            flush_cold_batch(cold_batch_indices, cold_batch_raw)
        pbar.close()

    elapsed = time.time() - t_start
    print(f"\n[SUCCESS] Mixed quantization completed in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")
    print(f"  Hot Binary : {hot_bin_path} ({os.path.getsize(hot_bin_path) / (1024**3):.2f} GB)")
    print(f"  Cold Binary: {cold_bin_path} ({os.path.getsize(cold_bin_path) / (1024**3):.2f} GB)")

if __name__ == "__main__":
    main()
