#!/usr/bin/env python3
import json
import os
import sys
import numpy as np
import torch

def main():
    qwen_manifest_path = "models/qwen3_8_27b/moecher_manifest_qwen.json"
    qwen_dense_bin = "models/qwen3_8_27b/attention_dense_layers.bin"
    qwen_tok_path = "models/qwen3_8_27b/tokenizer.json"

    ds_manifest_path = "models/deepseek_v4_flash_q4/moecher_manifest_mixed.json"
    ds_dense_bin = "models/deepseek_v4_flash_q4/attention_dense_layers_q4.bin"
    ds_tok_path = "models/deepseek_v4_flash_q4/tokenizer.json"

    out_bridge_bin = sys.argv[1] if len(sys.argv) > 1 else "models/frankenstin/bridge_fc2.bin"

    print("Step 1: Finding common vocabulary between Qwen and DeepSeek...")
    with open(qwen_tok_path, "r") as f:
        q_vocab = json.load(f)["model"]["vocab"]
    with open(ds_tok_path, "r") as f:
        d_vocab = json.load(f)["model"]["vocab"]

    common_tokens = sorted(list(set(q_vocab.keys()) & set(d_vocab.keys())))
    print(f"Found {len(common_tokens)} common tokens.")

    # Select the top 40,000 common tokens with lowest token IDs (most frequent words/BPEs)
    # Filter out single characters or special tokens if desired, or sort by sum of IDs
    common_tokens.sort(key=lambda t: q_vocab[t] + d_vocab[t])
    selected_tokens = common_tokens[:50000]
    print(f"Selected {len(selected_tokens)} most frequent common tokens for semantic manifold alignment.")

    q_indices = np.array([q_vocab[t] for t in selected_tokens], dtype=np.int64)
    d_indices = np.array([d_vocab[t] for t in selected_tokens], dtype=np.int64)

    print("Step 2: Loading embedding tensors from mmap...")
    with open(qwen_manifest_path, "r") as f:
        q_mf = json.load(f)
    q_emb_meta = q_mf["dense_tensors"]["model.language_model.embed_tokens.weight"]
    
    with open(ds_manifest_path, "r") as f:
        ds_mf = json.load(f)
    ds_emb_meta = ds_mf["dense_tensors"]["embed.weight"]

    # Load Qwen embeddings [248320, 5120] in BF16
    q_mmap = np.memmap(qwen_dense_bin, dtype=np.uint16, mode='r', offset=q_emb_meta["offset"], shape=(q_emb_meta["shape"][0], q_emb_meta["shape"][1]))
    # Load DeepSeek embeddings [129280, 4096] in BF16
    ds_mmap = np.memmap(ds_dense_bin, dtype=np.uint16, mode='r', offset=ds_emb_meta["offset"], shape=(ds_emb_meta["shape"][0], ds_emb_meta["shape"][1]))

    print("Extracting selected token embeddings to float32...")
    # Convert uint16 BF16 to float32 using torch
    q_sub_u16 = torch.from_numpy(np.array(q_mmap[q_indices], copy=True))
    d_sub_u16 = torch.from_numpy(np.array(ds_mmap[d_indices], copy=True))

    q_sub_bf16 = q_sub_u16.view(torch.bfloat16).to(torch.float32) # [50000, 5120]
    d_sub_bf16 = d_sub_u16.view(torch.bfloat16).to(torch.float32) # [50000, 4096]

    print(f"Shapes: Qwen sub-embeddings {q_sub_bf16.shape}, DeepSeek sub-embeddings {d_sub_bf16.shape}")

    # Center embeddings
    q_mean = q_sub_bf16.mean(dim=0, keepdim=True)
    d_mean = d_sub_bf16.mean(dim=0, keepdim=True)
    X = q_sub_bf16 - q_mean # [N, 5120]
    Y = d_sub_bf16 - d_mean # [N, 4096]

    # Use CPU to avoid CUDA OOM alongside running 85GB server engine
    device = torch.device("cpu")
    print(f"Computing Ridge Alignment Projector P on {device}...")
    X_dev = X.to(device)
    Y_dev = Y.to(device)

    # Solve P^T = (X^T X + lambda I)^-1 X^T Y -> [5120, 4096]
    lambda_reg = 1e-2
    XtX = torch.matmul(X_dev.T, X_dev) # [5120, 5120]
    XtX.diagonal().add_(lambda_reg * torch.trace(XtX) / 5120.0)
    XtY = torch.matmul(X_dev.T, Y_dev) # [5120, 4096]

    # Solve XtX * PT = XtY
    PT = torch.linalg.solve(XtX, XtY) # [5120, 4096]
    P = PT.T # [4096, 5120]

    # Bias offset for centering: b_align = d_mean - q_mean @ PT
    b_align = d_mean.to(device) - torch.matmul(q_mean.to(device), PT) # [1, 4096]
    b_align = b_align.squeeze(0) # [4096]

    print(f"Alignment Projector P computed: shape {P.shape}, norm {torch.norm(P).item():.4f}")

    print("Step 3: Loading Qwen's trained visual merger fc2 weights...")
    q_fc2_w_meta = q_mf["dense_tensors"]["model.visual.merger.linear_fc2.weight"]
    q_fc2_b_meta = q_mf["dense_tensors"]["model.visual.merger.linear_fc2.bias"]

    q_fc2_w_u16 = np.memmap(qwen_dense_bin, dtype=np.uint16, mode='r', offset=q_fc2_w_meta["offset"], shape=(q_fc2_w_meta["shape"][0], q_fc2_w_meta["shape"][1]))
    q_fc2_b_u16 = np.memmap(qwen_dense_bin, dtype=np.uint16, mode='r', offset=q_fc2_b_meta["offset"], shape=(q_fc2_b_meta["shape"][0],))

    W_qwen = torch.from_numpy(np.array(q_fc2_w_u16, copy=True)).view(torch.bfloat16).to(torch.float32).to(device) # [5120, 4608]
    b_qwen = torch.from_numpy(np.array(q_fc2_b_u16, copy=True)).view(torch.bfloat16).to(torch.float32).to(device) # [5120]

    print(f"W_qwen: {W_qwen.shape}, b_qwen: {b_qwen.shape}")

    # Compose: W_bridge = P @ W_qwen -> [4096, 4608]
    # b_bridge = P @ b_qwen + b_align -> [4096]
    W_bridge = torch.matmul(P, W_qwen) # [4096, 4608]
    b_bridge = torch.matmul(P, b_qwen) + b_align # [4096]

    print(f"Bridge Projector W shape: {W_bridge.shape}, norm: {W_bridge.norm().item():.4f}")
    print(f"Bridge Projector b shape: {b_bridge.shape}, norm: {b_bridge.norm().item():.4f}")
    print(f"Bridge Projector row norm mean: {W_bridge.norm(dim=1).mean().item():.4f}")

    # Convert to bfloat16
    W_bridge_bf16 = W_bridge.to(torch.bfloat16).view(torch.uint16).cpu().numpy().tobytes()
    b_bridge_bf16 = b_bridge.to(torch.bfloat16).view(torch.uint16).cpu().numpy().tobytes()

    os.makedirs(os.path.dirname(out_bridge_bin), exist_ok=True)
    with open(out_bridge_bin, "wb") as f:
        f.write(W_bridge_bf16)
        f.write(b_bridge_bf16)

    total_bytes = len(W_bridge_bf16) + len(b_bridge_bf16)
    print(f"Successfully generated {out_bridge_bin} ({total_bytes} bytes = {total_bytes / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()

