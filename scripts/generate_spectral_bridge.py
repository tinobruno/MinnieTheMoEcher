#!/usr/bin/env python3
import json
import os
import sys
import numpy as np
import torch

def generate_spectral_bridge():
    qwen_manifest_path = "models/qwen3_8_27b/moecher_manifest_qwen.json"
    qwen_dense_bin = "models/qwen3_8_27b/attention_dense_layers.bin"
    qwen_tok_path = "models/qwen3_8_27b/tokenizer.json"

    ds_manifest_path = "models/deepseek_v4_flash_q4/moecher_manifest_mixed.json"
    ds_dense_bin = "models/deepseek_v4_flash_q4/attention_dense_layers_q4.bin"
    ds_tok_path = "models/deepseek_v4_flash_q4/tokenizer.json"

    out_bridge_bin = "models/frankenstin/bridge_fc2.bin"

    print("Step 1: Loading shared vocabulary...")
    with open(qwen_tok_path) as f: q_vocab = json.load(f)["model"]["vocab"]
    with open(ds_tok_path) as f: d_vocab = json.load(f)["model"]["vocab"]
    common = sorted(list(set(q_vocab.keys()) & set(d_vocab.keys())), key=lambda t: q_vocab[t] + d_vocab[t])[:50000]
    qi = [q_vocab[t] for t in common]
    di = [d_vocab[t] for t in common]

    with open(qwen_manifest_path) as f: q_mf = json.load(f)
    with open(ds_manifest_path) as f: ds_mf = json.load(f)
    qm = q_mf["dense_tensors"]["model.language_model.embed_tokens.weight"]
    dm = ds_mf["dense_tensors"]["embed.weight"]

    print("Step 2: Loading embedding tensors...")
    X = torch.from_numpy(np.array(np.memmap(qwen_dense_bin, dtype=np.uint16, mode="r", offset=qm["offset"], shape=tuple(qm["shape"]))[qi], copy=True)).view(torch.bfloat16).float()
    Y = torch.from_numpy(np.array(np.memmap(ds_dense_bin, dtype=np.uint16, mode="r", offset=dm["offset"], shape=tuple(dm["shape"]))[di], copy=True)).view(torch.bfloat16).float()

    q_mean = X.mean(dim=0, keepdim=True)
    d_mean = Y.mean(dim=0, keepdim=True)
    X_c = X - q_mean
    Y_c = Y - d_mean

    X_n = torch.nn.functional.normalize(X_c, dim=1)
    Y_n = torch.nn.functional.normalize(Y_c, dim=1)

    print("Step 3: Computing SVD of cross-covariance matrix...")
    M = torch.matmul(X_n.T, Y_n)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    k = 512
    p = 0.2
    weights = (S[:k] / S[0]) ** p
    PT = torch.matmul(U[:, :k] * weights, Vh[:k, :])
    P = PT.T
    b_align = d_mean.squeeze(0) - torch.matmul(q_mean, PT).squeeze(0)

    print("Step 4: Composing with Qwen visual merger linear_fc2...")
    fc2_w_m = q_mf["dense_tensors"]["model.visual.merger.linear_fc2.weight"]
    fc2_b_m = q_mf["dense_tensors"]["model.visual.merger.linear_fc2.bias"]
    W_qwen = torch.from_numpy(np.array(np.memmap(qwen_dense_bin, dtype=np.uint16, mode="r", offset=fc2_w_m["offset"], shape=tuple(fc2_w_m["shape"])), copy=True)).view(torch.bfloat16).float()
    b_qwen = torch.from_numpy(np.array(np.memmap(qwen_dense_bin, dtype=np.uint16, mode="r", offset=fc2_b_m["offset"], shape=tuple(fc2_b_m["shape"])), copy=True)).view(torch.bfloat16).float()

    W_bridge = torch.matmul(P, W_qwen) # [4096, 4608]
    b_bridge = torch.matmul(P, b_qwen) + b_align # [4096]

    # Calibration scale factor to ensure visual token norms match DeepSeek text embedding norm ~4.5
    scale = 0.856299
    W_bridge_scaled = W_bridge * scale
    b_bridge_scaled = b_bridge * scale

    print(f"Step 5: Writing calibrated bridge to {out_bridge_bin}...")
    W_out = W_bridge_scaled.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
    b_out = b_bridge_scaled.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()

    with open(out_bridge_bin, "wb") as f:
        f.write(W_out)
        f.write(b_out)

    total_bytes = len(W_out) + len(b_out)
    print(f"Done! Saved {total_bytes} bytes ({total_bytes / (1024*1024):.2f} MB) to {out_bridge_bin}")

if __name__ == "__main__":
    generate_spectral_bridge()
