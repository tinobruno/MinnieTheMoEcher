#!/usr/bin/env python3
import json
import os
import sys
import numpy as np
import torch

def train_and_save_projector(lam=1.0, out_bin="models/frankenstin/bridge_fc2.bin"):
    print("=== Training FrankensTin Vision Bridge Projector (Phase 2: Bayesian MAP) ===")
    
    # 1. Load dataset
    with open("scratch/alignment_dataset.json") as f:
        dataset = json.load(f)

    ds_dense_bin = "models/deepseek_v4_flash_q4/attention_dense_layers_q4.bin"
    embed_mmap = np.memmap(ds_dense_bin, dtype=np.uint16, mode="r", offset=0, shape=(129280, 4096))

    with open("models/deepseek_v4_flash_q4/tokenizer.json") as f:
        vocab = json.load(f)["model"]["vocab"]
    inv_vocab = {v: k for k, v in vocab.items()}

    X_list = []
    Y_list = []
    weights = []
    meta = []

    print(f"Loading {len(dataset)} scenes from dataset...")
    for scene in dataset:
        bin_path = scene["bin_path"]
        if not os.path.exists(bin_path):
            continue
        
        # Load [576, 4608] features in BF16
        raw_feat = np.fromfile(bin_path, dtype=np.uint16).reshape(576, 4608)
        feat_t = torch.from_numpy(raw_feat).view(torch.bfloat16).float()

        is_real = ("bmw" in bin_path) or ("healey" in bin_path) or ("ferrari" in bin_path) or ("apple" in bin_path)
        w_val = 15.0 if is_real else 1.0

        for pair in scene["pairs"]:
            pidx = pair["patch_idx"]
            tid = pair["token_id"]
            name = pair.get("name", str(tid))

            x_vec = feat_t[pidx] # [4608]
            # Homogeneous coordinate: [4609]
            x_homo = torch.cat([x_vec, torch.tensor([1.0], dtype=torch.float32)])

            y_vec = torch.from_numpy(np.array(embed_mmap[tid], copy=True)).view(torch.bfloat16).float()

            X_list.append(x_homo)
            Y_list.append(y_vec)
            weights.append(w_val)
            meta.append((name, tid, is_real, bin_path, pidx))

    X = torch.stack(X_list, dim=1) # [4609, N]
    Y = torch.stack(Y_list, dim=1) # [4096, N]
    W_diag = torch.tensor(weights, dtype=torch.float32) # [N]
    N = X.shape[1]
    print(f"Loaded {N} grounded pairs. X: {X.shape}, Y: {Y.shape}")

    # 2. Load Prior (Clean Semantic Ridge Projector)
    prior_bin = "scratch/bridge_clean_prior.bin" if os.path.exists("scratch/bridge_clean_prior.bin") else out_bin
    print(f"Loading prior from {prior_bin}...")
    raw_prior = np.fromfile(prior_bin, dtype=np.uint16)
    W_prior = torch.from_numpy(raw_prior[:4096 * 4608].reshape(4096, 4608)).view(torch.bfloat16).float()
    b_prior = torch.from_numpy(raw_prior[4096 * 4608: 4096 * 4608 + 4096]).view(torch.bfloat16).float()
    W_tilde_prior = torch.cat([W_prior, b_prior.unsqueeze(1)], dim=1) # [4096, 4609]
    print(f"Prior loaded: {W_tilde_prior.shape}, Fro norm = {torch.norm(W_tilde_prior).item():.4f}")

    # 3. Solve MAP Normal Equations
    # min_W || (W X - Y) W_diag^{1/2} ||_F^2 + lam || W - W_prior ||_F^2
    # W (X W_diag X^T + lam I) = Y W_diag X^T + lam W_prior
    print(f"Solving MAP normal equations with lambda = {lam}...")
    X_weighted = X * W_diag.unsqueeze(0) # [4609, N]
    XXT = torch.matmul(X_weighted, X.T)  # [4609, 4609]
    YXT = torch.matmul(Y, X_weighted.T)  # [4096, 4609]

    A = XXT + lam * torch.eye(4609)
    B = YXT + lam * W_tilde_prior

    W_tilde_opt = torch.linalg.solve(A, B.T).T # [4096, 4609]
    print(f"Optimization solved successfully. Optimized W_tilde shape: {W_tilde_opt.shape}")

    # 4. Decompose into W and b
    W_opt = W_tilde_opt[:, :4608] # [4096, 4608]
    b_opt = W_tilde_opt[:, 4608]  # [4096]

    # 5. Evaluate grounded targets
    Y_hat = torch.matmul(W_tilde_opt, X)
    cos_sim = torch.nn.functional.cosine_similarity(Y_hat, Y, dim=0)
    print(f"\n--- Validation Metrics ---")
    print(f"Overall Grounded Cosine Sim: {cos_sim.mean().item():.4f}")
    
    real_targets = []
    for i, (name, tid, is_real, bpath, pidx) in enumerate(meta):
        if is_real:
            real_targets.append((name, tid, cos_sim[i].item(), bpath, pidx))
    print(f"Real Target Mean Cosine Sim: {np.mean([c for _, _, c, _, _ in real_targets]):.4f}")
    print("\nSample Real Target Accuracies:")
    for name, tid, c, bpath, pidx in real_targets:
        r = pidx // 24
        c_idx = pidx % 24
        print(f"  {os.path.basename(bpath):24s} Patch({r:02d}, {c_idx:02d}) {name:8s} -> cos = {c:.4f}")

    # 6. Save to binary
    print(f"\nWriting updated bridge weights to {out_bin}...")
    os.system(f"cp {out_bin} {out_bin}.bak")
    
    W_bytes = W_opt.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
    b_bytes = b_opt.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()

    with open(out_bin, "wb") as f:
        f.write(W_bytes)
        f.write(b_bytes)

    total_bytes = len(W_bytes) + len(b_bytes)
    print(f"Success! Saved {total_bytes} bytes ({total_bytes / (1024*1024):.2f} MB) to {out_bin}")

if __name__ == "__main__":
    train_and_save_projector(lam=5.0)
