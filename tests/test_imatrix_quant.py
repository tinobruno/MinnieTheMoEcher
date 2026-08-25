#!/usr/bin/env python3
"""
Test imatrix parser and weighted quantization in scripts/quantize_experts.py
"""

import os
import sys
import struct
import tempfile
import torch
import numpy as np

# Add scripts directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from quantize_experts import ImatrixLoader, quantize_to_int2_asymmetric


def create_synthetic_dat_file(path, n_layers=2, n_experts=4, n_cols=256):
    """
    Creates a valid binary .dat imatrix file with simulated expert activation data.
    """
    with open(path, "wb") as f:
        # Total entries: n_layers * 3 (gate, up, down)
        entries = []
        for l in range(n_layers):
            entries.append((f"blk.{l}.ffn_gate_exps.weight", n_experts * n_cols))
            entries.append((f"blk.{l}.ffn_up_exps.weight", n_experts * n_cols))
            entries.append((f"blk.{l}.ffn_down_exps.weight", n_experts * n_cols))

        f.write(struct.pack("<i", len(entries)))

        for name, nval in entries:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<i", 100)) # ncall
            f.write(struct.pack("<i", nval)) # nval
            # Generate random non-negative importance values
            vals = torch.rand(nval, dtype=torch.float32) + 0.1
            f.write(vals.numpy().tobytes())


def test_imatrix_loader():
    print("Testing ImatrixLoader...")
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        create_synthetic_dat_file(tmp_path, n_layers=2, n_experts=4, n_cols=256)
        loader = ImatrixLoader(tmp_path)
        assert len(loader.entries) == 6, f"Expected 6 entries, got {len(loader.entries)}"

        # Test lookup for layer 0, expert 2, gate part (w1)
        w = loader.get_expert_weights(layer_idx=0, expert_idx=2, part_name="w1.weight", cols=256)
        assert w is not None, "Failed to retrieve expert weights"
        assert w.shape[0] == 256, f"Expected shape [256], got {w.shape}"
        assert torch.isclose(w.mean(), torch.tensor(1.0), atol=1e-3), f"Expected normalized mean ~1.0, got {w.mean()}"

        # Test lookup for down part (w2)
        w2 = loader.get_expert_weights(layer_idx=1, expert_idx=3, part_name="w2.weight", cols=256)
        assert w2 is not None
        assert w2.shape[0] == 256

        # Test missing layer
        w_none = loader.get_expert_weights(layer_idx=99, expert_idx=0, part_name="w1.weight", cols=256)
        assert w_none is None

        print("✓ ImatrixLoader passed successfully!")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_weighted_quantization():
    print("Testing quantize_to_int2_asymmetric...")
    torch.manual_seed(42)
    rows = 128
    cols = 256
    block_size = 64

    x = torch.randn(rows, cols, dtype=torch.float32)

    # 1. Unweighted quantization
    packed, scale, xmin = quantize_to_int2_asymmetric(x, block_size=block_size, col_weights=None)
    assert packed.shape == (rows, cols // 4)
    assert scale.shape == (rows, cols // block_size)
    assert xmin.shape == (rows, cols // block_size)

    # 2. Weighted quantization: columns 0..31 are given 100x importance
    weights = torch.ones(cols, dtype=torch.float32)
    weights[0:32] = 100.0
    weights = weights / weights.mean()

    packed_w, scale_w, xmin_w = quantize_to_int2_asymmetric(x, block_size=block_size, col_weights=weights)
    assert packed_w.shape == (rows, cols // 4)
    assert scale_w.shape == (rows, cols // block_size)

    # Reconstruct unweighted vs weighted for the high-importance columns in block 0
    # Unpack block 0
    b0_q = (packed[:, 0:16].unsqueeze(2) >> torch.tensor([0, 2, 4, 6])) & 3
    b0_q = b0_q.view(rows, 64)
    recon_unweighted = b0_q * scale[:, 0:1].float() + xmin[:, 0:1].float()
    err_unweighted = ((x[:, 0:32] - recon_unweighted[:, 0:32]) ** 2).mean().item()

    b0_qw = (packed_w[:, 0:16].unsqueeze(2) >> torch.tensor([0, 2, 4, 6])) & 3
    b0_qw = b0_qw.view(rows, 64)
    recon_weighted = b0_qw * scale_w[:, 0:1].float() + xmin_w[:, 0:1].float()
    err_weighted = ((x[:, 0:32] - recon_weighted[:, 0:32]) ** 2).mean().item()

    print(f"  MSE on critical columns: Unweighted = {err_unweighted:.5f}, Weighted = {err_weighted:.5f}")
    assert err_weighted <= err_unweighted + 1e-4, "Weighted MSE did not prioritize critical columns!"

    print("✓ quantize_to_int2_asymmetric passed successfully!")


if __name__ == "__main__":
    test_imatrix_loader()
    test_weighted_quantization()
    print("\nAll tests passed successfully!")
