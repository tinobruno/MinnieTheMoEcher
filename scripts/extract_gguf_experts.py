#!/usr/bin/env python3
"""
Extracts IQ2_XXS and Q2_K MoE expert weights directly from GGUF file
(e.g., model.gguf) into MinnieTheMoEcher binary format (moe_experts_iq2.bin)
and creates the corresponding manifest (moecher_manifest_iq2.json).
"""

import os
import sys
import mmap
import json
import struct
import argparse
from tqdm import tqdm

def parse_gguf_header(f):
    magic, ver, n_tensors, n_kv = struct.unpack('<4sIQQ', f.read(24))
    if magic != b'GGUF':
        raise ValueError(f"Invalid GGUF magic: {magic}")
    
    def read_str():
        l = struct.unpack('<Q', f.read(8))[0]
        return f.read(l).decode('utf-8')

    def skip_val(vtype):
        if vtype in (0, 1, 7): f.seek(1, 1)
        elif vtype in (2, 3): f.seek(2, 1)
        elif vtype in (4, 5, 6): f.seek(4, 1)
        elif vtype in (10, 11, 12): f.seek(8, 1)
        elif vtype == 8:
            l = struct.unpack('<Q', f.read(8))[0]
            f.seek(l, 1)
        elif vtype == 9:
            etype, elen = struct.unpack('<IQ', f.read(12))
            for _ in range(elen):
                skip_val(etype)

    alignment = 32
    for _ in range(n_kv):
        k = read_str()
        vtype = struct.unpack('<I', f.read(4))[0]
        if k == 'general.alignment':
            alignment = struct.unpack('<I', f.read(4))[0]
        else:
            skip_val(vtype)

    tensors = {}
    for _ in range(n_tensors):
        name = read_str()
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype, offset = struct.unpack('<IQ', f.read(12))
        tensors[name] = {'dims': dims, 'type': ttype, 'offset': offset}

    data_start = (f.tell() + alignment - 1) // alignment * alignment
    return tensors, data_start

def main():
    parser = argparse.ArgumentParser(description="Extract MoE expert weights from GGUF to MoEcher binary")
    parser.add_argument("--gguf", type=str, default="model.gguf", help="Path to GGUF file")
    parser.add_argument("--manifest-in", type=str, default="moecher_manifest.json", help="Input manifest template")
    parser.add_argument("--output-bin", type=str, default="moe_experts_iq2.bin", help="Output expert binary path")
    parser.add_argument("--output-manifest", type=str, default="moecher_manifest_iq2.json", help="Output manifest path")
    args = parser.parse_args()

    print(f"Opening GGUF file: {args.gguf}")
    file_size = os.path.getsize(args.gguf)
    
    with open(args.gguf, "rb") as f:
        tensors, data_start = parse_gguf_header(f)

    print(f"GGUF parsed: {len(tensors)} tensors, tensor data starts at byte {data_start}")

    # Inspect expert dimensions
    # Layer 0..42, Gate [4096, 2048, 256], Up [4096, 2048, 256], Down [2048, 4096, 256]
    n_layers = 43
    n_experts = 256

    w1_bytes = 2048 * (4096 // 256) * 66 # 2,162,688
    w3_bytes = 2048 * (4096 // 256) * 66 # 2,162,688
    w2_bytes = 4096 * (2048 // 256) * 84 # 2,752,512
    expert_total_bytes = w1_bytes + w3_bytes + w2_bytes # 7,077,888
    
    assert expert_total_bytes % 4096 == 0, f"Expert size {expert_total_bytes} not page aligned"
    print(f"Expert layout: w1={w1_bytes} bytes (IQ2_XXS), w3={w3_bytes} bytes (IQ2_XXS), w2={w2_bytes} bytes (Q2_K)")
    print(f"Total per expert: {expert_total_bytes} bytes ({expert_total_bytes // 4096} pages)")

    total_bin_size = n_layers * n_experts * expert_total_bytes
    print(f"Creating output binary {args.output_bin} ({total_bin_size / (1024**3):.2f} GiB)...")

    with open(args.gguf, "rb") as f_in, open(args.output_bin, "w+b") as f_out:
        # Preallocate output file
        f_out.truncate(total_bin_size)
        f_out.flush()

        mm_in = mmap.mmap(f_in.fileno(), 0, access=mmap.ACCESS_READ)
        mm_out = mmap.mmap(f_out.fileno(), 0, access=mmap.ACCESS_WRITE)

        pbar = tqdm(total=n_layers * n_experts, desc="Extracting MoE experts")

        for layer in range(n_layers):
            gate_key = f"blk.{layer}.ffn_gate_exps.weight"
            up_key   = f"blk.{layer}.ffn_up_exps.weight"
            down_key = f"blk.{layer}.ffn_down_exps.weight"

            gate_info = tensors[gate_key]
            up_info   = tensors[up_key]
            down_info = tensors[down_key]

            gate_base = data_start + gate_info['offset']
            up_base   = data_start + up_info['offset']
            down_base = data_start + down_info['offset']

            for exp in range(n_experts):
                out_offset = (layer * n_experts + exp) * expert_total_bytes

                # 1. Gate (w1)
                g_src = gate_base + exp * w1_bytes
                mm_out[out_offset : out_offset + w1_bytes] = mm_in[g_src : g_src + w1_bytes]

                # 2. Up (w3)
                u_src = up_base + exp * w3_bytes
                mm_out[out_offset + w1_bytes : out_offset + w1_bytes + w3_bytes] = mm_in[u_src : u_src + w3_bytes]

                # 3. Down (w2)
                d_src = down_base + exp * w2_bytes
                mm_out[out_offset + w1_bytes + w3_bytes : out_offset + expert_total_bytes] = mm_in[d_src : d_src + w2_bytes]

                pbar.update(1)

        pbar.close()
        mm_out.flush()
        mm_out.close()
        mm_in.close()

    print(f"Successfully extracted {n_layers * n_experts} experts to {args.output_bin}")

    # Generate manifest
    print(f"Generating manifest {args.output_manifest}...")
    with open(args.manifest_in, "r") as f:
        manifest = json.load(f)

    manifest["expert_bin"] = os.path.abspath(args.output_bin)
    manifest["model_config"]["expert_dtype"] = "iq2_xxs"
    manifest["expert_layout"]["block_size"] = expert_total_bytes
    manifest["expert_layout"]["part_order"] = ["w1.weight", "w3.weight", "w2.weight"]
    manifest["expert_layout"]["parts"] = {
        "w1.weight": {
            "offset_in_block": 0,
            "nbytes": w1_bytes,
            "dtype": "IQ2_XXS",
            "shape": [2048, 4096]
        },
        "w3.weight": {
            "offset_in_block": w1_bytes,
            "nbytes": w3_bytes,
            "dtype": "IQ2_XXS",
            "shape": [2048, 4096]
        },
        "w2.weight": {
            "offset_in_block": w1_bytes + w3_bytes,
            "nbytes": w2_bytes,
            "dtype": "Q2_K",
            "shape": [4096, 2048]
        }
    }

    with open(args.output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {args.output_manifest}")

if __name__ == "__main__":
    main()
