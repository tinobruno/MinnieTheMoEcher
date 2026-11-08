#!/usr/bin/env python3
"""
Convert Qwen3.8-Flash-Next GGUF shards into MinnieTheMoEcher native binary pack:
  - resident.bin: All always-active weights (embeddings, output head, Layer 1 PLE projections,
                  48 layers of GR, GDN/QSA, Router, and Shared Experts), 256-byte aligned.
  - experts.bin:  All 12,288 routed experts (48 layers x 256 experts), each packed as a
                  single contiguous [gate | up | down] binary block for single-DMA streaming.
  - expert_freq.bin & imatrix.dat.counts.json: Per-expert activation frequency ranking.
  - moecher_manifest.json: Standard MinnieTheMoEcher model manifest.
"""

import json
import os
import struct
import sys
import time
from pathlib import Path

GGML_ROW_CALC = {
    0:  lambda cols: cols * 4,                # F32
    1:  lambda cols: cols * 2,                # F16
    2:  lambda cols: (cols // 32) * 18,       # Q4_0
    3:  lambda cols: (cols // 32) * 20,       # Q4_1
    6:  lambda cols: (cols // 32) * 22,       # Q5_0
    7:  lambda cols: (cols // 32) * 24,       # Q5_1
    8:  lambda cols: (cols // 32) * 34,       # Q8_0
    10: lambda cols: (cols // 256) * 84,      # Q2_K
    11: lambda cols: (cols // 256) * 110,     # Q3_K
    12: lambda cols: (cols // 256) * 144,     # Q4_K
    13: lambda cols: (cols // 256) * 176,     # Q5_K
    14: lambda cols: (cols // 256) * 210,     # Q6_K
    16: lambda cols: (cols // 256) * 66,      # IQ2_XXS
    17: lambda cols: (cols // 256) * 74,      # IQ2_XS
    18: lambda cols: (cols // 256) * 98,      # IQ3_XXS
    19: lambda cols: (cols // 256) * 50,      # IQ1_S
    20: lambda cols: (cols // 32) * 18,       # IQ4_NL
    21: lambda cols: (cols // 256) * 110,     # IQ3_S
    22: lambda cols: (cols // 256) * 82,      # IQ2_S
    23: lambda cols: (cols // 256) * 136,     # IQ4_XS
    29: lambda cols: (cols // 256) * 56,      # IQ1_M
    30: lambda cols: cols * 2,                # BF16
    40: lambda cols: (cols // 128) * 18,      # Q1_0
    42: lambda cols: (cols // 64) * 18,       # Q2_0
}

GGML_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1",
    8: "q8_0", 10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k",
    16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl",
    21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 29: "iq1_m", 30: "bf16",
    40: "q1_0", 42: "q2_0",
}


def parse_gguf_header(path: Path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"Not a GGUF file: {path}"
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        alignment = 32

        def read_str():
            (length,) = struct.unpack("<Q", f.read(8))
            return f.read(length).decode("utf-8")

        def skip_val(vtype):
            if vtype in (0, 1, 7):
                f.seek(1, os.SEEK_CUR)
            elif vtype in (2, 3):
                f.seek(2, os.SEEK_CUR)
            elif vtype in (4, 5, 6):
                f.seek(4, os.SEEK_CUR)
            elif vtype in (10, 11, 12):
                f.seek(8, os.SEEK_CUR)
            elif vtype == 8:
                (length,) = struct.unpack("<Q", f.read(8))
                f.seek(length, os.SEEK_CUR)
            elif vtype == 9:
                etype, cnt = struct.unpack("<IQ", f.read(12))
                for _ in range(cnt):
                    skip_val(etype)
            else:
                raise ValueError(f"Unknown GGUF KV type {vtype}")

        for _ in range(n_kv):
            k = read_str()
            (vtype,) = struct.unpack("<I", f.read(4))
            if k == "general.alignment" and vtype == 4:
                (alignment,) = struct.unpack("<I", f.read(4))
            else:
                skip_val(vtype)

        tensors = {}
        for _ in range(n_tensors):
            name = read_str()
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type, rel_offset = struct.unpack("<IQ", f.read(12))
            n_rows = 1
            for d in dims[1:]:
                n_rows *= d
            row_bytes = GGML_ROW_CALC[ggml_type](dims[0])
            nbytes = row_bytes * n_rows
            tensors[name] = {
                "name": name,
                "dims": dims,
                "ggml_type": ggml_type,
                "dtype": GGML_TYPE_NAMES.get(ggml_type, f"type_{ggml_type}"),
                "rel_offset": rel_offset,
                "nbytes": nbytes,
            }

        header_end = f.tell()
        data_offset = (header_end + alignment - 1) & ~(alignment - 1)
        for ti in tensors.values():
            ti["abs_offset"] = data_offset + ti["rel_offset"]

        return tensors


def main():
    t0 = time.time()
    model_dir = Path(r"e:\dev\MinnieTheMoEcher\models\qwen3.8_flash_next")
    shard1_path = model_dir / "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
    shard2_path = model_dir / "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf"

    print(f"[1/4] Parsing GGUF headers from {shard1_path.name} and {shard2_path.name}...")
    s1_tensors = parse_gguf_header(shard1_path)
    s2_tensors = parse_gguf_header(shard2_path) if shard2_path.exists() else {}
    print(f"      Found {len(s1_tensors)} tensors in Shard 1, {len(s2_tensors)} tensors in Shard 2.")

    resident_path = model_dir / "resident.bin"
    experts_path = model_dir / "experts.bin"

    dense_manifest = {}
    expert_layers = []

    # Separate resident tensors vs routed expert tensors
    routed_names = set()
    for l in range(48):
        routed_names.add(f"blk.{l}.ffn_gate_exps.weight")
        routed_names.add(f"blk.{l}.ffn_up_exps.weight")
        routed_names.add(f"blk.{l}.ffn_down_exps.weight")

    resident_items = [
        ti for name, ti in sorted(s1_tensors.items(), key=lambda kv: kv[1]["abs_offset"])
        if name not in routed_names
    ]

    expected_res_size = sum((ti["nbytes"] + 255) & ~255 for ti in resident_items)
    cur_res_offset = 0
    if resident_path.exists() and resident_path.stat().st_size == expected_res_size:
        print(f"[2/4] {resident_path.name} ({expected_res_size / (1024**3):.3f} GiB) already up-to-date, reusing...")
        for ti in resident_items:
            dense_manifest[ti["name"]] = {
                "offset": cur_res_offset,
                "nbytes": ti["nbytes"],
                "ggml_type": ti["ggml_type"],
                "dtype": ti["dtype"],
                "shape": ti["dims"],
            }
            cur_res_offset += (ti["nbytes"] + 255) & ~255
    else:
        print(f"[2/4] Writing {len(resident_items)} resident tensors to {resident_path.name}...")
        with open(shard1_path, "rb") as f_in, open(resident_path, "wb") as f_res:
            for ti in resident_items:
                f_in.seek(ti["abs_offset"])
                raw = f_in.read(ti["nbytes"])
                assert len(raw) == ti["nbytes"], f"Short read on {ti['name']}"
                f_res.write(raw)

                dense_manifest[ti["name"]] = {
                    "offset": cur_res_offset,
                    "nbytes": ti["nbytes"],
                    "ggml_type": ti["ggml_type"],
                    "dtype": ti["dtype"],
                    "shape": ti["dims"],
                }
                cur_res_offset += ti["nbytes"]
                # Pad to 256-byte boundary
                pad = (256 - (cur_res_offset % 256)) % 256
                if pad > 0:
                    f_res.write(b"\x00" * pad)
                    cur_res_offset += pad

        print(f"      Wrote {resident_path.name}: {cur_res_offset / (1024**3):.3f} GiB in {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"[3/4] Repacking 48 x 256 = 12,288 routed experts into contiguous [gate | up | down] blocks in {experts_path.name}...")
    cur_exp_offset = 0
    n_experts = 256
    with open(shard1_path, "rb") as f_in, open(experts_path, "wb") as f_exp:
        for l in range(48):
            t_g = s1_tensors[f"blk.{l}.ffn_gate_exps.weight"]
            t_u = s1_tensors[f"blk.{l}.ffn_up_exps.weight"]
            t_d = s1_tensors[f"blk.{l}.ffn_down_exps.weight"]

            g_eb = t_g["nbytes"] // n_experts
            u_eb = t_u["nbytes"] // n_experts
            d_eb = t_d["nbytes"] // n_experts
            blob_bytes = g_eb + u_eb + d_eb

            f_in.seek(t_g["abs_offset"])
            g_raw = memoryview(f_in.read(t_g["nbytes"]))
            f_in.seek(t_u["abs_offset"])
            u_raw = memoryview(f_in.read(t_u["nbytes"]))
            f_in.seek(t_d["abs_offset"])
            d_raw = memoryview(f_in.read(t_d["nbytes"]))

            layer_buf = bytearray(n_experts * blob_bytes)
            mv = memoryview(layer_buf)
            for e in range(n_experts):
                base = e * blob_bytes
                mv[base : base + g_eb] = g_raw[e * g_eb : (e + 1) * g_eb]
                mv[base + g_eb : base + g_eb + u_eb] = u_raw[e * u_eb : (e + 1) * u_eb]
                mv[base + g_eb + u_eb : base + blob_bytes] = d_raw[e * d_eb : (e + 1) * d_eb]

            f_exp.write(layer_buf)

            expert_layers.append({
                "layer_id": l,
                "offset": cur_exp_offset,
                "blob_bytes": blob_bytes,
                "gate_bytes": g_eb,
                "up_bytes": u_eb,
                "down_bytes": d_eb,
                "gate_type": t_g["ggml_type"],
                "up_type": t_u["ggml_type"],
                "down_type": t_d["ggml_type"],
            })
            cur_exp_offset += n_experts * blob_bytes
            if (l + 1) % 8 == 0 or l == 47:
                print(f"      Repacked layer {l + 1:2d}/48 ({cur_exp_offset / (1024**3):.2f} GiB)...")

    print(f"      Wrote {experts_path.name}: {cur_exp_offset / (1024**3):.3f} GiB in {time.time() - t1:.1f}s")

    # 4. Generate expert_freq.bin & imatrix.dat.counts.json
    print("[4/4] Generating expert_freq.bin, imatrix.dat.counts.json, and moecher_manifest.json...")
    n_layers = 48
    total_experts = n_layers * n_experts
    counts = [1] * total_experts
    ranked_pairs = []

    strp_candidates = [
        Path(r"C:\Users\tino\.gemini\antigravity\brain\c0560bb6-2171-4365-9e54-6e178c6c4c8f\scratch\Strata\data\expert-profile-coder.bin"),
        Path(r"C:\Users\tino\.gemini\antigravity\brain\c0560bb6-2171-4365-9e54-6e178c6c4c8f\scratch\Strata\data\expert-profile.bin"),
    ]
    for cand in strp_candidates:
        if cand.exists():
            with open(cand, "rb") as fp:
                magic = fp.read(4)
                if magic == b"STRP":
                    ver, nl, ne, want, n_ranked = struct.unpack("<5I", fp.read(20))
                    if nl == n_layers and ne == n_experts:
                        raw_pairs = struct.unpack(f"<{n_ranked * 2}H", fp.read(n_ranked * 4))
                        for rank in range(n_ranked):
                            l_id = raw_pairs[2 * rank]
                            e_id = raw_pairs[2 * rank + 1]
                            score = total_experts - rank
                            counts[l_id * n_experts + e_id] = score
                            ranked_pairs.append([int(l_id), int(e_id)])
                        print(f"      Loaded empirical routing profile ({n_ranked} ranked experts) from {cand.name}")
                        break

    freq_bin_path = model_dir / "expert_freq.bin"
    with open(freq_bin_path, "wb") as fp:
        fp.write(struct.pack("<iiq", n_layers, n_experts, 100000))
        fp.write(struct.pack(f"<{total_experts}I", *counts))

    counts_json = {}
    for l in range(n_layers):
        counts_json[f"layer_{l}"] = counts[l * n_experts : (l + 1) * n_experts]
    with open(model_dir / "imatrix.dat.counts.json", "w", encoding="utf-8") as fp:
        json.dump(counts_json, fp)

    ple_info = None
    if "per_layer_token_embd.weight" in s2_tensors:
        pt = s2_tensors["per_layer_token_embd.weight"]
        ple_info = {
            "file": shard2_path.name,
            "offset": pt["abs_offset"],
            "nbytes": pt["nbytes"],
            "ggml_type": pt["ggml_type"],
            "dtype": pt["dtype"],
            "shape": pt["dims"],
        }

    manifest = {
        "model_name": "Qwen 3.8 Flash-Next (GSQ-RCO-IQ1_M)",
        "model_id": "qwen3.8-flash-next",
        "tokenizer": "tokenizer.json",
        "dense_bin": "resident.bin",
        "expert_bin": "experts.bin",
        "expert_freq_bin": "expert_freq.bin",
        "mmproj_gguf": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
        "ple_ngram_table": ple_info,
        "model_config": {
            "architecture": "qwen4exp",
            "model_name": "Qwen 3.8 Flash-Next (GSQ-RCO-IQ1_M)",
            "model_id": "qwen3.8-flash-next",
            "vocab_size": 248320,
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "moe_intermediate_size": 640,
            "moe_shared_expert_intermediate_size": 640,
            "n_routed_experts": 256,
            "num_experts_per_tok": 10,
            "n_shared_experts": 1,
            "hc_mult": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000.0,
            "max_seq_len": 16384,
            "bos_token_id": 248044,
            "eos_token_id": 248046,
            "expert_dtype": "gsq_rco_iq1_m",
            "default_temperature": 0.7,
            "default_top_p": 0.95,
            "default_top_k": 20,
            "default_min_p": 0.0,
            "default_repetition_penalty": 1.0,
            "default_reasoning_effort": "medium",
            "default_thinking_budget": 2048,
        },
        "dense_tensors": dense_manifest,
        "expert_layout": {
            "n_layers": 48,
            "n_experts": 256,
            "total_bytes": cur_exp_offset,
            "layers": expert_layers,
        },
    }

    manifest_path = model_dir / "moecher_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"Done in {time.time() - t0:.1f}s! Created resident.bin, experts.bin, expert_freq.bin, imatrix.dat.counts.json, and moecher_manifest.json.")


if __name__ == "__main__":
    main()
