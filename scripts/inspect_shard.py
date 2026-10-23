import struct
import json
import sys
from pathlib import Path

def inspect(path):
    p = Path(path)
    if not p.exists():
        print(f"File {p} does not exist.")
        return
    with open(p, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    print(f"\n--- {p.name} ({len(header)} keys) ---")
    for k, v in list(header.items())[:25]:
        if k == "__metadata__":
            continue
        print(f"{k}: dtype={v.get('dtype')}, shape={v.get('shape')}")

def inspect_config():
    p = Path("E:/dev/nemotron_bolt_24gb_raw/config.json")
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        c = json.load(f)
    if "llm_config" in c:
        c = c["llm_config"]
    bcs = c.get("block_configs", [])
    print(f"Total layers in block_configs: {len(bcs)}")
    for i, b in enumerate(bcs):
        btype = b.get("block_type")
        if btype == "moe":
            n_routed = b.get("n_routed_experts")
            inter = b.get("moe_intermediate_size")
            latent = b.get("moe_latent_size", c.get("moe_latent_size", 1024))
            print(f"Layer {i}: MoE n_routed={n_routed}, inter={inter}, latent={latent}")

def inspect_index():
    p = Path("E:/dev/nemotron_bolt_24gb_raw/model.safetensors.index.json")
    if not p.exists():
        p = Path("C:/Users/tino/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/snapshots/c6cc2e8ffa353403213d6b9b18aaf8eb684d705a/model.safetensors.index.json")
    if not p.exists():
        print("Index file not found.")
        return
    with open(p, "r", encoding="utf-8") as f:
        idx = json.load(f)["weight_map"]
    print(f"Index loaded with {len(idx)} keys.")
    print("Layer 1 non-expert tensors:")
    for k, v in idx.items():
        if "layers.1.mixer." in k and "experts." not in k:
            print(f"  {k} -> {v}")
    print("\nLayer 1 fc / shared tensors:")
    for k, v in idx.items():
        if "layers.1." in k and ("fc" in k or "shared" in k):
            print(f"  {k} -> {v}")

import numpy as np
import time

def benchmark_q8_0():
    raw = np.random.bytes(1024 * 128 * 34)
    t0 = time.time()
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    scales = blocks[:, :2].view(np.float16).astype(np.float32)
    quants = blocks[:, 2:].view(np.int8).astype(np.float32)
    res = (quants * scales).astype(np.float32)
    # convert float32 to bfloat16:
    # float32 has 1 sign, 8 exponent, 23 mantissa.
    # bfloat16 is the top 16 bits of float32!
    bf16_bytes = (res.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    dt = time.time() - t0
    print(f"Dequantized 4.45MB Q8_0 -> 8.38MB BF16 in {dt*1000:.2f}ms! (out bytes: {len(bf16_bytes)})")

def inspect_manifest():
    p = Path("models/nemotron_3.5_bolt_24gb/moecher_manifest.json")
    if not p.exists():
        print("Manifest not found")
        return
    with open(p, "r", encoding="utf-8") as f:
        m = json.load(f)
    dt = m.get("dense_tensors", {})
    print(f"Total dense tensors in manifest: {len(dt)}")
    keys = list(dt.keys())
    print("Sample keys:")
    for k in keys[:15]:
        print(f"  {k} ({dt[k].get('dtype')}, {dt[k].get('shape')})")
    fc_keys = [k for k in keys if "fc" in k or "shared" in k or "gate" in k]
    print(f"\nFound {len(fc_keys)} fc/shared/gate keys. Sample:")
    for k in fc_keys[:15]:
        print(f"  {k}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "manifest":
        inspect_manifest()
    elif len(sys.argv) > 1 and sys.argv[1] == "bench":
        benchmark_q8_0()
    elif len(sys.argv) > 1 and sys.argv[1] == "config":
        inspect_config()
    elif len(sys.argv) > 1 and sys.argv[1] == "index":
        inspect_index()
    else:
        inspect_manifest()
