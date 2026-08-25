import os
import sys
sys.path.insert(0, '.')
import torch
import json
from scripts.quantize_experts_iq2 import (
    dequantize_fp4_gpu,
    quantize_iq2_xxs_gpu,
    quantize_q2_k_gpu,
    DEVICE
)

with open("moecher_manifest.json", "r") as f:
    manifest = json.load(f)

expert_bin_path = manifest["expert_bin"]
layout = manifest["expert_layout"]
parts = layout["parts"]
src_block_size = layout["block_size"]

with open(expert_bin_path, "rb") as fin:
    block_data = fin.read(src_block_size)

# w1
w1_packed = torch.frombuffer(block_data[parts["w1.weight"]["offset_in_block"] : parts["w1.weight"]["offset_in_block"] + parts["w1.weight"]["nbytes"]], dtype=torch.uint8).clone().to(DEVICE).view(2048, 2048)
w1_scales = torch.frombuffer(block_data[parts["w1.scale"]["offset_in_block"] : parts["w1.scale"]["offset_in_block"] + parts["w1.scale"]["nbytes"]], dtype=torch.uint8).clone().to(DEVICE).view(2048, 128)
w1_f32 = dequantize_fp4_gpu(w1_packed, w1_scales, 2048, 4096, 128)

# w3
w3_packed = torch.frombuffer(block_data[parts["w3.weight"]["offset_in_block"] : parts["w3.weight"]["offset_in_block"] + parts["w3.weight"]["nbytes"]], dtype=torch.uint8).clone().to(DEVICE).view(2048, 2048)
w3_scales = torch.frombuffer(block_data[parts["w3.scale"]["offset_in_block"] : parts["w3.scale"]["offset_in_block"] + parts["w3.scale"]["nbytes"]], dtype=torch.uint8).clone().to(DEVICE).view(2048, 128)
w3_f32 = dequantize_fp4_gpu(w3_packed, w3_scales, 2048, 4096, 128)

# w2
w2_packed = torch.frombuffer(block_data[parts["w2.weight"]["offset_in_block"] : parts["w2.weight"]["offset_in_block"] + parts["w2.weight"]["nbytes"]], dtype=torch.uint8).clone().to(DEVICE).view(4096, 1024)
w2_scales = torch.frombuffer(block_data[parts["w2.scale"]["offset_in_block"] : parts["w2.scale"]["offset_in_block"] + parts["w2.scale"]["nbytes"]], dtype=torch.uint8).clone().to(DEVICE).view(4096, 64)
w2_f32 = dequantize_fp4_gpu(w2_packed, w2_scales, 4096, 2048, 64)

w1_bytes = quantize_iq2_xxs_gpu(w1_f32)
w3_bytes = quantize_iq2_xxs_gpu(w3_f32)
w2_bytes = quantize_q2_k_gpu(w2_f32)

with open("moe_test_expert.bin", "wb") as fout:
    fout.write(w1_bytes)
    fout.write(w3_bytes)
    fout.write(w2_bytes)

print(f"Wrote moe_test_expert.bin: {len(w1_bytes) + len(w3_bytes) + len(w2_bytes)} bytes")
