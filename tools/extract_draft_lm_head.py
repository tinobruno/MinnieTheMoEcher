"""
Extract and requantize the draft lm_head from the full lm_head.

The full lm_head is [248320, 5120] in BF16 = 2.4 GB.
The draft lm_head is [40000, 5120] in INT8 with scales = ~200 MB + 2.5 MB scales.

This extracts only the rows corresponding to draft_vocab_ids and quantizes to INT8.
"""
import json
import struct
import numpy as np
import os
import sys

def extract_draft_lm_head(manifest_path, draft_vocab_path, output_path):
    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Load draft vocab IDs
    with open(draft_vocab_path, 'r') as f:
        draft_ids = json.load(f)
    
    draft_size = len(draft_ids)
    print(f"Draft vocabulary size: {draft_size}")
    
    # Find lm_head tensor info
    tensors = manifest['dense_tensors']
    lm_head_info = tensors.get('lm_head.weight')
    if not lm_head_info:
        print("ERROR: lm_head.weight not found in manifest")
        return False
    
    vocab_size = lm_head_info['shape'][0]  # 248320
    hidden_size = lm_head_info['shape'][1]  # 5120
    offset = lm_head_info['offset']
    dtype = lm_head_info['dtype']
    
    print(f"Full lm_head: [{vocab_size}, {hidden_size}], dtype={dtype}, offset={offset}")
    
    # Open the binary file
    bin_path = os.path.join(os.path.dirname(manifest_path), manifest.get('dense_bin', 'attention_dense_layers_q4.bin'))
    print(f"Reading from: {bin_path}")
    
    # Read the full lm_head - it's BF16
    row_bytes = hidden_size * 2  # 2 bytes per BF16
    
    # Extract only the draft rows
    draft_rows_bf16 = np.zeros((draft_size, hidden_size), dtype=np.uint16)  # Raw BF16 as uint16
    
    with open(bin_path, 'rb') as f:
        for i, full_id in enumerate(draft_ids):
            row_offset = offset + full_id * row_bytes
            f.seek(row_offset)
            row_data = f.read(row_bytes)
            draft_rows_bf16[i] = np.frombuffer(row_data, dtype=np.uint16)
            
            if i % 5000 == 0:
                print(f"  Extracted {i}/{draft_size} rows...")
    
    print(f"Extracted {draft_size} rows from lm_head")
    
    # Convert BF16 to FP32 for quantization
    def bf16_to_fp32(bf16_array):
        """Convert BF16 (as uint16) to FP32"""
        fp32 = np.zeros(bf16_array.shape, dtype=np.float32)
        # BF16 is just FP32 with bottom 16 bits zeroed
        fp32_view = fp32.view(np.uint32)
        fp32_view[:] = bf16_array.astype(np.uint32) << 16
        return fp32
    
    draft_rows_fp32 = bf16_to_fp32(draft_rows_bf16)
    print(f"Converted to FP32: shape={draft_rows_fp32.shape}")
    
    # Quantize to INT8 with per-row scales (symmetric quantization)
    block_size = 128  # Group size for quantization
    n_groups_per_row = (hidden_size + block_size - 1) // block_size
    
    draft_rows_int8 = np.zeros((draft_size, hidden_size), dtype=np.int8)
    scales = np.zeros((draft_size, n_groups_per_row), dtype=np.float32)
    
    for i in range(draft_size):
        for g in range(n_groups_per_row):
            start = g * block_size
            end = min(start + block_size, hidden_size)
            group = draft_rows_fp32[i, start:end]
            
            max_abs = np.max(np.abs(group))
            if max_abs < 1e-10:
                scale = 1.0
            else:
                scale = max_abs / 127.0
            
            scales[i, g] = scale
            draft_rows_int8[i, start:end] = np.clip(np.round(group / scale), -127, 127).astype(np.int8)
        
        if i % 5000 == 0:
            print(f"  Quantized {i}/{draft_size} rows...")
    
    print(f"Quantized to INT8: weights shape={draft_rows_int8.shape}, scales shape={scales.shape}")
    
    # Save as binary file
    # Format: [draft_size: u32] [hidden_size: u32] [block_size: u32] [n_groups: u32]
    #         [int8_weights: draft_size * hidden_size bytes]
    #         [fp32_scales: draft_size * n_groups * 4 bytes]
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<IIII', draft_size, hidden_size, block_size, n_groups_per_row))
        f.write(draft_rows_int8.tobytes())
        f.write(scales.astype(np.float32).tobytes())
    
    total_size = 16 + draft_size * hidden_size + draft_size * n_groups_per_row * 4
    print(f"Saved draft lm_head to {output_path}: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    
    # Also save the BF16 version (for comparison and as fallback)
    bf16_path = output_path.replace('.bin', '_bf16.bin')
    with open(bf16_path, 'wb') as f:
        f.write(struct.pack('<II', draft_size, hidden_size))
        f.write(draft_rows_bf16.tobytes())
    
    bf16_size = 8 + draft_size * hidden_size * 2
    print(f"Saved BF16 draft lm_head to {bf16_path}: {bf16_size:,} bytes ({bf16_size/1024/1024:.1f} MB)")
    
    return True

if __name__ == '__main__':
    manifest_path = 'f:/Moecher/models/qwen3_8_27b_q4/moecher_manifest.json'
    draft_vocab_path = 'f:/Moecher/models/qwen3_8_27b_q4/draft_vocab_ids.json'
    output_path = 'f:/Moecher/models/qwen3_8_27b_q4/draft_lm_head_int8.bin'
    
    extract_draft_lm_head(manifest_path, draft_vocab_path, output_path)
