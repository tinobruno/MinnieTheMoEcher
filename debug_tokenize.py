#!/usr/bin/env python3
"""Check tokenization against C++ engine."""
from transformers import AutoTokenizer

model_path = "/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Manual chat template (from DeepSeek)
prompt = "<|begin▁of▁sentence|><|User|>Hi<|Assistant|>"
print(f"Prompt: {repr(prompt)}")

input_ids = tokenizer.encode(prompt)
print(f"Token IDs: {input_ids}")
print(f"Num tokens: {len(input_ids)}")

for i, tid in enumerate(input_ids):
    print(f"  [{i}] {tid} -> {repr(tokenizer.decode([tid]))}")

# Check what the C++ engine sees (token 162 at position 0)
print(f"\nToken 162 decodes to: {repr(tokenizer.decode([162]))}")
print(f"Token 35119 decodes to: {repr(tokenizer.decode([35119]))}")
print(f"Token 49721 decodes to: {repr(tokenizer.decode([49721]))}")
print(f"Token 108080 decodes to: {repr(tokenizer.decode([108080]))}")
print(f"Token 39149 decodes to: {repr(tokenizer.decode([39149]))}")
