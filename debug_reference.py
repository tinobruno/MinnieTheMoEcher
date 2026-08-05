#!/usr/bin/env python3
"""Reference computation for layer 0, BOS token to validate C++ engine."""
import json, numpy as np

def bf16_to_f32(raw, count):
    u16 = np.frombuffer(raw[:count*2], dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return np.frombuffer(u32.tobytes(), dtype=np.float32)

m = json.load(open('/home/tinobruno/minniethemoecher/moecher_manifest.json'))
mc = m['model_config']
dt = m['dense_tensors']
with open('/home/tinobruno/minniethemoecher/attention_dense_layers.bin', 'rb') as f:
    data = f.read()

dim = mc['hidden_size']
hc = mc.get('hc_mult', 4)
eps = mc.get('rms_norm_eps', 1e-6)
hc_eps = mc.get('hc_eps', 1e-6)
sinkhorn_iters = mc.get('hc_sinkhorn_iters', 5)

def load_bf16(name, count):
    info = dt[name]
    raw = data[info['offset']:info['offset']+info['nbytes']]
    return bf16_to_f32(raw, count)

def load_f32(name, count):
    info = dt[name]
    raw = data[info['offset']:info['offset']+info['nbytes']]
    return np.frombuffer(raw[:count*4], dtype=np.float32)

def rms_norm(x, w, eps=1e-6):
    rms = np.sqrt(np.mean(x**2) + eps)
    return (x / rms) * w

# Embedding for BOS token (id=0)
emb_info = dt['embed.weight']
emb_raw = data[emb_info['offset']:emb_info['offset']+dim*2]
embed = bf16_to_f32(emb_raw, dim)
print(f"Embed[0]: norm={np.linalg.norm(embed):.6f}, first5={embed[:5]}")

# HC state: replicate embedding
hc_state = np.tile(embed, (hc, 1))
print(f"HC state init: each copy norm={np.linalg.norm(hc_state[0]):.6f}")

# HC pre for attention (layer 0)
hc_fn = load_f32('layers.0.hc_attn_fn', 24 * hc * dim).reshape(24, hc * dim)
hc_base = load_f32('layers.0.hc_attn_base', 24)
hc_scale = load_f32('layers.0.hc_attn_scale', 3)

x_flat = hc_state.flatten()
rms_val = np.sqrt(np.mean(x_flat**2) + hc_eps)
x_normed = x_flat / rms_val

mixes = x_normed @ hc_fn.T  # [24]

pre_raw = mixes[:hc]
post_raw = mixes[hc:2*hc]
comb_raw = mixes[2*hc:]

pre = 1.0 / (1.0 + np.exp(-(pre_raw * hc_scale[0] + hc_base[:hc]))) + hc_eps
post = 1.0 / (1.0 + np.exp(-(post_raw * hc_scale[1] + hc_base[hc:2*hc]))) + hc_eps
comb_mat = 1.0 / (1.0 + np.exp(-(comb_raw * hc_scale[2] + hc_base[2*hc:]))) + hc_eps
comb_mat = comb_mat.reshape(hc, hc)

for _ in range(sinkhorn_iters):
    comb_mat /= comb_mat.sum(axis=1, keepdims=True)
    comb_mat /= comb_mat.sum(axis=0, keepdims=True)

print(f"\nHC pre attn weights: {pre}")
print(f"HC post attn weights: {post}")

# Weighted sum: hidden = sum(pre[i] * hc_state[i])
hidden = sum(pre[i] * hc_state[i] for i in range(hc))
print(f"\nHC pre output: norm={np.linalg.norm(hidden):.6f}, first5={hidden[:5]}")

# Attention norm
attn_norm_w = load_bf16('layers.0.attn_norm.weight', dim)
hidden_normed = rms_norm(hidden, attn_norm_w, eps)
print(f"After attn_norm: norm={np.linalg.norm(hidden_normed):.6f}, first5={hidden_normed[:5]}")

# Compare with C++ output:
# C++ showed: embed norm=10.937148
# C++ showed: L0 hc_pre_attn norm=?
# C++ showed: L0 attn_normed norm=1.932495 (with old token), should be different now with BOS=0
print(f"\n--- Reference values for BOS token (id=0) ---")
print(f"embed norm: {np.linalg.norm(embed):.6f}")
print(f"hc_pre_attn norm: {np.linalg.norm(hidden):.6f}")
print(f"attn_normed norm: {np.linalg.norm(hidden_normed):.6f}")
