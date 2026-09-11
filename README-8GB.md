# DeepSeek V4 Flash - 8GB GPU Optimizations

This document outlines potential optimizations to run the DeepSeek V4 Flash model comfortably within 8GB of VRAM, avoiding extreme workarounds (like INT4 embeddings) that heavily degrade model intelligence.

## Current Memory Breakdown (Baseline)
- **Dense Layers (Q4)**: ~3.9 GB
- **Embeddings + LM Head (BF16)**: ~2.12 GB (1.06 GB each)
- **Total Footprint**: ~6.22 GB
- **Engine Pre-allocation Overhead**: ~2.0 GB
- **Total**: ~8.22 GB (Crashes or severely throttles an 8GB GPU)

## Optimization Strategies

### 1. Quantize Embeddings & LM Head to FP8
Instead of using INT4 for embeddings (which causes massive precision loss and hallucinatory behavior), quantize both the `embed_tokens` and `lm_head` from BF16 (2 bytes) to **FP8** (1 byte).
- **Savings**: ~1.06 GB VRAM.
- **Accuracy**: Minimal degradation. FP8 preserves the precision of these highly sensitive layers much better than INT4.
- **Performance**: Can slightly increase generation speed (`tok/s`), as it cuts the massive memory bandwidth requirement of the final logit computation in half.

### 2. Disable/Strip MTP (Multi-Token Prediction) Layers
DeepSeek V4 includes several extra layers (`mtp.0.*`, `mtp.1.*`, etc.) that act as a built-in draft model for speculative decoding.
- **Savings**: Several hundred megabytes.
- **Implementation**: Add an engine flag (`--disable-mtp`) or strip these tensors from the Q4 manifest entirely so they are never loaded into VRAM.
- **Tradeoff**: You lose the token generation speedup provided by speculative decoding, but the base model's intelligence and context recall remain 100% intact.

### 3. Tune Engine Pre-allocations
The `moecher` engine aggressively pre-allocates memory for experts and the KV cache. We can pass stricter constraints to the engine:
- **KV Cache Size**: Pass `--max-seq-len 4096` or `8192` (instead of 32K+) to enforce a strict upper limit on KV cache memory.
- **Expert Cache Limit**: Reduce the active GPU expert cache pool (e.g., set `--max-vram 5` instead of `6`).
- **Tradeoff**: This ensures the engine respects the 8GB limit, but having fewer experts cached in VRAM means the engine must fetch from DRAM more frequently, which lowers your `tok/s`.
