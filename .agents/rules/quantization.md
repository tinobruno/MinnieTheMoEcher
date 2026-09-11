# Quantization Policy

## DeepSeek Q4 Manifest Creation

When running the `scripts/quantize_deepseek_dense.py` script to generate a Q4 manifest for `moecher`, you **MUST ALWAYS** pass the `--no-embed-quant` flag. You should also consider passing the `--no-indexer-quant` flag depending on the context length requirements.

### Rationale:

#### 1. Embeddings & Head (`--no-embed-quant`)
* **Engine Compatibility**: The C++/CUDA backend (`src/cuda/activations.cu` and `src/server_single.cpp`) currently lacks INT4 implementations for `embedding_cuda` and `embedding_broadcast_device_id_cuda`. It expects `bfloat16` for embeddings. Failing to pass this flag will result in an `illegal memory access` crash.
* **Precision & Intelligence**: Quantizing `embed_tokens` (input) and `lm_head` (output) heavily degrades the model's coherence and perplexity for a relatively small VRAM saving (~1.5GB total). 
* **Footprint**: Using `--no-embed-quant` leaves the embeddings and head in BF16/FP8, yielding a Dense Bin size of ~6.22 GB. With standard engine overhead (~2.0 GB), the total footprint for the core dense layers sits at **~8.22 GB**. This comfortably allows DeepSeek V4 Flash to fit inside a 12GB GPU while maintaining a massive L1 Expert Cache.

#### 2. CSA Indexers (`--no-indexer-quant`)
* **Memory Savings**: The model contains 21 indexer tensors (`indexer.wq_b.weight`). Quantizing them from FP8 to INT4 saves ~4.2 MB per tensor, for a total of **~88 MB** saved across the entire model.
* **Accuracy Tradeoff**: The indexer computes the queries used to search the compressed KV cache for older tokens. Quantizing it to INT4 adds noise to those queries, which might occasionally cause the model to fetch the wrong historical context block when dealing with massive prompts.
* **Conclusion**: Because 88 MB is a negligible memory saving for a model of this size, it is recommended to use `--no-indexer-quant` to skip quantizing the indexer. This preserves the original FP8 precision and maximizes long-context retrieval accuracy. (Note: The C++ engine *does* safely support INT4 indexers if you choose to quantize them, but the accuracy penalty is generally not worth the tiny memory saving).

**Command Example:**
```bash
python scripts/quantize_deepseek_dense.py /path/to/model/dir --no-embed-quant --no-indexer-quant
```
