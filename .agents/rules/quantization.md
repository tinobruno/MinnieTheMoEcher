# MinnieTheMoEcher Agent Rules

All agents and contributors MUST read and follow the authoritative rules in [`RULES.md`](../../RULES.md) at the repository root.

Key highlights:
1. **Zero User-Installed Dependencies (`Rule #1`)**: End users only run `moecher` (`moecher.exe`) and model weights. External runtimes must be auto-managed inside `.moecher_runtime/`.
2. **Clean Root Directory (`Rule #2`)**: Never leave scratch files, plans, notes, logs, or test scripts in the project root. Use `docs/plans/`, `docs/notes/`, `docs/`, `Articles/`, `benchmarks/`, `presentations/`, `scripts/`, `tests/`, or `scratch/`.
3. **Static-Linked Modularity (`Rule #3` & `Rule #4`)**: Single binary, zero heap allocations on decode hot paths, quiet console logging by default (`g_quiet = true`), and `.inl` model isolation in `src/models/`.
4. **Producer Defaults & VRAM Headroom (`Rule #5`)**: Auto-configure producer sampling defaults per model, keep `mtp_k = 0` (auto `1` on 16GB GPUs, `4` on >=24GB GPUs), and scope Hybrid CPU Expert Offload strictly to `Qwen 3.8 Flash-Next` (`QWEN4_EXP`).
5. **Quantization Policy (`Rule #6`)**: Always pass `--no-embed-quant --no-indexer-quant` when running `scripts/quantize_deepseek_dense.py`.
