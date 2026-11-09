@echo off
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH%"

echo Starting llama-cli with -ngl 48 (spilling 40 layers to CPU)...
E:\dev\llama.cpp\build\bin\llama-cli.exe ^
  -m "E:\models\nemotron_bolt_24gb\NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf" ^
  -ngl 48 ^
  -p "Explain the difference between SSD and RAM in simple terms." ^
  -n 50 ^
  --temp 0.0 ^
  --no-warmup > benchmark_llama_ngl48.log 2>&1

echo Done! Output saved to benchmark_llama_ngl48.log
