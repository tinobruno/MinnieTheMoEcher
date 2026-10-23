@echo off
setlocal
chcp 65001 >nul
echo ===============================================================================
echo   Starting Moecher Server with NVIDIA Nemotron 3.5 Bolt (74B-A9B 24GB IQ2_XXS)
echo   Architecture: nemotron_h_puzzle / LatentMoE + Mamba-2 SSM
echo   Mode: 100%% Memory Resident (16GB VRAM + 32GB DRAM, ZERO NVMe SSD Reads)
echo   Web UI: http://localhost:8001
echo ===============================================================================

.\moecher.exe -m models\nemotron_3.5_bolt_24gb\moecher_manifest.json --port 8001 %*
