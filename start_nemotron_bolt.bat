@echo off
setlocal
chcp 65001 >nul
echo ===============================================================================
echo   Starting Moecher Server with NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4)
echo   Architecture: nemotron_h_puzzle / LatentMoE + Mamba-2 SSM
echo   Web UI: http://localhost:8001
echo ===============================================================================
if "%1"=="24gb" (
    echo [INFO] Target: 24GB VRAM (100%% In-VRAM Resident Execution)
    .\moecher.exe -m models\nemotron_3.5_bolt\moecher_manifest_24gb.json --port 8001 %*
) else (
    echo [INFO] Target: 16GB VRAM (Hybrid 18 VRAM / 22 DRAM Lookahead Streaming)
    .\moecher.exe -m models\nemotron_3.5_bolt\moecher_manifest_16gb.json --port 8001 %*
)
