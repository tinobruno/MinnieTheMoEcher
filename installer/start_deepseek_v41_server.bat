@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
echo ===============================================================================
echo   Starting Moecher Server: DeepSeek-V4.1-Flash (552B / 14B Active)
echo   Hardware Target: NVIDIA RTX PRO 6000 96GB + 64GB RAM + NVMe SSD Direct I/O
echo   Memory Hierarchy: L1 VRAM (46 GB) + L2 DRAM (Auto Max) + L3 SSD (Direct I/O)
echo   Web UI: http://localhost:8001
echo ===============================================================================
start "Moecher DeepSeek-V4.1 Server" /high moecher.exe -m F:\Moecher\models\deepseek_v41_nvfp4\moecher_manifest.json --cache-gb 46 --mtp-k 4 --port 8001 %*
