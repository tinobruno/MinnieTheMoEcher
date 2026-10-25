@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
echo ===============================================================================
echo   Starting Moecher Server: DeepSeek-V4.1-Flash (Mixed NVFP4 Hot + IQ2 Cold)
echo   Hardware Target: NVIDIA RTX PRO 6000 96GB + 64GB RAM + NVMe SSD Direct I/O
echo   Memory Hierarchy: 3,840 Hot in VRAM (67GB) + 4,660 Cold in DRAM + NVMe Direct I/O
echo   Web UI: http://localhost:8001
echo ===============================================================================
start "Moecher DeepSeek-V4.1 Mixed Server" /high moecher.exe -m F:\Moecher\models\deepseek_v41_nvfp4\moecher_manifest_mixed.json --mtp-k 4 --port 8001 %*
