@echo off
setlocal
chcp 65001 >nul
echo ===============================================================================
echo   Starting Moecher Server: DeepSeek-V4.1-Flash (Mixed NVFP4 Hot + IQ2 Cold)
echo   Hardware Target: NVIDIA RTX PRO 6000 96GB + 64GB RAM + NVMe SSD Direct I/O
echo   Memory Hierarchy: 3,840 Hot in VRAM (67GB) + 4,660 Cold in DRAM + NVMe Direct I/O
echo   Web UI: http://localhost:8001
echo ===============================================================================
start "Moecher DeepSeek-V4.1 Server" /high moecher.exe -m F:\Moecher\models\deepseek_v41_nvfp4\moecher_manifest_mixed.json --no-spec --port 8001 %*
