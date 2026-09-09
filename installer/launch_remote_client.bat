@echo off
setlocal
chcp 65001 >nul

set SERVER_HOST=%1
if "%SERVER_HOST%"=="" set SERVER_HOST=zetaorangews

echo ===============================================================================
echo   Launching Browser with Forward Proxy configured to %SERVER_HOST%:8002
echo   Web UI: http://%SERVER_HOST%:8001
echo ===============================================================================

where msedge >nul 2>nul
if %errorlevel% equ 0 (
    start msedge.exe --proxy-server="http://%SERVER_HOST%:8002" "http://%SERVER_HOST%:8001"
    exit /b 0
)

where chrome >nul 2>nul
if %errorlevel% equ 0 (
    start chrome.exe --proxy-server="http://%SERVER_HOST%:8002" "http://%SERVER_HOST%:8001"
    exit /b 0
)

echo [INFO] Neither Edge nor Chrome was found in PATH.
echo Launch your browser manually with:
echo --proxy-server="http://%SERVER_HOST%:8002" "http://%SERVER_HOST%:8001"
pause
