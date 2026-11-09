@echo off
setlocal
cd /d "%~dp0"
echo =======================================================================
echo  MinnieTheMoEcher vs. llama.cpp Automated Benchmark Suite
echo =======================================================================
echo.

set PYTHON_BIN=C:\ComfyUI\.venv\Scripts\python.exe

if not exist "%PYTHON_BIN%" (
    set PYTHON_BIN=python
)

"%PYTHON_BIN%" scripts\run_full_benchmark.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Benchmark encountered an error (code: %ERRORLEVEL%).
) else (
    echo.
    echo [+] Benchmark completed successfully!
)

pause
