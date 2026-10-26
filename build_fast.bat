@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if %errorlevel% neq 0 exit /b 1

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"

if not exist src\embedded_web.hpp (
    echo [INFO] Embedding Web UI assets into binary...
    python scripts\embed_web.py
    if %errorlevel% neq 0 exit /b 1
)

if not exist build (
    cmake -B build -G Ninja ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe" ^
        -DCMAKE_CUDA_ARCHITECTURES="86;89;90;100;120"
    if %errorlevel% neq 0 exit /b 1
)

cmake --build build --config Release
if %errorlevel% neq 0 exit /b 1

if exist build\moecher.exe (
    copy /y build\moecher.exe moecher.exe >nul
    echo [INFO] Copied build\moecher.exe to root folder.
) else if exist build\Release\moecher.exe (
    copy /y build\Release\moecher.exe moecher.exe >nul
    echo [INFO] Copied build\Release\moecher.exe to root folder.
)

echo [SUCCESS] Fast build completed successfully!
