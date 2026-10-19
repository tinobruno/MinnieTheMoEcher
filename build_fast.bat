@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if %errorlevel% neq 0 exit /b 1

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

if not exist src\embedded_web.hpp (
    echo [INFO] Embedding Web UI assets into binary...
    python scripts\embed_web.py
    if %errorlevel% neq 0 exit /b 1
)

if not exist build\build.ninja (
    cmake -B build -G Ninja ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DCMAKE_CXX_COMPILER="C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe" ^
        -DCMAKE_C_COMPILER="C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe" ^
        -DCMAKE_CUDA_HOST_COMPILER="C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe" ^
        -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe" ^
        -DCMAKE_CUDA_ARCHITECTURES="86;89;90;120a"
    if %errorlevel% neq 0 exit /b 1
)

cmake --build build --config Release
if %errorlevel% neq 0 exit /b 1

if exist build\Release\moecher.exe (
    copy /y build\Release\moecher.exe moecher.exe >nul
    echo [INFO] Copied build\Release\moecher.exe to root folder.
    if exist "F:\Moecher" (
        copy /y build\Release\moecher.exe "F:\Moecher\moecher.exe" >nul 2>nul
        echo [INFO] Copied to F:\Moecher\moecher.exe
    )
) else if exist build\moecher.exe (
    copy /y build\moecher.exe moecher.exe >nul
    echo [INFO] Copied build\moecher.exe to root folder.
    if exist "F:\Moecher" (
        copy /y build\moecher.exe "F:\Moecher\moecher.exe" >nul 2>nul
        echo [INFO] Copied to F:\Moecher\moecher.exe
    )
)

echo [SUCCESS] Fast build completed successfully!
