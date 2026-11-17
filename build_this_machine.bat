@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo   Building MinnieTheMoEcher for this machine (2x RTX 3090 sm_86)
echo ================================================================

REM 1. Initialize MSVC Environment
if exist "D:\dev\msvc\setup_x64.bat" (
    call "D:\dev\msvc\setup_x64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo [ERROR] MSVC environment not found!
    exit /b 1
)

REM 2. Configure CUDA and Toolchain Paths
if exist "C:\Users\tino\scoop\apps\cuda\current" (
    set "CUDA_PATH=C:\Users\tino\scoop\apps\cuda\current"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
)

set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;C:\Users\tino\scoop\apps\python\current;C:\Users\tino\scoop\shims;%PATH%"

REM 3. Embed Web UI Assets
echo [INFO] Embedding Web UI assets into binary...
python scripts\embed_web.py
if %errorlevel% neq 0 (
    echo [ERROR] Failed to embed web assets!
    exit /b 1
)

REM 4. Configure CMake with Ninja & sm_86
if not exist build (
    cmake -B build -G Ninja ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DCMAKE_CXX_COMPILER="D:/dev/msvc/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe" ^
        -DCMAKE_CUDA_COMPILER="%CUDA_PATH%/bin/nvcc.exe" ^
        -DCMAKE_CUDA_HOST_COMPILER="D:/dev/msvc/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/cl.exe" ^
        -DCMAKE_CUDA_ARCHITECTURES="86" ^
        -DCMAKE_MT="D:/dev/msvc/Windows Kits/10/bin/10.0.28000.0/x64/mt.exe" ^
        -DCMAKE_RC_COMPILER="D:/dev/msvc/Windows Kits/10/bin/10.0.28000.0/x64/rc.exe"
    if %errorlevel% neq 0 (
        echo [ERROR] CMake configuration failed!
        exit /b 1
    )
)

REM 5. Compile moecher
cmake --build build --config Release --target moecher
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    exit /b 1
)

REM 6. Copy binary and runtime DLLs
if exist build\moecher.exe (
    copy /y build\moecher.exe moecher.exe >nul
    echo [INFO] Copied build\moecher.exe to root folder.
    if exist "%CUDA_PATH%\bin\x64\cublas64_13.dll" (
        copy /y "%CUDA_PATH%\bin\x64\cublas64_13.dll" .\ >nul
        copy /y "%CUDA_PATH%\bin\x64\cublasLt64_13.dll" .\ >nul
        copy /y "%CUDA_PATH%\bin\x64\cudart64_13.dll" .\ >nul
    )
)

echo ================================================================
echo [SUCCESS] MinnieTheMoEcher built successfully for this machine!
echo ================================================================
