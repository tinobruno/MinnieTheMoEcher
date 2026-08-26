# ─────────────────────────────────────────────────────────────────────────────
#  build_installer.ps1 — Build Spanned Moecher Windows Setup Package
# ─────────────────────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Building Moecher Windows Setup Package (Spanned .bin Slices)" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# 1. Locate Inno Setup Compiler (ISCC.exe)
$IsccPaths = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
)

$IsccExe = $IsccPaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $IsccExe) {
    Write-Error "ISCC.exe not found! Please ensure Inno Setup is installed."
}

Write-Host "Found Inno Setup Compiler: $IsccExe" -ForegroundColor Green

# 2. Verify all prerequisite files
$RequiredFiles = @(
    "build\Release\moecher.exe",
    "installer\Moecher_Setup.iss",
    "installer\start_qwen_server.bat",
    "installer\test_qwen.bat",
    "graphics\moecher.ico",
    "graphics\installer_wizard.bmp",
    "graphics\installer_small.bmp",
    "web\index.html",
    "models\qwen3_8_27b_q4\moecher_manifest_qwen_q4.json",
    "models\qwen3_8_27b_q4\tokenizer.json",
    "models\qwen3_8_27b_q4\attention_dense_layers_q4.bin"
)

foreach ($file in $RequiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Error "Missing required source file: $file"
    }
}

# 3. Clean and prepare dist output directory
$DistDir = "D:\dev\minniemoe\MinnieTheMoEcher\dist"
if (Test-Path $DistDir) {
    Remove-Item -Path "$DistDir\*" -Recurse -Force -ErrorAction SilentlyContinue
} else {
    New-Item -ItemType Directory -Path $DistDir | Out-Null
}

# 4. Compile Inno Setup package
Write-Host "Compiling Inno Setup package (packaging ~18.7 GB into .bin slices)..." -ForegroundColor Cyan
$IssScript = "D:\dev\minniemoe\MinnieTheMoEcher\installer\Moecher_Setup.iss"

& $IsccExe $IssScript

if ($LASTEXITCODE -ne 0) {
    Write-Error "Installer compilation failed with exit code $LASTEXITCODE"
}

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  Moecher Spanned Setup Package Created Successfully!" -ForegroundColor Green
Write-Host "  Destination: $DistDir" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
