# -----------------------------------------------------------------------------
#  build_installer.ps1 - Build Lightweight Moecher Windows Online Installer
# -----------------------------------------------------------------------------

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Building Lightweight Moecher Windows Installer" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

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
    "installer\start_deepseek_server.bat",
    "installer\test_qwen.bat",
    "installer\download_model.ps1",
    "graphics\moecher.ico",
    "graphics\installer_wizard.bmp",
    "graphics\installer_small.bmp",
    "web\index.html"
)

foreach ($file in $RequiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Error "Missing required source file: $file"
    }
}

# 3. Clean and prepare dist output directory
$RepoRoot = Resolve-Path "$PSScriptRoot\.."
$DistDir = "$RepoRoot\dist"
if (Test-Path $DistDir) {
    Remove-Item -Path "$DistDir\*" -Recurse -Force -ErrorAction SilentlyContinue
} else {
    New-Item -ItemType Directory -Path $DistDir | Out-Null
}

# 4. Compile Inno Setup package
Write-Host "Compiling Inno Setup package..." -ForegroundColor Cyan
$IssScript = "$PSScriptRoot\Moecher_Setup.iss"

& $IsccExe $IssScript

if ($LASTEXITCODE -ne 0) {
    Write-Error "Installer compilation failed with exit code $LASTEXITCODE"
}

$setupExe = "$DistDir\Moecher-Setup.exe"
if (Test-Path $setupExe) {
    $setupItem = Get-Item $setupExe
    $setupMB = [math]::Round($setupItem.Length / 1MB, 2)
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "  Moecher Installer Created Successfully!" -ForegroundColor Green
    Write-Host "  Installer Path: $setupExe ($setupMB MB)" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
}
