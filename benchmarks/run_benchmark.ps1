# PowerShell benchmark runner wrapper
param(
    [switch]$Quick,
    [switch]$NoVision,
    [string]$Engine = "both",
    [string]$OutputJson = "benchmark_results.json"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonBin = "C:\ComfyUI\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonBin)) {
    $PythonBin = (Get-Command python -ErrorAction SilentlyContinue).Source
}

if (-not $PythonBin) {
    Write-Error "Python executable not found."
    exit 1
}

$ArgsList = @("scripts\run_full_benchmark.py", "--engine", $Engine, "--output-json", $OutputJson)
if ($Quick) { $ArgsList += "--quick" }
if ($NoVision) { $ArgsList += "--no-vision" }

Write-Host "Starting benchmark with Python: $PythonBin" -ForegroundColor Cyan
& $PythonBin $ArgsList
