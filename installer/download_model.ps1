# ─────────────────────────────────────────────────────────────────────────────
#  download_model.ps1 — Moecher On-Demand Model Downloader
# ─────────────────────────────────────────────────────────────────────────────

param (
    [Parameter(Mandatory=$true)]
    [ValidateSet("qwen", "deepseek", "both")]
    [string]$Model,

    [string]$DestDir = "$PSScriptRoot\models",
    [string]$Username = "TinoBruno"
)

$ErrorActionPreference = "Stop"

$ModelConfigs = @{
    "qwen" = @{
        "Repo"  = "$Username/moecher-qwen-3.8-27b-q4"
        "Dir"   = "$DestDir\qwen3_8_27b_q4"
        "Files" = @(
            "moecher_manifest.json",
            "tokenizer.json",
            "attention_dense_layers_q4.bin"
        )
    }
    "deepseek" = @{
        "Repo"  = "$Username/moecher-deepseek-v4-flash-iq2"
        "Dir"   = "$DestDir\deepseek_v4_flash_iq2"
        "Files" = @(
            "moecher_manifest.json",
            "tokenizer.json",
            "attention_dense_layers.bin",
            "moe_experts_iq2.bin"
        )
    }
}

function Download-HuggingFaceFile {
    param (
        [string]$RepoId,
        [string]$FileName,
        [string]$TargetFolder
    )

    $url = "https://huggingface.co/$RepoId/resolve/main/$FileName"
    $outFile = Join-Path $TargetFolder $FileName

    if (-not (Test-Path $TargetFolder)) {
        New-Item -ItemType Directory -Path $TargetFolder -Force | Out-Null
    }

    if (Test-Path $outFile) {
        Write-Host "  [EXISTS] $FileName already present. Skipping." -ForegroundColor Green
        return
    }

    Write-Host "  Downloading: $FileName from https://huggingface.co/$RepoId ..." -ForegroundColor Cyan

    $tempFile = "$outFile.download"
    $curlExe = Get-Command "curl.exe" -ErrorAction SilentlyContinue

    if ($curlExe) {
        # Use curl with resume, follow-redirects, and progress bar
        & curl.exe -L -C - "$url" -o "$tempFile" --progress-bar
        if ($LASTEXITCODE -eq 0) {
            Move-Item "$tempFile" "$outFile" -Force
            Write-Host "  [DONE] $FileName successfully downloaded." -ForegroundColor Green
        } else {
            Write-Error "Failed to download $FileName from $url"
        }
    } else {
        # PowerShell Fallback
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($url, $tempFile)
        Move-Item "$tempFile" "$outFile" -Force
        Write-Host "  [DONE] $FileName successfully downloaded." -ForegroundColor Green
    }
}

$targets = if ($Model -eq "both") { @("qwen", "deepseek") } else { @($Model) }

foreach ($t in $targets) {
    $cfg = $ModelConfigs[$t]
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  Downloading Model: $($cfg.Repo)" -ForegroundColor Cyan
    Write-Host "  Destination: $($cfg.Dir)" -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

    foreach ($f in $cfg.Files) {
        Download-HuggingFaceFile -RepoId $cfg.Repo -FileName $f -TargetFolder $cfg.Dir
    }
}

Write-Host "Model download process finished." -ForegroundColor Green
