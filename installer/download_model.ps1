# -----------------------------------------------------------------------------
#  download_model.ps1 - Moecher On-Demand Model Downloader
# -----------------------------------------------------------------------------

param (
    [Parameter(Mandatory=$true)]
    [ValidateSet("qwen", "qwen_13g", "deepseek", "deepseek_q4", "nemotron", "nemotron_bolt", "both")]
    [string]$Model,

    [string]$DestDir = "",
    [string]$Username = "TinoBruno"
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Determine robust absolute destination directory
if (-not $DestDir -or $DestDir.Trim() -eq "") {
    $scriptDir = $PSScriptRoot
    if (-not $scriptDir -and $MyInvocation.MyCommand.Path) {
        $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    if (-not $scriptDir) {
        $scriptDir = (Get-Location).Path
    }
    $DestDir = Join-Path $scriptDir "models"
}

# Resolve to full absolute path to avoid any drive-root relative paths
$DestDir = [System.IO.Path]::GetFullPath($DestDir)

$ModelConfigs = @{
    "nemotron_bolt" = @{
        "Repo"  = "$Username/moecher-nemotron-3.5-bolt-nvfp4"
        "Dir"   = (Join-Path $DestDir "nemotron_3.5_bolt")
        "Files" = @(
            "moecher_manifest_16gb.json",
            "moecher_manifest_24gb.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "expert_scales.bin",
            "expert_freq.bin",
            "expert_profile.json",
            "attention_dense_layers.bin",
            "expert_weights.bin.001",
            "expert_weights.bin.002"
        )
    }
    "nemotron" = @{
        "Repo"  = "$Username/moecher-nemotron-3.5-lightning-vision"
        "Dir"   = (Join-Path $DestDir "nemotron_3.5_lightning")
        "Files" = @(
            "moecher_manifest.json",
            "tokenizer.json",
            "expert_scales.bin",
            "attention_dense_layers.bin",
            "expert_weights.bin"
        )
    }
    "qwen" = @{
        "Repo"  = "$Username/moecher-qwen-3.8-27b-q4"
        "Dir"   = (Join-Path $DestDir "qwen3_8_27b_q4")
        "Files" = @(
            "moecher_manifest.json",
            "tokenizer.json",
            "attention_dense_layers_q4.bin",
            "draft_vocab_ids.bin",
            "draft_lm_head_int8_bf16.bin"
        )
    }
    "qwen_13g" = @{
        "Repo"  = "$Username/moecher-qwen-3.8-27b-vision-13g"
        "Dir"   = (Join-Path $DestDir "qwen3_8_27b_vision_13g")
        "Files" = @(
            "moecher_manifest.json",
            "tokenizer.json",
            "attention_dense_layers.bin",
            "draft_lm_head_int8_bf16.bin",
            "draft_lm_head_int8.bin",
            "draft_vocab_ids.bin",
            "draft_vocab_ids.json",
            "draft_vocab_ids_mapping.json",
            "draft_vocab_ids_reverse.bin"
        )
    }
    "deepseek" = @{
        "Repo"  = "$Username/moecher-deepseek-v4-flash-iq2"
        "Dir"   = (Join-Path $DestDir "deepseek_v4_flash_iq2")
        "Files" = @(
            "moecher_manifest.json",
            "tokenizer.json",
            "attention_dense_layers.bin",
            "moe_experts_iq2.bin"
        )
    }
    "deepseek_q4" = @{
        "Repo"  = "$Username/moecher-deepseek-v4-flash-q4"
        "Dir"   = (Join-Path $DestDir "deepseek_v4_flash_q4")
        "Files" = @(
            "moecher_manifest.json",
            "tokenizer.json",
            "attention_dense_layers_q4.bin",
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
        $localSize = (Get-Item $outFile).Length
        if ($localSize -gt 0) {
            Write-Host "  [SKIP] $FileName already exists ($([math]::Round($localSize / 1MB, 1)) MB)." -ForegroundColor Yellow
            return
        }
    }

    $tempFile = "$outFile.downloading"
    Write-Host "Downloading: $FileName from https://huggingface.co/$RepoId ..." -ForegroundColor Cyan

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
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  Downloading Model: $($cfg.Repo)" -ForegroundColor Cyan
    Write-Host "  Destination Folder: $($cfg.Dir)" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan

    foreach ($f in $cfg.Files) {
        Download-HuggingFaceFile -RepoId $cfg.Repo -FileName $f -TargetFolder $cfg.Dir
    }

    # Automatically assemble split expert_weights.bin for Nemotron Bolt
    if ($t -eq "nemotron_bolt") {
        $p1 = Join-Path $cfg.Dir "expert_weights.bin.001"
        $p2 = Join-Path $cfg.Dir "expert_weights.bin.002"
        $target = Join-Path $cfg.Dir "expert_weights.bin"
        if ((Test-Path $p1) -and (Test-Path $p2)) {
            Write-Host "================================================================" -ForegroundColor Cyan
            Write-Host "  Assembling expert_weights.bin from split parts..." -ForegroundColor Cyan
            Write-Host "================================================================" -ForegroundColor Cyan
            cmd.exe /c "copy /b `"$p1`"+`"$p2`" `"$target`""
            if ($LASTEXITCODE -eq 0 -and (Test-Path $target)) {
                Remove-Item $p1, $p2 -Force
                Write-Host "  [DONE] expert_weights.bin successfully assembled ($([math]::Round((Get-Item $target).Length / 1GB, 2)) GB)." -ForegroundColor Green
            } else {
                Write-Error "Failed to assemble expert_weights.bin from parts"
            }
        }
    }
    if ($t -eq "nemotron") {
        $delDir = Join-Path $DestDir "frankenstin\vision"
        $delRepo = "$Username/frankenstin-vision-delegate"
        $delFiles = @(
            "moecher_manifest.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "chat_template.json",
            "merges.txt",
            "vocab.json",
            "attention_dense_layers.bin"
        )
        Write-Host "================================================================" -ForegroundColor Cyan
        Write-Host "  Downloading Vision Delegate: $delRepo" -ForegroundColor Cyan
        Write-Host "  Destination Folder: $delDir" -ForegroundColor Cyan
        Write-Host "================================================================" -ForegroundColor Cyan
        foreach ($df in $delFiles) {
            Download-HuggingFaceFile -RepoId $delRepo -FileName $df -TargetFolder $delDir
        }
    }
}

Write-Host "Model download process finished." -ForegroundColor Green
