# ═══════════════════════════════════════════════════════════════════════════════
# convert_nemotron_lightning.ps1 — PowerShell Manifest Generator for Nemotron 3.5
# ═══════════════════════════════════════════════════════════════════════════════
param(
    [string]$OutputDir = "models/nemotron_3.5_lightning",
    [string]$ModelName = "nvidia/nemotron-3.5-lightning-30b-a3b",
    [string]$VisionDelegatePath = "models/frankenstin/vision/moecher_manifest.json"
)

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$numLayers = 48
$layerTypes = @()
for ($i = 0; $i -lt $numLayers; $i++) {
    if ($i % 4 -eq 3) {
        $layerTypes += 0 # Attention layer
    } else {
        $layerTypes += 1 # Mamba-2 SSM layer
    }
}

$manifest = @{
    model_config = @{
        model_name = "Nemotron-3.5-Lightning-30B-A3B-FrankensTin"
        model_id = $ModelName
        architecture = "NEMOTRON_35"
        vocab_size = 131072
        hidden_size = 4096
        num_hidden_layers = $numLayers
        num_attention_heads = 32
        num_key_value_heads = 8
        head_dim = 128
        intermediate_size = 11008
        moe_intermediate_size = 2048
        n_routed_experts = 32
        num_experts_per_tok = 6
        rms_norm_eps = 1e-6
        rope_theta = 500000.0
        max_seq_len = 65536
        original_seq_len = 65536
        expert_dtype = "int3"
        ssm_d_state = 128
        ssm_d_conv = 4
        ssm_expand = 2
        ssm_dt_rank = 256
        layer_types = $layerTypes
        has_vision = $true
        vision_delegate = $VisionDelegatePath
    }
    dense_bin = "attention_dense_layers.bin"
    expert_bin = "expert_weights.bin"
    tokenizer = "tokenizer.json"
    dense_tensors = @{}
    expert_parts = @{
        gate = @{ dtype = "int3"; offset = 0 }
        up   = @{ dtype = "int3"; offset = 1 }
        down = @{ dtype = "int3"; offset = 2 }
    }
}

$manifestPath = Join-Path $OutputDir "moecher_manifest.json"
$manifest | ConvertTo-Json -Depth 10 | Set-Content -Path $manifestPath -Encoding utf8

Write-Host "[Nemotron Generator] Generated manifest successfully at: $manifestPath" -ForegroundColor Green
Write-Host "  Architecture: NEMOTRON_35 (Hybrid Mamba-2 SSM + MoE)"
Write-Host "  Layers: $numLayers (36 Mamba-2 SSM layers, 12 Attention layers)"
Write-Host "  Vision Delegate: $VisionDelegatePath"
