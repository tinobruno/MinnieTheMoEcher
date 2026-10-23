Add-Type -AssemblyName System.Speech

$outDir = "e:\dev\MinnieTheMoEcher\voice_clips"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$scenes = @(
    @{
        Id = "voice_01"
        Text = "Did llama dot cpp really just make 125B MoE models four times faster? Kai recently analyzed the new MoE expert cache in PR 29887. But what happens when we benchmark this on a real consumer card, the RTX 5060 Ti 16GB, with NVIDIA 24 gigabyte Nemotron 3.5 Bolt?"
    },
    @{
        Id = "voice_02"
        Text = "Here is the architectural reality. Nemotron 3.5 Bolt packs 496 routed experts per layer across 40 layers. That is 19,840 total experts. On a 16 gigabyte GPU, a 6 gigabyte cache can only hold 4,500 expert slices. That results in a devastating 78 percent cache miss rate."
    },
    @{
        Id = "voice_03"
        Text = "This is exactly what Kai warned about at timestamp 23:17. In llama dot cpp, every single miss triggers a synchronous cudaStreamWaitEvent. The GPU compute graph halts, Tensor Cores freeze, and generation collapses to just 3.5 tokens per second."
    },
    @{
        Id = "voice_04"
        Text = "MinnieTheMoEcher solves this through asynchronous double-buffered DMA. While Layer N computes Mamba-2 SSM and Attention on Tensor Cores, Layer N plus 1 experts are pre-streamed in parallel over dedicated DMA streams from pinned host DRAM. PCIe transfer latency is completely hidden."
    },
    @{
        Id = "voice_05"
        Text = "The final result: Minnie delivers 13.51 tokens per second decode - 3.86 times faster than llama dot cpp MoE cache - with zero percent CPU burn. Massive 70B MoE inference on a sub-500-dollar consumer GPU is now a reality."
    }
)

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.SelectVoice("Microsoft David Desktop")
} catch {
    Write-Host "Using default system voice"
}

$synth.Rate = 0

foreach ($s in $scenes) {
    $wavPath = Join-Path $outDir "$($s.Id).wav"
    $synth.SetOutputToWaveFile($wavPath)
    $synth.Speak($s.Text)
    Write-Host "Generated $($s.Id).wav"
}

$synth.Dispose()
Write-Host "All voice clips generated successfully."
