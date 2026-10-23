Add-Type -AssemblyName System.Drawing

$outDir = "e:\dev\MinnieTheMoEcher\video_frames"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$width = 1920
$height = 1080

function Create-Slide {
    param(
        [string]$Filename,
        [scriptblock]$DrawBlock
    )
    $bmp = New-Object System.Drawing.Bitmap($width, $height)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic

    # Deep dark background with subtle gradient
    $bgBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
        (New-Object System.Drawing.Point(0, 0)),
        (New-Object System.Drawing.Point($width, $height)),
        [System.Drawing.Color]::FromArgb(255, 7, 8, 14),
        [System.Drawing.Color]::FromArgb(255, 14, 18, 30)
    )
    $g.FillRectangle($bgBrush, 0, 0, $width, $height)

    # Top HUD Bar
    $hudFont = New-Object System.Drawing.Font("Consolas", 14, [System.Drawing.FontStyle]::Bold)
    $hudTagBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 240, 255))
    $hudSpecBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 148, 163, 184))
    
    # Red recording dot
    $recDotBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 51, 102))
    $g.FillEllipse($recDotBrush, 80, 55, 14, 14)
    $g.DrawString("KAI-STYLE TECH BREAKDOWN // RESEARCH REPORT", $hudFont, $hudTagBrush, 105, 52)
    $g.DrawString("NVIDIA RTX 5060 Ti 16GB // NEMOTRON 3.5 BOLT 74B (24GB)", $hudFont, $hudSpecBrush, 1180, 52)

    # Top border line
    $penCyan = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(80, 0, 240, 255), 2)
    $g.DrawLine($penCyan, 80, 85, 1840, 85)

    # Run the custom slide content
    & $DrawBlock $g

    # Bottom watermark line
    $penMuted = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(40, 255, 255, 255), 1)
    $g.DrawLine($penMuted, 80, 1000, 1840, 1000)
    $watermarkFont = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Regular)
    $watermarkBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 100, 116, 139))
    $g.DrawString("MinnieTheMoEcher Bare-Metal CUDA Engine vs. llama.cpp Upstream (b11845)", $watermarkFont, $watermarkBrush, 80, 1015)
    $g.DrawString("Reference: https://youtube.com/watch?v=shl8FFSoANE (@23:17)", $watermarkFont, $watermarkBrush, 1280, 1015)

    # Clean up
    $g.Dispose()
    $filepath = Join-Path $outDir $Filename
    $bmp.Save($filepath, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    Write-Host "Created slide: $filepath"
}

# ----------------- SLIDE 1 -----------------
Create-Slide "slide_01.png" {
    param($g)
    $titleFont = New-Object System.Drawing.Font("Segoe UI", 48, [System.Drawing.FontStyle]::Bold)
    $subFont = New-Object System.Drawing.Font("Segoe UI", 22, [System.Drawing.FontStyle]::Regular)
    $monoFont = New-Object System.Drawing.Font("Consolas", 18, [System.Drawing.FontStyle]::Bold)

    $whiteBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
    $cyanBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 240, 255))
    $mutedBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 203, 213, 225))
    $goldBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 183, 3))

    $g.DrawString("THE 4X MoE MYTH", $monoFont, $goldBrush, 80, 200)
    $g.DrawString("Did llama.cpp Really Just Make", $titleFont, $whiteBrush, 75, 250)
    $g.DrawString("125B MoE Models 4x Faster?", $titleFont, $cyanBrush, 75, 340)

    $g.DrawString("Kai recently broke down PR #29887 and the new --moe-cache-mib feature in llama.cpp,", $subFont, $mutedBrush, 80, 480)
    $g.DrawString("claiming up to 4x throughput boosts by caching host experts in GPU VRAM.", $subFont, $mutedBrush, 80, 525)

    # Card
    $cardBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 20, 24, 38))
    $cardPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 0, 240, 255), 2)
    $g.FillRectangle($cardBrush, 80, 620, 1760, 250)
    $g.DrawRectangle($cardPen, 80, 620, 1760, 250)

    $cardHeadFont = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Bold)
    $cardBodyFont = New-Object System.Drawing.Font("Segoe UI", 16, [System.Drawing.FontStyle]::Regular)
    $g.DrawString("THE EMPIRICAL QUESTION:", $cardHeadFont, $cyanBrush, 120, 660)
    $g.DrawString("What happens when we run this on a real consumer card (RTX 5060 Ti 16GB + 32GB RAM)", $cardBodyFont, $whiteBrush, 120, 715)
    $g.DrawString("with the official 24GB NVIDIA Nemotron 3.5 Bolt (74B total / 9B active)?", $cardBodyFont, $whiteBrush, 120, 755)
    $g.DrawString("And why did Kai warning at 23:17 (Prompts, drafts and loops) become the key to the benchmark?", $cardBodyFont, $goldBrush, 120, 800)
}

# ----------------- SLIDE 2 -----------------
Create-Slide "slide_02.png" {
    param($g)
    $titleFont = New-Object System.Drawing.Font("Segoe UI", 44, [System.Drawing.FontStyle]::Bold)
    $subFont = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Regular)
    $monoFont = New-Object System.Drawing.Font("Consolas", 18, [System.Drawing.FontStyle]::Bold)
    $boxFont = New-Object System.Drawing.Font("Segoe UI", 22, [System.Drawing.FontStyle]::Bold)

    $whiteBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
    $crimsonBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 51, 102))
    $cyanBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 240, 255))
    $purpleBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 180, 100, 255))
    $mutedBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 203, 213, 225))
    $goldBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 183, 3))

    $g.DrawString("THE ARCHITECTURAL REALITY", $monoFont, $crimsonBrush, 80, 160)
    $g.DrawString("The 19,840 Expert Wall", $titleFont, $whiteBrush, 75, 205)
    $g.DrawString("Nemotron 3.5 Bolt 74B features 496 routed experts per layer across 40 layers = 19,840 experts!", $subFont, $mutedBrush, 80, 290)

    # Box 1: GPU VRAM
    $box1Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 15, 23, 42))
    $box1Pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 0, 240, 255), 3)
    $g.FillRectangle($box1Brush, 120, 380, 680, 420)
    $g.DrawRectangle($box1Pen, 120, 380, 680, 420)

    $g.DrawString("RTX 5060 Ti (16GB VRAM)", $boxFont, $cyanBrush, 160, 420)
    $g.DrawString("Base Layers and KV Cache: 6.4 GB", $subFont, $whiteBrush, 160, 490)
    $g.DrawString("Available for MoE Cache: 6.0 GB", $subFont, $whiteBrush, 160, 540)
    $g.DrawString("Max Cached Experts: ~4,500 slices", $subFont, $crimsonBrush, 160, 610)
    $g.DrawString("CACHE HIT RATE: ONLY 22%", $boxFont, $crimsonBrush, 160, 680)

    # Connector / PCIe
    $g.DrawString("<-- PCIe Gen 4/5 -->", $monoFont, $crimsonBrush, 850, 540)
    $g.DrawString("78% MISS RATE", $monoFont, $whiteBrush, 880, 580)
    $g.DrawString("Constant Stalls", $monoFont, $crimsonBrush, 880, 615)

    # Box 2: Host RAM
    $box2Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 20, 15, 38))
    $box2Pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 180, 100, 255), 3)
    $g.FillRectangle($box2Brush, 1120, 380, 680, 420)
    $g.DrawRectangle($box2Pen, 1120, 380, 680, 420)

    $g.DrawString("System DDR5 RAM (32GB)", $boxFont, $purpleBrush, 1160, 420)
    $g.DrawString("Host-Resident Experts: 15,340 slices", $subFont, $whiteBrush, 1160, 490)
    $g.DrawString("Weight Footprint: ~15 GB", $subFont, $whiteBrush, 1160, 540)
    $g.DrawString("Non-pageable host memory pool", $subFont, $mutedBrush, 1160, 610)
    $g.DrawString("COLD EXPERTS WAITING", $boxFont, $purpleBrush, 1160, 680)

    $bottomNoteFont = New-Object System.Drawing.Font("Segoe UI", 16, [System.Drawing.FontStyle]::Italic)
    $g.DrawString("With an SLRU cache holding only 22% of experts, 78% of all queries trigger cold PCIe transfers!", $bottomNoteFont, $goldBrush, 120, 840)
}

# ----------------- SLIDE 3 -----------------
Create-Slide "slide_03.png" {
    param($g)
    $titleFont = New-Object System.Drawing.Font("Segoe UI", 44, [System.Drawing.FontStyle]::Bold)
    $subFont = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Regular)
    $monoFont = New-Object System.Drawing.Font("Consolas", 18, [System.Drawing.FontStyle]::Bold)
    $statFont = New-Object System.Drawing.Font("Segoe UI", 36, [System.Drawing.FontStyle]::Bold)

    $whiteBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
    $crimsonBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 51, 102))
    $amberBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 183, 3))
    $mutedBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 203, 213, 225))

    $g.DrawString("KAI WARNING AT TIMESTAMP 23:17", $monoFont, $amberBrush, 80, 160)
    $g.DrawString("Anatomy of a Cache Miss: The Synchronous Stall", $titleFont, $whiteBrush, 75, 205)
    $g.DrawString("Kai warned about draft tokens, prompt prefill, and loops creating severe cache miss penalties.", $subFont, $mutedBrush, 80, 290)

    # Problem Box
    $stallBoxBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 35, 15, 25))
    $stallBoxPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 255, 51, 102), 3)
    $g.FillRectangle($stallBoxBrush, 100, 360, 1720, 460)
    $g.DrawRectangle($stallBoxPen, 100, 360, 1720, 460)

    $g.DrawString("IN LLAMA.CPP: EVERY MISS STALLS THE GPU GRAPH", $monoFont, $crimsonBrush, 140, 400)
    $g.DrawString("cudaStreamWaitEvent() halts the entire CUDA stream until the missing expert finishes PCIe copy.", $subFont, $whiteBrush, 140, 450)
    $g.DrawString("Instead of 1 overlap transfer, the GPU undergoes up to 48 sequential waits per token generation!", $subFont, $whiteBrush, 140, 495)

    # Stats Row
    $g.DrawString("PREFILL THROUGHPUT", $monoFont, $mutedBrush, 140, 580)
    $g.DrawString("1.7 tok/s", $statFont, $crimsonBrush, 140, 620)

    $g.DrawString("DECODE GENERATION", $monoFont, $mutedBrush, 650, 580)
    $g.DrawString("3.5 tok/s", $statFont, $crimsonBrush, 650, 620)

    $g.DrawString("GPU UTILIZATION", $monoFont, $mutedBrush, 1200, 580)
    $g.DrawString("9% @ 20 Watts", $statFont, $crimsonBrush, 1200, 620)

    $g.DrawString("RESULT: llama.cpp MoE cache is SLOWER than CPU layer offloading (6.9 tok/s) due to stalls!", $subFont, $amberBrush, 140, 730)
}

# ----------------- SLIDE 4 -----------------
Create-Slide "slide_04.png" {
    param($g)
    $titleFont = New-Object System.Drawing.Font("Segoe UI", 44, [System.Drawing.FontStyle]::Bold)
    $subFont = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Regular)
    $monoFont = New-Object System.Drawing.Font("Consolas", 18, [System.Drawing.FontStyle]::Bold)
    $boxFont = New-Object System.Drawing.Font("Segoe UI", 22, [System.Drawing.FontStyle]::Bold)

    $whiteBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
    $cyanBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 240, 255))
    $greenBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 230, 118))
    $purpleBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 180, 100, 255))
    $mutedBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 203, 213, 225))

    $g.DrawString("THE ARCHITECTURAL SOLUTION", $monoFont, $greenBrush, 80, 160)
    $g.DrawString("Minnie Asynchronous Double-Buffered DMA", $titleFont, $whiteBrush, 75, 205)
    $g.DrawString("Completely decouples memory transfers from Tensor Core compute using dual overlapping streams.", $subFont, $mutedBrush, 80, 290)

    # Box 1: Stream A
    $box1Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 10, 28, 40))
    $box1Pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 0, 240, 255), 3)
    $g.FillRectangle($box1Brush, 120, 370, 780, 380)
    $g.DrawRectangle($box1Pen, 120, 370, 780, 380)

    $g.DrawString("STREAM A: GPU COMPUTE", $boxFont, $cyanBrush, 160, 410)
    $g.DrawString("- Layer N: Mamba-2 SSM Execution", $subFont, $whiteBrush, 160, 480)
    $g.DrawString("- Dense Multi-Head Attention", $subFont, $whiteBrush, 160, 530)
    $g.DrawString("- 100% Tensor Cores Active (120W+)", $subFont, $greenBrush, 160, 580)
    $g.DrawString("ZERO WAIT ON PCIE", $boxFont, $greenBrush, 160, 650)

    # Box 2: Stream B
    $box2Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 24, 15, 45))
    $box2Pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 180, 100, 255), 3)
    $g.FillRectangle($box2Brush, 980, 370, 780, 380)
    $g.DrawRectangle($box2Pen, 980, 370, 780, 380)

    $g.DrawString("STREAM B: ASYNC DMA PREFETCH", $boxFont, $purpleBrush, 1020, 410)
    $g.DrawString("- Layer N+1: Routed Experts Pre-Streamed", $subFont, $whiteBrush, 1020, 480)
    $g.DrawString("- Pinned Host DRAM (Non-pageable L2 pool)", $subFont, $whiteBrush, 1020, 530)
    $g.DrawString("- Speculative Prefetch Pipeline", $subFont, $whiteBrush, 1020, 580)
    $g.DrawString("PCIE LATENCY 100% MASKED", $boxFont, $cyanBrush, 1020, 650)

    $bottomBoxBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 15, 30, 20))
    $g.FillRectangle($bottomBoxBrush, 120, 780, 1640, 90)
    $g.DrawString("RESULT: 13.51 tok/s decode with 0% CPU load - full hardware parallelism on a sub-$500 GPU!", $subFont, $greenBrush, 160, 810)
}

# ----------------- SLIDE 5 -----------------
Create-Slide "slide_05.png" {
    param($g)
    $titleFont = New-Object System.Drawing.Font("Segoe UI", 44, [System.Drawing.FontStyle]::Bold)
    $subFont = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Regular)
    $monoFont = New-Object System.Drawing.Font("Consolas", 18, [System.Drawing.FontStyle]::Bold)
    $barNameFont = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Bold)
    $barStatFont = New-Object System.Drawing.Font("Consolas", 22, [System.Drawing.FontStyle]::Bold)

    $whiteBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
    $cyanBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 240, 255))
    $crimsonBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 51, 102))
    $goldBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 183, 3))
    $mutedBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 148, 163, 184))

    $g.DrawString("HEAD-TO-HEAD BENCHMARK RESULTS", $monoFont, $cyanBrush, 80, 140)
    $g.DrawString("NVIDIA Nemotron 3.5 Bolt 74B (24GB) on RTX 5060 Ti 16GB", $titleFont, $whiteBrush, 75, 180)
    $g.DrawString("Prompt: Explain difference between SSD and RAM (Decode tok/s)", $subFont, $mutedBrush, 80, 255)

    # Bar 1: Minnie
    $b1Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 15, 30, 48))
    $g.FillRectangle($b1Brush, 80, 310, 1760, 110)
    $minnieBarFill = New-Object System.Drawing.Drawing2D.LinearGradientBrush((New-Object System.Drawing.Point(540, 0)), (New-Object System.Drawing.Point(1450, 0)), [System.Drawing.Color]::FromArgb(255, 0, 240, 255), [System.Drawing.Color]::FromArgb(255, 58, 134, 255))
    $g.FillRectangle($minnieBarFill, 540, 340, 910, 50)
    $g.DrawString("MinnieTheMoEcher", $barNameFont, $cyanBrush, 110, 345)
    $g.DrawString("13.51 tok/s (3.86x) [0% CPU]", $barStatFont, $cyanBrush, 1470, 345)

    # Bar 2: llama.cpp -ngl 48
    $b2Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 20, 24, 34))
    $g.FillRectangle($b2Brush, 80, 440, 1760, 110)
    $llamaCpuFill = New-Object System.Drawing.Drawing2D.LinearGradientBrush((New-Object System.Drawing.Point(540, 0)), (New-Object System.Drawing.Point(1005, 0)), [System.Drawing.Color]::FromArgb(255, 255, 183, 3), [System.Drawing.Color]::FromArgb(255, 251, 133, 0))
    $g.FillRectangle($llamaCpuFill, 540, 470, 465, 50)
    $g.DrawString("llama.cpp (-ngl 48 CPU)", $barNameFont, $goldBrush, 110, 475)
    $g.DrawString("6.90 tok/s [100% CPU]", $barStatFont, $goldBrush, 1030, 475)

    # Bar 3: llama.cpp MoE Cache
    $b3Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 28, 18, 28))
    $g.FillRectangle($b3Brush, 80, 570, 1760, 110)
    $llamaMoeFill = New-Object System.Drawing.Drawing2D.LinearGradientBrush((New-Object System.Drawing.Point(540, 0)), (New-Object System.Drawing.Point(775, 0)), [System.Drawing.Color]::FromArgb(255, 255, 51, 102), [System.Drawing.Color]::FromArgb(255, 255, 117, 140))
    $g.FillRectangle($llamaMoeFill, 540, 600, 235, 50)
    $g.DrawString("llama.cpp (--moe-cache-mib)", $barNameFont, $crimsonBrush, 110, 605)
    $g.DrawString("3.50 tok/s [PCIe Wait]", $barStatFont, $crimsonBrush, 800, 605)

    # Bar 4: llama.cpp -ngl 88
    $b4Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 18, 20, 28))
    $g.FillRectangle($b4Brush, 80, 700, 1760, 110)
    $llamaWddmFill = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 80, 90, 110))
    $g.FillRectangle($llamaWddmFill, 540, 730, 55, 50)
    $g.DrawString("llama.cpp (-ngl 88 WDDM)", $barNameFont, $mutedBrush, 110, 735)
    $g.DrawString("< 1.00 tok/s [Paging Stall]", $barStatFont, $mutedBrush, 620, 735)

    # Bottom summary box
    $summaryBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 10, 24, 38))
    $summaryPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 0, 240, 255), 2)
    $g.FillRectangle($summaryBrush, 80, 840, 1760, 90)
    $g.DrawRectangle($summaryPen, 80, 840, 1760, 90)
    $g.DrawString("Minnie Prefill: 21.54 tok/s (12.6x faster than MoE Cache) - Decode: 13.51 tok/s - 0% CPU Burn", $subFont, $cyanBrush, 140, 870)
}

Write-Host "All 5 slide frames successfully generated in $outDir"
