Add-Type -AssemblyName System.Drawing

$outDir = "e:\dev\MinnieTheMoEcher\bolt_video_frames"
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

    # Deep dark background with subtle high-tech gradient
    $bgBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
        (New-Object System.Drawing.Point(0, 0)),
        (New-Object System.Drawing.Point($width, $height)),
        [System.Drawing.Color]::FromArgb(255, 6, 8, 14),
        [System.Drawing.Color]::FromArgb(255, 12, 18, 32)
    )
    $g.FillRectangle($bgBrush, 0, 0, $width, $height)

    # Top HUD Bar
    $hudFont = New-Object System.Drawing.Font("Consolas", 14, [System.Drawing.FontStyle]::Bold)
    $hudTagBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 240, 255))
    $hudSpecBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 148, 163, 184))
    
    # Green NVIDIA style indicator dot
    $recDotBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 118, 185, 0)) # NVIDIA Green
    $g.FillEllipse($recDotBrush, 80, 55, 14, 14)
    $g.DrawString("MINNIE THE MOECHER // ARCHITECTURE & BENCHMARK REPORT", $hudFont, $hudTagBrush, 105, 52)
    $g.DrawString("NVIDIA NEMOTRON 3.5 BOLT 74B // RTX 5060 Ti 16GB", $hudFont, $hudSpecBrush, 1220, 52)

    # Top border line
    $penCyan = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(80, 0, 240, 255), 2)
    $g.DrawLine($penCyan, 80, 85, 1840, 85)

    # Run custom slide content
    & $DrawBlock $g

    # Bottom watermark line
    $penMuted = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(40, 255, 255, 255), 1)
    $g.DrawLine($penMuted, 80, 1000, 1840, 1000)
    $watermarkFont = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Regular)
    $watermarkBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 100, 116, 139))
    $g.DrawString("MinnieTheMoEcher: Bare-Metal CUDA & Metal Inference Engine", $watermarkFont, $watermarkBrush, 80, 1015)
    $g.DrawString("Open Source on GitHub: github.com/tinobruno/MinnieTheMoEcher", $watermarkFont, $watermarkBrush, 1260, 1015)

    $g.Dispose()
    $filepath = Join-Path $outDir $Filename
    $bmp.Save($filepath, [System.Drawing.Imaging.ImageFormat]::Png)
    $bmp.Dispose()
    Write-Host "Created slide: $filepath"
}

# ----------------- SLIDE 1: Thank You NVIDIA -----------------
Create-Slide "slide_01.png" {
    param($g)
    $titleFont = New-Object System.Drawing.Font("Segoe UI", 48, [System.Drawing.FontStyle]::Bold)
    $subFont = New-Object System.Drawing.Font("Segoe UI", 22, [System.Drawing.FontStyle]::Regular)
    $monoFont = New-Object System.Drawing.Font("Consolas", 18, [System.Drawing.FontStyle]::Bold)

    $whiteBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
    $nvGreenBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 118, 185, 0)) # NVIDIA Green
    $cyanBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 240, 255))
    $mutedBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 203, 213, 225))
    $goldBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 183, 3))

    $g.DrawString("OPEN-SOURCE MILESTONE", $monoFont, $nvGreenBrush, 80, 190)
    $g.DrawString("Thank You, NVIDIA!", $titleFont, $nvGreenBrush, 75, 235)
    $g.DrawString("Finally, a 70B Model Running on a $400 GPU", $titleFont, $whiteBrush, 75, 330)

    $g.DrawString("NVIDIA recently released the 24GB Nemotron 3.5 Bolt (74B Total / 9B Active),", $subFont, $mutedBrush, 80, 470)
    $g.DrawString("proving that frontier-class intelligence can run on local consumer hardware.", $subFont, $mutedBrush, 80, 515)

    # Card
    $cardBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 16, 24, 38))
    $cardPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 118, 185, 0), 2)
    $g.FillRectangle($cardBrush, 80, 610, 1760, 260)
    $g.DrawRectangle($cardPen, 80, 610, 1760, 260)

    $cardHeadFont = New-Object System.Drawing.Font("Segoe UI", 22, [System.Drawing.FontStyle]::Bold)
    $cardBodyFont = New-Object System.Drawing.Font("Segoe UI", 17, [System.Drawing.FontStyle]::Regular)
    $g.DrawString("THE CONSUMER HARDWARE REALITY:", $cardHeadFont, $cyanBrush, 120, 650)
    $g.DrawString("- Tested Hardware: NVIDIA GeForce RTX 5060 Ti 16GB + 32GB DDR5 RAM", $cardBodyFont, $whiteBrush, 120, 710)
    $g.DrawString("- Target Model: NVIDIA-Nemotron-3.5-Bolt-74B-A9B (20.5 GB GGUF / 25 GB Raw)", $cardBodyFont, $whiteBrush, 120, 755)
    $g.DrawString("- Result: 13.51 tok/s decode throughput with 100% GPU Tensor Cores active!", $cardBodyFont, $goldBrush, 120, 800)
}

# ----------------- SLIDE 2: Model Architecture -----------------
Create-Slide "slide_02.png" {
    param($g)
    $titleFont = New-Object System.Drawing.Font("Segoe UI", 44, [System.Drawing.FontStyle]::Bold)
    $subFont = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Regular)
    $monoFont = New-Object System.Drawing.Font("Consolas", 18, [System.Drawing.FontStyle]::Bold)
    $boxFont = New-Object System.Drawing.Font("Segoe UI", 22, [System.Drawing.FontStyle]::Bold)

    $whiteBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
    $cyanBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 240, 255))
    $purpleBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 180, 100, 255))
    $mutedBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 203, 213, 225))
    $goldBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 183, 3))

    $g.DrawString("HYBRID STATE SPACE & LATENT MOE", $monoFont, $cyanBrush, 80, 150)
    $g.DrawString("The Nemotron 3.5 Bolt Architecture", $titleFont, $whiteBrush, 75, 195)
    $g.DrawString("A radical departure from standard dense Transformers, combining SSM speed with massive MoE capacity.", $subFont, $mutedBrush, 80, 280)

    # Box 1: Mamba-2 SSM
    $box1Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 15, 25, 42))
    $box1Pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 0, 240, 255), 2)
    $g.FillRectangle($box1Brush, 120, 360, 500, 450)
    $g.DrawRectangle($box1Pen, 120, 360, 500, 450)

    $g.DrawString("1. Mamba-2 SSM", $boxFont, $cyanBrush, 150, 400)
    $g.DrawString("Linear State Space mixer", $subFont, $mutedBrush, 150, 450)
    $g.DrawString("- O(1) state recurrence", $subFont, $whiteBrush, 150, 510)
    $g.DrawString("- Conv1D token mixing", $subFont, $whiteBrush, 150, 560)
    $g.DrawString("- Blazing fast prefill", $subFont, $whiteBrush, 150, 610)
    $g.DrawString("21.5 tok/s Prefill", $boxFont, $cyanBrush, 150, 710)

    # Box 2: GQA Attention
    $box2Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 20, 20, 45))
    $box2Pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 157, 78, 221), 2)
    $g.FillRectangle($box2Brush, 660, 360, 500, 450)
    $g.DrawRectangle($box2Pen, 660, 360, 500, 450)

    $g.DrawString("2. GQA Attention", $boxFont, $purpleBrush, 690, 400)
    $g.DrawString("Grouped Query Attention", $subFont, $mutedBrush, 690, 450)
    $g.DrawString("- Full sequence reasoning", $subFont, $whiteBrush, 690, 510)
    $g.DrawString("- RoPE position embeddings", $subFont, $whiteBrush, 690, 560)
    $g.DrawString("- Compact KV footprint", $subFont, $whiteBrush, 690, 610)
    $g.DrawString("Precision Reasoning", $boxFont, $purpleBrush, 690, 710)

    # Box 3: LatentMoE
    $box3Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 28, 20, 35))
    $box3Pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 255, 183, 3), 2)
    $g.FillRectangle($box3Brush, 1200, 360, 600, 450)
    $g.DrawRectangle($box3Pen, 1200, 360, 600, 450)

    $g.DrawString("3. LatentMoE (496 Exp)", $boxFont, $goldBrush, 1230, 400)
    $g.DrawString("Fine-Grained Expert Routing", $subFont, $mutedBrush, 1230, 450)
    $g.DrawString("- 40 MoE layers (19,840 exp)", $subFont, $whiteBrush, 1230, 510)
    $g.DrawString("- Top-9 routed active experts", $subFont, $whiteBrush, 1230, 560)
    $g.DrawString("- Dual latent projections", $subFont, $whiteBrush, 1230, 610)
    $g.DrawString("Massive Capacity", $boxFont, $goldBrush, 1230, 710)

    $g.DrawString("VRAM L1: 6.4 GB (Dense + SSM)  |  Pinned Host DRAM L2: 13.5 GB (MoE Experts)", $subFont, $whiteBrush, 120, 850)
}

# ----------------- SLIDE 3: Benchmark Showdown -----------------
Create-Slide "slide_03.png" {
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

    $g.DrawString("HEAD-TO-HEAD BENCHMARK SHOWDOWN", $monoFont, $cyanBrush, 80, 140)
    $g.DrawString("Nemotron 3.5 Bolt (24GB) on RTX 5060 Ti 16GB", $titleFont, $whiteBrush, 75, 180)
    $g.DrawString("Prompt: Explain the difference between SSD and RAM in simple terms. (Decode Speed)", $subFont, $mutedBrush, 80, 255)

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
    $g.DrawString("Minnie Prefill: 21.54 tok/s (12.6x faster than MoE Cache) - Decode: 13.51 tok/s - 0% CPU Load", $subFont, $cyanBrush, 140, 870)
}

# ----------------- SLIDE 4: Why Minnie is Faster -----------------
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

    $g.DrawString("THE ENGINE ARCHITECTURE", $monoFont, $greenBrush, 80, 150)
    $g.DrawString("Why Minnie Runs 3.86x Faster", $titleFont, $whiteBrush, 75, 195)
    $g.DrawString("How bare-metal asynchronous pipelining eliminates the PCIe serialization bottleneck.", $subFont, $mutedBrush, 80, 280)

    # Box 1: Double-Buffered DMA
    $box1Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 10, 28, 40))
    $box1Pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 0, 240, 255), 2)
    $g.FillRectangle($box1Brush, 120, 360, 780, 380)
    $g.DrawRectangle($box1Pen, 120, 360, 780, 380)

    $g.DrawString("1. Double-Buffered Async DMA", $boxFont, $cyanBrush, 160, 400)
    $g.DrawString("- While Layer N computes SSM and Attention,", $subFont, $whiteBrush, 160, 470)
    $g.DrawString("  Layer N+1 experts are pre-streamed in parallel.", $subFont, $whiteBrush, 160, 515)
    $g.DrawString("- Dedicated CUDA copy stream overlaps transfer.", $subFont, $whiteBrush, 160, 560)
    $g.DrawString("- Zero pipeline stalls waiting on PCIe.", $subFont, $greenBrush, 160, 605)
    $g.DrawString("100% LATENCY MASKED", $boxFont, $greenBrush, 160, 665)

    # Box 2: Zero Driver Paging
    $box2Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 24, 15, 45))
    $box2Pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 180, 100, 255), 2)
    $g.FillRectangle($box2Brush, 980, 360, 780, 380)
    $g.DrawRectangle($box2Pen, 980, 360, 780, 380)

    $g.DrawString("2. Pinned DRAM L2 Pool", $boxFont, $purpleBrush, 1020, 400)
    $g.DrawString("- Expert weights locked via VirtualAlloc.", $subFont, $whiteBrush, 1020, 470)
    $g.DrawString("- Non-pageable physical RAM allocation.", $subFont, $whiteBrush, 1020, 515)
    $g.DrawString("- Bypasses Windows WDDM 3.x memory virtualization.", $subFont, $whiteBrush, 1020, 560)
    $g.DrawString("- 100% Tensor Core execution (0% CPU core load).", $subFont, $cyanBrush, 1020, 605)
    $g.DrawString("ZERO WDDM THRASHING", $boxFont, $purpleBrush, 1020, 665)

    $bottomBoxBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 15, 30, 20))
    $g.FillRectangle($bottomBoxBrush, 120, 780, 1640, 90)
    $g.DrawString("Key Difference: llama.cpp pauses the GPU on every miss (48 waits/tok). Minnie streams in parallel.", $subFont, $greenBrush, 160, 810)
}

# ----------------- SLIDE 5: Next Steps (Apple Silicon) -----------------
Create-Slide "slide_05.png" {
    param($g)
    $titleFont = New-Object System.Drawing.Font("Segoe UI", 44, [System.Drawing.FontStyle]::Bold)
    $subFont = New-Object System.Drawing.Font("Segoe UI", 20, [System.Drawing.FontStyle]::Regular)
    $monoFont = New-Object System.Drawing.Font("Consolas", 18, [System.Drawing.FontStyle]::Bold)
    $boxFont = New-Object System.Drawing.Font("Segoe UI", 22, [System.Drawing.FontStyle]::Bold)

    $whiteBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
    $cyanBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 0, 240, 255))
    $appleBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 230, 235, 245))
    $mutedBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 203, 213, 225))
    $goldBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 255, 183, 3))

    $g.DrawString("ROADMAP & FUTURE HORIZONS", $monoFont, $goldBrush, 80, 150)
    $g.DrawString("Next Frontier: Apple Silicon Metal Port", $titleFont, $whiteBrush, 75, 195)
    $g.DrawString("Bringing Nemotron 3.5 Bolt to Apple M-Series with native Metal and Unified Memory.", $subFont, $mutedBrush, 80, 280)

    # Box 1: Apple Silicon Advantage
    $box1Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 20, 25, 38))
    $box1Pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 0, 240, 255), 2)
    $g.FillRectangle($box1Brush, 120, 360, 780, 420)
    $g.DrawRectangle($box1Pen, 120, 360, 780, 420)

    $g.DrawString("The Apple Silicon Advantage", $boxFont, $cyanBrush, 160, 400)
    $g.DrawString("- Unified Memory Architecture (UMA):", $subFont, $whiteBrush, 160, 470)
    $g.DrawString("  Zero PCIe bus bottleneck; CPU and GPU share DRAM.", $subFont, $mutedBrush, 160, 515)
    $g.DrawString("- Up to 128GB+ memory on Max and Ultra chips.", $subFont, $whiteBrush, 160, 560)
    $g.DrawString("- 100% zero-copy MoE expert routing.", $subFont, $whiteBrush, 160, 605)
    $g.DrawString("- No PCIe transfers required whatsoever.", $subFont, $cyanBrush, 160, 650)
    $g.DrawString("ZERO-COPY UMA SHARING", $boxFont, $cyanBrush, 160, 715)

    # Box 2: Metal Engine Features
    $box2Brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 24, 20, 40))
    $box2Pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 255, 183, 3), 2)
    $g.FillRectangle($box2Brush, 980, 360, 780, 420)
    $g.DrawRectangle($box2Pen, 980, 360, 780, 420)

    $g.DrawString("Upcoming Metal Features", $boxFont, $goldBrush, 1020, 400)
    $g.DrawString("- MetalPerformancePrimitives (MPP) Tiled Engine.", $subFont, $whiteBrush, 1020, 470)
    $g.DrawString("- Native <metal_tensor> hardware acceleration.", $subFont, $whiteBrush, 1020, 515)
    $g.DrawString("- On-the-fly threadgroup SRAM dequantization.", $subFont, $whiteBrush, 1020, 560)
    $g.DrawString("- Mamba-2 SSM SIMD-shuffled threadgroup kernels.", $subFont, $whiteBrush, 1020, 605)
    $g.DrawString("- Target: 15-20 tok/s on M-Series Max.", $subFont, $goldBrush, 1020, 650)
    $g.DrawString("HARDWARE ACCELERATION", $boxFont, $goldBrush, 1020, 715)

    $bottomBoxBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 15, 25, 40))
    $g.FillRectangle($bottomBoxBrush, 120, 820, 1640, 70)
    $g.DrawString("Next Steps: Completing the Metal Mamba-2 kernel and releasing the macOS universal build!", $subFont, $cyanBrush, 160, 842)
}

Write-Host "All 5 slide frames successfully generated in $outDir"
