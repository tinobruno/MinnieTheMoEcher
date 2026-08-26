# ─────────────────────────────────────────────────────────────────────────────
#  make_icon.ps1 — Convert JPG/PNG logo into Windows multi-res .ico file
# ─────────────────────────────────────────────────────────────────────────────

Add-Type -AssemblyName System.Drawing

$SourcePath = "D:\dev\minniemoe\MinnieTheMoEcher\graphics\moecher_logo_b_clear.jpg"
$OutIcoPath = "D:\dev\minniemoe\MinnieTheMoEcher\graphics\moecher.ico"
$OutPngPath = "D:\dev\minniemoe\MinnieTheMoEcher\graphics\moecher_logo.png"

if (-not (Test-Path $SourcePath)) {
    Write-Error "Source image not found: $SourcePath"
}

$srcBitmap = [System.Drawing.Bitmap]::FromFile($SourcePath)
$width = $srcBitmap.Width
$height = $srcBitmap.Height

Write-Host "Loaded source image: ${width}x${height}" -ForegroundColor Cyan

# Create 32-bit transparent ARGB bitmap
$cleanBitmap = New-Object System.Drawing.Bitmap($width, $height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$cx = $width / 2.0
$cy = $height / 2.0
# The outer black circle radius is around 415 pixels
$circleRadius = $width * 0.415

for ($y = 0; $y -lt $height; $y++) {
    for ($x = 0; $x -lt $width; $x++) {
        $p = $srcBitmap.GetPixel($x, $y)
        $dx = $x - $cx
        $dy = $y - $cy
        $dist = [math]::Sqrt($dx * $dx + $dy * $dy)

        $r = $p.R
        $g = $p.G
        $b = $p.B

        # If it's pure black/dark line (the drawing strokes)
        $brightness = ($r + $g + $b) / 3.0

        if ($dist -gt ($circleRadius + 4)) {
            # Outside the badge circle: transparent
            $cleanBitmap.SetPixel($x, $y, [System.Drawing.Color]::FromArgb(0, 0, 0, 0))
        } elseif ($dist -gt $circleRadius) {
            # Antialias boundary
            $alpha = [int](255.0 * (1.0 - (($dist - $circleRadius) / 4.0)))
            if ($brightness -lt 100) {
                $cleanBitmap.SetPixel($x, $y, [System.Drawing.Color]::FromArgb($alpha, $r, $g, $b))
            } else {
                $cleanBitmap.SetPixel($x, $y, [System.Drawing.Color]::FromArgb(0, 0, 0, 0))
            }
        } else {
            # Inside badge
            if ($brightness -lt 120) {
                # Dark line art
                $cleanBitmap.SetPixel($x, $y, [System.Drawing.Color]::FromArgb(255, $r, $g, $b))
            } else {
                # Background inside badge: white fill with smooth blend
                $cleanBitmap.SetPixel($x, $y, [System.Drawing.Color]::FromArgb(255, 255, 255, 255))
            }
        }
    }
}

# Save crisp 32-bit PNG with transparent background
$cleanBitmap.Save($OutPngPath, [System.Drawing.Imaging.ImageFormat]::Png)
Write-Host "Saved transparent PNG to $OutPngPath" -ForegroundColor Green

# Prepare icon resolutions: 256, 128, 64, 48, 32, 16
$sizes = @(256, 128, 64, 48, 32, 16)
$pngStreams = @()

foreach ($sz in $sizes) {
    $resized = New-Object System.Drawing.Bitmap($sz, $sz, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($resized)
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
    $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $g.Clear([System.Drawing.Color]::Transparent)
    $g.DrawImage($cleanBitmap, 0, 0, $sz, $sz)
    $g.Dispose()

    $ms = New-Object System.IO.MemoryStream
    $resized.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    $pngStreams += ,@($sz, $ms.ToArray())
    $resized.Dispose()
}

$cleanBitmap.Dispose()
$srcBitmap.Dispose()

# Write .ICO file
$fs = [System.IO.File]::Create($OutIcoPath)
$bw = New-Object System.IO.BinaryWriter($fs)

# ICONDIR header
$bw.Write([uint16]0)                # Reserved
$bw.Write([uint16]1)                # Type 1 = Icon
$bw.Write([uint16]$pngStreams.Count) # Image count

$headerSize = 6
$dirEntrySize = 16
$currentOffset = $headerSize + ($dirEntrySize * $pngStreams.Count)

# Write ICONDIRENTRY list
foreach ($item in $pngStreams) {
    $sz = $item[0]
    $bytes = $item[1]
    $bWidth = if ($sz -ge 256) { 0 } else { [byte]$sz }
    $bHeight = if ($sz -ge 256) { 0 } else { [byte]$sz }

    $bw.Write([byte]$bWidth)            # Width
    $bw.Write([byte]$bHeight)           # Height
    $bw.Write([byte]0)                  # Color count
    $bw.Write([byte]0)                  # Reserved
    $bw.Write([uint16]1)                # Color planes
    $bw.Write([uint16]32)               # Bits per pixel
    $bw.Write([uint32]$bytes.Length)    # Image size
    $bw.Write([uint32]$currentOffset)   # Image offset

    $currentOffset += $bytes.Length
}

# Write PNG byte payloads
foreach ($item in $pngStreams) {
    $bytes = $item[1]
    $bw.Write($bytes)
}

$bw.Flush()
$bw.Close()
$fs.Close()

Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  Successfully generated Windows Icon: $OutIcoPath" -ForegroundColor Green
Write-Host "  Included resolutions: 256x256, 128x128, 64x64, 48x48, 32x32, 16x16" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor Green
