# ─────────────────────────────────────────────────────────────────────────────
#  make_installer_graphics.ps1 — Generate Inno Setup wizard bitmap assets
# ─────────────────────────────────────────────────────────────────────────────

Add-Type -AssemblyName System.Drawing

$SourcePath = "D:\dev\minniemoe\MinnieTheMoEcher\graphics\moecher_logo.png"
$WizardBmpPath = "D:\dev\minniemoe\MinnieTheMoEcher\graphics\installer_wizard.bmp"
$SmallBmpPath = "D:\dev\minniemoe\MinnieTheMoEcher\graphics\installer_small.bmp"

if (-not (Test-Path $SourcePath)) {
    Write-Error "Source image not found: $SourcePath"
}

$logo = [System.Drawing.Bitmap]::FromFile($SourcePath)

# 1. WizardImageFile: 164 x 314 (Standard Inno Setup Welcome/Finished sidebar)
$wizardWidth = 164
$wizardHeight = 314
$wizardBmp = New-Object System.Drawing.Bitmap($wizardWidth, $wizardHeight, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$g1 = [System.Drawing.Graphics]::FromImage($wizardBmp)
$g1.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$g1.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality

# Background: Crisp clean gradient
$brush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
    (New-Object System.Drawing.Point(0, 0)),
    (New-Object System.Drawing.Point(0, $wizardHeight)),
    [System.Drawing.Color]::FromArgb(245, 247, 250),
    [System.Drawing.Color]::FromArgb(220, 225, 235)
)
$g1.FillRectangle($brush, 0, 0, $wizardWidth, $wizardHeight)
$brush.Dispose()

# Draw centered walrus logo (140x140 at center)
$logoSize = 140
$logoX = ($wizardWidth - $logoSize) / 2
$logoY = ($wizardHeight - $logoSize) / 2
$g1.DrawImage($logo, $logoX, $logoY, $logoSize, $logoSize)
$g1.Dispose()

$wizardBmp.Save($WizardBmpPath, [System.Drawing.Imaging.ImageFormat]::Bmp)
$wizardBmp.Dispose()
Write-Host "Created Inno Setup sidebar banner: $WizardBmpPath" -ForegroundColor Green

# 2. WizardSmallImageFile: 55 x 55 (Standard Inno Setup header logo)
$smallWidth = 55
$smallHeight = 55
$smallBmp = New-Object System.Drawing.Bitmap($smallWidth, $smallHeight, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$g2 = [System.Drawing.Graphics]::FromImage($smallBmp)
$g2.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$g2.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality

# White background
$g2.Clear([System.Drawing.Color]::White)

# Draw fitted logo (48x48 centered)
$sSize = 48
$sX = ($smallWidth - $sSize) / 2
$sY = ($smallHeight - $sSize) / 2
$g2.DrawImage($logo, $sX, $sY, $sSize, $sSize)
$g2.Dispose()

$smallBmp.Save($SmallBmpPath, [System.Drawing.Imaging.ImageFormat]::Bmp)
$smallBmp.Dispose()
$logo.Dispose()

Write-Host "Created Inno Setup header icon: $SmallBmpPath" -ForegroundColor Green
