$ffmpeg = "C:\Program Files (x86)\CoolerMaster\MasterPlus\ffmpeg.exe"
$baseDir = "e:\dev\MinnieTheMoEcher"
$framesDir = Join-Path $baseDir "video_frames"
$voiceDir = Join-Path $baseDir "voice_clips"
$scenesDir = Join-Path $baseDir "scene_clips"

if (-not (Test-Path $scenesDir)) { New-Item -ItemType Directory -Path $scenesDir | Out-Null }

$concatList = Join-Path $scenesDir "scenes_list.txt"
$concatContent = @()

1..5 | ForEach-Object {
    $i = $_
    $slideImg = Join-Path $framesDir "slide_0$i.png"
    $voiceWav = Join-Path $voiceDir "voice_0$i.wav"
    $sceneMp4 = Join-Path $scenesDir "scene_0$i.mp4"

    Write-Host "Rendering Scene $i with voiceover..."
    # Render slide image looped over the voice clip duration with a tiny 0.5s trailing silence buffer
    & $ffmpeg -loop 1 -i $slideImg -i $voiceWav -c:v libx264 -preset fast -tune stillimage -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k -shortest -y $sceneMp4 2>&1 | Out-Null
    
    # Store path for concat demuxer (using forward slashes)
    $normalizedPath = $sceneMp4 -replace '\\', '/'
    $concatContent += "file '$normalizedPath'"
}

$concatContent | Set-Content -Path $concatList -Encoding ASCII
Write-Host "Concat list written to $concatList"

$finalMp4 = Join-Path $baseDir "kai_style_moe_benchmark.mp4"
Write-Host "Concatenating scenes into final video: $finalMp4"
& $ffmpeg -f concat -safe 0 -i $concatList -c copy -y $finalMp4

Write-Host "Done! Final video generated at: $finalMp4"
