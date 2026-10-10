param([string]$ProcessName = "moecher")

$samples = (Get-Counter "\GPU Process Memory(*)\*").CounterSamples
$matched = $samples | Where-Object { $_.Path -like "*Dedicated Usage*" -or $_.Path -like "*Shared Usage*" }

$procList = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
if (-not $procList) {
    Write-Output "Process $ProcessName not found."
    exit 0
}

foreach ($p in $procList) {
    $pidStr = "pid_$($p.Id)_"
    $dedSamples = $samples | Where-Object { $_.Path -like "*($pidStr*)\dedicated usage" }
    $shaSamples = $samples | Where-Object { $_.Path -like "*($pidStr*)\shared usage" }
    
    $totalDed = ($dedSamples | Measure-Object -Property CookedValue -Sum).Sum
    $totalSha = ($shaSamples | Measure-Object -Property CookedValue -Sum).Sum
    
    [PSCustomObject]@{
        ProcessName = $p.ProcessName
        PID = $p.Id
        WorkingSetMB = [math]::Round($p.WorkingSet64 / 1MB, 2)
        DedicatedVRAM_MB = [math]::Round($totalDed / 1MB, 2)
        SharedPCIe_MB = [math]::Round($totalSha / 1MB, 2)
    } | Format-Table -AutoSize
}
