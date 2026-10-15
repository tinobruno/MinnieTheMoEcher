param(
    [Parameter(Mandatory=$true)][string]$Path,
    [Parameter(Mandatory=$true)][int]$StartLine,
    [Parameter(Mandatory=$true)][int]$EndLine,
    [string]$Replacement = ""
)

$lines = [System.IO.File]::ReadAllLines($Path)
$newLines = [System.Collections.Generic.List[string]]::new()

for ($i = 0; $i -lt $lines.Length; $i++) {
    $lineNum = $i + 1
    if ($lineNum -ge $StartLine -and $lineNum -le $EndLine) {
        if ($lineNum -eq $StartLine -and $Replacement -ne "") {
            $repLines = $Replacement -split "`r?`n"
            foreach ($rl in $repLines) {
                $newLines.Add($rl)
            }
        }
        continue
    }
    $newLines.Add($lines[$i])
}

[System.IO.File]::WriteAllLines($Path, $newLines)
Write-Host "Successfully replaced lines $StartLine to $EndLine in $Path. New line count: $($newLines.Count)"
