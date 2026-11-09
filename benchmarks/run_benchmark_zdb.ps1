param(
    [string]$BaseUrl = "http://127.0.0.1:8001",
    [int]$Iterations = 5
)

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " BENCHMARK: Zero Decode Bypass (ZDB) vs. Standard Autoregressive Decoding       " -ForegroundColor Cyan
Write-Host " Proving the Sub-50ms Non-Autoregressive 'System 1' Inference Speedup Theory   " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$testCases = @(
    @{
        Name = "Loan Risk Approval (Binary 2-Way)"
        Prompt = @"
You are an expert automated underwriting engine. Evaluate applicant Jane Doe:
Credit Score: 790, Annual Income: $145,000, Debt-to-Income: 18%, Late Payments: 0, Requested: $15,000 for Home Improvement.
Based strictly on underwriting guidelines, is the loan APPROVED or REJECTED?
Answer with the single decision word:
"@
        Choices = @("APPROVED", "REJECTED")
    },
    @{
        Name = "Sentiment Analysis (3-Way)"
        Prompt = @"
Classify the sentiment of this customer feedback into POSITIVE, NEGATIVE, or NEUTRAL:
"The GPU inference speed with this engine is unbelievable, sub-30ms latency on a single 3090 RTX!"
Sentiment:
"@
        Choices = @("POSITIVE", "NEGATIVE", "NEUTRAL")
    },
    @{
        Name = "Agent Tool Dispatcher (4-Way)"
        Prompt = @"
You are an intelligent agent dispatcher. Available tools: [web_search, read_file, fetch_url, none].
User query: "What is the stock price of NVDA today?"
Which tool should be called?
Selected tool:
"@
        Choices = @("web_search", "read_file", "fetch_url", "none")
    },
    @{
        Name = "Compliance Threat Classification (4-Way)"
        Prompt = @"
Analyze this security log event for compliance risk: [CRITICAL, HIGH, MEDIUM, LOW].
Log: "Failed SSH login attempt 15 times within 10 seconds from unrecognized foreign IP 198.51.100.24 targeting root account."
Risk level:
"@
        Choices = @("CRITICAL", "HIGH", "MEDIUM", "LOW")
    }
)

$results = @()

foreach ($tc in $testCases) {
    Write-Host "`n>>> Running Benchmark Case: $($tc.Name) ($($tc.Choices.Count) choices) <<<" -ForegroundColor Yellow

    $zdbTimes = @()
    $stdTimes = @()
    $zdbDecisions = @()
    $stdOutputs = @()

    # Warmup
    try {
        $warmPayload = @{
            prompt = $tc.Prompt
            choices = $tc.Choices
            format = "json"
            bypass_prefix_cache = $false
        } | ConvertTo-Json -Depth 5
        $null = Invoke-RestMethod -Uri "$BaseUrl/v1/decide" -Method Post -Body $warmPayload -ContentType "application/json"
    } catch {}

    for ($i = 1; $i -le $Iterations; $i++) {
        # 1. ZDB Mode (Zero Decode Bypass)
        $zdbPayload = @{
            prompt = $tc.Prompt
            choices = $tc.Choices
            format = "json"
            bypass_prefix_cache = $false
        } | ConvertTo-Json -Depth 5

        $swZdb = [System.Diagnostics.Stopwatch]::StartNew()
        $resZdb = Invoke-RestMethod -Uri "$BaseUrl/v1/decide" -Method Post -Body $zdbPayload -ContentType "application/json"
        $swZdb.Stop()
        $zdbTimes += $swZdb.Elapsed.TotalMilliseconds
        $zdbDecisions += $resZdb.decision

        # 2. Standard Autoregressive Mode (ZDB Disabled via /v1/chat/completions)
        $stdPayload = @{
            model = "minnie-moe"
            messages = @(
                @{
                    role = "user"
                    content = $tc.Prompt
                }
            )
            max_tokens = 32
            temperature = 0.0
            enable_thinking = $false
            reasoning_effort = "none"
            zero_decode_bypass = $false
        } | ConvertTo-Json -Depth 5

        $swStd = [System.Diagnostics.Stopwatch]::StartNew()
        $resStd = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" -Method Post -Body $stdPayload -ContentType "application/json"
        $swStd.Stop()
        $stdTimes += $swStd.Elapsed.TotalMilliseconds
        $stdContent = $resStd.choices[0].message.content.Trim()
        $stdOutputs += $stdContent
    }

    # Calculate statistics
    $sortedZdb = $zdbTimes | Sort-Object
    $sortedStd = $stdTimes | Sort-Object
    $medZdb = $sortedZdb[[math]::Floor($sortedZdb.Count / 2)]
    $medStd = $sortedStd[[math]::Floor($sortedStd.Count / 2)]
    $avgZdb = ($zdbTimes | Measure-Object -Average).Average
    $avgStd = ($stdTimes | Measure-Object -Average).Average
    $speedup = if ($medZdb -gt 0) { [math]::Round($medStd / $medZdb, 2) } else { 0 }

    $lastZdbDec = $zdbDecisions[-1]
    $lastStdOut = $stdOutputs[-1]
    if ($lastStdOut.Length -gt 35) { $lastStdOut = $lastStdOut.Substring(0, 32) + "..." }

    $results += [PSCustomObject]@{
        TestCase        = $tc.Name
        ChoicesCount    = $tc.Choices.Count
        ZDB_Median_ms   = [math]::Round($medZdb, 1)
        Std_Median_ms   = [math]::Round($medStd, 1)
        Speedup_Factor  = "$($speedup)x"
        ZDB_Decision    = $lastZdbDec
        Std_Output      = $lastStdOut
    }

    Write-Host "  -> ZDB Median Latency : $([math]::Round($medZdb, 1)) ms (0 decode passes)" -ForegroundColor Green
    Write-Host "  -> Std Median Latency : $([math]::Round($medStd, 1)) ms" -ForegroundColor Magenta
    Write-Host "  -> Speedup Multiplier : $($speedup)x faster" -ForegroundColor Cyan
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host " SUMMARY BENCHMARK RESULTS: ZERO DECODE BYPASS EMPIRICAL VALIDATION             " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$results | Format-Table -AutoSize | Out-String | Write-Host -ForegroundColor Green

# Output Markdown Table for Artifacts/Reports
Write-Host "`n--- Markdown Report Table ---`n" -ForegroundColor Yellow
Write-Host "| Test Case | Choices | ZDB Latency (0 decode) | Std Latency (Autoregressive) | Speedup Factor | ZDB Decision |"
Write-Host "| :--- | :---: | :---: | :---: | :---: | :--- |"
foreach ($r in $results) {
    Write-Host "| $($r.TestCase) | $($r.ChoicesCount) | **$($r.ZDB_Median_ms) ms** | $($r.Std_Median_ms) ms | **$($r.Speedup_Factor)** | `$($r.ZDB_Decision)` |"
}
