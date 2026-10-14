# Formalized Verification Suite for MinnieTheMoEcher Hybrid Arbiter
# Replicating the exact scenarios tested by the user on bare-metal GPU

$BaseUrl = "http://127.0.0.1:8001"

$PromptGood = @"
Evaluate applicant for loan risk:
- Applicant: Jane Doe
- Credit Score: 790
- Annual Income: $145,000
- Debt-to-Income (DTI): 18%
- Late Payments (past 24 mo): 0
- Requested Amount: $15,000

Underwriting Decision:
"@

$PromptBad = @"
Evaluate applicant for loan risk:
- Applicant: Jane Doe
- Credit Score: 79
- Annual Income: $14,000
- Debt-to-Income (DTI): 18%
- Late Payments (past 24 mo): 0
- Requested Amount: $150,000

Underwriting Decision:
"@

$PromptBadWithRule = @"
Evaluate applicant for loan risk:
- Applicant: Jane Doe
- Credit Score: 79
- Annual Income: $14,000
- Debt-to-Income (DTI): 18%
- Late Payments (past 24 mo): 0
- Requested Amount: $150,000

Rule: If credit score is below 600 or requested amount exceeds annual income, the decision must be REJECTED.

Underwriting Decision:
"@

$Choices = @("APPROVED", "REJECTED")

function Test-ChatCompletion {
    param(
        [string]$Name,
        [string]$Prompt,
        [bool]$ZdbEnabled,
        [bool]$AutoFallback = $true
    )

    Write-Host "`n===========================================================================" -ForegroundColor Cyan
    Write-Host " [TEST] $Name" -ForegroundColor Yellow
    Write-Host " ZDB: $ZdbEnabled | Auto-Fallback: $AutoFallback" -ForegroundColor DarkGray
    Write-Host "===========================================================================" -ForegroundColor Cyan

    $bodyObj = @{
        model = "minnie-moe"
        messages = @(
            @{ role = "user"; content = $Prompt }
        )
        stream = $false
        temperature = 0.0
        enable_thinking = $true
        max_tokens = 512
    }

    if ($ZdbEnabled) {
        $bodyObj["zero_decode_bypass"] = $true
        $bodyObj["choices"] = $Choices
        $bodyObj["auto_fallback"] = $AutoFallback
        $bodyObj["min_confidence"] = 0.70
        $bodyObj["max_top1_gap"] = 0.50
    }

    $jsonBody = $bodyObj | ConvertTo-Json -Depth 5
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" -Method Post -Body $jsonBody -ContentType "application/json" -TimeoutSec 120
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        return $null
    }
    $sw.Stop()

    $elapsedMs = $sw.Elapsed.TotalMilliseconds
    $content = $resp.choices[0].message.content
    $zdb = $resp.zero_decode_bypass

    Write-Host " Wall Latency:      $([math]::Round($elapsedMs, 2)) ms" -ForegroundColor White
    if ($zdb -and $zdb.enabled) {
        $arb = $zdb.arbiter
        Write-Host " ZDB Fast Path:     SUCCESS (0 decode passes)" -ForegroundColor Green
        Write-Host " Gate Passed:       $($arb.gate_passed)" -ForegroundColor Cyan
        Write-Host " System Path:       $($arb.system_path)" -ForegroundColor Cyan
        Write-Host " Winning Choice:    '$($zdb.selected_choice)'" -ForegroundColor Yellow
        Write-Host " Confidence:        $([math]::Round($zdb.confidence * 100, 2))%" -ForegroundColor White
        Write-Host " Top-1 Logit Gap:   $([math]::Round($arb.top1_gap, 2))" -ForegroundColor DarkYellow
    } else {
        Write-Host " System 2 Fallback: ACTIVE (Autoregressive Thinking & Reasoning)" -ForegroundColor Magenta
        $first100 = if ($content.Length -gt 250) { $content.Substring(0, 250) } else { $content }
        Write-Host " Output Preview:    $first100..." -ForegroundColor Gray
    }
    return $resp
}

function Test-DecideEndpoint {
    param(
        [string]$Name,
        [string]$Prompt,
        [bool]$AutoFallback = $true
    )

    Write-Host "`n---------------------------------------------------------------------------" -ForegroundColor DarkCyan
    Write-Host " [/v1/decide] $Name (auto_fallback=$AutoFallback)" -ForegroundColor Cyan
    Write-Host "---------------------------------------------------------------------------" -ForegroundColor DarkCyan

    $bodyObj = @{
        prompt = $Prompt
        choices = $Choices
        auto_fallback = $AutoFallback
        min_confidence = 0.70
        max_top1_gap = 0.50
        format = "json"
    }

    $jsonBody = $bodyObj | ConvertTo-Json -Depth 5
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-RestMethod -Uri "$BaseUrl/v1/decide" -Method Post -Body $jsonBody -ContentType "application/json" -TimeoutSec 120
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
        return $null
    }
    $sw.Stop()

    $elapsedMs = $sw.Elapsed.TotalMilliseconds
    $arb = $resp.arbiter

    Write-Host " Decision:          '$($resp.decision)'" -ForegroundColor Yellow
    Write-Host " Gate Passed:       $($arb.gate_passed)" -ForegroundColor $(if ($arb.gate_passed) { "Green" } else { "Magenta" })
    Write-Host " System Path:       $($arb.system_path)" -ForegroundColor Cyan
    Write-Host " Decode Passes:     $($resp.decode_passes)" -ForegroundColor White
    Write-Host " Confidence:        $([math]::Round($resp.confidence * 100, 2))%" -ForegroundColor White
    Write-Host " Top-1 Logit Gap:   $([math]::Round($arb.top1_gap, 2))" -ForegroundColor DarkYellow
    Write-Host " Total Latency:     $([math]::Round($elapsedMs, 2)) ms" -ForegroundColor White

    return $resp
}

Write-Host "`n===========================================================================" -ForegroundColor Green
Write-Host " STARTING FORMALIZED VERIFICATION OF HYBRID ARBITER ON BARE METAL" -ForegroundColor Green
Write-Host "===========================================================================" -ForegroundColor Green

# 1. Good Applicant (Score 790, Income $145k) -> Should pass gate & return APPROVED in <100ms
Test-ChatCompletion -Name "Good Applicant (Score 790, $145k) -> Fast Path" -Prompt $PromptGood -ZdbEnabled $true -AutoFallback $true

# 2. Bad Applicant (Score 79, Income $14k, Req $150k) -> Gate should FAIL, auto-fallback to System 2 decode & DENIED/REJECTED
Test-ChatCompletion -Name "Bad Applicant (Score 79, $14k, Req $150k) -> Arbiter Fallback" -Prompt $PromptBad -ZdbEnabled $true -AutoFallback $true

# 3. Bad Applicant with Explicit Rule in prompt -> Gate should PASS & return REJECTED in <100ms
Test-ChatCompletion -Name "Bad Applicant with Explicit Rule -> Fast Path" -Prompt $PromptBadWithRule -ZdbEnabled $true -AutoFallback $true

# 4. Direct /v1/decide endpoint tests
Test-DecideEndpoint -Name "Good Applicant via /v1/decide" -Prompt $PromptGood -AutoFallback $true
Test-DecideEndpoint -Name "Bad Applicant via /v1/decide" -Prompt $PromptBad -AutoFallback $true
Test-DecideEndpoint -Name "Bad Applicant with Rule via /v1/decide" -Prompt $PromptBadWithRule -AutoFallback $true

Write-Host "`n===========================================================================" -ForegroundColor Green
Write-Host " VERIFICATION SUITE FINISHED" -ForegroundColor Green
Write-Host "===========================================================================" -ForegroundColor Green
