@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Starting MinnieTheMoEcher Nemotron Bolt Benchmark
echo ===================================================

start /b .\moecher.exe -m models\nemotron_3.5_bolt_24gb\moecher_manifest.json --port 8001 > minnie_server.log 2>&1

echo Waiting for server to initialize...
:wait_loop
timeout /t 2 /nobreak >nul
curl.exe -s http://localhost:8001/ >nul 2>&1
if errorlevel 1 goto wait_loop

echo Server is online! Sending benchmark query...
curl.exe -s -X POST http://localhost:8001/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"nemotron\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain the difference between SSD and RAM in simple terms.\"}],\"max_tokens\":50,\"temperature\":0.0}" > benchmark_minnie_result.json

echo Benchmark finished!
type benchmark_minnie_result.json
