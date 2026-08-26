@echo off
title Moecher - Test Prompt
cd /d "%~dp0"

echo ======================================================================
echo   Testing Moecher API (http://127.0.0.1:8001/v1/chat/completions)
echo ======================================================================
echo.

curl -s -X POST http://127.0.0.1:8001/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"qwen3.8-27b-q4\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 4-line poem about the deep ocean.\"}],\"temperature\":0.7,\"enable_thinking\":false}"

echo.
echo.
pause
