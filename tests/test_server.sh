#!/bin/bash
./build/moecher ../weights 8001 > server.log 2>&1 &
SERVER_PID=$!
sleep 5
curl -s -X POST http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}], "max_tokens": 15, "temperature": 0.0}' > out.json
kill $SERVER_PID
cat out.json
