#!/bin/bash
kill -9 $(pidof moecher)
nsys profile -y 5 -d 20 -o moecher_profile --force-overwrite true ./build/moecher --manifest moecher_manifest_q2.json --max-vram 95.0 > logs/nsys.log 2>&1 &
SERVER_PID=$!
sleep 15
python3 test_curl.py
kill -9 $SERVER_PID
