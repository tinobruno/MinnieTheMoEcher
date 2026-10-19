#!/usr/bin/env python3
"""
Forwarder to scripts/upload_nemotron_to_hf.py
"""
import os
import sys

script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "upload_nemotron_to_hf.py"))
if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, script_path] + sys.argv[1:])
