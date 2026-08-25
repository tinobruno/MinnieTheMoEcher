import subprocess
import time

p = subprocess.Popen(["python3", "chat.py", "--raw"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
stdout, stderr = p.communicate("hello\n/quit\n")
print(stdout)
