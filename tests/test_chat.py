import pexpect
import sys
import re

child = pexpect.spawn('python3 chat.py --show-reasoning', encoding='utf-8')
child.expect('You ❯')
child.sendline('/system You are a helpful assistant.')
child.expect('You ❯')
child.sendline('/temp 0.6')
child.expect('You ❯')
child.sendline('what is commodore Amiga?')
child.expect('Assistant ❯', timeout=120)

output = ""
while True:
    try:
        chunk = child.read_nonblocking(size=1024, timeout=30)
        output += chunk
        if "You ❯" in output:
            break
    except pexpect.TIMEOUT:
        print("Timeout waiting for output")
        break
    except pexpect.EOF:
        break

# Look for ANSI DIM codes (ESC[2m)
has_dim = '\033[2m' in output
if has_dim:
    dim_content = re.search(r'\033\[2m(.*?)(\033\[0m|$)', output, re.DOTALL)
    if dim_content:
        print("REASONING BLOCK:")
        print(dim_content.group(1).strip())
    else:
        print("DIM content format mismatch.")
else:
    print("NO DIM CONTENT FOUND.")

print("\n--- FINAL CONTENT ---")
# Get text after the last RESET code
last_reset = output.rfind('\033[0m')
if last_reset != -1:
    final_text = output[last_reset+4:]
    # Remove any other ansi codes
    final_text = re.sub(r'\x1b\[[0-9;]*m', '', final_text)
    print(final_text.replace("You ❯", "").strip())
