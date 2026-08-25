import pexpect
import sys

child = pexpect.spawn('python3 chat.py --show-reasoning', encoding='utf-8')
child.expect('You ❯')
child.sendline('/temp 0.6')
child.expect('You ❯')
child.sendline('what is commodore Amiga?')

output = ""
while True:
    try:
        chunk = child.read_nonblocking(size=1024, timeout=30)
        output += chunk
        if "You ❯" in output:
            break
    except pexpect.TIMEOUT:
        break
    except pexpect.EOF:
        break

print("RAW BYTES:")
print(repr(output))
