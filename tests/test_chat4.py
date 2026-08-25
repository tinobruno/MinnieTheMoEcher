import pexpect
import sys

child = pexpect.spawn('python3 chat.py --show-reasoning', encoding='utf-8')
child.expect('You ❯')
child.sendline('/temp 0.1')
child.expect('You ❯')
child.sendline('Hello, who are you?')

output = ""
while True:
    try:
        chunk = child.read_nonblocking(size=1024, timeout=30)
        output += chunk
        sys.stdout.write(chunk)
        sys.stdout.flush()
        if "You ❯" in output:
            break
    except pexpect.TIMEOUT:
        break
    except pexpect.EOF:
        break
