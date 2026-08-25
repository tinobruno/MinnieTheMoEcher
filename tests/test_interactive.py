import pexpect
import sys
child = pexpect.spawn("python3 chat.py", encoding="utf-8")
child.expect("You ❯")
child.sendline("hello")
child.expect("You ❯", timeout=20)
print(child.before)
child.sendline("what is the capital of Italy?")
child.expect("You ❯", timeout=20)
print(child.before)
child.sendline("/quit")
