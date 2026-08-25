import pexpect
import sys

child = pexpect.spawn('python3 chat.py --hide-reasoning --temperature 0.0', encoding='utf-8')
child.logfile_read = sys.stdout

child.expect('You ❯')
print("\n--- Sending Turn 1 ---")
child.sendline('what is commodore Amiga ?')

child.expect('You ❯', timeout=120)
print("\n--- Sending Turn 2 ---")
child.sendline('was it 16 bit or 32 bit ?')

child.expect('You ❯', timeout=120)
print("\n--- Finished ---")
child.sendline('/quit')
