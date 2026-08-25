import pexpect
import sys
import re

def test_chat():
    child = pexpect.spawn('python3 chat.py --show-reasoning', encoding='utf-8')
    child.logfile = sys.stdout

    child.expect(r'You ❯', timeout=10)
    print("--- Sending first message ---")
    child.sendline("Hello")
    
    child.expect(r'You ❯', timeout=60)
    print("--- Sending second message ---")
    child.sendline("what is commodore Amiga")
    
    child.expect(r'You ❯', timeout=60)
    print("--- Test Complete ---")
    child.sendline("/quit")
    child.expect(pexpect.EOF)

test_chat()
