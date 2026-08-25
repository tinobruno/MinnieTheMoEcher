import pexpect
import sys

child = pexpect.spawn('python3 chat.py --show-reasoning --repetition-penalty 1.0', encoding='utf-8')
child.logfile = sys.stdout

# Wait for prompt
child.expect('You ❯')

# Send first message
print("\n--- Sending first message ---")
child.sendline("Hello")

# Wait for prompt again
child.expect('You ❯')

# Send second message
print("\n--- Sending second message ---")
child.sendline("what is commodore Amiga")

# Wait for final output
child.expect('You ❯')

print("\n--- Test Complete ---")
child.sendline("/quit")
child.expect(pexpect.EOF)
