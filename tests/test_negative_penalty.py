import math

logits = [-20.0, -2.0, 10.0]
for l in logits:
    print(f"Logit: {l}, Penalized: {l / 1.1}")
