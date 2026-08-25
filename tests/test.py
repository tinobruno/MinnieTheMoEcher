with open("src/cuda/activations.cu") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "gemv_int2_kernel" in line:
            print(f"Line {i}: {line.strip()}")
