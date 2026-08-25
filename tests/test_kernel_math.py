import torch
import numpy as np
import struct

def bfloat16_to_float(bf16_int):
    # Pad to 32 bits and interpret as float
    return struct.unpack('!f', struct.pack('!I', (bf16_int & 0xFFFF) << 16))[0]

def float_to_bfloat16(f):
    # Pack as float, extract top 16 bits
    return struct.unpack('!I', struct.pack('!f', f))[0] >> 16

# test our unpacking
x_q_final = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.uint8)

packed = (x_q_final[:, 0::4] | 
         (x_q_final[:, 1::4] << 2) | 
         (x_q_final[:, 2::4] << 4) | 
         (x_q_final[:, 3::4] << 6))

print("Packed:", packed)

b = packed[0, 0].item()

v0 = b & 0x03
v1 = (b >> 2) & 0x03
v2 = (b >> 4) & 0x03
v3 = (b >> 6) & 0x03

print("Unpacked from b:", v0, v1, v2, v3)

b2 = packed[0, 1].item()
v4 = b2 & 0x03
v5 = (b2 >> 2) & 0x03
v6 = (b2 >> 4) & 0x03
v7 = (b2 >> 6) & 0x03
print("Unpacked from b2:", v4, v5, v6, v7)

