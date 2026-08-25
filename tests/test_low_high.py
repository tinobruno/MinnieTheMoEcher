import math
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )

dim = 64
base = 10000
original = 4096

low_dim = yarn_find_correction_dim(32, dim, base, original)
high_dim = yarn_find_correction_dim(1, dim, base, original)

print("low_dim:", low_dim)
print("high_dim:", high_dim)
