import numpy as np

from src.mask import split_gru_mask


print("=" * 70)
print("IMPULSE GUARD - GRU COMPLEX MASK TEST")
print("=" * 70)

# Simulate GRU output:
# batch = 1
# time frames = 100
# outputs = 44
gru_output = np.random.randn(
    1,
    100,
    44
).astype(np.float32)

complex_mask = split_gru_mask(
    gru_output
)

print()
print("GRU output shape:", gru_output.shape)
print("Complex mask shape:", complex_mask.shape)
print("Complex mask dtype:", complex_mask.dtype)

print()
print("Expected GRU output:")
print("(1, 100, 44)")

print()
print("Expected complex mask:")
print("(1, 100, 22)")

print()
print("Example mask:")
print(complex_mask[0, 0])

print()
print("Real part shape:",
      complex_mask.real.shape)

print("Imaginary part shape:",
      complex_mask.imag.shape)

print()
print("DONE")
print("=" * 70)