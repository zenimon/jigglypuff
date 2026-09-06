import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from src.mask import (
    split_gru_mask,
    expand_subband_mask
)

from src.subbands import (
    get_bark_band_centers
)


print("=" * 70)
print("IMPULSE GUARD - SUBBAND MASK EXPANSION TEST")
print("=" * 70)

# Simulated GRU output
gru_output = np.random.randn(
    1,
    100,
    44
).astype(np.float32)

# 44 values -> 22 complex values
subband_mask = split_gru_mask(
    gru_output
)

# Get the exact Bark-band centers
center_frequencies = get_bark_band_centers()

# 22 complex values -> 257 complex values
fullband_mask = expand_subband_mask(
    subband_mask,
    center_frequencies
)

print()
print("GRU output shape:")
print(gru_output.shape)

print()
print("Sub-band mask shape:")
print(subband_mask.shape)

print()
print("Full-band mask shape:")
print(fullband_mask.shape)

print()
print("Full-band mask dtype:")
print(fullband_mask.dtype)

print()
print("Expected:")
print("GRU:           (1, 100, 44)")
print("Sub-band mask: (1, 100, 22)")
print("Full mask:     (1, 100, 257)")

print()
print("Example expanded mask:")
print(fullband_mask[0, 0])

print()
print("Minimum magnitude:",
      np.abs(fullband_mask).min())

print("Maximum magnitude:",
      np.abs(fullband_mask).max())

print()
print("DONE")
print("=" * 70)