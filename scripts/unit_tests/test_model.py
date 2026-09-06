import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from src.model import create_gru_model


model = create_gru_model()

print("=" * 70)
print("IMPULSE GUARD - GRU MODEL TEST")
print("=" * 70)

print()
print("Model summary:")

model.summary()

print()
print("Testing with dummy feature sequence...")

# 1 sample
# 100 time frames
# 44 features per frame
dummy_input = np.random.randn(
    1,
    100,
    44
).astype(np.float32)

output = model(dummy_input)

print()
print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)

print()
print("Expected input:")
print("(batch, time_frames, 44)")

print()
print("Expected output:")
print("(batch, time_frames, 44)")

print()
print("DONE")
print("=" * 70)