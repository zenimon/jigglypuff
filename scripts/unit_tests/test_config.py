import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import (
    SAMPLE_RATE,
    FRAME_SIZE,
    HOP_SIZE,
    FFT_SIZE,
    NUM_FREQ_BINS,
    NUM_SUBBANDS,
    GRU_HIDDEN_SIZE,
)

print("Sample rate:", SAMPLE_RATE)
print("Frame size:", FRAME_SIZE)
print("Hop size:", HOP_SIZE)
print("FFT size:", FFT_SIZE)
print("Frequency bins:", NUM_FREQ_BINS)
print("Subbands:", NUM_SUBBANDS)
print("GRU hidden size:", GRU_HIDDEN_SIZE)