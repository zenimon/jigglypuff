import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from src.config import SAMPLE_RATE, FFT_SIZE, NUM_FREQ_BINS, NUM_SUBBANDS


# ============================================================
# CURRENT EQUAL-WIDTH MAPPING
# ============================================================

equal_edges = np.linspace(
    0,
    NUM_FREQ_BINS,
    NUM_SUBBANDS + 1,
    dtype=int
)


# ============================================================
# BARK SCALE
# ============================================================

def hz_to_bark(frequency):
    """
    Convert frequency in Hz to the Bark psychoacoustic scale.
    """
    return (
        13.0 * np.arctan(0.00076 * frequency)
        + 3.5 * np.arctan((frequency / 7500.0) ** 2)
    )


def bark_to_hz(bark):
    """
    Convert Bark scale back approximately to Hz.
    """
    frequencies = np.linspace(0, SAMPLE_RATE / 2, 100000)

    bark_values = hz_to_bark(frequencies)

    return np.interp(
        bark,
        bark_values,
        frequencies
    )


# Create 22 equally spaced Bark bands
max_bark = hz_to_bark(SAMPLE_RATE / 2)

bark_edges = np.linspace(
    0,
    max_bark,
    NUM_SUBBANDS + 1
)

bark_hz_edges = np.array([
    bark_to_hz(bark)
    for bark in bark_edges
])


# ============================================================
# CONVERT BARK EDGES TO FFT BIN INDICES
# ============================================================

bark_bin_edges = np.round(
    bark_hz_edges / SAMPLE_RATE * FFT_SIZE
).astype(int)

bark_bin_edges = np.clip(
    bark_bin_edges,
    0,
    NUM_FREQ_BINS - 1
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("IMPULSE GUARD BAND MAPPING")
print("========================================")

print("Sample rate:", SAMPLE_RATE, "Hz")
print("FFT size:", FFT_SIZE)
print("Frequency bins:", NUM_FREQ_BINS)
print("Number of subbands:", NUM_SUBBANDS)
print("Frequency range: 0 -", SAMPLE_RATE / 2, "Hz")


print("\n========================================")
print("CURRENT EQUAL-WIDTH BANDS")
print("========================================")

for i in range(NUM_SUBBANDS):

    start_bin = equal_edges[i]
    end_bin = equal_edges[i + 1] - 1

    start_hz = start_bin * SAMPLE_RATE / FFT_SIZE
    end_hz = end_bin * SAMPLE_RATE / FFT_SIZE

    print(
        f"Band {i + 1:02d}: "
        f"bins {start_bin:3d}-{end_bin:3d} | "
        f"{start_hz:7.1f}-{end_hz:7.1f} Hz"
    )


print("\n========================================")
print("BARK BANDS")
print("========================================")

for i in range(NUM_SUBBANDS):

    start_bin = bark_bin_edges[i]
    end_bin = bark_bin_edges[i + 1]

    start_hz = bark_hz_edges[i]
    end_hz = bark_hz_edges[i + 1]

    print(
        f"Band {i + 1:02d}: "
        f"bins {start_bin:3d}-{end_bin:3d} | "
        f"{start_hz:7.1f}-{end_hz:7.1f} Hz"
    )


print("\n========================================")
print("DONE")
print("========================================")