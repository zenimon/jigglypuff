import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import librosa

from src.config import (
    SAMPLE_RATE,
    FFT_SIZE,
    NUM_FREQ_BINS,
    NUM_SUBBANDS
)

from src.stft import compute_stft


# ========================================
# BARK SCALE
# ========================================

def hz_to_bark(frequency):
    return (
        13.0 * np.arctan(0.00076 * frequency)
        + 3.5 * np.arctan(
            (frequency / 7500.0) ** 2
        )
    )


def bark_to_hz(bark):
    frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        10000
    )

    barks = hz_to_bark(frequencies)

    return np.interp(
        bark,
        barks,
        frequencies
    )


# ========================================
# CREATE BARK FILTERBANK
# ========================================

def create_bark_filterbank():

    min_bark = hz_to_bark(0)
    max_bark = hz_to_bark(SAMPLE_RATE / 2)

    bark_points = np.linspace(
        min_bark,
        max_bark,
        NUM_SUBBANDS + 2
    )

    hz_points = bark_to_hz(
        bark_points
    )

    bin_points = np.floor(
        (FFT_SIZE + 1)
        * hz_points
        / SAMPLE_RATE
    ).astype(int)

    bin_points = np.clip(
        bin_points,
        0,
        NUM_FREQ_BINS - 1
    )

    filters = np.zeros(
        (NUM_SUBBANDS, NUM_FREQ_BINS),
        dtype=np.float32
    )

    for band in range(NUM_SUBBANDS):

        left = bin_points[band]
        center = bin_points[band + 1]
        right = bin_points[band + 2]

        # Rising slope
        if center > left:

            for k in range(left, center):

                filters[band, k] = (
                    (k - left)
                    / (center - left)
                )

        # Falling slope
        if right > center:

            for k in range(center, right):

                filters[band, k] = (
                    (right - k)
                    / (right - center)
                )

    return filters


# ========================================
# LOAD AUDIO
# ========================================

audio, sr = librosa.load(
    "data/training/clean/speech1.wav",
    sr=SAMPLE_RATE,
    mono=True
)

complex_stft = compute_stft(audio)

magnitude = np.abs(
    complex_stft
)


# ========================================
# BARK ENERGY
# ========================================

power = magnitude ** 2

filters = create_bark_filterbank()

band_power = (
    filters @ power
)

log_band_power = 10 * np.log10(
    band_power + 1e-10
)


# ========================================
# TEMPORAL ENERGY CHANGE
# ========================================

# Difference between consecutive
# Bark-energy frames.

temporal_difference = (
    log_band_power[:, 1:]
    - log_band_power[:, :-1]
)


# ========================================
# ABSOLUTE TEMPORAL CHANGE
# ========================================

absolute_change = np.abs(
    temporal_difference
)


# ========================================
# FINAL 44 FEATURES
# ========================================

features_44 = np.concatenate(
    [
        log_band_power[:, 1:],
        temporal_difference
    ],
    axis=0
)


# ========================================
# PRINT SHAPES
# ========================================

print("=" * 75)
print("IMPULSE GUARD - TEMPORAL FEATURE TEST")
print("=" * 75)

print("STFT shape:", complex_stft.shape)

print(
    "Bark energy shape:",
    log_band_power.shape
)

print(
    "Temporal difference shape:",
    temporal_difference.shape
)

print(
    "Absolute change shape:",
    absolute_change.shape
)

print(
    "Final 44-feature shape:",
    features_44.shape
)


# ========================================
# TEMPORAL FEATURE STATISTICS
# ========================================

print()
print("=" * 75)
print("TEMPORAL FEATURE STATISTICS")
print("=" * 75)

print(
    "Minimum:",
    temporal_difference.min()
)

print(
    "Maximum:",
    temporal_difference.max()
)

print(
    "Mean:",
    temporal_difference.mean()
)

print(
    "Std:",
    temporal_difference.std()
)

print(
    "Absolute mean:",
    absolute_change.mean()
)

print(
    "Absolute maximum:",
    absolute_change.max()
)

print(
    "Any NaN:",
    np.isnan(temporal_difference).any()
)

print(
    "Any Inf:",
    np.isinf(temporal_difference).any()
)


# ========================================
# PER-BAND STATISTICS
# ========================================

print()
print("=" * 75)
print("PER-BAND TEMPORAL CHANGE")
print("=" * 75)

for i in range(NUM_SUBBANDS):

    band = temporal_difference[i]

    print(
        f"Band {i + 1:02d}: "
        f"Mean={np.mean(band):8.3f} | "
        f"Std={np.std(band):8.3f} | "
        f"Abs mean={np.mean(np.abs(band)):8.3f} | "
        f"Abs max={np.max(np.abs(band)):8.3f}"
    )


# ========================================
# FINAL FEATURE STATISTICS
# ========================================

print()
print("=" * 75)
print("FINAL 44-FEATURE STATISTICS")
print("=" * 75)

print(
    "Minimum:",
    features_44.min()
)

print(
    "Maximum:",
    features_44.max()
)

print(
    "Mean:",
    features_44.mean()
)

print(
    "Std:",
    features_44.std()
)

print(
    "Any NaN:",
    np.isnan(features_44).any()
)

print(
    "Any Inf:",
    np.isinf(features_44).any()
)


print()
print("=" * 75)
print("DONE")
print("=" * 75)