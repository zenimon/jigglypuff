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

    max_frequency = SAMPLE_RATE / 2

    min_bark = hz_to_bark(0)
    max_bark = hz_to_bark(max_frequency)

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
        (
            NUM_SUBBANDS,
            NUM_FREQ_BINS
        ),
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

phase = np.angle(
    complex_stft
)


# ========================================
# BARK FILTERBANK
# ========================================

filters = create_bark_filterbank()


# ========================================
# 22 LOG MAGNITUDE FEATURES
# ========================================

power = magnitude ** 2

band_power = (
    filters @ power
)

log_band_power = 10 * np.log10(
    band_power + 1e-10
)


# ========================================
# PHASE DIFFERENCE
# ========================================

# Phase difference between consecutive
# STFT frames.

phase_difference = np.angle(
    np.exp(
        1j * (
            phase[:, 1:]
            - phase[:, :-1]
        )
    )
)


# ========================================
# WEIGHT PHASE DIFFERENCE BY MAGNITUDE
# ========================================

magnitude_current = magnitude[:, 1:]

weighted_phase = (
    magnitude_current
    * np.abs(phase_difference)
)


# ========================================
# PROJECT PHASE INFORMATION INTO
# THE 22 BARK BANDS
# ========================================

band_phase_change = (
    filters @ weighted_phase
)


# Normalize by band magnitude so that
# high-energy bins do not dominate.

band_magnitude = (
    filters @ magnitude_current
)

band_phase_change = (
    band_phase_change
    / (band_magnitude + 1e-8)
)


# ========================================
# FINAL 44-FEATURE CANDIDATE
# ========================================

# The phase feature has one fewer frame
# because it compares frame t with t-1.

features_44 = np.concatenate(
    [
        log_band_power[:, 1:],
        band_phase_change
    ],
    axis=0
)


# ========================================
# PRINT SHAPES
# ========================================

print("=" * 75)
print("IMPULSE GUARD - PHASE FEATURE TEST")
print("=" * 75)

print("STFT shape:", complex_stft.shape)

print(
    "Log Bark energy shape:",
    log_band_power.shape
)

print(
    "Raw phase shape:",
    phase.shape
)

print(
    "Phase difference shape:",
    phase_difference.shape
)

print(
    "Band phase feature shape:",
    band_phase_change.shape
)

print(
    "Final 44-feature shape:",
    features_44.shape
)


# ========================================
# STATISTICS
# ========================================

print()
print("=" * 75)
print("PHASE FEATURE STATISTICS")
print("=" * 75)

print(
    "Phase difference minimum:",
    phase_difference.min()
)

print(
    "Phase difference maximum:",
    phase_difference.max()
)

print(
    "Phase difference mean:",
    phase_difference.mean()
)

print(
    "Phase difference std:",
    phase_difference.std()
)

print()
print(
    "Band phase feature minimum:",
    band_phase_change.min()
)

print(
    "Band phase feature maximum:",
    band_phase_change.max()
)

print(
    "Band phase feature mean:",
    band_phase_change.mean()
)

print(
    "Band phase feature std:",
    band_phase_change.std()
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


# ========================================
# CHECK EACH FEATURE RANGE
# ========================================

print()
print("=" * 75)
print("PER-FEATURE CHECK")
print("=" * 75)

for i in range(NUM_SUBBANDS):

    energy_feature = features_44[i]

    phase_feature = features_44[
        NUM_SUBBANDS + i
    ]

    print(
        f"Band {i + 1:02d}: "
        f"Energy mean={np.mean(energy_feature):8.2f}, "
        f"std={np.std(energy_feature):7.2f} | "
        f"Phase mean={np.mean(phase_feature):8.4f}, "
        f"std={np.std(phase_feature):7.4f}"
    )


print()
print("=" * 75)
print("DONE")
print("=" * 75)