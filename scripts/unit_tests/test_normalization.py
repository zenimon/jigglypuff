import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import librosa
import numpy as np
import os

from src.stft import compute_stft
from src.subbands import extract_subband_features

from src.feature_normalization import (
    calculate_statistics,
    normalize_features,
    save_statistics,
    load_statistics
)


# -----------------------------------------
# Load audio
# -----------------------------------------

audio, sr = librosa.load(
    "data/training/clean/speech1.wav",
    sr=16000,
    mono=True
)


# -----------------------------------------
# STFT
# -----------------------------------------

complex_stft = compute_stft(audio)


# -----------------------------------------
# Extract 44 features
# -----------------------------------------

features = extract_subband_features(
    complex_stft
)


# -----------------------------------------
# Calculate statistics
# -----------------------------------------

mean, std = calculate_statistics(
    features
)


# -----------------------------------------
# Save statistics
# -----------------------------------------

statistics_path = (
    "data/metadata/feature_normalization.json"
)

save_statistics(
    mean,
    std,
    statistics_path
)


# -----------------------------------------
# Load statistics again
# -----------------------------------------

loaded_mean, loaded_std = load_statistics(
    statistics_path
)


# -----------------------------------------
# Normalize
# -----------------------------------------

normalized_features = normalize_features(
    features,
    loaded_mean,
    loaded_std
)


# -----------------------------------------
# Print results
# -----------------------------------------

print("=" * 70)
print("IMPULSE GUARD - FEATURE NORMALIZATION TEST")
print("=" * 70)

print("Original shape:",
      features.shape)

print("Normalized shape:",
      normalized_features.shape)

print()

print("Statistics saved to:")
print(statistics_path)

print()

print("=" * 70)
print("NORMALIZED FEATURES")
print("=" * 70)

print("Minimum:",
      normalized_features.min())

print("Maximum:",
      normalized_features.max())

print("Mean:",
      normalized_features.mean())

print("Std:",
      normalized_features.std())

print()

print("Any NaN:",
      np.isnan(normalized_features).any())

print("Any Inf:",
      np.isinf(normalized_features).any())

print()

print("Loaded mean shape:",
      loaded_mean.shape)

print("Loaded std shape:",
      loaded_std.shape)

print()

print("=" * 70)
print("CHECK")
print("=" * 70)

print(
    "Mean statistics match:",
    np.allclose(mean, loaded_mean)
)

print(
    "Std statistics match:",
    np.allclose(std, loaded_std)
)

print("=" * 70)
print("DONE")
print("=" * 70)