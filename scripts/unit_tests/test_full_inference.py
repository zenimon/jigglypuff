import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import librosa
import soundfile as sf

from src.stft import compute_stft
from src.istft import compute_istft
from src.subbands import get_bark_band_centers, extract_subband_features
from src.feature_normalization import load_statistics
from src.model import create_gru_model
from src.mask import (
    split_gru_mask,
    expand_subband_mask,
    apply_complex_mask
)


# ============================================================
# PATHS
# ============================================================

audio_path = "data/tests/automatic_5db.wav"
output_path = "data/tests/untrained_gru_output.wav"

statistics_path = (
    "data/metadata/feature_normalization.json"
)


# ============================================================
# LOAD AUDIO
# ============================================================

audio, sr = librosa.load(
    audio_path,
    sr=16000,
    mono=True
)

print("=" * 70)
print("IMPULSE GUARD - FULL UNTRAINED INFERENCE TEST")
print("=" * 70)

print()
print("Input audio:", audio_path)
print("Sample rate:", sr)
print("Audio samples:", len(audio))
print("Audio duration:", len(audio) / sr)


# ============================================================
# STFT
# ============================================================

noisy_stft = compute_stft(audio)

print()
print("Noisy STFT shape:", noisy_stft.shape)


# ============================================================
# SUB-BAND FEATURES
# ============================================================

features = extract_subband_features(
    noisy_stft
)

print()
print("Sub-band feature shape:", features.shape)


# ============================================================
# NORMALIZATION
# ============================================================

mean, std = load_statistics(
    statistics_path
)

normalized_features = (
    (features - mean) / std
).astype(np.float32)

print()
print("Normalized feature shape:",
      normalized_features.shape)

print("Normalized feature mean:",
      np.mean(normalized_features))

print("Normalized feature std:",
      np.std(normalized_features))


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

model_input = normalized_features[np.newaxis, :, :]

print()
print("GRU input shape:", model_input.shape)


# ============================================================
# CREATE GRU
# ============================================================

model = create_gru_model()

gru_output = model(
    model_input,
    training=False
).numpy()

print()
print("GRU output shape:", gru_output.shape)


# ============================================================
# 44 OUTPUTS → 22 COMPLEX MASKS
# ============================================================

subband_mask = split_gru_mask(
    gru_output
)

print()
print("Sub-band complex mask shape:",
      subband_mask.shape)


# ============================================================
# 22 → 257 FREQUENCY BINS
# ============================================================

center_frequencies = get_bark_band_centers()

fullband_mask = expand_subband_mask(
    subband_mask,
    center_frequencies
)

print()
print("Expanded mask shape:",
      fullband_mask.shape)


# ============================================================
# ALIGN MASK WITH STFT
# ============================================================
#
# Features are generated from:
#
#     frame 1 - frame 0
#
# Therefore there is one fewer feature frame than
# the original STFT.
#
# STFT:
#     (257, 4842)
#
# Mask:
#     (1, 4841, 257)
#
# We remove the batch dimension and transpose
# to frequency × time:
#
#     (257, 4841)
#
# Then use STFT frames 1 onward.
# ============================================================

fullband_mask = fullband_mask[0]

fullband_mask = fullband_mask.T

aligned_noisy_stft = noisy_stft

print()
print("Aligned noisy STFT shape:",
      aligned_noisy_stft.shape)

print("Aligned mask shape:",
      fullband_mask.shape)


# ============================================================
# APPLY COMPLEX MASK
# ============================================================

enhanced_stft = apply_complex_mask(
    aligned_noisy_stft,
    fullband_mask
)

print()
print("Enhanced STFT shape:",
      enhanced_stft.shape)


# ============================================================
# ISTFT
# ============================================================

enhanced_audio = compute_istft(
    enhanced_stft
)

print()
print("Enhanced audio length:",
      len(enhanced_audio))


# ============================================================
# SAVE
# ============================================================

sf.write(
    output_path,
    enhanced_audio,
    sr,
    subtype="PCM_16"
)

print()
print("Saved:", output_path)


# ============================================================
# MASK STATISTICS
# ============================================================

mask_magnitude = np.abs(fullband_mask)

print()
print("Mask magnitude:")
print("Minimum:", np.min(mask_magnitude))
print("Maximum:", np.max(mask_magnitude))
print("Mean:", np.mean(mask_magnitude))


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("FULL INFERENCE PIPELINE TEST COMPLETE")
print("=" * 70)