import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import librosa
import soundfile as sf

from src.stft import compute_stft
from src.istft import compute_istft
from src.mask import (
    create_ideal_complex_mask,
    apply_complex_mask
)


# =============================
# FILE PATHS
# =============================

clean_path = "data/tests/clean_5db.wav"
noisy_path = "data/tests/automatic_5db.wav"

output_path = "data/tests/ideal_mask_5db.wav"


# =============================
# LOAD AUDIO
# =============================

clean_audio, sr = librosa.load(
    clean_path,
    sr=16000,
    mono=True
)

noisy_audio, _ = librosa.load(
    noisy_path,
    sr=16000,
    mono=True
)


# =============================
# CHECK AUDIO LENGTHS
# =============================

if len(clean_audio) != len(noisy_audio):
    raise ValueError(
        f"Audio length mismatch: "
        f"clean {len(clean_audio)}, "
        f"noisy {len(noisy_audio)}"
    )


# =============================
# COMPUTE STFT
# =============================

clean_stft = compute_stft(clean_audio)

noisy_stft = compute_stft(noisy_audio)


# =============================
# CREATE IDEAL COMPLEX MASK
# =============================

ideal_mask = create_ideal_complex_mask(
    clean_stft,
    noisy_stft,
    max_magnitude=2.0
)


# =============================
# APPLY IDEAL MASK
# =============================

enhanced_stft = apply_complex_mask(
    noisy_stft,
    ideal_mask
)


# =============================
# COMPUTE ISTFT
# =============================

enhanced_audio = compute_istft(
    enhanced_stft,
    length=len(clean_audio)
)


# =============================
# SAVE OUTPUT
# =============================

sf.write(
    output_path,
    enhanced_audio,
    sr,
    subtype="PCM_16"
)


# =============================
# CALCULATE ERRORS
# =============================

stft_error = clean_stft - enhanced_stft

stft_mse = np.mean(
    np.abs(stft_error) ** 2
)

audio_error = clean_audio - enhanced_audio

audio_mse = np.mean(
    audio_error ** 2
)

input_rms = np.sqrt(
    np.mean(noisy_audio ** 2)
)

output_rms = np.sqrt(
    np.mean(enhanced_audio ** 2)
)

clean_rms = np.sqrt(
    np.mean(clean_audio ** 2)
)

mask_magnitude = np.abs(ideal_mask)


# =============================
# DISPLAY RESULTS
# =============================

print("=" * 70)
print("IMPULSE GUARD - IDEAL COMPLEX MASK TEST")
print("=" * 70)

print("Clean audio:", clean_path)
print("Noisy audio:", noisy_path)
print("Output audio:", output_path)

print()
print("Clean STFT shape:", clean_stft.shape)
print("Noisy STFT shape:", noisy_stft.shape)
print("Ideal mask shape:", ideal_mask.shape)
print("Enhanced STFT shape:", enhanced_stft.shape)

print()
print("Ideal mask magnitude:")
print("Minimum:", np.min(mask_magnitude))
print("Maximum:", np.max(mask_magnitude))
print("Mean:", np.mean(mask_magnitude))

print()
print("Clean RMS:", clean_rms)
print("Noisy RMS:", input_rms)
print("Enhanced RMS:", output_rms)

print()
print("STFT MSE:", stft_mse)
print("Audio MSE:", audio_mse)

print()
print("Saved:", output_path)

print("=" * 70)
print("DONE")
print("=" * 70)