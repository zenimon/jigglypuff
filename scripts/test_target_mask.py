
import numpy as np
import librosa

from src.stft import compute_stft
from src.target_mask import (
    create_subband_complex_target,
    complex_target_to_real
)


clean_path = "data/tests/clean_5db.wav"
noisy_path = "data/tests/automatic_5db.wav"


# ==========================================
# LOAD AUDIO
# ==========================================

clean_audio, clean_sr = librosa.load(
    clean_path,
    sr=16000,
    mono=True
)

noisy_audio, noisy_sr = librosa.load(
    noisy_path,
    sr=16000,
    mono=True
)

print("=" * 70)
print("IMPULSE GUARD - SUB-BAND TARGET MASK TEST")
print("=" * 70)

print()
print("Clean audio length:", len(clean_audio))
print("Noisy audio length:", len(noisy_audio))


# ==========================================
# STFT
# ==========================================

clean_stft = compute_stft(
    clean_audio
)

noisy_stft = compute_stft(
    noisy_audio
)

print()
print("Clean STFT shape:", clean_stft.shape)
print("Noisy STFT shape:", noisy_stft.shape)


# ==========================================
# CREATE TARGET MASK
# ==========================================

target_mask = create_subband_complex_target(
    clean_stft,
    noisy_stft
)

print()
print("Complex target mask shape:",
      target_mask.shape)

print("Complex target dtype:",
      target_mask.dtype)


# ==========================================
# CONVERT TO 44 REAL VALUES
# ==========================================

target = complex_target_to_real(
    target_mask
)

print()
print("Real training target shape:",
      target.shape)

print("Target dtype:",
      target.dtype)


# ==========================================
# NUMERICAL CHECKS
# ==========================================

target_magnitude = np.abs(
    target_mask
)

print()
print("Target mask magnitude:")
print("Minimum:",
      np.min(target_magnitude))

print("Maximum:",
      np.max(target_magnitude))

print("Mean:",
      np.mean(target_magnitude))

print()
print("Target real minimum:",
      np.min(target[..., :22]))

print("Target real maximum:",
      np.max(target[..., :22]))

print("Target imaginary minimum:",
      np.min(target[..., 22:]))

print("Target imaginary maximum:",
      np.max(target[..., 22:]))

print()
print("Any NaN:",
      np.any(np.isnan(target)))

print("Any Inf:",
      np.any(np.isinf(target)))


# ==========================================
# FINAL CHECK
# ==========================================

expected_frames = noisy_stft.shape[1]

assert target_mask.shape == (
    expected_frames,
    22
)

assert target.shape == (
    expected_frames,
    44
)

assert not np.any(
    np.isnan(target)
)

assert not np.any(
    np.isinf(target)
)

print()
print("=" * 70)
print("TARGET MASK TEST PASSED")
print("=" * 70)
