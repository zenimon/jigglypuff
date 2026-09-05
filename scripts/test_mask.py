import numpy as np
import librosa
import soundfile as sf

from src.stft import compute_stft
from src.istft import compute_istft
from src.mask import (
    create_frequency_mask,
    apply_complex_mask
)


# =============================
# FILE PATHS
# =============================

audio_path = "data/tests/automatic_5db.wav"
output_path = "data/tests/masked_frequency_5db.wav"


# =============================
# LOAD AUDIO
# =============================

audio, sr = librosa.load(
    audio_path,
    sr=16000,
    mono=True
)


# =============================
# COMPUTE STFT
# =============================

complex_stft = compute_stft(audio)


# =============================
# CREATE FREQUENCY MASK
# =============================

complex_mask = create_frequency_mask(
    complex_stft
)


# =============================
# APPLY COMPLEX MASK
# =============================

enhanced_stft = apply_complex_mask(
    complex_stft,
    complex_mask
)


# =============================
# COMPUTE ISTFT
# =============================

enhanced_audio = compute_istft(
    enhanced_stft,
    length=len(audio)
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
# CALCULATE RMS
# =============================

input_rms = np.sqrt(
    np.mean(audio ** 2)
)

output_rms = np.sqrt(
    np.mean(enhanced_audio ** 2)
)


# =============================
# DISPLAY RESULTS
# =============================

print("=" * 70)
print("IMPULSE GUARD - FREQUENCY-DEPENDENT COMPLEX MASK TEST")
print("=" * 70)

print("Input audio:", audio_path)
print("Output audio:", output_path)

print()
print("STFT shape:", complex_stft.shape)
print("Mask shape:", complex_mask.shape)
print("Enhanced STFT shape:", enhanced_stft.shape)

print()
print("Frequency mask:")
print("0–300 Hz:       0.2 + 0j")
print("300–3000 Hz:    0.8 + 0j")
print("3000–8000 Hz:   0.3 + 0j")

print()
print("Input RMS:", input_rms)
print("Output RMS:", output_rms)

print()
print("Saved:", output_path)

print("=" * 70)
print("DONE")
print("=" * 70)