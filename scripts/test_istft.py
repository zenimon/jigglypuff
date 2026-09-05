import numpy as np
import librosa
import soundfile as sf

from src.stft import compute_stft
from src.istft import compute_istft


# -----------------------------------------
# Load original audio
# -----------------------------------------

audio, sr = librosa.load(
    "data/training/clean/speech1.wav",
    sr=16000,
    mono=True
)


# -----------------------------------------
# Compute STFT
# -----------------------------------------

complex_stft = compute_stft(
    audio
)


# -----------------------------------------
# Reconstruct using ISTFT
# -----------------------------------------

reconstructed_audio = compute_istft(
    complex_stft,
    length=len(audio)
)


# -----------------------------------------
# Calculate reconstruction error
# -----------------------------------------

difference = (
    audio - reconstructed_audio
)

mse = np.mean(
    difference ** 2
)

max_error = np.max(
    np.abs(difference)
)

original_rms = np.sqrt(
    np.mean(audio ** 2)
)

reconstructed_rms = np.sqrt(
    np.mean(reconstructed_audio ** 2)
)


# -----------------------------------------
# Save reconstructed audio
# -----------------------------------------

output_path = (
    "data/tests/reconstructed.wav"
)

sf.write(
    output_path,
    reconstructed_audio,
    sr,
    subtype="PCM_16"
)


# -----------------------------------------
# Print results
# -----------------------------------------

print("=" * 70)
print("IMPULSE GUARD - STFT / ISTFT TEST")
print("=" * 70)

print("Original length:",
      len(audio))

print("Reconstructed length:",
      len(reconstructed_audio))

print()

print("STFT shape:",
      complex_stft.shape)

print()

print("Original RMS:",
      original_rms)

print("Reconstructed RMS:",
      reconstructed_rms)

print()

print("Mean squared error:",
      mse)

print("Maximum absolute error:",
      max_error)

print()

print("Saved reconstructed audio:")
print(output_path)

print("=" * 70)
print("DONE")
print("=" * 70)