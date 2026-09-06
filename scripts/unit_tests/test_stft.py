import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import librosa

from src.stft import compute_stft


audio, sr = librosa.load(
    "data/training/clean/speech1.wav",
    sr=16000,
    mono=True
)

complex_stft = compute_stft(audio)

print("Sample rate:", sr)
print("Audio samples:", len(audio))
print("Audio duration:", len(audio) / sr, "seconds")
print("Complex STFT shape:", complex_stft.shape)
print("Complex STFT dtype:", complex_stft.dtype)