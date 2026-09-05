import librosa
import numpy as np

from src.stft import compute_stft
from src.subbands import extract_subband_features


audio, sr = librosa.load(
    "data/training/clean/speech1.wav",
    sr=16000,
    mono=True
)

complex_stft = compute_stft(audio)

features = extract_subband_features(complex_stft)

print("STFT shape:", complex_stft.shape)
print("Subband feature shape:", features.shape)
print("Feature dtype:", features.dtype)
print("Feature minimum:", features.min())
print("Feature maximum:", features.max())
print("Any NaN:", np.isnan(features).any())