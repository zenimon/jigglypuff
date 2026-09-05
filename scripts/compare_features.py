import librosa
import numpy as np

from src.stft import compute_stft
from src.subbands import create_subband_mapping


audio, sr = librosa.load(
    "data/training/clean/speech1.wav",
    sr=16000,
    mono=True
)

complex_stft = compute_stft(audio)

mapping = create_subband_mapping()


# ============================================================
# 1. CURRENT REPRESENTATION
#    22 mean(real) + 22 mean(imag)
# ============================================================

current_features = np.zeros(
    (complex_stft.shape[1], 44),
    dtype=np.float32
)

for subband_index, frequency_bins in enumerate(mapping):

    values = complex_stft[frequency_bins, :]

    real_mean = np.mean(
        values.real,
        axis=0
    )

    imag_mean = np.mean(
        values.imag,
        axis=0
    )

    current_features[:, subband_index] = real_mean
    current_features[:, 22 + subband_index] = imag_mean


# ============================================================
# 2. MAGNITUDE + PHASE
# ============================================================

magnitude_phase = np.zeros(
    (complex_stft.shape[1], 44),
    dtype=np.float32
)

for subband_index, frequency_bins in enumerate(mapping):

    values = complex_stft[frequency_bins, :]

    magnitude = np.abs(values)
    phase = np.angle(values)

    magnitude_mean = np.mean(
        magnitude,
        axis=0
    )

    phase_mean = np.mean(
        phase,
        axis=0
    )

    magnitude_phase[:, subband_index] = magnitude_mean
    magnitude_phase[:, 22 + subband_index] = phase_mean


# ============================================================
# 3. MAGNITUDE + SIN/COS PHASE
# ============================================================

magnitude_sincos = np.zeros(
    (complex_stft.shape[1], 44),
    dtype=np.float32
)

for subband_index, frequency_bins in enumerate(mapping):

    values = complex_stft[frequency_bins, :]

    magnitude = np.abs(values)
    phase = np.angle(values)

    magnitude_mean = np.mean(
        magnitude,
        axis=0
    )

    sin_phase_mean = np.mean(
        np.sin(phase),
        axis=0
    )

    cos_phase_mean = np.mean(
        np.cos(phase),
        axis=0
    )

    magnitude_sincos[:, subband_index] = magnitude_mean

    # Temporary diagnostic:
    # average sin and cos together into one phase feature
    phase_feature = (
        sin_phase_mean + cos_phase_mean
    ) / 2.0

    magnitude_sincos[:, 22 + subband_index] = phase_feature


# ============================================================
# RESULTS
# ============================================================

print("\n========== SHAPES ==========")

print(
    "Current:",
    current_features.shape
)

print(
    "Magnitude + phase:",
    magnitude_phase.shape
)

print(
    "Magnitude + sin/cos:",
    magnitude_sincos.shape
)


print("\n========== CURRENT ==========")

print(
    "Min:",
    current_features.min()
)

print(
    "Max:",
    current_features.max()
)

print(
    "Mean:",
    current_features.mean()
)

print(
    "Std:",
    current_features.std()
)

print(
    "NaN:",
    np.isnan(current_features).any()
)


print("\n========== MAGNITUDE + PHASE ==========")

print(
    "Min:",
    magnitude_phase.min()
)

print(
    "Max:",
    magnitude_phase.max()
)

print(
    "Mean:",
    magnitude_phase.mean()
)

print(
    "Std:",
    magnitude_phase.std()
)

print(
    "NaN:",
    np.isnan(magnitude_phase).any()
)


print("\n========== MAGNITUDE + SIN/COS ==========")

print(
    "Min:",
    magnitude_sincos.min()
)

print(
    "Max:",
    magnitude_sincos.max()
)

print(
    "Mean:",
    magnitude_sincos.mean()
)

print(
    "Std:",
    magnitude_sincos.std()
)

print(
    "NaN:",
    np.isnan(magnitude_sincos).any()
)