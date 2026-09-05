import numpy as np
import librosa

from src.config import (
    SAMPLE_RATE,
    FFT_SIZE,
    NUM_FREQ_BINS,
    NUM_SUBBANDS
)

from src.stft import compute_stft


def hz_to_bark(frequency):
    return (
        13.0 * np.arctan(0.00076 * frequency)
        + 3.5 * np.arctan((frequency / 7500.0) ** 2)
    )


def bark_to_hz(bark):
    frequencies = np.linspace(0, 8000, 10000)
    barks = hz_to_bark(frequencies)

    return np.interp(
        bark,
        barks,
        frequencies
    )


# ========================================
# LOAD AUDIO
# ========================================

audio, sr = librosa.load(
    "data/training/clean/speech1.wav",
    sr=SAMPLE_RATE,
    mono=True
)

complex_stft = compute_stft(audio)

# Power of each complex STFT bin
power = np.abs(complex_stft) ** 2


# ========================================
# CREATE BARK BANDS
# ========================================

max_frequency = SAMPLE_RATE / 2

min_bark = hz_to_bark(0)
max_bark = hz_to_bark(max_frequency)

bark_edges = np.linspace(
    min_bark,
    max_bark,
    NUM_SUBBANDS + 1
)

hz_edges = bark_to_hz(bark_edges)

# Convert frequency boundaries to FFT-bin boundaries
bin_edges = np.round(
    hz_edges / (SAMPLE_RATE / FFT_SIZE)
).astype(int)

# Keep bins within valid range
bin_edges = np.clip(
    bin_edges,
    0,
    NUM_FREQ_BINS
)


# ========================================
# TOTAL SPECTRAL POWER
# ========================================

total_power = np.sum(power)


# ========================================
# PRINT RESULTS
# ========================================

print("=" * 70)
print("IMPULSE GUARD BARK BAND ENERGY")
print("=" * 70)

print("Sample rate:", SAMPLE_RATE, "Hz")
print("FFT size:", FFT_SIZE)
print("Frequency bins:", NUM_FREQ_BINS)
print("Number of subbands:", NUM_SUBBANDS)
print("STFT shape:", complex_stft.shape)

print()
print("Total spectral power:", total_power)

print()
print("=" * 70)

for i in range(NUM_SUBBANDS):

    # Band uses [start_bin, end_bin)
    start_bin = bin_edges[i]
    end_bin = bin_edges[i + 1]

    # Safety check
    if end_bin <= start_bin:
        continue

    band_power = power[
        start_bin:end_bin,
        :
    ]

    # Total energy contained in this band
    band_power_total = np.sum(
        band_power
    )

    # Average power per STFT value
    mean_power = np.mean(
        band_power
    )

    # Convert average power to dB
    power_db = 10 * np.log10(
        mean_power + 1e-12
    )

    # Percentage of total spectral energy
    percentage = (
        band_power_total
        / total_power
    ) * 100

    # Convert FFT bins back to frequency
    start_hz = (
        start_bin
        * SAMPLE_RATE
        / FFT_SIZE
    )

    end_hz = (
        end_bin
        * SAMPLE_RATE
        / FFT_SIZE
    )

    print(
        f"Band {i + 1:02d}: "
        f"{start_hz:7.1f}-{end_hz:7.1f} Hz | "
        f"Bins {start_bin:3d}-{end_bin - 1:3d} | "
        f"Power: {power_db:7.2f} dB | "
        f"Share: {percentage:6.2f}%"
    )


print("=" * 70)
print("DONE")
print("=" * 70)