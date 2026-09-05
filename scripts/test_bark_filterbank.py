import numpy as np
import librosa

from src.config import (
    SAMPLE_RATE,
    FFT_SIZE,
    NUM_FREQ_BINS,
    NUM_SUBBANDS
)

from src.stft import compute_stft


# ========================================
# BARK SCALE
# ========================================

def hz_to_bark(frequency):
    return (
        13.0 * np.arctan(0.00076 * frequency)
        + 3.5 * np.arctan(
            (frequency / 7500.0) ** 2
        )
    )


def bark_to_hz(bark):
    frequencies = np.linspace(
        0,
        SAMPLE_RATE / 2,
        10000
    )

    barks = hz_to_bark(frequencies)

    return np.interp(
        bark,
        barks,
        frequencies
    )


# ========================================
# CREATE TRIANGULAR BARK FILTERBANK
# ========================================

def create_bark_filterbank():

    max_frequency = SAMPLE_RATE / 2

    min_bark = hz_to_bark(0)
    max_bark = hz_to_bark(max_frequency)

    # 24 points for 22 overlapping filters
    bark_points = np.linspace(
        min_bark,
        max_bark,
        NUM_SUBBANDS + 2
    )

    hz_points = bark_to_hz(
        bark_points
    )

    bin_points = np.floor(
        (FFT_SIZE + 1)
        * hz_points
        / SAMPLE_RATE
    ).astype(int)

    bin_points = np.clip(
        bin_points,
        0,
        NUM_FREQ_BINS - 1
    )

    filters = np.zeros(
        (
            NUM_SUBBANDS,
            NUM_FREQ_BINS
        ),
        dtype=np.float32
    )

    for band in range(NUM_SUBBANDS):

        left = bin_points[band]
        center = bin_points[band + 1]
        right = bin_points[band + 2]

        # Rising slope
        if center > left:

            for k in range(left, center):

                filters[band, k] = (
                    (k - left)
                    / (center - left)
                )

        # Falling slope
        if right > center:

            for k in range(center, right):

                filters[band, k] = (
                    (right - k)
                    / (right - center)
                )

    return filters, hz_points, bin_points


# ========================================
# LOAD AUDIO
# ========================================

audio, sr = librosa.load(
    "data/training/clean/speech1.wav",
    sr=SAMPLE_RATE,
    mono=True
)

complex_stft = compute_stft(audio)

magnitude = np.abs(
    complex_stft
)

power = magnitude ** 2


# ========================================
# CREATE FILTERBANK
# ========================================

filters, hz_points, bin_points = (
    create_bark_filterbank()
)


# ========================================
# APPLY FILTERBANK
# ========================================

# Shape:
# filters     = (22, 257)
# power       = (257, time)
#
# Result:
# band_power  = (22, time)

band_power = (
    filters @ power
)


# ========================================
# LOG COMPRESSION
# ========================================

log_band_power = 10 * np.log10(
    band_power + 1e-10
)


# ========================================
# PRINT INFORMATION
# ========================================

print("=" * 75)
print("IMPULSE GUARD - BARK FILTERBANK")
print("=" * 75)

print("Sample rate:", SAMPLE_RATE, "Hz")
print("FFT size:", FFT_SIZE)
print("Frequency bins:", NUM_FREQ_BINS)
print("Number of Bark bands:", NUM_SUBBANDS)
print("STFT shape:", complex_stft.shape)
print("Filterbank shape:", filters.shape)
print("Band-power shape:", band_power.shape)
print("Log feature shape:", log_band_power.shape)

print()
print("=" * 75)
print("BARK FILTERS")
print("=" * 75)

for i in range(NUM_SUBBANDS):

    left_hz = hz_points[i]
    center_hz = hz_points[i + 1]
    right_hz = hz_points[i + 2]

    nonzero_bins = np.where(
        filters[i] > 0
    )[0]

    if len(nonzero_bins) > 0:

        first_bin = nonzero_bins[0]
        last_bin = nonzero_bins[-1]

    else:

        first_bin = -1
        last_bin = -1

    print(
        f"Band {i + 1:02d}: "
        f"{left_hz:7.1f}-"
        f"{center_hz:7.1f}-"
        f"{right_hz:7.1f} Hz | "
        f"Bins {first_bin:3d}-"
        f"{last_bin:3d}"
    )


# ========================================
# STATISTICS
# ========================================

print()
print("=" * 75)
print("FILTERBANK FEATURE STATISTICS")
print("=" * 75)

print(
    "Minimum:",
    np.min(log_band_power)
)

print(
    "Maximum:",
    np.max(log_band_power)
)

print(
    "Mean:",
    np.mean(log_band_power)
)

print(
    "Std:",
    np.std(log_band_power)
)

print(
    "Any NaN:",
    np.isnan(log_band_power).any()
)

print(
    "Any Inf:",
    np.isinf(log_band_power).any()
)


# ========================================
# AVERAGE ENERGY PER BAND
# ========================================

print()
print("=" * 75)
print("AVERAGE ENERGY PER BARK BAND")
print("=" * 75)

for i in range(NUM_SUBBANDS):

    average_db = np.mean(
        log_band_power[i]
    )

    print(
        f"Band {i + 1:02d}: "
        f"{average_db:8.2f} dB"
    )


print()
print("=" * 75)
print("DONE")
print("=" * 75)