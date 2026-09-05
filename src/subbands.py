
import numpy as np

from src.config import (
    SAMPLE_RATE,
    FFT_SIZE,
    NUM_FREQ_BINS,
    NUM_SUBBANDS
)


def hz_to_bark(frequency):
    """
    Convert frequency in Hz to Bark scale.
    """

    return (
        13.0 * np.arctan(0.00076 * frequency)
        + 3.5 * np.arctan(
            (frequency / 7500.0) ** 2
        )
    )


def bark_to_hz(bark):
    """
    Convert Bark scale back to frequency in Hz.
    """

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


def create_bark_filterbank():
    """
    Create 22 triangular Bark filters.

    Returns:
        filters:
            Shape (22, 257)
    """

    min_bark = hz_to_bark(0)

    max_bark = hz_to_bark(
        SAMPLE_RATE / 2
    )

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

            for k in range(
                left,
                center
            ):

                filters[band, k] = (
                    (k - left)
                    / (center - left)
                )

        # Falling slope
        if right > center:

            for k in range(
                center,
                right
            ):

                filters[band, k] = (
                    (right - k)
                    / (right - center)
                )

    return filters


def get_bark_band_centers():
    """
    Return the center frequency of each
    Bark sub-band.

    Returns:
        numpy array with shape (22,)
    """

    min_bark = hz_to_bark(0)

    max_bark = hz_to_bark(
        SAMPLE_RATE / 2
    )

    bark_points = np.linspace(
        min_bark,
        max_bark,
        NUM_SUBBANDS + 2
    )

    hz_points = bark_to_hz(
        bark_points
    )

    # The center points are everything
    # except the first and last boundary.

    center_frequencies = (
        hz_points[1:-1]
    )

    return center_frequencies.astype(
        np.float32
    )


def extract_subband_features(complex_stft):
    """
    Convert complex STFT into 44-dimensional
    Bark + temporal features.

    Input:
        complex_stft:
            Shape (257, time_frames)

    Output:
        features:
            Shape (time_frames, 44)

        First 22:
            log Bark-band energy

        Next 22:
            temporal energy difference

    The first frame has no previous frame,
    so its temporal difference is defined as zero.
    """

    # -----------------------------------------
    # 1. Magnitude
    # -----------------------------------------

    magnitude = np.abs(
        complex_stft
    )

    # -----------------------------------------
    # 2. Power spectrum
    # -----------------------------------------

    power = magnitude ** 2

    # -----------------------------------------
    # 3. Bark filterbank
    # -----------------------------------------

    filters = create_bark_filterbank()

    # -----------------------------------------
    # 4. Calculate energy in each Bark band
    # -----------------------------------------

    band_power = filters @ power

    # -----------------------------------------
    # 5. Convert to logarithmic scale
    # -----------------------------------------

    log_band_power = (
        10.0
        * np.log10(
            band_power + 1e-10
        )
    )

    # -----------------------------------------
    # 6. Calculate temporal change
    # -----------------------------------------
    #
    # We keep the same number of frames
    # as the STFT.
    #
    # Frame 0:
    #     temporal difference = 0
    #
    # Frame 1:
    #     frame 1 - frame 0
    #
    # Frame 2:
    #     frame 2 - frame 1
    #
    # etc.
    # -----------------------------------------

    temporal_difference = np.zeros_like(
        log_band_power
    )

    temporal_difference[:, 1:] = (
        log_band_power[:, 1:]
        - log_band_power[:, :-1]
    )

    # -----------------------------------------
    # 7. Energy features
    # -----------------------------------------
    #
    # Keep ALL frames.
    #
    # Shape:
    #     (22, time_frames)
    # -----------------------------------------

    energy_features = (
        log_band_power
    )

    # -----------------------------------------
    # 8. Combine
    # -----------------------------------------
    #
    # 22 energy features
    #
    # +
    #
    # 22 temporal features
    #
    # =
    #
    # 44 features
    #
    # Shape before transpose:
    #     (44, time_frames)
    # -----------------------------------------

    features = np.concatenate(
        [
            energy_features,
            temporal_difference
        ],
        axis=0
    )

    # -----------------------------------------
    # 9. Transpose
    # -----------------------------------------
    #
    # Before:
    #     (44, time_frames)
    #
    # After:
    #     (time_frames, 44)
    # -----------------------------------------

    features = features.T

    return features.astype(
        np.float32
    )
