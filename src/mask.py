import numpy as np

from src.config import SAMPLE_RATE, FFT_SIZE


def create_constant_complex_mask(
    complex_stft,
    real_value=1.0,
    imag_value=0.0
):
    """
    Create a constant complex mask.
    """

    mask = np.full(
        complex_stft.shape,
        real_value + 1j * imag_value,
        dtype=np.complex64
    )

    return mask


def create_frequency_mask(complex_stft):
    """
    Create a frequency-dependent complex mask.

    0–300 Hz       -> 0.2 + 0j
    300–3000 Hz    -> 0.8 + 0j
    3000–8000 Hz   -> 0.3 + 0j
    """

    num_bins, num_frames = complex_stft.shape

    frequencies = np.fft.rfftfreq(
        FFT_SIZE,
        d=1.0 / SAMPLE_RATE
    )

    mask_values = np.zeros(
        num_bins,
        dtype=np.complex64
    )

    for k, frequency in enumerate(frequencies):

        if frequency < 300:
            real_value = 0.2

        elif frequency < 3000:
            real_value = 0.8

        else:
            real_value = 0.3

        mask_values[k] = real_value + 0j

    mask = np.tile(
        mask_values[:, np.newaxis],
        (1, num_frames)
    )

    return mask


def create_ideal_complex_mask(
    clean_stft,
    noisy_stft,
    max_magnitude=2.0
):
    """
    Create an ideal complex ratio mask.

    M = Clean STFT / Noisy STFT

    Applying:
        Enhanced STFT = Noisy STFT × M
    """

    if clean_stft.shape != noisy_stft.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"clean {clean_stft.shape}, "
            f"noisy {noisy_stft.shape}"
        )

    epsilon = 1e-8

    ideal_mask = (
        clean_stft
        / (noisy_stft + epsilon)
    )

    # Limit extreme mask magnitudes.
    mask_magnitude = np.abs(ideal_mask)

    scale = np.minimum(
        1.0,
        max_magnitude / (mask_magnitude + epsilon)
    )

    ideal_mask = ideal_mask * scale

    return ideal_mask.astype(np.complex64)


def apply_complex_mask(complex_stft, complex_mask):
    """
    Apply a complex mask to a complex STFT.

    Enhanced STFT = Noisy STFT × Complex Mask
    """

    if complex_stft.shape != complex_mask.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"STFT {complex_stft.shape}, "
            f"mask {complex_mask.shape}"
        )

    enhanced_stft = complex_stft * complex_mask





    return enhanced_stft.astype(np.complex64)

def split_gru_mask(gru_output):
    """
    Convert 44 real-valued GRU outputs into
    22 complex sub-band mask values.

    Input:
        (..., 44)

    Output:
        (..., 22) complex64
    """

    if gru_output.shape[-1] != 44:
        raise ValueError(
            f"Expected 44 GRU outputs, "
            f"got {gru_output.shape[-1]}"
        )

    real_mask = gru_output[..., :22]
    imag_mask = gru_output[..., 22:44]

    complex_mask = (
        real_mask
        + 1j * imag_mask
    )

    return complex_mask.astype(np.complex64)

def expand_subband_mask(
    subband_mask,
    center_frequencies,
    num_bins=257,
    sample_rate=16000,
    fft_size=512
):
    """
    Expand a 22-band complex mask to a
    full 257-bin complex frequency mask.

    Input:
        subband_mask:
            (..., 22) complex values

        center_frequencies:
            (22,) Bark-band center frequencies

    Output:
        (..., 257) complex64
    """

    subband_mask = np.asarray(
        subband_mask,
        dtype=np.complex64
    )

    center_frequencies = np.asarray(
        center_frequencies,
        dtype=np.float32
    )

    if subband_mask.shape[-1] != len(center_frequencies):
        raise ValueError(
            "Number of mask bands must match "
            "number of center frequencies."
        )

    frequencies = np.fft.rfftfreq(
        fft_size,
        d=1.0 / sample_rate
    )

    real_part = np.real(subband_mask)
    imag_part = np.imag(subband_mask)

    expanded_real = np.empty(
        subband_mask.shape[:-1] + (num_bins,),
        dtype=np.float32
    )

    expanded_imag = np.empty(
        subband_mask.shape[:-1] + (num_bins,),
        dtype=np.float32
    )

    flat_real = real_part.reshape(-1, real_part.shape[-1])
    flat_imag = imag_part.reshape(-1, imag_part.shape[-1])

    flat_expanded_real = expanded_real.reshape(
        -1,
        num_bins
    )

    flat_expanded_imag = expanded_imag.reshape(
        -1,
        num_bins
    )

    for i in range(flat_real.shape[0]):

        flat_expanded_real[i] = np.interp(
            frequencies,
            center_frequencies,
            flat_real[i]
        )

        flat_expanded_imag[i] = np.interp(
            frequencies,
            center_frequencies,
            flat_imag[i]
        )

    expanded_mask = (
        flat_expanded_real
        + 1j * flat_expanded_imag
    )

    expanded_mask = expanded_mask.reshape(
        subband_mask.shape[:-1] + (num_bins,)
    )

    return expanded_mask.astype(np.complex64)