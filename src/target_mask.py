
import numpy as np

from src.subbands import create_bark_filterbank


def create_subband_complex_target(
    clean_stft,
    noisy_stft
):
    """
    Create a 22-band complex target mask.

    The clean and noisy STFTs are first projected
    into the same Bark sub-bands.

    For each sub-band:

        Clean_band = W @ Clean_STFT
        Noisy_band = W @ Noisy_STFT

    Then:

        Target_mask =
            Clean_band / Noisy_band

    Input:
        clean_stft:
            Shape (257, time_frames)

        noisy_stft:
            Shape (257, time_frames)

    Output:
        target_mask:
            Shape (time_frames, 22)

        dtype:
            complex64
    """

    # -----------------------------------------
    # 1. Check STFT shapes
    # -----------------------------------------

    if clean_stft.shape != noisy_stft.shape:
        raise ValueError(
            f"Clean and noisy STFT shapes must match. "
            f"Got clean={clean_stft.shape}, "
            f"noisy={noisy_stft.shape}"
        )

    # -----------------------------------------
    # 2. Create Bark filterbank
    # -----------------------------------------

    filters = create_bark_filterbank()

    # Shape:
    #     (22, 257)

    # -----------------------------------------
    # 3. Project clean STFT into sub-bands
    # -----------------------------------------

    clean_subbands = (
        filters @ clean_stft
    )

    # Shape:
    #     (22, time_frames)

    # -----------------------------------------
    # 4. Project noisy STFT into sub-bands
    # -----------------------------------------

    noisy_subbands = (
        filters @ noisy_stft
    )

    # Shape:
    #     (22, time_frames)

    # -----------------------------------------
    # 5. Calculate complex ratio mask
    # -----------------------------------------

    epsilon = 1e-8

    target_mask = (
        clean_subbands
        / (
            noisy_subbands
            + epsilon
        )
    )

    # -----------------------------------------
    # 6. Limit extreme values
    # -----------------------------------------
    #
    # Very small noisy-band energy can produce
    # extremely large ratios.
    #
    # We limit the magnitude to 2.0.
    #
    # This keeps training targets numerically
    # stable.
    # -----------------------------------------

    max_magnitude = 2.0

    magnitude = np.abs(
        target_mask
    )

    scale = np.minimum(
        1.0,
        max_magnitude
        / (magnitude + epsilon)
    )

    target_mask = (
        target_mask
        * scale
    )

    # -----------------------------------------
    # 7. Transpose
    # -----------------------------------------
    #
    # Before:
    #
    #     (22, time_frames)
    #
    # After:
    #
    #     (time_frames, 22)
    # -----------------------------------------

    target_mask = target_mask.T

    return target_mask.astype(
        np.complex64
    )


def complex_target_to_real(
    target_mask
):
    """
    Convert a complex 22-band target mask
    into 44 real-valued training targets.

    Input:
        target_mask:
            Shape (..., 22)

    Output:
        target:
            Shape (..., 44)

    Layout:

        [real_0 ... real_21,
         imag_0 ... imag_21]
    """

    if target_mask.shape[-1] != 22:
        raise ValueError(
            f"Expected 22 complex sub-band masks, "
            f"got {target_mask.shape[-1]}"
        )

    real_part = np.real(
        target_mask
    )

    imag_part = np.imag(
        target_mask
    )

    target = np.concatenate(
        [
            real_part,
            imag_part
        ],
        axis=-1
    )

    return target.astype(
        np.float32
    )