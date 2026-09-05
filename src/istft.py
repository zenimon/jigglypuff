import numpy as np
import librosa

from src.config import (
    HOP_SIZE,
    FRAME_SIZE
)


def compute_istft(complex_stft, length=None):
    """
    Convert complex STFT back into time-domain audio.

    Input:
        complex_stft:
            Shape (257, time_frames)

    Output:
        audio:
            1D time-domain signal
    """

    audio = librosa.istft(
        complex_stft,
        hop_length=HOP_SIZE,
        win_length=FRAME_SIZE,
        window="hann",
        center=False,
        length=length
    )

    return audio.astype(np.float32)