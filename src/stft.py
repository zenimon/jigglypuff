import numpy as np
import librosa

from src.config import FRAME_SIZE, HOP_SIZE, FFT_SIZE


def compute_stft(audio):
    complex_stft = librosa.stft(
        audio,
        n_fft=FFT_SIZE,
        hop_length=HOP_SIZE,
        win_length=FRAME_SIZE,
        window="hann",
        center=False,
    )

    return complex_stft