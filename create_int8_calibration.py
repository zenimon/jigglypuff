import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path

from src.dataset_loader import get_mixture_pairs, load_normalization_stats
from src.stft import compute_stft
from src.subbands import extract_subband_features

SAMPLE_RATE = 16000
NUM_CALIBRATION_FILES = 200
MAX_FRAMES_PER_FILE = 20

FLOAT_MODEL = "models/gru_subband/streaming_gru_subband_float32.tflite"
OUT = Path("models/gru_subband/int8_calibration.npz")


def main():
    pairs = get_mixture_pairs("train")
    mean, std = load_normalization_stats()

    interpreter = tf.lite.Interpreter(
        model_path=FLOAT_MODEL
    )
    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    feature_input = next(
        x for x in inputs if x["shape"][1] == 44
    )
    state_input = next(
        x for x in inputs if x["shape"][1] == 64
    )
    state_output = next(
        x for x in outputs if x["shape"][1] == 64
    )

    indices = np.linspace(
        0,
        len(pairs) - 1,
        NUM_CALIBRATION_FILES,
        dtype=int,
    )

    calibration_features = []
    calibration_states = []

    for count, idx in enumerate(indices, 1):
        _, noisy_path = pairs[idx]

        audio, _ = librosa.load(
            noisy_path,
            sr=SAMPLE_RATE,
            mono=True,
        )

        stft = compute_stft(audio)

        features = extract_subband_features(stft)

        features = (
            (features - mean)
            / (std + 1e-8)
        ).astype(np.float32)

        if not np.isfinite(features).all():
            raise ValueError(
                f"Non-finite features: {noisy_path}"
            )

        frame_indices = np.linspace(
            0,
            len(features) - 1,
            min(MAX_FRAMES_PER_FILE, len(features)),
            dtype=int,
        )

        hidden = np.zeros(
            (1, 64),
            dtype=np.float32,
        )

        for frame_idx in frame_indices:
            x = features[frame_idx:frame_idx + 1]

            calibration_features.append(x.copy())
            calibration_states.append(hidden.copy())

            interpreter.set_tensor(
                feature_input["index"],
                x,
            )
            interpreter.set_tensor(
                state_input["index"],
                hidden,
            )
            interpreter.invoke()

            hidden = interpreter.get_tensor(
                state_output["index"]
            ).copy()

        if count % 25 == 0:
            print(
                f"Processed {count}/{NUM_CALIBRATION_FILES}"
            )

    calibration_features = np.concatenate(
        calibration_features,
        axis=0,
    )

    calibration_states = np.concatenate(
        calibration_states,
        axis=0,
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUT,
        features=calibration_features,
        hidden_states=calibration_states,
    )

    print()
    print("Saved:", OUT)
    print("Features shape:", calibration_features.shape)
    print("States shape:", calibration_states.shape)
    print("Features dtype:", calibration_features.dtype)
    print("States dtype:", calibration_states.dtype)
    print(
        "Feature range:",
        float(calibration_features.min()),
        "to",
        float(calibration_features.max()),
    )
    print(
        "State range:",
        float(calibration_states.min()),
        "to",
        float(calibration_states.max()),
    )


if __name__ == "__main__":
    main()
