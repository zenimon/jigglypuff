import os
import glob
import numpy as np
import soundfile as sf
import tensorflow as tf

from src.config import SAMPLE_RATE, FFT_SIZE, NUM_FREQ_BINS
from src.stft import compute_stft
from src.istft import compute_istft
from src.subbands import (
    extract_subband_features,
    get_bark_band_centers,
)
from src.feature_normalization import load_statistics
from src.mask import (
    split_gru_mask,
    expand_subband_mask,
    apply_complex_mask,
)


MODEL_PATH = "models/gru_subband/best_gru_subband.keras"

TEST_NOISY_DIR = "data/mixtures/test/noisy"
OUTPUT_DIR = "data/enhanced/test"

NORMALIZATION_PATH = "data/metadata/feature_normalization.json"


def load_model():
    print(f"Loading model: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        f"Model loaded successfully "
        f"({model.count_params():,} parameters)"
    )

    return model


def normalize_features(features, mean, std):
    return (
        (features - mean)
        / (std + 1e-8)
    ).astype(np.float32)


def infer_single(
    model,
    noisy_audio,
    feature_mean,
    feature_std
):
    """
    Run complete Impulse Guard inference.

    noisy audio
        -> STFT
        -> 44 Bark features
        -> normalization
        -> GRU
        -> 22 complex mask values
        -> 257-bin mask
        -> complex masking
        -> ISTFT
    """

    original_length = len(noisy_audio)

    # -----------------------------------------
    # 1. STFT
    # -----------------------------------------

    noisy_stft = compute_stft(
        noisy_audio
    )

    if noisy_stft.shape[0] != NUM_FREQ_BINS:
        raise ValueError(
            f"Expected {NUM_FREQ_BINS} frequency bins, "
            f"got {noisy_stft.shape[0]}"
        )

    # -----------------------------------------
    # 2. Extract 44-dimensional features
    # -----------------------------------------

    features = extract_subband_features(
        noisy_stft
    )

    # features:
    # (time_frames, 44)

    if features.shape[-1] != 44:
        raise ValueError(
            f"Expected 44 features, "
            f"got {features.shape[-1]}"
        )

    # -----------------------------------------
    # 3. Normalize using TRAINING statistics
    # -----------------------------------------

    features = normalize_features(
        features,
        feature_mean,
        feature_std
    )

    # -----------------------------------------
    # 4. Add batch dimension
    # -----------------------------------------

    model_input = features[
        np.newaxis,
        ...,
    ]

    # -----------------------------------------
    # 5. GRU inference
    # -----------------------------------------

    gru_output = model(
        model_input,
        training=False
    ).numpy()

    # Shape:
    # (1, time_frames, 44)

    # -----------------------------------------
    # 6. Remove batch dimension
    # -----------------------------------------

    gru_output = gru_output[0]

    # Shape:
    # (time_frames, 44)

    # -----------------------------------------
    # 7. Split 44 outputs
    # into 22 real + 22 imaginary
    # -----------------------------------------

    subband_mask = split_gru_mask(
        gru_output
    )

    # Shape:
    # (time_frames, 22)

    # -----------------------------------------
    # 8. Expand 22-band mask
    # to 257 FFT bins
    # -----------------------------------------

    center_frequencies = (
        get_bark_band_centers()
    )

    full_mask = expand_subband_mask(
        subband_mask,
        center_frequencies,
        num_bins=NUM_FREQ_BINS,
        sample_rate=SAMPLE_RATE,
        fft_size=FFT_SIZE,
    )

    # Shape:
    # (time_frames, 257)

    # -----------------------------------------
    # 9. Transpose mask
    # to STFT layout
    # -----------------------------------------

    full_mask = full_mask.T

    # Shape:
    # (257, time_frames)

    # -----------------------------------------
    # 10. Apply complex mask
    # -----------------------------------------

    enhanced_stft = apply_complex_mask(
        noisy_stft,
        full_mask
    )

    # -----------------------------------------
    # 11. ISTFT
    # -----------------------------------------

    enhanced_audio = compute_istft(
        enhanced_stft,
        length=original_length
    )

    # -----------------------------------------
    # 12. Safety checks
    # -----------------------------------------

    enhanced_audio = np.nan_to_num(
        enhanced_audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    # Prevent accidental clipping.
    peak = np.max(
        np.abs(enhanced_audio)
    )

    if peak > 0.98:
        enhanced_audio *= (
            0.98 / peak
        )

    return enhanced_audio


def main():

    print(
        "\n========================================"
    )
    print(
        " Impulse Guard - Test Set Inference"
    )
    print(
        "========================================\n"
    )

    # -----------------------------------------
    # Load model
    # -----------------------------------------

    model = load_model()

    # -----------------------------------------
    # Load training normalization statistics
    # -----------------------------------------

    print(
        f"Loading normalization stats: "
        f"{NORMALIZATION_PATH}"
    )

    feature_mean, feature_std = (
        load_statistics(
            NORMALIZATION_PATH
        )
    )

    feature_mean = np.asarray(
        feature_mean,
        dtype=np.float32
    )

    feature_std = np.asarray(
        feature_std,
        dtype=np.float32
    )

    if feature_mean.shape != (44,):
        raise ValueError(
            f"Expected mean shape (44,), "
            f"got {feature_mean.shape}"
        )

    if feature_std.shape != (44,):
        raise ValueError(
            f"Expected std shape (44,), "
            f"got {feature_std.shape}"
        )

    print("Normalization statistics loaded.")

    # -----------------------------------------
    # Find test noisy files
    # -----------------------------------------

    noisy_files = sorted(
        glob.glob(
            os.path.join(
                TEST_NOISY_DIR,
                "*.wav"
            )
        )
    )

    if not noisy_files:
        raise RuntimeError(
            f"No WAV files found in "
            f"{TEST_NOISY_DIR}"
        )

    print(
        f"Found {len(noisy_files):,} test noisy files."
    )

    # -----------------------------------------
    # Create output directory
    # -----------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # -----------------------------------------
    # Process test set
    # -----------------------------------------

    success = 0
    failures = 0

    for index, noisy_path in enumerate(
        noisy_files,
        start=1
    ):

        filename = os.path.basename(
            noisy_path
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        try:

            noisy_audio, sr = sf.read(
                noisy_path,
                dtype="float32"
            )

            if noisy_audio.ndim > 1:
                noisy_audio = np.mean(
                    noisy_audio,
                    axis=1
                )

            if sr != SAMPLE_RATE:
                raise ValueError(
                    f"Expected {SAMPLE_RATE} Hz, "
                    f"got {sr} Hz"
                )

            enhanced_audio = infer_single(
                model,
                noisy_audio,
                feature_mean,
                feature_std
            )

            sf.write(
                output_path,
                enhanced_audio,
                SAMPLE_RATE,
                subtype="PCM_16"
            )

            success += 1

            if (
                index <= 5
                or index % 100 == 0
                or index == len(noisy_files)
            ):
                print(
                    f"[{index:,}/{len(noisy_files):,}] "
                    f"{filename}"
                )

        except Exception as e:

            failures += 1

            print(
                f"[FAILED] {filename}: {e}"
            )

    # -----------------------------------------
    # Summary
    # -----------------------------------------

    print(
        "\n========================================"
    )
    print(
        " Inference Complete"
    )
    print(
        "========================================"
    )

    print(
        f"Successful: {success:,}"
    )

    print(
        f"Failed:     {failures:,}"
    )

    print(
        f"Output:     {OUTPUT_DIR}"
    )

    print(
        "========================================\n"
    )

    if failures > 0:
        print(
            "WARNING: Some files failed. "
            "Check the [FAILED] messages above."
        )


if __name__ == "__main__":
    main()
