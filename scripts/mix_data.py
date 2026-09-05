import numpy as np
import librosa
import soundfile as sf
import json
import os


# ======================================================
# 1. Save metadata to JSONL
# ======================================================

def save_metadata(metadata_path, metadata):

    os.makedirs(
        os.path.dirname(metadata_path),
        exist_ok=True
    )

    with open(metadata_path, "a") as f:
        f.write(
            json.dumps(metadata) + "\n"
        )


# ======================================================
# 2. Generate the next unique sample ID
# ======================================================

def get_next_sample_id(metadata_path):

    # If metadata file doesn't exist yet
    if not os.path.exists(metadata_path):
        return "IG_000001"

    # Read existing metadata
    with open(metadata_path, "r") as f:
        lines = f.readlines()

    # If file exists but is empty
    if not lines:
        return "IG_000001"

    # Read the last sample
    last_metadata = json.loads(lines[-1])

    last_id = last_metadata["sample_id"]

    # Extract number from IG_000001
    number = int(
        last_id.split("_")[1]
    )

    # Generate next ID
    return f"IG_{number + 1:06d}"


# ======================================================
# 3. Mix clean speech with noise
# ======================================================

def mix_audio(
    clean_path,
    noise_path,
    clean_output_path,
    noisy_output_path,
    target_snr
):

    # --------------------------------------------------
    # 1. Load clean speech and noise
    # --------------------------------------------------

    clean, _ = librosa.load(
        clean_path,
        sr=16000,
        mono=True
    )

    noise, _ = librosa.load(
        noise_path,
        sr=16000,
        mono=True
    )

    # --------------------------------------------------
    # 2. Make noise long enough
    # --------------------------------------------------

    repeat_count = int(
        np.ceil(
            len(clean) / len(noise)
        )
    )

    noise = np.tile(
        noise,
        repeat_count
    )

    # Match noise length to speech
    noise = noise[:len(clean)]

    # --------------------------------------------------
    # 3. Calculate original RMS values
    # --------------------------------------------------

    clean_rms = np.sqrt(
        np.mean(clean ** 2)
    )

    noise_rms = np.sqrt(
        np.mean(noise ** 2)
    )

    # --------------------------------------------------
    # 4. Calculate required noise RMS
    # --------------------------------------------------

    target_noise_rms = clean_rms / (
        10 ** (target_snr / 20)
    )

    # --------------------------------------------------
    # 5. Scale the noise
    # --------------------------------------------------

    scale = (
        target_noise_rms /
        noise_rms
    )

    noise_scaled = noise * scale

    # --------------------------------------------------
    # 6. Create noisy speech
    # --------------------------------------------------

    noisy = clean + noise_scaled

    # --------------------------------------------------
    # 7. Check for clipping
    # --------------------------------------------------

    peak = np.max(
        np.abs(noisy)
    )

    print(
        "Peak before clipping protection:",
        peak
    )

    if peak > 1.0:

        gain = 1.0 / peak

        clean = clean * gain
        noise_scaled = noise_scaled * gain
        noisy = noisy * gain

        print("Clipping detected.")
        print("Applied gain:", gain)

    else:

        print("No clipping.")

    # --------------------------------------------------
    # 8. Verify final SNR
    # --------------------------------------------------

    final_clean_rms = np.sqrt(
        np.mean(clean ** 2)
    )

    final_noise_rms = np.sqrt(
        np.mean(noise_scaled ** 2)
    )

    actual_snr = 20 * np.log10(
        final_clean_rms /
        final_noise_rms
    )

    # --------------------------------------------------
    # 9. Print results
    # --------------------------------------------------

    print()

    print(
        "========== MIXING RESULT =========="
    )

    print(
        "Target SNR:",
        target_snr,
        "dB"
    )

    print(
        "Actual SNR:",
        actual_snr,
        "dB"
    )

    print(
        "Clean RMS:",
        final_clean_rms
    )

    print(
        "Noise RMS:",
        final_noise_rms
    )

    print(
        "Final peak:",
        np.max(np.abs(noisy))
    )

    print(
        "===================================="
    )

    # --------------------------------------------------
    # 10. Save clean and noisy audio
    # --------------------------------------------------

    sf.write(
        clean_output_path,
        clean,
        16000,
        subtype="PCM_16"
    )

    sf.write(
        noisy_output_path,
        noisy,
        16000,
        subtype="PCM_16"
    )

    print(
        "Saved clean:",
        clean_output_path
    )

    print(
        "Saved noisy:",
        noisy_output_path
    )

    # --------------------------------------------------
    # 11. Create metadata
    # --------------------------------------------------

    metadata_path = (
        "data/metadata/samples.jsonl"
    )

    sample_id = get_next_sample_id(
        metadata_path
    )

    metadata = {

        "sample_id": sample_id,

        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "duration_sec": len(clean) / 16000,
            "clean_file": clean_output_path,
            "noisy_file": noisy_output_path,
            "enhanced_file": None
        },

        "speech": {
            "language": "unknown",
            "speaker_id": "unknown",
            "gender": "unknown",
            "source": "test"
        },

        "noise": {
            "type": "unknown",
            "source": "test",
            "noise_file": noise_path,
            "category": "unknown"
        },

        "mixing": {
            "target_snr_db": target_snr,
            "actual_snr_db": float(
                actual_snr
            ),
            "speech_rms": float(
                final_clean_rms
            ),
            "noise_rms": float(
                final_noise_rms
            ),
            "mixing_method": "rms_scaling"
        },

        "impulses": [],

        "split": "test",

        "generation": {
            "generator_version": "mix_v1.0"
        }
    }

    # --------------------------------------------------
    # 12. Save metadata
    # --------------------------------------------------

    save_metadata(
        metadata_path,
        metadata
    )

    print(
        "Saved metadata:",
        metadata_path
    )


# ======================================================
# TEST
# ======================================================

snr_levels = [-5, 0, 5, 10, 15, 20]

for snr in snr_levels:

    mix_audio(
        "data/tests/processed.wav",
        "data/tests/noise.wav",
        f"data/tests/clean_{snr}db.wav",
        f"data/tests/automatic_{snr}db.wav",
        snr
    )