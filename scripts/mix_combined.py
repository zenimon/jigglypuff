import numpy as np
import librosa
import soundfile as sf
import json
import os


# ======================================================
# 1. Save metadata
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
# 2. Mix clean speech with normal noise
# ======================================================

def mix_normal_noise(
    clean,
    noise,
    target_snr_db
):

    # Repeat noise if it is shorter than speech

    if len(noise) < len(clean):

        repeats = int(
            np.ceil(len(clean) / len(noise))
        )

        noise = np.tile(
            noise,
            repeats
        )

    # Make noise exactly as long as speech

    noise = noise[:len(clean)]

    # Calculate RMS

    speech_rms = np.sqrt(
        np.mean(clean ** 2)
    )

    noise_rms = np.sqrt(
        np.mean(noise ** 2)
    )

    # Avoid division by zero

    if noise_rms == 0:

        raise ValueError(
            "Noise RMS is zero."
        )

    # Calculate required noise RMS

    target_noise_rms = (
        speech_rms /
        (10 ** (target_snr_db / 20))
    )

    # Scale noise

    noise = (
        noise *
        target_noise_rms /
        noise_rms
    )

    # Mix

    noisy = clean + noise

    return noisy, noise


# ======================================================
# 3. Add impulse
# ======================================================

def add_impulse(
    noisy,
    impulse,
    impulse_gain=0.5
):

    # Normalize impulse peak

    impulse_peak = np.max(
        np.abs(impulse)
    )

    if impulse_peak > 0:

        impulse = (
            impulse /
            impulse_peak
        )

    # Control impulse strength

    impulse = impulse * impulse_gain

    # Select random insertion point

    max_start = (
        len(noisy) -
        len(impulse)
    )

    if max_start <= 0:

        raise ValueError(
            "Impulse is longer than speech."
        )

    start = np.random.randint(
        0,
        max_start
    )

    # Add impulse

    mixed = noisy.copy()

    mixed[
        start:start + len(impulse)
    ] += impulse

    return mixed, impulse, start


# ======================================================
# 4. Main mixing function
# ======================================================

def create_combined_sample(
    target_snr_db,
    impulse_gain,
    sample_number
):

    clean_path = (
        "data/training/clean/speech1.wav"
    )

    normal_noise_path = (
        "data/tests/noise.wav"
    )

    impulse_path = (
        "data/impulses/impulse1.wav"
    )

    # --------------------------------------------------
    # Unique sample name
    # --------------------------------------------------

    sample_id = (
        f"COMBINED_{sample_number:03d}"
    )

    output_filename = (
        f"{sample_id}_snr_{target_snr_db}db_gain_{impulse_gain}.wav"
    )

    output_path = os.path.join(
        "data/training/noisy",
        output_filename
    )

    # --------------------------------------------------
    # Load audio
    # --------------------------------------------------

    clean, _ = librosa.load(
        clean_path,
        sr=16000,
        mono=True
    )

    normal_noise, _ = librosa.load(
        normal_noise_path,
        sr=16000,
        mono=True
    )

    impulse, _ = librosa.load(
        impulse_path,
        sr=16000,
        mono=True
    )

    # --------------------------------------------------
    # Add normal noise
    # --------------------------------------------------

    noisy, scaled_noise = mix_normal_noise(
        clean,
        normal_noise,
        target_snr_db
    )

    # --------------------------------------------------
    # Add impulse
    # --------------------------------------------------

    mixed, impulse, start = add_impulse(
        noisy,
        impulse,
        impulse_gain
    )

    # --------------------------------------------------
    # Prevent clipping
    # --------------------------------------------------

    peak = np.max(
        np.abs(mixed)
    )

    if peak > 1.0:

        gain = 1.0 / peak

        clean = clean * gain
        noisy = noisy * gain
        mixed = mixed * gain
        scaled_noise = scaled_noise * gain
        impulse = impulse * gain

        print("Clipping detected.")
        print("Applied gain:", gain)

    else:

        print("No clipping.")

    # --------------------------------------------------
    # Calculate timing
    # --------------------------------------------------

    start_time = start / 16000

    impulse_duration = (
        len(impulse) / 16000
    )

    end_time = (
        start_time +
        impulse_duration
    )

    # --------------------------------------------------
    # Calculate RMS and actual SNR
    # --------------------------------------------------

    speech_rms = np.sqrt(
        np.mean(clean ** 2)
    )

    noise_rms = np.sqrt(
        np.mean(scaled_noise ** 2)
    )

    actual_snr_db = (
        20 *
        np.log10(
            speech_rms /
            noise_rms
        )
    )

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    metadata = {

        "sample_id": sample_id,

        "audio": {

            "sample_rate": 16000,

            "channels": 1,

            "duration_sec": len(clean) / 16000,

            "clean_file": clean_path,

            "noisy_file": output_path,

            "enhanced_file": None
        },

        "speech": {

            "language": "unknown",

            "speaker_id": "unknown",

            "gender": "unknown",

            "source": "test"
        },

        "noise": {

            "type": "mixed",

            "source": "test",

            "noise_file": normal_noise_path,

            "category": "normal_plus_impulsive"
        },

        "mixing": {

            "target_snr_db": target_snr_db,

            "actual_snr_db": float(
                actual_snr_db
            ),

            "speech_rms": float(
                speech_rms
            ),

            "noise_rms": float(
                noise_rms
            ),

            "mixing_method":
                "rms_scaling_plus_impulse",

            "impulse_gain": impulse_gain
        },

        "impulses": [

            {

                "event_id": "imp_001",

                "type": "unknown",

                "start_sec": float(
                    start_time
                ),

                "end_sec": float(
                    end_time
                ),

                "duration_ms": float(
                    impulse_duration * 1000
                )
            }
        ],

        "split": "test",

        "generation": {

            "generator_version":
                "combined_v1.1"

        }
    }

    # --------------------------------------------------
    # Print result
    # --------------------------------------------------

    print()

    print(
        "========== COMBINED MIX =========="
    )

    print(
        "Sample ID:",
        sample_id
    )

    print(
        "Target SNR:",
        target_snr_db,
        "dB"
    )

    print(
        "Actual SNR:",
        actual_snr_db,
        "dB"
    )

    print(
        "Impulse gain:",
        impulse_gain
    )

    print(
        "Impulse start:",
        start_time,
        "sec"
    )

    print(
        "Impulse end:",
        end_time,
        "sec"
    )

    print(
        "Impulse duration:",
        impulse_duration,
        "seconds"
    )

    print(
        "Final peak:",
        np.max(np.abs(mixed))
    )

    print(
        "=================================="
    )

    # --------------------------------------------------
    # Save audio
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    sf.write(
        output_path,
        mixed,
        16000,
        subtype="PCM_16"
    )

    print(
        "Saved:",
        output_path
    )

    # --------------------------------------------------
    # Save metadata
    # --------------------------------------------------

    metadata_path = (
        "data/metadata/samples.jsonl"
    )

    save_metadata(
        metadata_path,
        metadata
    )

    print(
        "Saved metadata:",
        metadata_path
    )


# ======================================================
# Run
# ======================================================

if __name__ == "__main__":

    snr_levels = [
        -5,
        0,
        5,
        10,
        15,
        20
    ]

    impulse_gains = [
        0.1,
        0.25,
        0.5,
        0.75,
        1.0
    ]

    sample_number = 1

    for snr in snr_levels:

        for gain in impulse_gains:

            create_combined_sample(
                target_snr_db=snr,
                impulse_gain=gain,
                sample_number=sample_number
            )

            sample_number += 1