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
# 2. Add impulse to clean speech
# ======================================================

def add_impulse(
    clean_path,
    impulse_path,
    output_path,
    impulse_gain=1.0
):

    # --------------------------------------------------
    # 1. Load clean speech
    # --------------------------------------------------

    clean, _ = librosa.load(
        clean_path,
        sr=16000,
        mono=True
    )

    # --------------------------------------------------
    # 2. Load impulse
    # --------------------------------------------------

    impulse, _ = librosa.load(
        impulse_path,
        sr=16000,
        mono=True
    )

    # --------------------------------------------------
    # 3. Normalize impulse
    # --------------------------------------------------

    impulse_peak = np.max(
        np.abs(impulse)
    )

    if impulse_peak > 0:

        impulse = (
            impulse /
            impulse_peak
        )

    # Apply controllable impulse strength
    impulse = impulse * impulse_gain

    # --------------------------------------------------
    # 4. Choose random insertion point
    # --------------------------------------------------

    max_start = (
        len(clean) -
        len(impulse)
    )

    if max_start <= 0:

        raise ValueError(
            "Impulse is longer than clean speech."
        )

    start = np.random.randint(
        0,
        max_start
    )

    # --------------------------------------------------
    # 5. Create noisy audio
    # --------------------------------------------------

    noisy = clean.copy()

    noisy[
        start:start + len(impulse)
    ] += impulse

    # --------------------------------------------------
    # 6. Prevent clipping
    # --------------------------------------------------

    peak = np.max(
        np.abs(noisy)
    )

    if peak > 1.0:

        gain = 1.0 / peak

        clean = clean * gain

        noisy = noisy * gain

        print(
            "Clipping detected."
        )

        print(
            "Applied gain:",
            gain
        )

    else:

        print(
            "No clipping."
        )

    # --------------------------------------------------
    # 7. Calculate impulse timing
    # --------------------------------------------------

    start_time = (
        start / 16000
    )

    impulse_duration = (
        len(impulse) / 16000
    )

    end_time = (
        start_time +
        impulse_duration
    )

    duration_ms = (
        impulse_duration * 1000
    )

    # --------------------------------------------------
    # 8. Calculate impulse information
    # --------------------------------------------------

    impulse_rms = np.sqrt(
        np.mean(impulse ** 2)
    )

    speech_rms = np.sqrt(
        np.mean(clean ** 2)
    )

    # --------------------------------------------------
    # 9. Create metadata
    # --------------------------------------------------

    metadata = {

        "sample_id": "IMPULSE_TEST_001",

        "audio": {

            "sample_rate": 16000,

            "channels": 1,

            "duration_sec": (
                len(clean) / 16000
            ),

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

            "type": "impulsive",

            "source": "test",

            "noise_file": impulse_path,

            "category": "unknown"
        },

        "mixing": {

            "target_snr_db": None,

            "actual_snr_db": None,

            "speech_rms": float(
                speech_rms
            ),

            "noise_rms": float(
                impulse_rms
            ),

            "mixing_method": "impulse_addition",

            "impulse_gain": float(
                impulse_gain
            )
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
                    duration_ms
                )
            }
        ],

        "split": "test",

        "generation": {

            "generator_version":
                "impulse_v1.1"
        }
    }

    # --------------------------------------------------
    # 10. Print information
    # --------------------------------------------------

    print()

    print(
        "========== IMPULSE MIX =========="
    )

    print(
        "Speech duration:",
        len(clean) / 16000,
        "seconds"
    )

    print(
        "Impulse gain:",
        impulse_gain
    )

    print(
        "Impulse RMS:",
        impulse_rms
    )

    print(
        "Speech RMS:",
        speech_rms
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
        np.max(
            np.abs(noisy)
        )
    )

    print(
        "================================="
    )

    # --------------------------------------------------
    # 11. Save noisy audio
    # --------------------------------------------------

    sf.write(
        output_path,
        noisy,
        16000,
        subtype="PCM_16"
    )

    print(
        "Saved:",
        output_path
    )

    # --------------------------------------------------
    # 12. Save metadata
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
# TEST
# ======================================================

add_impulse(
    "data/training/clean/speech1.wav",
    "data/impulses/impulse1.wav",
    "data/training/noisy/impulse_test.wav",
    impulse_gains = [0.1, 0.25, 0.5, 0.75, 1.0]
)