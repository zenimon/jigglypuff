#!/usr/bin/env python3

"""
Impulse Guard - Production Impulse Mixer

Generates speech + impulsive-noise mixtures.

Impulsive noise:
    - UrbanSound8K gunshot recordings

Important:
    - Speech and gunshot always come from the same split.
    - Exact impulse onset/offset is stored.
    - Gunshot recordings are cropped around their actual
      high-energy event before insertion.
"""

from pathlib import Path
import json
import random

import librosa
import numpy as np
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

ROOT = Path("data/raw/Impulse_Guard/ImpulseGuard_dataset")
SPLIT_DIR = Path("data/splits")

OUTPUT_ROOT = Path("data/mixtures")
METADATA_PATH = Path("data/metadata/samples.jsonl")

IMPULSE_LABEL_ROOT = Path("data/impulse_labels")

# Gain controls impulse strength.
IMPULSE_GAINS = [
    0.20,
    0.50,
    1.00,
    1.25,
    1.55,
    2.00,
]


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_dataset_path(path_string):
    """
    Resolve paths from split/metadata files.

    Supports:

        Noise/...
        Speech/...
        data/raw/Impulse_Guard/ImpulseGuard_dataset/...
    """

    path = Path(path_string)

    if path.is_absolute():
        return path

    if str(path).startswith("data/raw/"):
        return path

    return ROOT / path


# ============================================================
# SPLIT LOADING
# ============================================================

def load_split_file(filename):
    """
    Load a split file and verify every file exists.
    """

    path = SPLIT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}"
        )

    files = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            resolved = resolve_dataset_path(
                line
            )

            if not resolved.exists():

                raise FileNotFoundError(
                    f"File listed in {path} does not exist:\n"
                    f"{resolved}"
                )

            files.append(resolved)

    if not files:

        raise RuntimeError(
            f"No files found in {path}"
        )

    return files


# ============================================================
# METADATA
# ============================================================

def load_jsonl(path):
    """
    Load JSONL file.
    """

    records = []

    if not path.exists():

        raise FileNotFoundError(
            f"Metadata file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# SAMPLE ID
# ============================================================

def get_next_sample_id():
    """
    Find next available IG_XXXXXX ID.
    """

    max_id = 0

    if METADATA_PATH.exists():

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError:
                    continue

                sample_id = record.get(
                    "sample_id",
                    "",
                )

                if not sample_id.startswith(
                    "IG_"
                ):
                    continue

                try:

                    number = int(
                        sample_id[3:]
                    )

                    max_id = max(
                        max_id,
                        number,
                    )

                except ValueError:
                    pass

    return max_id + 1


# ============================================================
# AUDIO
# ============================================================

def load_audio(path):
    """
    Load audio as:
        mono
        16 kHz
        float32
    """

    audio, sr = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    return audio.astype(
        np.float32
    )


def calculate_rms(audio):

    if len(audio) == 0:
        return 0.0

    return float(
        np.sqrt(
            np.mean(audio ** 2)
            + 1e-12
        )
    )


# ============================================================
# IMPULSE EXTRACTION
# ============================================================
def extract_impulse_event(
    impulse,
    threshold_ratio=0.08,
):
    """
    Detect the actual high-energy impulse using a short-time
    RMS/envelope rather than a raw peak-amplitude threshold.

    This is more robust when the gunshot recording contains
    background noise or a long tail.

    Returns:
        event_audio
        start_sample
        end_sample
    """

    if len(impulse) == 0:
        raise ValueError(
            "Impulse audio is empty."
        )

    # --------------------------------------------------------
    # 1. Calculate short-time RMS envelope
    # --------------------------------------------------------

    frame_size = int(
        0.010 * SAMPLE_RATE
    )  # 10 ms

    hop_size = int(
        0.005 * SAMPLE_RATE
    )  # 5 ms

    if len(impulse) < frame_size:
        return (
            impulse,
            0,
            len(impulse),
        )

    rms_values = []

    for start in range(
        0,
        len(impulse) - frame_size + 1,
        hop_size,
    ):

        frame = impulse[
            start:start + frame_size
        ]

        rms = np.sqrt(
            np.mean(frame ** 2)
            + 1e-12
        )

        rms_values.append(rms)

    rms_values = np.asarray(
        rms_values,
        dtype=np.float32,
    )

    if len(rms_values) == 0:
        return (
            impulse,
            0,
            len(impulse),
        )

    # --------------------------------------------------------
    # 2. Determine energy threshold
    # --------------------------------------------------------

    peak_rms = np.max(
        rms_values
    )

    if peak_rms <= 1e-8:
        raise ValueError(
            "Impulse audio has negligible energy."
        )

    threshold = (
        threshold_ratio
        * peak_rms
    )

    active = (
        rms_values >= threshold
    )

    active_indices = np.flatnonzero(
        active
    )

    if len(active_indices) == 0:
        return (
            impulse,
            0,
            len(impulse),
        )

    # --------------------------------------------------------
    # 3. Convert active RMS frames to samples
    # --------------------------------------------------------

    start_frame = int(
        active_indices[0]
    )

    end_frame = int(
        active_indices[-1]
    )

    start = (
        start_frame
        * hop_size
    )

    end = (
        end_frame
        * hop_size
        + frame_size
    )

    # --------------------------------------------------------
    # 4. Add 20 ms padding
    # --------------------------------------------------------

    padding = int(
        0.020 * SAMPLE_RATE
    )

    start = max(
        0,
        start - padding,
    )

    end = min(
        len(impulse),
        end + padding,
    )

    event = impulse[
        start:end
    ]

    return (
        event,
        start,
        end,
    )


    
# ============================================================
# IMPULSE MIXING
# ============================================================

def add_impulse_arrays(
    base_audio,
    impulse_audio,
    impulse_gain=0.5,
):
    """
    Add a detected impulse event to an audio array.

    Returns:
        base_audio: clipped/scaled base signal
        noisy_audio: base signal with impulse
        metadata: impulse timing and gain information
    """

    base_audio = np.asarray(
        base_audio,
        dtype=np.float32,
    ).copy()

    impulse_audio = np.asarray(
        impulse_audio,
        dtype=np.float32,
    ).copy()

    if len(base_audio) == 0:
        raise ValueError("Base audio is empty.")

    if len(impulse_audio) == 0:
        raise ValueError("Impulse audio is empty.")

    # Extract the actual impulse event.
    (
        event,
        event_start,
        event_end,
    ) = extract_impulse_event(
        impulse_audio
    )

    if len(event) == 0:
        raise ValueError("No impulse event detected.")

    # Normalize event peak.
    event_peak = np.max(
        np.abs(event)
    )

    if event_peak <= 1e-8:
        raise ValueError(
            "Impulse event has extremely low amplitude."
        )

    event = event / event_peak
    event = event * float(impulse_gain)

    # Make noisy copy.
    noisy_audio = base_audio.copy()

    # Choose insertion position.
    if len(event) >= len(base_audio):

        event = event[:len(base_audio)]

        insert_position = 0

    else:

        max_position = (
            len(base_audio)
            - len(event)
        )

        insert_position = random.randint(
            0,
            max_position,
        )

    insert_end = (
        insert_position
        + len(event)
    )

    # Add impulse.
    noisy_audio[
        insert_position:insert_end
    ] += event

    # Clipping protection.
    peak = max(
        np.max(np.abs(base_audio)),
        np.max(np.abs(noisy_audio)),
    )

    if peak > 0.999:

        scale = 0.999 / peak

        base_audio = base_audio * scale
        noisy_audio = noisy_audio * scale

    return (
        base_audio.astype(np.float32),
        noisy_audio.astype(np.float32),
        {
            "impulse_gain": float(
                impulse_gain
            ),
            "impulse_source_start_sample": int(
                event_start
            ),
            "impulse_source_end_sample": int(
                event_end
            ),
            "impulse_onset_sample": int(
                insert_position
            ),
            "impulse_offset_sample": int(
                insert_end
            ),
            "impulse_duration_samples": int(
                len(event)
            ),
            "impulse_duration_sec": float(
                len(event) / SAMPLE_RATE
            ),
        },
    )




def add_impulse(
    clean_path,
    impulse_path,
    clean_output_path,
    noisy_output_path,
    impulse_gain=0.5,
):
    """
    Insert one extracted impulse event into clean speech.

    Returns metadata about the event.
    """

    clean = load_audio(
        clean_path
    )

    impulse = load_audio(
        impulse_path
    )

    if len(clean) == 0:

        raise ValueError(
            f"Empty clean audio: {clean_path}"
        )

    if len(impulse) == 0:

        raise ValueError(
            f"Empty impulse audio: {impulse_path}"
        )

    # --------------------------------------------------------
    # Extract actual impulse event.
    # --------------------------------------------------------

    (
        impulse_event,
        event_start,
        event_end,
    ) = extract_impulse_event(
        impulse
    )

    # --------------------------------------------------------
    # Normalize impulse peak.
    # --------------------------------------------------------

    impulse_peak = np.max(
        np.abs(impulse_event)
    )

    if impulse_peak <= 1e-8:

        raise ValueError(
            "Extracted impulse has negligible amplitude."
        )

    impulse_event = (
        impulse_event / impulse_peak
    )

    # Apply requested gain.
    impulse_event = (
        impulse_event * impulse_gain
    )

    # --------------------------------------------------------
    # Make a copy of clean audio.
    # --------------------------------------------------------

    noisy = clean.copy()

    # --------------------------------------------------------
    # Choose random insertion position.
    #
    # Keep the entire impulse inside the speech clip.
    # --------------------------------------------------------

    if len(impulse_event) >= len(clean):

        # Rare edge case.
        # Center/crop the impulse.
        impulse_event = (
            impulse_event[:len(clean)]
        )

        insert_start = 0

    else:

        max_start = (
            len(clean)
            - len(impulse_event)
        )

        insert_start = random.randint(
            0,
            max_start,
        )

    insert_end = (
        insert_start
        + len(impulse_event)
    )

    # --------------------------------------------------------
    # Add impulse.
    # --------------------------------------------------------

    noisy[
        insert_start:insert_end
    ] += impulse_event

    # --------------------------------------------------------
    # Clipping protection.
    #
    # Scale clean + noisy together.
    # This preserves the relative impulse relationship.
    # --------------------------------------------------------

    peak = max(
        np.max(np.abs(clean)),
        np.max(np.abs(noisy)),
    )

    if peak > 0.999:

        scale = 0.999 / peak

        clean = clean * scale
        noisy = noisy * scale

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    clean_output_path = Path(
        clean_output_path
    )

    noisy_output_path = Path(
        noisy_output_path
    )

    clean_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    noisy_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(clean_output_path),
        clean,
        SAMPLE_RATE,
        subtype="PCM_16",
    )

    sf.write(
        str(noisy_output_path),
        noisy,
        SAMPLE_RATE,
        subtype="PCM_16",
    )

    # --------------------------------------------------------
    # Exact event timing.
    # --------------------------------------------------------

    onset_sample = insert_start
    offset_sample = insert_end

    onset_sec = (
        onset_sample
        / SAMPLE_RATE
    )

    offset_sec = (
        offset_sample
        / SAMPLE_RATE
    )

    duration_sec = (
        offset_sample
        - onset_sample
    ) / SAMPLE_RATE

    return {
        "onset_sample": int(
            onset_sample
        ),

        "offset_sample": int(
            offset_sample
        ),

        "onset_sec": float(
            onset_sec
        ),

        "offset_sec": float(
            offset_sec
        ),

        "duration_sec": float(
            duration_sec
        ),

        "impulse_gain": float(
            impulse_gain
        ),

        "impulse_event_source_start_sample": int(
            event_start
        ),

        "impulse_event_source_end_sample": int(
            event_end
        ),

        "audio_duration_sec": float(
            len(clean) / SAMPLE_RATE
        ),
    }


# ============================================================
# PRODUCTION IMPULSE DATASET
# ============================================================

def generate_production_impulse_mixtures(
    split,
    num_samples,
    seed=None,
):
    """
    Generate production speech + gunshot mixtures.

    Speech and gunshots are taken from the same split.
    """

    if split not in {
        "train",
        "validation",
        "test",
    }:

        raise ValueError(
            "split must be train, validation, or test"
        )

    if num_samples <= 0:

        raise ValueError(
            "num_samples must be > 0"
        )

    if seed is not None:

        random.seed(seed)
        np.random.seed(seed)

    # --------------------------------------------------------
    # Load speech split.
    # --------------------------------------------------------

    speech_files = load_split_file(
        f"speech_{split}.txt"
    )

    # --------------------------------------------------------
    # Load noise split.
    # --------------------------------------------------------

    noise_files = load_split_file(
        f"noise_{split}.txt"
    )

    noise_set = {
        str(path)
        for path in noise_files
    }

    # --------------------------------------------------------
    # Load noise metadata.
    # --------------------------------------------------------

    noise_records = load_jsonl(
        Path(
            "data/metadata/"
            "noise_metadata.jsonl"
        )
    )

    # --------------------------------------------------------
    # Keep only gunshot records that
    # actually belong to this split.
    # --------------------------------------------------------

    gunshot_records = []

    for record in noise_records:

        if record.get(
            "category"
        ) != "gunshot":

            continue

        file_value = record.get(
            "file"
        )

        if not file_value:
            continue

        path = resolve_dataset_path(
            file_value
        )

        # Critical leakage protection.
        if str(path) not in noise_set:
            continue

        gunshot_records.append(
            record
        )

    if not gunshot_records:

        raise RuntimeError(
            f"No gunshot recordings found "
            f"in {split} split."
        )

    print(
        f"Gunshots available in {split}: "
        f"{len(gunshot_records)}"
    )

    # --------------------------------------------------------
    # Output directories.
    # --------------------------------------------------------

    clean_dir = (
        OUTPUT_ROOT
        / split
        / "clean"
    )

    noisy_dir = (
        OUTPUT_ROOT
        / split
        / "noisy"
    )

    clean_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    noisy_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Onset label file.
    # --------------------------------------------------------

    IMPULSE_LABEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    onset_path = (
        IMPULSE_LABEL_ROOT
        / f"{split}_onsets.jsonl"
    )

    # --------------------------------------------------------
    # Sample IDs.
    # --------------------------------------------------------

    next_id = get_next_sample_id()

    generated = []

    # --------------------------------------------------------
    # Generate.
    # --------------------------------------------------------

    for index in range(
        num_samples
    ):

        sample_id = (
            f"IG_{next_id:06d}"
        )

        next_id += 1

        # Random speech from same split.
        speech_path = random.choice(
            speech_files
        )

        # Random gunshot from same split.
        gunshot_record = random.choice(
            gunshot_records
        )

        gunshot_path = resolve_dataset_path(
            gunshot_record["file"]
        )

        # Safety.
        if str(gunshot_path) not in noise_set:

            raise RuntimeError(
                "Split leakage detected!\n"
                f"Gunshot: {gunshot_path}\n"
                f"Split: {split}"
            )

        impulse_gain = random.choice(
            IMPULSE_GAINS
        )

        clean_output = (
            clean_dir
            / f"{sample_id}.wav"
        )

        noisy_output = (
            noisy_dir
            / f"{sample_id}.wav"
        )

        event = add_impulse(
            clean_path=speech_path,
            impulse_path=gunshot_path,
            clean_output_path=clean_output,
            noisy_output_path=noisy_output,
            impulse_gain=impulse_gain,
        )

        # ----------------------------------------------------
        # Rich mixture metadata.
        # ----------------------------------------------------

        record = {

            "sample_id": sample_id,

            "split": split,

            "clean_path": str(
                clean_output
            ),

            "noisy_path": str(
                noisy_output
            ),

            "speech": {
                "path": str(
                    speech_path
                ),
                "source": "LibriSpeech",
            },

            "impulse": {
                "path": str(
                    gunshot_path
                ),
                "source": gunshot_record.get(
                    "source"
                ),
                "category": "gunshot",
                "type": "impulsive",
                "original_duration_sec":
                    gunshot_record.get(
                        "duration_sec"
                    ),
            },

            "impulse_gain": event[
                "impulse_gain"
            ],

            "onset_sample": event[
                "onset_sample"
            ],

            "offset_sample": event[
                "offset_sample"
            ],

            "onset_sec": event[
                "onset_sec"
            ],

            "offset_sec": event[
                "offset_sec"
            ],

            "impulse_duration_sec": event[
                "duration_sec"
            ],

            "duration_sec": event[
                "audio_duration_sec"
            ],

            "sample_rate": SAMPLE_RATE,

            "mixture_type": "impulse",

            "generator": "mix_impulse_v2.0",
        }

        generated.append(
            record
        )

        # ----------------------------------------------------
        # samples.jsonl
        # ----------------------------------------------------

        METADATA_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            METADATA_PATH,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        # ----------------------------------------------------
        # Onset labels.
        # ----------------------------------------------------

        onset_record = {

            "sample_id": sample_id,

            "split": split,

            "noisy_path": str(
                noisy_output
            ),

            "onset_sample": event[
                "onset_sample"
            ],

            "offset_sample": event[
                "offset_sample"
            ],

            "onset_sec": event[
                "onset_sec"
            ],

            "offset_sec": event[
                "offset_sec"
            ],

            "impulse_duration_sec":
                event[
                    "duration_sec"
                ],

            "label": "gunshot",

        }

        with open(
            onset_path,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                json.dumps(
                    onset_record
                )
                + "\n"
            )

        print(
            f"[{index + 1:>3}/{num_samples}] "
            f"{sample_id} | "
            f"gain {impulse_gain:.2f} | "
            f"onset {event['onset_sec']:>7.3f}s | "
            f"duration {event['duration_sec']:.3f}s"
        )

    print(
        "\nImpulse generation complete."
    )

    print(
        f"Split: {split}"
    )

    print(
        f"Samples: {num_samples}"
    )

    print(
        f"Onset labels: {onset_path}"
    )

    return generated


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # First production smoke test.
    # Generate only 20 samples.

    generate_production_impulse_mixtures(
        split="train",
        num_samples=20,
        seed=42,
    )