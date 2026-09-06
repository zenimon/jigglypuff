#!/usr/bin/env python3

"""
Impulse Guard - Production Normal Noise Mixer

Generates clean/noisy speech mixtures for:
    - stationary noise
    - non-stationary noise

Excluded here:
    - impulsive/gunshot noise
      -> handled separately by mix_impulse.py

Production requirements:
    - split-aware
    - no cross-split mixing
    - randomized speech/noise selection
    - balanced noise-category sampling
    - randomized SNR
    - unique sample IDs
    - rich metadata
    - clipping protection
    - actual SNR verification
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

SNR_CHOICES = [-5, 0, 5, 10, 15, 20]

# Gunshot is intentionally excluded.
# It will be handled by mix_impulse.py.
NORMAL_NOISE_CATEGORIES = [
    "engine_idling",
    "drone",
    "musan",
    "siren",
    "wind",
]


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_dataset_path(path_string):
    """
    Resolve paths appearing in split files.

    Supported formats:

        Speech/Clean_Speech/...
        Noise/...
        data/raw/Impulse_Guard/ImpulseGuard_dataset/...

    Returns:
        Path
    """

    path = Path(path_string)

    if path.is_absolute():
        return path

    # Already relative to repository root.
    if str(path).startswith("data/raw/"):
        return path

    # Relative to dataset root.
    return ROOT / path


# ============================================================
# SPLIT LOADING
# ============================================================

def load_split_file(filename):
    """
    Load a split file and resolve every path.

    Example:
        speech_train.txt
        noise_train.txt
    """

    path = SPLIT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    files = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            resolved = resolve_dataset_path(line)

            if not resolved.exists():
                raise FileNotFoundError(
                    f"File listed in {path} does not exist:\n{resolved}"
                )

            files.append(resolved)

    if not files:
        raise RuntimeError(f"No files found in split: {path}")

    return files


# ============================================================
# METADATA LOADING
# ============================================================

def load_jsonl(path):
    """
    Load JSONL metadata.
    """

    records = []

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def build_noise_category_index():
    """
    Build:

        category -> list of metadata records

    using noise_metadata.jsonl.
    """

    metadata_path = Path("data/metadata/noise_metadata.jsonl")

    records = load_jsonl(metadata_path)

    categories = {}

    for record in records:
        category = record.get("category")

        if category not in NORMAL_NOISE_CATEGORIES:
            continue

        file_value = record.get("file")

        if not file_value:
            continue

        categories.setdefault(category, []).append(record)

    return categories


# ============================================================
# SAMPLE ID
# ============================================================

def get_next_sample_id():
    """
    Find the next IG_XXXXXX sample ID.

    Existing metadata is scanned so IDs remain unique
    across multiple runs.
    """

    max_id = 0

    if METADATA_PATH.exists():

        with open(METADATA_PATH, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sample_id = record.get("sample_id", "")

                if sample_id.startswith("IG_"):

                    try:
                        number = int(sample_id[3:])
                        max_id = max(max_id, number)

                    except ValueError:
                        pass

    return max_id + 1


# ============================================================
# AUDIO HELPERS
# ============================================================

def load_audio(path):
    """
    Load audio as mono 16 kHz float32.
    """

    audio, sr = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = audio.astype(np.float32)

    return audio


def calculate_rms(audio):
    """
    Calculate RMS safely.
    """

    if len(audio) == 0:
        return 0.0

    return float(np.sqrt(np.mean(audio ** 2) + 1e-12))


def calculate_snr_db(clean, noise):
    """
    Calculate actual SNR between clean and noise.
    """

    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power <= 1e-12:
        return float("inf")

    return float(
        10.0 * np.log10(
            clean_power / noise_power
        )
    )


# ============================================================
# NORMAL-NOISE MIXING
# ============================================================

def mix_audio(
    clean_path,
    noise_path,
    clean_output_path,
    noisy_output_path,
    target_snr,
):
    """
    Mix clean speech with noise at a target SNR.

    This function is kept simple so it can also be used
    independently for testing.
    """

    clean = load_audio(clean_path)
    noise = load_audio(noise_path)

    if len(clean) == 0:
        raise ValueError(f"Empty clean audio: {clean_path}")

    if len(noise) == 0:
        raise ValueError(f"Empty noise audio: {noise_path}")

    # --------------------------------------------------------
    # Make noise at least as long as speech.
    # --------------------------------------------------------

    if len(noise) < len(clean):

        repetitions = int(
            np.ceil(len(clean) / len(noise))
        )

        noise = np.tile(noise, repetitions)

    # Random crop when noise is longer.
    if len(noise) > len(clean):

        max_start = len(noise) - len(clean)

        start = random.randint(0, max_start)

        noise = noise[start:start + len(clean)]

    else:

        noise = noise[:len(clean)]

    # --------------------------------------------------------
    # RMS scaling
    # --------------------------------------------------------

    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)

    if clean_rms <= 1e-8:
        raise ValueError(
            f"Clean audio has extremely low RMS: {clean_path}"
        )

    if noise_rms <= 1e-8:
        raise ValueError(
            f"Noise audio has extremely low RMS: {noise_path}"
        )

    target_noise_rms = (
        clean_rms /
        (10.0 ** (target_snr / 20.0))
    )

    noise = noise * (
        target_noise_rms / noise_rms
    )

    # --------------------------------------------------------
    # Mix
    # --------------------------------------------------------

    noisy = clean + noise

    # --------------------------------------------------------
    # Clipping protection
    #
    # Scale the entire mixture together so that:
    #     clean/noise relationship remains unchanged.
    # --------------------------------------------------------

    peak = max(
        np.max(np.abs(clean)),
        np.max(np.abs(noise)),
        np.max(np.abs(noisy)),
    )

    if peak > 0.999:

        scale = 0.999 / peak

        clean = clean * scale
        noise = noise * scale
        noisy = noisy * scale

    # --------------------------------------------------------
    # Actual SNR after scaling
    # --------------------------------------------------------

    actual_snr = calculate_snr_db(
        clean,
        noise,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    clean_output_path = Path(clean_output_path)
    noisy_output_path = Path(noisy_output_path)

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

    return {
        "target_snr_db": float(target_snr),
        "actual_snr_db": float(actual_snr),
        "duration_sec": float(len(clean) / SAMPLE_RATE),
    }


def scale_noise_to_snr(
    clean,
    noise,
    target_snr,
):
    """
    Prepare a noise signal at a target SNR relative to clean speech.

    This function only scales/crops the noise.
    It does NOT add the noise to the clean signal.

    This is useful when multiple noise sources need to be
    combined while keeping each noise level referenced to
    the original clean speech.
    """

    clean = np.asarray(
        clean,
        dtype=np.float32,
    ).copy()

    noise = np.asarray(
        noise,
        dtype=np.float32,
    ).copy()

    if len(clean) == 0:
        raise ValueError("Clean audio is empty.")

    if len(noise) == 0:
        raise ValueError("Noise audio is empty.")

    # Make noise at least as long as speech.
    if len(noise) < len(clean):
        repetitions = int(
            np.ceil(len(clean) / len(noise))
        )

        noise = np.tile(
            noise,
            repetitions,
        )

    # Random crop when noise is longer.
    if len(noise) > len(clean):
        max_start = len(noise) - len(clean)

        start = random.randint(
            0,
            max_start,
        )

        noise = noise[
            start:start + len(clean)
        ]

    else:
        noise = noise[:len(clean)]

    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)

    if clean_rms <= 1e-8:
        raise ValueError(
            "Clean audio has extremely low RMS."
        )

    if noise_rms <= 1e-8:
        raise ValueError(
            "Noise audio has extremely low RMS."
        )

    target_noise_rms = (
        clean_rms
        / (10.0 ** (target_snr / 20.0))
    )

    noise = noise * (
        target_noise_rms
        / noise_rms
    )

    return noise.astype(np.float32)

# ============================================================
# NOISE CATEGORY SAMPLING
# ============================================================

def mix_audio_arrays(
    clean,
    noise,
    target_snr,
):
    """
    Mix clean speech with noise in memory.

    This uses the same mixing logic as mix_audio(),
    but returns arrays instead of writing WAV files.

    Returns:
        clean: clipped/scaled clean signal
        noisy: clipped/scaled noisy signal
        metadata: target and actual SNR
    """

    clean = np.asarray(
        clean,
        dtype=np.float32,
    ).copy()

    noise = np.asarray(
        noise,
        dtype=np.float32,
    ).copy()

    if len(clean) == 0:
        raise ValueError("Clean audio is empty.")

    if len(noise) == 0:
        raise ValueError("Noise audio is empty.")

    # --------------------------------------------------------
    # Make noise at least as long as speech.
    # --------------------------------------------------------

    if len(noise) < len(clean):

        repetitions = int(
            np.ceil(len(clean) / len(noise))
        )

        noise = np.tile(
            noise,
            repetitions,
        )

    # Random crop when noise is longer.
    if len(noise) > len(clean):

        max_start = (
            len(noise)
            - len(clean)
        )

        start = random.randint(
            0,
            max_start,
        )

        noise = noise[
            start:start + len(clean)
        ]

    else:

        noise = noise[:len(clean)]

    # --------------------------------------------------------
    # RMS scaling
    # --------------------------------------------------------

    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)

    if clean_rms <= 1e-8:
        raise ValueError(
            "Clean audio has extremely low RMS."
        )

    if noise_rms <= 1e-8:
        raise ValueError(
            "Noise audio has extremely low RMS."
        )

    target_noise_rms = (
        clean_rms
        / (10.0 ** (target_snr / 20.0))
    )

    noise = noise * (
        target_noise_rms
        / noise_rms
    )

    # --------------------------------------------------------
    # Mix
    # --------------------------------------------------------

    noisy = clean + noise

    # --------------------------------------------------------
    # Clipping protection
    # --------------------------------------------------------

    peak = max(
        np.max(np.abs(clean)),
        np.max(np.abs(noise)),
        np.max(np.abs(noisy)),
    )

    if peak > 0.999:

        scale = 0.999 / peak

        clean = clean * scale
        noise = noise * scale
        noisy = noisy * scale

    # --------------------------------------------------------
    # Actual SNR
    # --------------------------------------------------------

    actual_snr = calculate_snr_db(
        clean,
        noise,
    )

    return (
        clean.astype(np.float32),
        noisy.astype(np.float32),
        {
            "target_snr_db": float(target_snr),
            "actual_snr_db": float(actual_snr),
            "duration_sec": float(
                len(clean) / SAMPLE_RATE
            ),
        },
    )
# ============================================================
# PRODUCTION DATASET GENERATION
# ============================================================

def generate_production_mixtures(
    split,
    num_samples,
    seed=None,
):
    """
    Generate production normal-noise mixtures.

    Example:

        generate_production_mixtures(
            split="train",
            num_samples=20,
            seed=42,
        )

    Important:
        Speech and noise are ALWAYS taken from the
        SAME split.
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
    # Load split-specific speech
    # --------------------------------------------------------

    speech_files = load_split_file(
        f"speech_{split}.txt"
    )

    # --------------------------------------------------------
    # Load split-specific noise
    # --------------------------------------------------------

    noise_files = load_split_file(
        f"noise_{split}.txt"
    )

    # Map actual paths -> metadata.
    noise_metadata = load_jsonl(
        Path("data/metadata/noise_metadata.jsonl")
    )

    metadata_by_path = {}

    for record in noise_metadata:

        file_value = record.get("file")

        if not file_value:
            continue

        resolved = resolve_dataset_path(
            file_value
        )

        metadata_by_path[
            str(resolved)
        ] = record

    # --------------------------------------------------------
    # Build category index ONLY from this split.
    #
    # This is important.
    # We must never accidentally choose a validation/test
    # recording while generating training data.
    # --------------------------------------------------------

    split_noise_by_category = {}

    split_noise_set = {
        str(path)
        for path in noise_files
    }

    for noise_path in noise_files:

        record = metadata_by_path.get(
            str(noise_path)
        )

        if record is None:
            continue

        category = record.get("category")

        if category not in NORMAL_NOISE_CATEGORIES:
            continue

        split_noise_by_category.setdefault(
            category,
            [],
        ).append(record)

    # --------------------------------------------------------
    # Verify categories
    # --------------------------------------------------------

    missing_categories = [
        category
        for category in NORMAL_NOISE_CATEGORIES
        if category not in split_noise_by_category
        or not split_noise_by_category[category]
    ]

    if missing_categories:

        print(
            "Warning: categories unavailable in "
            f"{split} split: {missing_categories}"
        )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    clean_dir = (
        OUTPUT_ROOT /
        split /
        "clean"
    )

    noisy_dir = (
        OUTPUT_ROOT /
        split /
        "noisy"
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
    # Sample IDs
    # --------------------------------------------------------

    next_id = get_next_sample_id()

    generated_records = []

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    for index in range(num_samples):

        sample_id = (
            f"IG_{next_id:06d}"
        )

        next_id += 1

        # Random speech from SAME split.
        speech_path = random.choice(
            speech_files
        )

        # Random category, balanced by category.
        category, noise_record = choose_noise_record(
            split_noise_by_category
        )

        noise_path = resolve_dataset_path(
            noise_record["file"]
        )

        # Safety check.
        if str(noise_path) not in split_noise_set:

            raise RuntimeError(
                "Split leakage detected!\n"
                f"Selected noise: {noise_path}\n"
                f"Split: {split}"
            )

        # Random SNR.
        target_snr = random.choice(
            SNR_CHOICES
        )

        clean_output = (
            clean_dir /
            f"{sample_id}.wav"
        )

        noisy_output = (
            noisy_dir /
            f"{sample_id}.wav"
        )

        mix_info = mix_audio(
            clean_path=speech_path,
            noise_path=noise_path,
            clean_output_path=clean_output,
            noisy_output_path=noisy_output,
            target_snr=target_snr,
        )

        # ----------------------------------------------------
        # Rich metadata
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
                "path": str(speech_path),
                "source": "LibriSpeech",
            },

            "noise": {
                "path": str(noise_path),
                "source": noise_record.get(
                    "source"
                ),
                "category": category,
                "type": noise_record.get(
                    "type"
                ),
                "duration_sec": noise_record.get(
                    "duration_sec"
                ),
            },

            "target_snr_db": mix_info[
                "target_snr_db"
            ],

            "actual_snr_db": mix_info[
                "actual_snr_db"
            ],

            "duration_sec": mix_info[
                "duration_sec"
            ],

            "sample_rate": SAMPLE_RATE,

            "mixture_type": "normal_noise",

            "generator": "mix_data_v2.0",
        }

        generated_records.append(
            record
        )

        # Append immediately so a crash doesn't
        # lose all previous metadata.
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

        print(
            f"[{index + 1:>3}/{num_samples}] "
            f"{sample_id} | "
            f"{category:<15} | "
            f"SNR {target_snr:>3} dB | "
            f"actual {mix_info['actual_snr_db']:>6.2f} dB"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    category_counts = {}

    for record in generated_records:

        category = record[
            "noise"
        ]["category"]

        category_counts[category] = (
            category_counts.get(
                category,
                0,
            )
            + 1
        )

    print("\nGeneration complete.")
    print(f"Split: {split}")
    print(f"Samples: {num_samples}")
    print("Categories:")

    for category, count in sorted(
        category_counts.items()
    ):

        print(
            f"  {category}: {count}"
        )

    return generated_records


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # For the first production test, generate ONLY 20.
    #
    # We will inspect:
    #   - files
    #   - metadata
    #   - categories
    #   - SNR
    #   - split correctness
    #
    # before generating thousands of samples.
    # --------------------------------------------------------

    generate_production_mixtures(
        split="train",
        num_samples=20,
        seed=42,
    )