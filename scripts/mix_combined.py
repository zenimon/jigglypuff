#!/usr/bin/env python3

"""
Impulse Guard - Production Combined Mixture Generator

Orchestrates the existing:

    scripts/mix_data.py
    scripts/mix_impulse.py

Mixture types:

    clean
    normal_noise
    two_normal_noises
    impulse
    normal_plus_impulse
    two_normal_plus_impulse

All speech/noise/impulse sources are split-aware.
"""

from pathlib import Path

import json
import random

import librosa
import numpy as np
import soundfile as sf

from scripts.mix_data import (
    load_split_file,
    load_jsonl,
    resolve_dataset_path,
    calculate_rms,
    calculate_snr_db,
    scale_noise_to_snr,
    NORMAL_NOISE_CATEGORIES,
    SNR_CHOICES,
)

from scripts.mix_impulse import (
    add_impulse_arrays,
    IMPULSE_GAINS,
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

CLIP_DURATION_SEC = 5.0

CLIP_SAMPLES = int(
    SAMPLE_RATE * CLIP_DURATION_SEC
)

OUTPUT_ROOT = Path(
    "data/mixtures"
)

METADATA_PATH = Path(
    "data/metadata/samples.jsonl"
)

NOISE_METADATA_PATH = Path(
    "data/metadata/noise_metadata.jsonl"
)


# ============================================================
# MIXTURE DISTRIBUTION
# ============================================================

MIXTURE_DISTRIBUTION = [
    ("clean", 0.10),
    ("normal_noise", 0.20),
    ("two_normal_noises", 0.15),
    ("impulse", 0.10),
    ("normal_plus_impulse", 0.25),
    ("two_normal_plus_impulse", 0.20),
]


# ============================================================
# SAMPLE ID
# ============================================================

def get_next_sample_id():
    """
    Find the next available IG_XXXXXX ID.
    """

    max_id = 0

    if not METADATA_PATH.exists():
        return 1

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

            if not sample_id.startswith("IG_"):
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
    Load audio as mono 16 kHz float32.
    """

    audio, _ = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.size == 0:
        raise ValueError(
            f"Audio is empty: {path}"
        )

    if not np.isfinite(audio).all():
        raise ValueError(
            f"Audio contains NaN/Inf: {path}"
        )

    return audio


def prepare_clean_audio(audio):
    """
    Make speech exactly 5 seconds.

    Longer:
        random crop

    Shorter:
        zero pad
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if len(audio) > CLIP_SAMPLES:

        max_start = (
            len(audio)
            - CLIP_SAMPLES
        )

        start = random.randint(
            0,
            max_start,
        )

        audio = audio[
            start:start + CLIP_SAMPLES
        ]

    elif len(audio) < CLIP_SAMPLES:

        padded = np.zeros(
            CLIP_SAMPLES,
            dtype=np.float32,
        )

        padded[:len(audio)] = audio

        audio = padded

    return audio.astype(
        np.float32
    )


# ============================================================
# NOISE METADATA
# ============================================================

def build_noise_metadata_index():
    """
    Build:

        category -> records

    from noise_metadata.jsonl.
    """

    records = load_jsonl(
        NOISE_METADATA_PATH
    )

    index = {}

    for record in records:

        category = record.get(
            "category"
        )

        if category not in NORMAL_NOISE_CATEGORIES:
            continue

        file_value = record.get(
            "file"
        )

        if not file_value:
            continue

        index.setdefault(
            category,
            [],
        ).append(record)

    return index


def build_split_noise_index(
    split,
    noise_metadata_index,
):
    """
    Restrict noise metadata to the requested split.
    """

    noise_files = load_split_file(
        f"noise_{split}.txt"
    )

    noise_set = {
        str(path)
        for path in noise_files
    }

    result = {}

    for category, records in (
        noise_metadata_index.items()
    ):

        for record in records:

            path = resolve_dataset_path(
                record["file"]
            )

            if str(path) not in noise_set:
                continue

            result.setdefault(
                category,
                [],
            ).append(record)

    return result


def choose_normal_noise(
    split_noise_index,
):
    """
    Balanced category selection.
    """

    available = [
        category
        for category in NORMAL_NOISE_CATEGORIES
        if (
            category in split_noise_index
            and split_noise_index[category]
        )
    ]

    if not available:
        raise RuntimeError(
            "No normal noise available."
        )

    category = random.choice(
        available
    )

    record = random.choice(
        split_noise_index[category]
    )

    path = resolve_dataset_path(
        record["file"]
    )

    return (
        category,
        record,
        path,
    )


def build_gunshot_records(split):
    """
    Get gunshot recordings belonging
    to the requested split.
    """

    noise_files = load_split_file(
        f"noise_{split}.txt"
    )

    noise_set = {
        str(path)
        for path in noise_files
    }

    records = load_jsonl(
        NOISE_METADATA_PATH
    )

    gunshots = []

    for record in records:

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

        if str(path) not in noise_set:
            continue

        gunshots.append(
            record
        )

    if not gunshots:
        raise RuntimeError(
            f"No gunshot recordings found "
            f"in {split} split."
        )

    return gunshots


# ============================================================
# CLIPPING PROTECTION
# ============================================================

def protect_clipping(
    clean,
    noisy,
):
    """
    Protect clean/noisy signals from clipping.

    The signals are scaled TOGETHER so their
    relationship is preserved.

    Target peak:
        0.98

    No hard clipping is performed.
    """

    clean = np.asarray(
        clean,
        dtype=np.float32,
    ).copy()

    noisy = np.asarray(
        noisy,
        dtype=np.float32,
    ).copy()

    if clean.size == 0:
        raise ValueError(
            "Clean audio is empty."
        )

    if noisy.size == 0:
        raise ValueError(
            "Noisy audio is empty."
        )

    if not np.isfinite(clean).all():
        raise ValueError(
            "Clean audio contains NaN/Inf."
        )

    if not np.isfinite(noisy).all():
        raise ValueError(
            "Noisy audio contains NaN/Inf."
        )

    clean_peak = float(
        np.max(
            np.abs(clean)
        )
    )

    noisy_peak = float(
        np.max(
            np.abs(noisy)
        )
    )

    peak = max(
        clean_peak,
        noisy_peak,
    )

    target_peak = 0.98

    if peak > target_peak:

        scale = (
            target_peak
            / peak
        )

        clean *= scale
        noisy *= scale

    else:

        scale = 1.0

    # --------------------------------------------------------
    # Final numerical verification
    # --------------------------------------------------------

    clean_peak_after = float(
        np.max(
            np.abs(clean)
        )
    )

    noisy_peak_after = float(
        np.max(
            np.abs(noisy)
        )
    )

    # Extremely small float32 rounding can theoretically
    # push the value a few ulps above 0.98. Apply one more
    # common scale if necessary.
    final_peak = max(
        clean_peak_after,
        noisy_peak_after,
    )

    if final_peak > target_peak:

        extra_scale = (
            target_peak
            / final_peak
        )

        clean *= extra_scale
        noisy *= extra_scale

        scale *= extra_scale

    clean = clean.astype(
        np.float32
    )

    noisy = noisy.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Final float32 verification
    # --------------------------------------------------------

    final_clean_peak = float(
        np.max(
            np.abs(clean)
        )
    )

    final_noisy_peak = float(
        np.max(
            np.abs(noisy)
        )
    )

    if final_clean_peak > 0.981:
        raise RuntimeError(
            "Clean clipping protection failed: "
            f"peak={final_clean_peak:.9f}"
        )

    if final_noisy_peak > 0.981:
        raise RuntimeError(
            "Noisy clipping protection failed: "
            f"peak={final_noisy_peak:.9f}"
        )

    return (
        clean,
        noisy,
        float(scale),
    )


# ============================================================
# TWO-NOISE MIXING
# ============================================================

def add_two_normal_noises(
    clean,
    noise1,
    noise2,
    target_snr,
):
    """
    Add two independent continuous noises.

    Both noises are scaled relative to
    the ORIGINAL clean speech.

    Each noise receives half of the
    requested total noise power.
    """

    clean_rms = calculate_rms(
        clean
    )

    total_target_power = (
        clean_rms ** 2
        / (
            10.0
            ** (
                target_snr
                / 10.0
            )
        )
    )

    individual_target_power = (
        total_target_power
        / 2.0
    )

    individual_target_snr = (
        10.0
        * np.log10(
            (
                clean_rms
                ** 2
            )
            / individual_target_power
        )
    )

    scaled1 = scale_noise_to_snr(
        clean,
        noise1,
        individual_target_snr,
    )

    scaled2 = scale_noise_to_snr(
        clean,
        noise2,
        individual_target_snr,
    )

    noisy = (
        clean
        + scaled1
        + scaled2
    )

    actual_noise = (
        scaled1
        + scaled2
    )

    actual_snr = calculate_snr_db(
        clean,
        actual_noise,
    )

    return (
        noisy.astype(np.float32),
        scaled1.astype(np.float32),
        scaled2.astype(np.float32),
        float(actual_snr),
    )


# ============================================================
# SINGLE SAMPLE GENERATION
# ============================================================

def generate_sample(
    split,
    mixture_type,
    speech_path,
    split_noise_index,
    gunshot_records,
):
    """
    Generate one complete 5-second mixture.
    """

    # --------------------------------------------------------
    # Clean speech
    # --------------------------------------------------------

    clean = prepare_clean_audio(
        load_audio(
            speech_path
        )
    )

    # --------------------------------------------------------
    # Clean sample
    # --------------------------------------------------------

    if mixture_type == "clean":

        noisy = clean.copy()

        return (
            clean,
            noisy,
            {
                "target_snr_db": None,
                "actual_snr_db": None,
                "noise_components": [],
                "impulse": None,
                "clipping_scale": 1.0,
            },
        )

    # --------------------------------------------------------
    # Target SNR
    # --------------------------------------------------------

    target_snr = random.choice(
        SNR_CHOICES
    )

    noise_components = []

    impulse_metadata = None

    # ========================================================
    # ONE NORMAL NOISE
    # ========================================================

    if mixture_type == "normal_noise":

        category, record, path = (
            choose_normal_noise(
                split_noise_index
            )
        )

        noise = load_audio(
            path
        )

        scaled_noise = (
            scale_noise_to_snr(
                clean,
                noise,
                target_snr,
            )
        )

        noisy = (
            clean
            + scaled_noise
        )

        noise_components.append(
            {
                "path": str(path),
                "source": record.get(
                    "source"
                ),
                "category": category,
                "type": record.get(
                    "type"
                ),
                "target_snr_db": float(
                    target_snr
                ),
            }
        )

    # ========================================================
    # TWO NORMAL NOISES
    # ========================================================

    elif mixture_type == "two_normal_noises":

        category1, record1, path1 = (
            choose_normal_noise(
                split_noise_index
            )
        )

        category2, record2, path2 = (
            choose_normal_noise(
                split_noise_index
            )
        )

        noise1 = load_audio(
            path1
        )

        noise2 = load_audio(
            path2
        )

        (
            noisy,
            scaled1,
            scaled2,
            actual_snr,
        ) = add_two_normal_noises(
            clean,
            noise1,
            noise2,
            target_snr,
        )

        noise_components.extend(
            [
                {
                    "path": str(path1),
                    "source": record1.get(
                        "source"
                    ),
                    "category": category1,
                    "type": record1.get(
                        "type"
                    ),
                },
                {
                    "path": str(path2),
                    "source": record2.get(
                        "source"
                    ),
                    "category": category2,
                    "type": record2.get(
                        "type"
                    ),
                },
            ]
        )

    # ========================================================
    # IMPULSE ONLY
    # ========================================================

    elif mixture_type == "impulse":

        gunshot = random.choice(
            gunshot_records
        )

        impulse_path = (
            resolve_dataset_path(
                gunshot["file"]
            )
        )

        gain = random.choice(
            IMPULSE_GAINS
        )

        (
            clean_after,
            noisy,
            impulse_info,
        ) = add_impulse_arrays(
            clean,
            load_audio(
                impulse_path
            ),
            impulse_gain=gain,
        )

        clean = clean_after

        impulse_metadata = {
            "path": str(
                impulse_path
            ),
            "source": gunshot.get(
                "source"
            ),
            "category": "gunshot",
            "type": "impulsive",
            **impulse_info,
        }

    # ========================================================
    # NORMAL + IMPULSE
    # ========================================================

    elif mixture_type == "normal_plus_impulse":

        category, record, path = (
            choose_normal_noise(
                split_noise_index
            )
        )

        noise = load_audio(
            path
        )

        scaled_noise = (
            scale_noise_to_snr(
                clean,
                noise,
                target_snr,
            )
        )

        base_noisy = (
            clean
            + scaled_noise
        )

        gunshot = random.choice(
            gunshot_records
        )

        impulse_path = (
            resolve_dataset_path(
                gunshot["file"]
            )
        )

        gain = random.choice(
            IMPULSE_GAINS
        )

        (
            _,
            noisy,
            impulse_info,
        ) = add_impulse_arrays(
            base_noisy,
            load_audio(
                impulse_path
            ),
            impulse_gain=gain,
        )

        impulse_metadata = {
            "path": str(
                impulse_path
            ),
            "source": gunshot.get(
                "source"
            ),
            "category": "gunshot",
            "type": "impulsive",
            **impulse_info,
        }

        noise_components.append(
            {
                "path": str(path),
                "source": record.get(
                    "source"
                ),
                "category": category,
                "type": record.get(
                    "type"
                ),
                "target_snr_db": float(
                    target_snr
                ),
            }
        )

    # ========================================================
    # TWO NORMAL + IMPULSE
    # ========================================================

    elif mixture_type == "two_normal_plus_impulse":

        category1, record1, path1 = (
            choose_normal_noise(
                split_noise_index
            )
        )

        category2, record2, path2 = (
            choose_normal_noise(
                split_noise_index
            )
        )

        noise1 = load_audio(
            path1
        )

        noise2 = load_audio(
            path2
        )

        (
            base_noisy,
            scaled1,
            scaled2,
            actual_snr,
        ) = add_two_normal_noises(
            clean,
            noise1,
            noise2,
            target_snr,
        )

        gunshot = random.choice(
            gunshot_records
        )

        impulse_path = (
            resolve_dataset_path(
                gunshot["file"]
            )
        )

        gain = random.choice(
            IMPULSE_GAINS
        )

        (
            _,
            noisy,
            impulse_info,
        ) = add_impulse_arrays(
            base_noisy,
            load_audio(
                impulse_path
            ),
            impulse_gain=gain,
        )

        impulse_metadata = {
            "path": str(
                impulse_path
            ),
            "source": gunshot.get(
                "source"
            ),
            "category": "gunshot",
            "type": "impulsive",
            **impulse_info,
        }

        noise_components.extend(
            [
                {
                    "path": str(path1),
                    "source": record1.get(
                        "source"
                    ),
                    "category": category1,
                    "type": record1.get(
                        "type"
                    ),
                },
                {
                    "path": str(path2),
                    "source": record2.get(
                        "source"
                    ),
                    "category": category2,
                    "type": record2.get(
                        "type"
                    ),
                },
            ]
        )

    else:

        raise ValueError(
            f"Unknown mixture type: "
            f"{mixture_type}"
        )

    # ========================================================
    # FINAL COMMON CLIPPING PROTECTION
    # ========================================================

    clean, noisy, clipping_scale = (
        protect_clipping(
            clean,
            noisy,
        )
    )

    # ========================================================
    # ACTUAL SNR
    # ========================================================

    if mixture_type in {
        "normal_noise",
        "two_normal_noises",
        "normal_plus_impulse",
        "two_normal_plus_impulse",
    }:

        total_difference = (
            noisy - clean
        )

        if mixture_type == "normal_noise":

            actual_snr = calculate_snr_db(
                clean,
                total_difference,
            )

        elif mixture_type == "two_normal_noises":

            actual_snr = calculate_snr_db(
                clean,
                total_difference,
            )

        else:

            # For impulse-containing mixtures,
            # the impulse changes the overall noise
            # energy, so we do not label this as the
            # original continuous-noise SNR.

            actual_snr = None

    else:

        actual_snr = None

    # ========================================================
    # FINAL RETURN
    # ========================================================

    return (
        clean,
        noisy,
        {
            "target_snr_db": float(
                target_snr
            ),
            "actual_snr_db": (
                float(actual_snr)
                if actual_snr is not None
                else None
            ),
            "noise_components": (
                noise_components
            ),
            "impulse": (
                impulse_metadata
            ),
            "clipping_scale": (
                clipping_scale
            ),
        },
    )


# ============================================================
# PRODUCTION GENERATOR
# ============================================================

def choose_mixture_type():
    """
    Randomly select a mixture type according
    to the production distribution.
    """

    value = random.random()

    cumulative = 0.0

    for mixture_type, probability in (
        MIXTURE_DISTRIBUTION
    ):

        cumulative += probability

        if value < cumulative:
            return mixture_type

    return MIXTURE_DISTRIBUTION[-1][0]


def generate_production_dataset(
    split,
    num_samples,
    seed=None,
):
    """
    Generate stored production mixtures.
    """

    if split not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "Invalid split."
        )

    if num_samples <= 0:
        raise ValueError(
            "num_samples must be > 0"
        )

    if seed is not None:

        random.seed(
            seed
        )

        np.random.seed(
            seed
        )

    print()
    print("=" * 60)

    print(
        f"Generating {split}: "
        f"{num_samples} samples"
    )

    print("=" * 60)

    # ========================================================
    # LOAD SPEECH
    # ========================================================

    speech_files = load_split_file(
        f"speech_{split}.txt"
    )

    # ========================================================
    # NOISE METADATA
    # ========================================================

    noise_metadata_index = (
        build_noise_metadata_index()
    )

    split_noise_index = (
        build_split_noise_index(
            split,
            noise_metadata_index,
        )
    )

    # ========================================================
    # GUNSHOTS
    # ========================================================

    gunshot_records = (
        build_gunshot_records(
            split
        )
    )

    print(
        f"Speech available : "
        f"{len(speech_files)}"
    )

    print(
        "Normal noise:"
    )

    for category in sorted(
        split_noise_index
    ):

        print(
            f"  {category:<15} "
            f"{len(split_noise_index[category])}"
        )

    print(
        f"Gunshots available: "
        f"{len(gunshot_records)}"
    )

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

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

    # ========================================================
    # IDS
    # ========================================================

    next_id = get_next_sample_id()

    generated_records = []

    counts = {}

    # ========================================================
    # GENERATE
    # ========================================================

    for index in range(
        num_samples
    ):

        sample_id = (
            f"IG_{next_id:06d}"
        )

        next_id += 1

        mixture_type = (
            choose_mixture_type()
        )

        speech_path = random.choice(
            speech_files
        )

        (
            clean,
            noisy,
            info,
        ) = generate_sample(
            split=split,
            mixture_type=mixture_type,
            speech_path=speech_path,
            split_noise_index=split_noise_index,
            gunshot_records=gunshot_records,
        )

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        clean = np.asarray(
            clean,
            dtype=np.float32,
        )

        noisy = np.asarray(
            noisy,
            dtype=np.float32,
        )

        if len(clean) != CLIP_SAMPLES:

            raise RuntimeError(
                f"Clean length error: "
                f"{sample_id}"
            )

        if len(noisy) != CLIP_SAMPLES:

            raise RuntimeError(
                f"Noisy length error: "
                f"{sample_id}"
            )

        if not np.isfinite(clean).all():

            raise RuntimeError(
                f"Clean contains NaN/Inf: "
                f"{sample_id}"
            )

        if not np.isfinite(noisy).all():

            raise RuntimeError(
                f"Noisy contains NaN/Inf: "
                f"{sample_id}"
            )

        # ----------------------------------------------------
        # Final safety normalization.
        #
        # This is intentionally done immediately before
        # writing the WAV files.
        # ----------------------------------------------------

        final_peak = max(
            float(
                np.max(
                    np.abs(clean)
                )
            ),
            float(
                np.max(
                    np.abs(noisy)
                )
            ),
        )

        final_target_peak = 0.98

        if final_peak > 0.981:

            final_scale = (
                final_target_peak
                / final_peak
            )

            clean *= final_scale
            noisy *= final_scale

            info["clipping_scale"] = (
                float(
                    info["clipping_scale"]
                )
                * float(final_scale)
            )

        # ----------------------------------------------------
        # Convert once more to float32 before checking.
        # ----------------------------------------------------

        clean = clean.astype(
            np.float32
        )

        noisy = noisy.astype(
            np.float32
        )

        noisy_peak = float(
            np.max(
                np.abs(noisy)
            )
        )

        clean_peak = float(
            np.max(
                np.abs(clean)
            )
        )

        # ----------------------------------------------------
        # Allow only a tiny float32 rounding tolerance.
        # Anything substantially above 0.98 would indicate
        # a real protection failure.
        # ----------------------------------------------------

        if clean_peak > 0.981:

            raise RuntimeError(
                f"Clean clipping: "
                f"{sample_id} "
                f"(peak={clean_peak:.9f})"
            )

        if noisy_peak > 0.981:

            raise RuntimeError(
                f"Noisy clipping: "
                f"{sample_id} "
                f"(peak={noisy_peak:.9f})"
            )

        # ====================================================
        # PATHS
        # ====================================================

        clean_path = (
            clean_dir
            / f"{sample_id}.wav"
        )

        noisy_path = (
            noisy_dir
            / f"{sample_id}.wav"
        )

        # ====================================================
        # SAVE AUDIO
        # ====================================================

        sf.write(
            str(clean_path),
            clean,
            SAMPLE_RATE,
            subtype="PCM_16",
        )

        sf.write(
            str(noisy_path),
            noisy,
            SAMPLE_RATE,
            subtype="PCM_16",
        )

        # ====================================================
        # METADATA
        # ====================================================

        record = {
            "sample_id": sample_id,

            "split": split,

            "clean_path": str(
                clean_path
            ),

            "noisy_path": str(
                noisy_path
            ),

            "speech": {
                "path": str(
                    speech_path
                ),
                "source": "LibriSpeech",
            },

            "mixture_type": (
                mixture_type
            ),

            "noise_components": (
                info[
                    "noise_components"
                ]
            ),

            "impulse": (
                info[
                    "impulse"
                ]
            ),

            "target_snr_db": (
                info[
                    "target_snr_db"
                ]
            ),

            "actual_snr_db": (
                info[
                    "actual_snr_db"
                ]
            ),

            "duration_sec": (
                CLIP_DURATION_SEC
            ),

            "sample_rate": (
                SAMPLE_RATE
            ),

            "channels": 1,

            "clipping_scale": (
                info[
                    "clipping_scale"
                ]
            ),

            "generator": (
                "mix_combined_v3.3"
            ),
        }

        generated_records.append(
            record
        )

        counts[
            mixture_type
        ] = (
            counts.get(
                mixture_type,
                0,
            )
            + 1
        )

        # ====================================================
        # METADATA APPEND
        # ====================================================

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

        # ====================================================
        # PROGRESS
        # ====================================================

        print(
            f"[{index + 1:>5}/"
            f"{num_samples}] "
            f"{sample_id} | "
            f"{mixture_type:<24}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print(
        "Generation complete."
    )

    print(
        f"Split: {split}"
    )

    print(
        f"Samples: {num_samples}"
    )

    print(
        "Mixture types:"
    )

    for mixture_type in sorted(
        counts
    ):

        print(
            f"  {mixture_type:<24}"
            f"{counts[mixture_type]}"
        )

    return generated_records


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate Impulse Guard "
            "production mixtures."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
        ],
        required=True,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    generate_production_dataset(
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
    )