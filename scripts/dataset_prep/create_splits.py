
import json
import random
import re
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# METADATA FILES
# =========================================================

SPEECH_METADATA = METADATA_DIR / "speech_metadata.jsonl"
NOISE_METADATA = METADATA_DIR / "noise_metadata.jsonl"


# =========================================================
# SETTINGS
# =========================================================

SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(
    TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0
) < 1e-9


# =========================================================
# LOAD JSONL
# =========================================================

def load_jsonl(file_path):

    records = []

    if not file_path.exists():
        print("\nERROR: Metadata file not found:")
        print(file_path)
        return records

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                records.append(json.loads(line))

    return records


# =========================================================
# BASIC SAMPLE SPLIT
# =========================================================

def split_records(records, rng):

    records = records.copy()
    rng.shuffle(records)

    total = len(records)

    train_end = int(total * TRAIN_RATIO)

    validation_end = int(
        total * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train = records[:train_end]
    validation = records[train_end:validation_end]
    test = records[validation_end:]

    return train, validation, test


# =========================================================
# GROUPED SPLIT
# =========================================================

def split_groups(records, group_function, rng):

    """
    Split records by group.

    Every record belonging to the same group stays
    in exactly one of train / validation / test.
    """

    groups = {}

    for record in records:

        group = group_function(record)

        groups.setdefault(group, []).append(record)

    group_items = list(groups.items())

    rng.shuffle(group_items)

    total_records = len(records)

    target_train = total_records * TRAIN_RATIO
    target_validation = total_records * VALIDATION_RATIO

    train = []
    validation = []
    test = []

    train_count = 0
    validation_count = 0

    for group, group_records in group_items:

        group_size = len(group_records)

        # Fill train first until approximately 70%.
        if train_count + group_size <= target_train:
            train.extend(group_records)
            train_count += group_size

        # Then validation until approximately 15%.
        elif validation_count + group_size <= target_validation:
            validation.extend(group_records)
            validation_count += group_size

        # Remaining groups go to test.
        else:
            test.extend(group_records)

    return train, validation, test


# =========================================================
# ENGLISH LIBRISPEECH GROUP
# =========================================================

def get_librispeech_speaker(record):

    """
    LibriSpeech path:

    .../librispeech_train-clean-100_wav/
        speaker/
            chapter/
                file.wav

    The speaker directory is therefore the leakage group.
    """

    path = Path(record["file"])

    parts = path.parts

    marker = "librispeech_train-clean-100_wav"

    if marker in parts:

        index = parts.index(marker)

        if index + 1 < len(parts):

            return parts[index + 1]

    return f"unknown_speaker::{record['file']}"


# =========================================================
# HINDI GROUP
# =========================================================

def get_hindi_group(record):

    """
    Hindi dataset currently has:

    hindi_speech/Audio/hindi_XXXXX.wav

    There is no reliable speaker/session identifier in the
    path or filename, so each recording is treated as its
    own group rather than inventing a speaker ID.
    """

    return f"hindi_file::{record['file']}"


# =========================================================
# URBANSOUND8K GROUP
# =========================================================

def get_urbansound_group(record):

    """
    UrbanSound8K filenames follow:

        recordingID-classID-occurrenceID-segmentID.wav

    Example:

        133797-6-0-0.wav
        133797-6-1-0.wav
        133797-6-2-0.wav

    All segments from recordingID 133797 must remain
    in the same split.
    """

    filename = Path(record["file"]).name

    match = re.match(r"^(\d+)-", filename)

    if match:

        recording_id = match.group(1)

        return f"urbansound8k::{recording_id}"

    # Fallback: keep an unrecognized file isolated.
    return f"urbansound8k_file::{filename}"


# =========================================================
# GENERIC SOURCE GROUP
# =========================================================

def get_source_group(record):

    """
    Generic grouping for datasets where the metadata source
    itself represents the recording collection.

    This prevents mixing source collections unnecessarily.
    """

    source = record.get("source", "unknown")

    return f"{source}::{record['file']}"


# =========================================================
# WRITE SPLIT FILE
# =========================================================

def write_split_file(filename, records):

    output_file = SPLITS_DIR / filename

    with open(output_file, "w", encoding="utf-8") as f:

        for record in records:
            f.write(record["file"] + "\n")

    return len(records)


# =========================================================
# LOAD DATA
# =========================================================

print("\n============================================")
print("         LOADING METADATA")
print("============================================")

speech_records = load_jsonl(SPEECH_METADATA)
noise_records = load_jsonl(NOISE_METADATA)

print("\nSpeech recordings:", len(speech_records))
print("Noise recordings:", len(noise_records))


if len(speech_records) == 0:

    print("\nERROR: No speech recordings found.")
    raise SystemExit(1)


if len(noise_records) == 0:

    print("\nERROR: No noise recordings found.")
    raise SystemExit(1)


rng = random.Random(SEED)


# =========================================================
# SPEECH SPLIT
# =========================================================

print("\n============================================")
print("             SPLITTING SPEECH")
print("============================================")

speech_train = []
speech_validation = []
speech_test = []


languages = sorted(
    set(
        record.get("language", "unknown")
        for record in speech_records
    )
)


for language in languages:

    language_records = [
        record
        for record in speech_records
        if record.get("language", "unknown") == language
    ]

    print(f"\nLanguage: {language}")
    print("  Total:", len(language_records))

    if language.lower() == "english":

        train, validation, test = split_groups(
            language_records,
            get_librispeech_speaker,
            rng
        )

        groups = {
            get_librispeech_speaker(record)
            for record in language_records
        }

        print("  Grouping: LibriSpeech speaker")
        print("  Speakers:", len(groups))

    elif language.lower() == "hindi":

        train, validation, test = split_groups(
            language_records,
            get_hindi_group,
            rng
        )

        print("  Grouping: individual recording")
        print("  Speaker metadata: unavailable")

    else:

        train, validation, test = split_records(
            language_records,
            rng
        )

        print("  Grouping: individual recording")

    speech_train.extend(train)
    speech_validation.extend(validation)
    speech_test.extend(test)

    print("  Train:", len(train))
    print("  Validation:", len(validation))
    print("  Test:", len(test))


# =========================================================
# NOISE SPLIT
# =========================================================

print("\n============================================")
print("              SPLITTING NOISE")
print("============================================")

noise_train = []
noise_validation = []
noise_test = []


categories = sorted(
    set(
        record.get("category", "unknown")
        for record in noise_records
    )
)


for category in categories:

    category_records = [
        record
        for record in noise_records
        if record.get("category", "unknown") == category
    ]

    source_values = {
        record.get("source", "unknown")
        for record in category_records
    }

    print(f"\nCategory: {category}")
    print("  Total:", len(category_records))
    print("  Sources:", ", ".join(sorted(source_values)))


    # -----------------------------------------------------
    # UrbanSound8K
    # -----------------------------------------------------

    if all(
        record.get("source") == "UrbanSound8K"
        for record in category_records
    ):

        train, validation, test = split_groups(
            category_records,
            get_urbansound_group,
            rng
        )

        print("  Grouping: UrbanSound8K recording ID")


    # -----------------------------------------------------
    # Glasgow / Drone
    #
    # Preserve the existing physical-drone split.
    # -----------------------------------------------------

    elif (
        category.lower() == "drone"
        and all(
            "University_of_Glasgow_Drone_Authentication"
            in record.get("source", "")
            for record in category_records
        )
    ):

        print("  Grouping: existing Glasgow drone split")

        drone_train_file = SPLITS_DIR / "drone_train.txt"
        drone_validation_file = SPLITS_DIR / "drone_validation.txt"
        drone_test_file = SPLITS_DIR / "drone_test.txt"

        if not (
            drone_train_file.exists()
            and drone_validation_file.exists()
            and drone_test_file.exists()
        ):
            raise RuntimeError(
                "Existing Glasgow drone split files are missing."
            )

        def load_split_paths(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return {
                    line.strip()
                    for line in f
                    if line.strip()
                }

        drone_train_paths = load_split_paths(drone_train_file)
        drone_validation_paths = load_split_paths(drone_validation_file)
        drone_test_paths = load_split_paths(drone_test_file)

        train = []
        validation = []
        test = []

        for record in category_records:
            record_path = record["file"]

            if record_path in drone_train_paths:
                train.append(record)

            elif record_path in drone_validation_paths:
                validation.append(record)

            elif record_path in drone_test_paths:
                test.append(record)

            else:
                raise RuntimeError(
                    f"Glasgow drone record missing from existing split: "
                    f"{record_path}"
                )

        print("  Existing split preserved:")
        print("    Train:", len(train))
        print("    Validation:", len(validation))
        print("    Test:", len(test))

    # -----------------------------------------------------
    # Other sources
    # -----------------------------------------------------

    else:

        train, validation, test = split_records(
            category_records,
            rng
        )

        print("  Grouping: individual recording")


    noise_train.extend(train)
    noise_validation.extend(validation)
    noise_test.extend(test)

    print("  Train:", len(train))
    print("  Validation:", len(validation))
    print("  Test:", len(test))


# =========================================================
# WRITE SPEECH SPLITS
# =========================================================

print("\n============================================")
print("          WRITING SPEECH SPLITS")
print("============================================")

write_split_file(
    "speech_train.txt",
    speech_train
)

write_split_file(
    "speech_validation.txt",
    speech_validation
)

write_split_file(
    "speech_test.txt",
    speech_test
)


# =========================================================
# WRITE NOISE SPLITS
# =========================================================

print("\n============================================")
print("           WRITING NOISE SPLITS")
print("============================================")

write_split_file(
    "noise_train.txt",
    noise_train
)

write_split_file(
    "noise_validation.txt",
    noise_validation
)

write_split_file(
    "noise_test.txt",
    noise_test
)


# =========================================================
# COMBINED SPLITS
# =========================================================

print("\n============================================")
print("          WRITING COMBINED SPLITS")
print("============================================")

train_combined = speech_train + noise_train
validation_combined = speech_validation + noise_validation
test_combined = speech_test + noise_test

rng.shuffle(train_combined)
rng.shuffle(validation_combined)
rng.shuffle(test_combined)

write_split_file(
    "train.txt",
    train_combined
)

write_split_file(
    "validation.txt",
    validation_combined
)

write_split_file(
    "test.txt",
    test_combined
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n============================================")
print("              FINAL SUMMARY")
print("============================================")

print("\nSPEECH")
print("  Train:", len(speech_train))
print("  Validation:", len(speech_validation))
print("  Test:", len(speech_test))
print("  Total:", len(speech_train) +
              len(speech_validation) +
              len(speech_test))

print("\nNOISE")
print("  Train:", len(noise_train))
print("  Validation:", len(noise_validation))
print("  Test:", len(noise_test))
print("  Total:", len(noise_train) +
              len(noise_validation) +
              len(noise_test))

print("\nCOMBINED")
print("  Train:", len(train_combined))
print("  Validation:", len(validation_combined))
print("  Test:", len(test_combined))
print("  Total:", len(train_combined) +
              len(validation_combined) +
              len(test_combined))

print("\n============================================")
print("              SPLIT COMPLETE")
print("============================================")
