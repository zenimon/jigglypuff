"""
Stored mixture dataset loader for Impulse Guard.

Reads pre-generated clean/noisy WAV pairs from:

    data/mixtures/
        train/
            clean/
            noisy/
        validation/
            clean/
            noisy/
        test/
            clean/
            noisy/

Pipeline:

    clean.wav + noisy.wav
        -> STFT
        -> noisy sub-band features
        -> clean/noisy sub-band complex target
        -> feature normalization

Returns:
    features: (time_frames, 44)
    target:   (time_frames, 44)
"""

from pathlib import Path
import json

import librosa
import numpy as np
import tensorflow as tf

from src.config import SAMPLE_RATE
from src.stft import compute_stft
from src.subbands import extract_subband_features
from src.target_mask import create_subband_complex_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MIXTURE_ROOT = PROJECT_ROOT / "data" / "mixtures"

NORMALIZATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "feature_normalization.json"
)

TARGET_DURATION_SECONDS = 5
TARGET_SAMPLES = SAMPLE_RATE * TARGET_DURATION_SECONDS

FEATURE_DIM = 44
TARGET_DIM = 44


# ============================================================
# STORED MIXTURE PATHS
# ============================================================

def get_split_directories(split):
    """
    Return clean/noisy directories for a dataset split.
    """

    valid_splits = {"train", "validation", "test"}

    if split not in valid_splits:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Expected one of: {sorted(valid_splits)}"
        )

    split_root = MIXTURE_ROOT / split

    clean_dir = split_root / "clean"
    noisy_dir = split_root / "noisy"

    if not clean_dir.exists():
        raise FileNotFoundError(
            f"Clean mixture directory does not exist: {clean_dir}"
        )

    if not noisy_dir.exists():
        raise FileNotFoundError(
            f"Noisy mixture directory does not exist: {noisy_dir}"
        )

    return clean_dir, noisy_dir


def get_mixture_pairs(split):
    """
    Find matching clean/noisy WAV pairs.

    Returns:
        list of tuples:
            [(clean_path, noisy_path), ...]
    """

    clean_dir, noisy_dir = get_split_directories(split)

    clean_files = {
        path.stem: path
        for path in clean_dir.glob("*.wav")
    }

    noisy_files = {
        path.stem: path
        for path in noisy_dir.glob("*.wav")
    }

    clean_ids = set(clean_files)
    noisy_ids = set(noisy_files)

    missing_noisy = clean_ids - noisy_ids
    missing_clean = noisy_ids - clean_ids

    if missing_noisy:
        raise RuntimeError(
            f"{len(missing_noisy)} clean mixtures have no noisy pair. "
            f"Examples: {sorted(missing_noisy)[:5]}"
        )

    if missing_clean:
        raise RuntimeError(
            f"{len(missing_clean)} noisy mixtures have no clean pair. "
            f"Examples: {sorted(missing_clean)[:5]}"
        )

    sample_ids = sorted(clean_ids)

    pairs = [
        (clean_files[sample_id], noisy_files[sample_id])
        for sample_id in sample_ids
    ]

    return pairs


# ============================================================
# AUDIO
# ============================================================

def load_audio(path):
    """
    Load mono audio at the project sample rate.
    """

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    audio = audio.astype(np.float32)

    if len(audio) == 0:
        raise ValueError(f"Audio file is empty: {path}")

    if not np.isfinite(audio).all():
        raise ValueError(
            f"Non-finite audio detected: {path}"
        )

    return audio


def validate_audio_pair(clean, noisy, clean_path, noisy_path):
    """
    Validate a clean/noisy pair.
    """

    if len(clean) != len(noisy):
        raise ValueError(
            "Clean/noisy length mismatch:\n"
            f"  clean: {clean_path} -> {len(clean)} samples\n"
            f"  noisy: {noisy_path} -> {len(noisy)} samples"
        )

    if len(clean) != TARGET_SAMPLES:
        raise ValueError(
            f"Expected {TARGET_SAMPLES} samples "
            f"({TARGET_DURATION_SECONDS}s), "
            f"but got {len(clean)}:\n"
            f"  {clean_path}"
        )

    if not np.isfinite(clean).all():
        raise ValueError(
            f"Clean audio contains NaN/Inf: {clean_path}"
        )

    if not np.isfinite(noisy).all():
        raise ValueError(
            f"Noisy audio contains NaN/Inf: {noisy_path}"
        )


# ============================================================
# NORMALIZATION
# ============================================================

def load_normalization_stats():
    """
    Load production feature normalization statistics.

    These statistics were calculated from TRAINING SPEECH
    features only and must be reused for train/validation/test.
    """

    if not NORMALIZATION_FILE.exists():
        raise FileNotFoundError(
            f"Normalization file not found: "
            f"{NORMALIZATION_FILE}"
        )

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        stats = json.load(f)

    mean = np.asarray(
        stats["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        stats["std"],
        dtype=np.float32,
    )

    if mean.shape != (FEATURE_DIM,):
        raise ValueError(
            f"Expected mean shape ({FEATURE_DIM},), "
            f"got {mean.shape}"
        )

    if std.shape != (FEATURE_DIM,):
        raise ValueError(
            f"Expected std shape ({FEATURE_DIM},), "
            f"got {std.shape}"
        )

    if not np.isfinite(mean).all():
        raise ValueError("Normalization mean contains NaN/Inf.")

    if not np.isfinite(std).all():
        raise ValueError("Normalization std contains NaN/Inf.")

    if np.any(std <= 0):
        raise ValueError(
            "Normalization std contains zero/negative values."
        )

    return mean, std


def normalize_features(features, mean, std):
    """
    Apply production feature normalization.
    """

    features = (
        features - mean
    ) / (std + 1e-8)

    features = features.astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError(
            "Normalized features contain NaN/Inf."
        )

    return features


# ============================================================
# ONE STORED SAMPLE
# ============================================================

def load_sample(
    clean_path,
    noisy_path,
    mean=None,
    std=None,
):
    """
    Load one stored mixture pair and convert it into
    model-ready features and target.

    Returns:

        features:
            (time_frames, 44)

        target:
            (time_frames, 44)
    """

    clean = load_audio(clean_path)
    noisy = load_audio(noisy_path)

    validate_audio_pair(
        clean,
        noisy,
        clean_path,
        noisy_path,
    )

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    clean_stft = compute_stft(clean)
    noisy_stft = compute_stft(noisy)

    if clean_stft.shape != noisy_stft.shape:
        raise ValueError(
            "STFT shape mismatch:\n"
            f"  clean: {clean_stft.shape}\n"
            f"  noisy: {noisy_stft.shape}"
        )

    # --------------------------------------------------------
    # INPUT FEATURES
    #
    # Only NOISY audio is used as model input.
    # --------------------------------------------------------

    features = extract_subband_features(
        noisy_stft
    )

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2-D features, got "
            f"{features.shape}"
        )

    if features.shape[-1] != FEATURE_DIM:
        raise ValueError(
            f"Expected {FEATURE_DIM} features, "
            f"got {features.shape[-1]}"
        )

    # --------------------------------------------------------
    # TARGET
    #
    # Clean + noisy STFT are used to construct the
    # sub-band complex mask target.
    #
    # Target shape:
    #   (time_frames, 22) complex
    #
    # Split into:
    #   22 real + 22 imaginary = 44
    # --------------------------------------------------------

    target_complex = create_subband_complex_target(
        clean_stft,
        noisy_stft,
    )

    target = np.concatenate(
        [
            np.real(target_complex),
            np.imag(target_complex),
        ],
        axis=-1,
    ).astype(np.float32)

    if target.ndim != 2:
        raise ValueError(
            f"Expected 2-D target, got {target.shape}"
        )

    if target.shape[-1] != TARGET_DIM:
        raise ValueError(
            f"Expected {TARGET_DIM} target values, "
            f"got {target.shape[-1]}"
        )

    # --------------------------------------------------------
    # Make sure time dimensions match
    # --------------------------------------------------------

    if features.shape[0] != target.shape[0]:
        raise ValueError(
            "Feature/target time mismatch:\n"
            f"  features: {features.shape}\n"
            f"  target:   {target.shape}"
        )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    if mean is None or std is None:
        mean, std = load_normalization_stats()

    features = normalize_features(
        features,
        mean,
        std,
    )

    # --------------------------------------------------------
    # Final finite check
    # --------------------------------------------------------

    if not np.isfinite(features).all():
        raise ValueError(
            f"Non-finite features from:\n"
            f"  {noisy_path}"
        )

    if not np.isfinite(target).all():
        raise ValueError(
            f"Non-finite target from:\n"
            f"  {clean_path}\n"
            f"  {noisy_path}"
        )

    return features, target


# ============================================================
# KERAS SEQUENCE
# ============================================================

class ImpulseGuardSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence for stored clean/noisy mixtures.

    No mixtures are generated here.

    The WAV files are already stored on disk.
    """

    def __init__(
        self,
        split,
        batch_size=8,
        shuffle=True,
    ):
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.pairs = get_mixture_pairs(split)

        if not self.pairs:
            raise RuntimeError(
                f"No mixture pairs found for split '{split}'."
            )

        self.mean, self.std = load_normalization_stats()

        self.indices = np.arange(
            len(self.pairs),
            dtype=np.int64,
        )

        self.on_epoch_end()

    def __len__(self):
        """
        Number of batches per epoch.
        """

        return int(
            np.ceil(
                len(self.pairs) / self.batch_size
            )
        )

    def __getitem__(self, index):
        """
        Load one batch.
        """

        start = index * self.batch_size
        end = min(
            start + self.batch_size,
            len(self.pairs),
        )

        batch_indices = self.indices[start:end]

        batch_features = []
        batch_targets = []

        for pair_index in batch_indices:

            clean_path, noisy_path = self.pairs[pair_index]

            features, target = load_sample(
                clean_path,
                noisy_path,
                self.mean,
                self.std,
            )

            batch_features.append(features)
            batch_targets.append(target)

        return (
            np.asarray(
                batch_features,
                dtype=np.float32,
            ),
            np.asarray(
                batch_targets,
                dtype=np.float32,
            ),
        )

    def on_epoch_end(self):
        """
        Shuffle mixture order between epochs.
        """

        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================
# FACTORY
# ============================================================

def create_training_sequence(
    batch_size=8,
    shuffle=True,
):
    """
    Create the stored training dataset.
    """

    return ImpulseGuardSequence(
        split="train",
        batch_size=batch_size,
        shuffle=shuffle,
    )


def create_validation_sequence(
    batch_size=8,
):
    """
    Create the stored validation dataset.
    """

    return ImpulseGuardSequence(
        split="validation",
        batch_size=batch_size,
        shuffle=False,
    )


def create_test_sequence(
    batch_size=8,
):
    """
    Create the stored test dataset.
    """

    return ImpulseGuardSequence(
        split="test",
        batch_size=batch_size,
        shuffle=False,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Impulse Guard - Stored Mixture Loader Test")
    print("=" * 60)

    print("\nLoading training mixture pairs...")

    pairs = get_mixture_pairs("train")

    print(f"Training pairs: {len(pairs)}")

    if not pairs:
        raise RuntimeError(
            "No training mixture pairs found."
        )

    clean_path, noisy_path = pairs[0]

    print("\nFirst pair:")
    print(f"Clean: {clean_path}")
    print(f"Noisy: {noisy_path}")

    print("\nLoading first sample...")

    mean, std = load_normalization_stats()

    features, target = load_sample(
        clean_path,
        noisy_path,
        mean,
        std,
    )

    print(f"Features shape: {features.shape}")
    print(f"Target shape:   {target.shape}")
    print(f"Features dtype: {features.dtype}")
    print(f"Target dtype:   {target.dtype}")
    print(
        f"Features finite: "
        f"{np.isfinite(features).all()}"
    )
    print(
        f"Target finite:   "
        f"{np.isfinite(target).all()}"
    )

    print("\nTesting Keras Sequence...")

    sequence = ImpulseGuardSequence(
        split="train",
        batch_size=2,
        shuffle=False,
    )

    batch_features, batch_targets = sequence[0]

    print(
        f"Batch features shape: "
        f"{batch_features.shape}"
    )

    print(
        f"Batch targets shape:  "
        f"{batch_targets.shape}"
    )

    print(
        f"Batch features finite: "
        f"{np.isfinite(batch_features).all()}"
    )

    print(
        f"Batch targets finite:   "
        f"{np.isfinite(batch_targets).all()}"
    )

    print("\nStored mixture loader test complete.")