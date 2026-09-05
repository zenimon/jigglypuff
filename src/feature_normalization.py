import numpy as np
import json
import os


def calculate_statistics(features):
    """
    Calculate mean and standard deviation
    for each feature dimension.

    Input:
        features: shape (time_frames, 44)

    Output:
        mean: shape (44,)
        std: shape (44,)
    """

    mean = np.mean(
        features,
        axis=0
    )

    std = np.std(
        features,
        axis=0
    )

    # Prevent division by zero
    std = np.maximum(
        std,
        1e-8
    )

    return (
        mean.astype(np.float32),
        std.astype(np.float32)
    )


def normalize_features(features, mean, std):
    """
    Normalize features using:

        normalized = (features - mean) / std
    """

    normalized = (
        features - mean
    ) / std

    return normalized.astype(
        np.float32
    )


def save_statistics(mean, std, output_path):
    """
    Save normalization statistics to JSON.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    data = {
        "feature_count": len(mean),
        "mean": mean.tolist(),
        "std": std.tolist()
    }

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


def load_statistics(input_path):
    """
    Load normalization statistics from JSON.

    Returns:
        mean: shape (44,)
        std: shape (44,)
    """

    with open(
        input_path,
        "r"
    ) as f:

        data = json.load(f)

    mean = np.array(
        data["mean"],
        dtype=np.float32
    )

    std = np.array(
        data["std"],
        dtype=np.float32
    )

    return mean, std