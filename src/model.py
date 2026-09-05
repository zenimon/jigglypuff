import tensorflow as tf

from src.config import (
    GRU_HIDDEN_SIZE,
    NUM_SUBBANDS
)


INPUT_FEATURES = NUM_SUBBANDS * 2
OUTPUT_FEATURES = NUM_SUBBANDS * 2


def create_gru_model():
    """
    Create the Impulse Guard sub-band GRU.

    Input:
        44 real-valued features per frame

    Output:
        44 real-valued values per frame
        22 real mask values
        22 imaginary mask values
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(None, INPUT_FEATURES)
        ),

        tf.keras.layers.GRU(
            GRU_HIDDEN_SIZE,
            return_sequences=True
        ),

        tf.keras.layers.Dense(
            OUTPUT_FEATURES
        )
    ])

    return model