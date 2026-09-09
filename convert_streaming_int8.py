import numpy as np
import tensorflow as tf
from pathlib import Path

SRC = "models/gru_subband/streaming_gru_subband.keras"
CALIB = "models/gru_subband/int8_calibration.npz"
OUT = "models/gru_subband/streaming_gru_subband_int8.tflite"

model = tf.keras.models.load_model(SRC)

data = np.load(CALIB)
features = data["features"].astype(np.float32)
hidden_states = data["hidden_states"].astype(np.float32)

def representative_dataset():
    for x, h in zip(features, hidden_states):
        yield [
            x.reshape(1, 44),
            h.reshape(1, 64),
        ]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

Path(OUT).write_bytes(tflite_model)

print("Saved:", OUT)
print("Size:", len(tflite_model), "bytes")
