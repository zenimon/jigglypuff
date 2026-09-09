import tensorflow as tf
from pathlib import Path

SRC = "models/gru_subband/streaming_gru_subband.keras"
OUT = "models/gru_subband/streaming_gru_subband_float32.tflite"

model = tf.keras.models.load_model(SRC)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Keep only standard TFLite operators.
# SELECT_TF_OPS is intentionally NOT used because ESP32 TFLite Micro
# cannot depend on TensorFlow Flex operators.
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]

tflite_model = converter.convert()

Path(OUT).write_bytes(tflite_model)

print("Saved:", OUT)
print("Size:", len(tflite_model), "bytes")
