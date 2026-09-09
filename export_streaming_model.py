import tensorflow as tf
from pathlib import Path

SRC = "models/gru_subband/best_gru_subband.keras"
OUT = "models/gru_subband/streaming_gru_subband.keras"

# Load frozen V1
base = tf.keras.models.load_model(SRC)

gru = base.get_layer("gru")
dense = base.get_layer("dense")

# Streaming inputs
x = tf.keras.Input(shape=(44,), dtype=tf.float32, name="features")
h = tf.keras.Input(shape=(64,), dtype=tf.float32, name="hidden_state")

# Use the EXISTING trained GRUCell
h_new, _ = gru.cell(x, [h], training=False)

# Use the EXISTING trained Dense layer
y = dense(h_new, training=False)

streaming_model = tf.keras.Model(
    inputs=[x, h],
    outputs=[y, h_new],
    name="streaming_gru_subband"
)

streaming_model.save(OUT)

print("Saved:", OUT)
streaming_model.summary()
