import os
import json
import tensorflow as tf

from src.dataset_loader import (
    create_training_sequence,
    create_validation_sequence,
)
from src.model import create_gru_model


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3

MODEL_DIR = "models/gru_subband"

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_gru_subband.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_gru_subband.keras"
)

HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "training_history.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("Impulse Guard - Subband GRU Training")
print("=" * 70)

print(f"Batch size:     {BATCH_SIZE}")
print(f"Epochs:         {EPOCHS}")
print(f"Learning rate:  {LEARNING_RATE}")
print()


# ============================================================
# GPU CHECK
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("WARNING: No GPU detected.")
    print("Training will run on CPU.")

print()


# ============================================================
# DATASET
# ============================================================

print("Loading training dataset...")

train_sequence = create_training_sequence(
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Training pairs:  {train_sequence.pairs}")
print(f"Training batches: {len(train_sequence)}")

print()

print("Loading validation dataset...")

validation_sequence = create_validation_sequence(
    batch_size=BATCH_SIZE,

)

print(f"Validation pairs:  {validation_sequence.pairs}")
print(f"Validation batches: {len(validation_sequence)}")

print()


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("Creating model")
print("=" * 70)

model = create_gru_model()

model.summary()

print()


# ============================================================
# LOSS
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="mse",
    metrics=["mae"]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    # Save the model with the lowest validation loss
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),

    # Reduce learning rate when validation stops improving
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),

    # Stop when validation stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print("=" * 70)
print("Starting training")
print("=" * 70)
print()

history = model.fit(
    train_sequence,
    validation_data=validation_sequence,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print()
print("=" * 70)
print("Saving final model")
print("=" * 70)

model.save(FINAL_MODEL_PATH)


# ============================================================
# SAVE HISTORY
# ============================================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(HISTORY_PATH, "w") as f:
    json.dump(
        history_data,
        f,
        indent=4
    )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print(f"Best model:")
print(f"  {BEST_MODEL_PATH}")

print()
print(f"Final model:")
print(f"  {FINAL_MODEL_PATH}")

print()
print(f"Training history:")
print(f"  {HISTORY_PATH}")

print()
print("Best validation loss:")
print(f"  {min(history.history['val_loss']):.6f}")

print()
print("Final training loss:")
print(f"  {history.history['loss'][-1]:.6f}")

print()
print("Final validation loss:")
print(f"  {history.history['val_loss'][-1]:.6f}")

print()
print("=" * 70)

