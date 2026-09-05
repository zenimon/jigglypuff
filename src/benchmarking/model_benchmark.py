import time
import numpy as np
import tensorflow as tf

from src.config import FRAME_SIZE, HOP_SIZE, SAMPLE_RATE


def compute_model_statistics(model: tf.keras.Model = None) -> dict:
    """
    Computes parameter count, estimated Flash footprint (Float32 & INT8),
    and FLOPS per 10ms frame for the Impulse Guard GRU model.
    """
    if model is None:
        from src.model import create_gru_model
        model = create_gru_model()

    trainable_params = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    non_trainable_params = int(np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]))
    total_params = trainable_params + non_trainable_params

    # Size estimates in Kilobytes (KB)
    float32_flash_kb = (total_params * 4) / 1024.0
    int8_flash_kb = (total_params * 1) / 1024.0

    # Approximate FLOPS for GRU: 3 * (input_dim * hidden_dim + hidden_dim^2 + hidden_dim) per timestep
    # Dense layer: input_dim * output_dim + output_dim
    # For Input=44, GRU=64, Dense=44:
    gru_flops = 3 * (44 * 64 + 64 * 64 + 64) * 2  # MACs -> FLOPS
    dense_flops = (64 * 44 + 44) * 2
    total_flops_per_frame = gru_flops + dense_flops

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "float32_flash_kb": float(float32_flash_kb),
        "int8_flash_kb": float(int8_flash_kb),
        "flops_per_frame": total_flops_per_frame,
    }


def measure_real_time_factor(model: tf.keras.Model = None, num_frames: int = 100) -> dict:
    """
    Measures CPU execution time and calculates Real-Time Factor (RTF).
    RTF = Execution Time / Audio Duration.
    RTF < 1.0 is required for real-time streaming.
    """
    if model is None:
        from src.model import create_gru_model
        model = create_gru_model()

    dummy_input = np.random.randn(1, num_frames, 44).astype(np.float32)

    # Warmup pass
    _ = model(dummy_input, training=False)

    # Benchmark pass
    start_time = time.perf_counter()
    _ = model(dummy_input, training=False)
    end_time = time.perf_counter()

    elapsed_sec = end_time - start_time
    audio_duration_sec = (num_frames * HOP_SIZE) / SAMPLE_RATE
    rtf = elapsed_sec / audio_duration_sec if audio_duration_sec > 0 else float('nan')
    latency_per_frame_ms = (elapsed_sec / num_frames) * 1000.0

    return {
        "num_frames": num_frames,
        "elapsed_sec": float(elapsed_sec),
        "audio_duration_sec": float(audio_duration_sec),
        "real_time_factor": float(rtf),
        "latency_per_frame_ms": float(latency_per_frame_ms),
        "is_realtime_capable": bool(rtf < 1.0)
    }
