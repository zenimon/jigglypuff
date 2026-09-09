import numpy as np
import tensorflow as tf

FLOAT_MODEL = "models/gru_subband/streaming_gru_subband_float32.tflite"
INT8_MODEL = "models/gru_subband/streaming_gru_subband_int8.tflite"
CALIB = "models/gru_subband/int8_calibration.npz"


def get_io(interpreter):
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    feature_input = next(x for x in inputs if x["shape"][1] == 44)
    state_input = next(x for x in inputs if x["shape"][1] == 64)

    mask_output = next(x for x in outputs if x["shape"][1] == 44)
    state_output = next(x for x in outputs if x["shape"][1] == 64)

    return feature_input, state_input, mask_output, state_output


def quantize(x, details):
    scale, zero_point = details["quantization"]
    q = np.round(x / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize(x, details):
    scale, zero_point = details["quantization"]
    return (x.astype(np.float32) - zero_point) * scale


float_interpreter = tf.lite.Interpreter(model_path=FLOAT_MODEL)
int8_interpreter = tf.lite.Interpreter(model_path=INT8_MODEL)

f_feat, f_state, f_mask, f_state_out = get_io(float_interpreter)
i_feat, i_state, i_mask, i_state_out = get_io(int8_interpreter)

data = np.load(CALIB)
features = data["features"]

num_frames = min(1000, len(features))

float_hidden = np.zeros((1, 64), dtype=np.float32)
int8_hidden = np.zeros((1, 64), dtype=np.float32)

mask_diffs = []
state_diffs = []

for n in range(num_frames):
    x = features[n:n+1].astype(np.float32)

    # FP32
    float_interpreter.set_tensor(
        f_feat["index"], x
    )
    float_interpreter.set_tensor(
        f_state["index"], float_hidden
    )
    float_interpreter.invoke()

    float_mask = float_interpreter.get_tensor(
        f_mask["index"]
    )
    float_hidden = float_interpreter.get_tensor(
        f_state_out["index"]
    )

    # INT8
    x_q = quantize(x, i_feat)
    state_q = quantize(int8_hidden, i_state)

    int8_interpreter.set_tensor(
        i_feat["index"], x_q
    )
    int8_interpreter.set_tensor(
        i_state["index"], state_q
    )
    int8_interpreter.invoke()

    int8_mask_q = int8_interpreter.get_tensor(
        i_mask["index"]
    )
    int8_state_q = int8_interpreter.get_tensor(
        i_state_out["index"]
    )

    int8_mask = dequantize(
        int8_mask_q, i_mask
    )
    int8_hidden = dequantize(
        int8_state_q, i_state_out
    )

    mask_diffs.append(
        np.max(np.abs(float_mask - int8_mask))
    )

    state_diffs.append(
        np.max(np.abs(float_hidden - int8_hidden))
    )

print("Frames tested:", num_frames)
print()
print("MASK")
print("Max abs diff :", max(mask_diffs))
print("Mean max diff:", np.mean(mask_diffs))
print()
print("HIDDEN STATE")
print("Max abs diff :", max(state_diffs))
print("Mean max diff:", np.mean(state_diffs))
