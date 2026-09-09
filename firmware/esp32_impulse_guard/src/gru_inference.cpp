#include "gru_inference.h"

#include <Arduino.h>
#include "esp32-hal-psram.h"

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern unsigned char models_gru_subband_streaming_gru_subband_int8_tflite[];
extern unsigned int models_gru_subband_streaming_gru_subband_int8_tflite_len;

namespace {

constexpr int kTensorArenaSize = 200 * 1024;

uint8_t* tensor_arena = nullptr;

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* features_tensor = nullptr;
TfLiteTensor* hidden_tensor = nullptr;
TfLiteTensor* hidden_output_tensor = nullptr;
TfLiteTensor* mask_output_tensor = nullptr;

int8_t hidden_state[GRU_HIDDEN_SIZE];

uint32_t last_latency_us = 0;

bool initialized = false;

} // namespace


void gru_init() {

    Serial.println();
    Serial.println("=== GRU INIT ===");

    // Allocate tensor arena in PSRAM
    tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);

    if (!tensor_arena) {
        Serial.println("ERROR: Failed to allocate GRU tensor arena.");
        return;
    }

    Serial.println("GRU tensor arena allocated in PSRAM.");

    // Load model
    model = tflite::GetModel(
        models_gru_subband_streaming_gru_subband_int8_tflite
    );

    if (!model) {
        Serial.println("ERROR: Failed to load GRU model.");
        return;
    }

    Serial.println("GRU model loaded.");

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Schema version mismatch.");
        return;
    }

    // Resolver — exactly the operators used by the validated V1 model
    static tflite::MicroMutableOpResolver<10> resolver;

    resolver.AddFullyConnected();
    resolver.AddSplit();
    resolver.AddStridedSlice();
    resolver.AddAdd();
    resolver.AddLogistic();
    resolver.AddMul();
    resolver.AddSub();
    resolver.AddTanh();

    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    interpreter = &static_interpreter;

    Serial.println("Creating GRU interpreter...");

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: GRU AllocateTensors() failed.");
        return;
    }

    Serial.println("GRU AllocateTensors() OK.");

    // Get tensors
    features_tensor = interpreter->input(0);
    hidden_tensor = interpreter->input(1);

    hidden_output_tensor = interpreter->output(0);
    mask_output_tensor = interpreter->output(1);

    if (!features_tensor ||
        !hidden_tensor ||
        !hidden_output_tensor ||
        !mask_output_tensor) {

        Serial.println("ERROR: Failed to obtain GRU tensors.");
        return;
    }

    // Verify expected INT8 model
    if (features_tensor->type != kTfLiteInt8 ||
        hidden_tensor->type != kTfLiteInt8 ||
        hidden_output_tensor->type != kTfLiteInt8 ||
        mask_output_tensor->type != kTfLiteInt8) {

        Serial.println("ERROR: GRU tensors are not INT8.");
        return;
    }

    // Initialize persistent hidden state.
    // IMPORTANT:
    // This is done ONCE, not once per frame.
    for (int i = 0; i < GRU_HIDDEN_SIZE; i++) {
        hidden_state[i] =
            (int8_t)hidden_tensor->params.zero_point;

        hidden_tensor->data.int8[i] =
            hidden_state[i];
    }

    Serial.println("Hidden state initialized.");

    Serial.println("=== GRU Quantization ===");

    Serial.print("Feature scale: ");
    Serial.println(features_tensor->params.scale, 9);

    Serial.print("Feature zero-point: ");
    Serial.println(features_tensor->params.zero_point);

    Serial.print("Mask scale: ");
    Serial.println(mask_output_tensor->params.scale, 9);

    Serial.print("Mask zero-point: ");
    Serial.println(mask_output_tensor->params.zero_point);

    Serial.print("Hidden scale: ");
    Serial.println(hidden_tensor->params.scale, 9);

    Serial.print("Hidden zero-point: ");
    Serial.println(hidden_tensor->params.zero_point);

    initialized = true;

    Serial.println("GRU initialization complete.");
}


bool gru_infer(
    const float* input_features,
    float* mask_output
) {

    if (!initialized ||
        !input_features ||
        !mask_output) {

        return false;
    }

    // ---------------------------------------------------------
    // 1. FLOAT → INT8 feature quantization
    //
    // q = round(real / scale) + zero_point
    // ---------------------------------------------------------

    const float input_scale =
        features_tensor->params.scale;

    const int input_zero_point =
        features_tensor->params.zero_point;

    for (int i = 0; i < GRU_INPUT_SIZE; i++) {

        float scaled =
            input_features[i] / input_scale;

        int32_t q =
            (int32_t)lroundf(scaled) +
            input_zero_point;

        // INT8 saturation
        if (q > 127) q = 127;
        if (q < -128) q = -128;

        features_tensor->data.int8[i] =
            (int8_t)q;
    }

    // ---------------------------------------------------------
    // 2. Restore persistent hidden state
    // ---------------------------------------------------------

    for (int i = 0; i < GRU_HIDDEN_SIZE; i++) {

        hidden_tensor->data.int8[i] =
            hidden_state[i];
    }

    // ---------------------------------------------------------
    // 3. Run GRU
    // ---------------------------------------------------------

    uint32_t start_us = micros();

    TfLiteStatus status =
        interpreter->Invoke();

    last_latency_us =
        micros() - start_us;

    if (status != kTfLiteOk) {

        Serial.println("ERROR: GRU Invoke() failed.");

        return false;
    }

    // ---------------------------------------------------------
    // 4. Save hidden state for NEXT frame
    // ---------------------------------------------------------

    for (int i = 0; i < GRU_HIDDEN_SIZE; i++) {

        hidden_state[i] =
            hidden_output_tensor->data.int8[i];
    }

    // ---------------------------------------------------------
    // 5. INT8 mask → FLOAT mask
    //
    // real = (q - zero_point) * scale
    // ---------------------------------------------------------

    const float output_scale =
        mask_output_tensor->params.scale;

    const int output_zero_point =
        mask_output_tensor->params.zero_point;

    for (int i = 0; i < GRU_OUTPUT_SIZE; i++) {

        int8_t q =
            mask_output_tensor->data.int8[i];

        mask_output[i] =
            ((float)q - output_zero_point) *
            output_scale;
    }

    return true;
}


uint32_t gru_get_last_latency_us() {

    return last_latency_us;
}
