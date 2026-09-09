#include <Arduino.h>
#include "esp32-hal-psram.h"

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

extern unsigned char models_gru_subband_streaming_gru_subband_int8_tflite[];
extern unsigned int models_gru_subband_streaming_gru_subband_int8_tflite_len;

constexpr int kTensorArenaSize = 200 * 1024;
uint8_t* tensor_arena = nullptr;

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println();
  Serial.println("=== ESP32 V1 GRU Inference Test ===");

  // Allocate tensor arena in PSRAM
  tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);

  if (tensor_arena == nullptr) {
    Serial.println("ERROR: PSRAM allocation failed");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("Tensor arena allocated in PSRAM.");

  // Load model
  const tflite::Model* model =
      tflite::GetModel(
          models_gru_subband_streaming_gru_subband_int8_tflite
      );

  if (model == nullptr) {
    Serial.println("ERROR: Model pointer is NULL");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("Model loaded.");

  // Check schema version
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Schema version mismatch. Model=");
    Serial.print(model->version());
    Serial.print(" Runtime=");
    Serial.println(TFLITE_SCHEMA_VERSION);

    while (true) {
      delay(1000);
    }
  }

  Serial.println("Schema version OK.");

  // Operator resolver
  tflite::MicroMutableOpResolver<10> resolver;

  resolver.AddFullyConnected();
  resolver.AddSplit();
  resolver.AddStridedSlice();
  resolver.AddAdd();
  resolver.AddLogistic();
  resolver.AddMul();
  resolver.AddSub();
  resolver.AddTanh();

  Serial.println("Creating interpreter...");

  // Create interpreter
  tflite::MicroInterpreter interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize
  );

  // Allocate tensors
  TfLiteStatus status = interpreter.AllocateTensors();

  if (status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed");
    Serial.println("Stopping safely.");

    while (true) {
      delay(1000);
    }
  }

  Serial.println("AllocateTensors() OK.");

  // --------------------------------------------------
  // Get model tensors
  // --------------------------------------------------

  TfLiteTensor* features = interpreter.input(0);
  TfLiteTensor* hidden = interpreter.input(1);

  TfLiteTensor* hidden_output = interpreter.output(0);
  TfLiteTensor* mask_output = interpreter.output(1);

  Serial.println();
  Serial.println("=== Tensor Information ===");

  Serial.print("Features type: ");
  Serial.println(features->type);

  Serial.print("Hidden type: ");
  Serial.println(hidden->type);

  Serial.print("Hidden output type: ");
  Serial.println(hidden_output->type);

  Serial.print("Mask output type: ");
  Serial.println(mask_output->type);

  Serial.print("Features shape: ");
  Serial.print(features->dims->data[0]);
  Serial.print(" x ");
  Serial.println(features->dims->data[1]);

  Serial.print("Hidden shape: ");
  Serial.print(hidden->dims->data[0]);
  Serial.print(" x ");
  Serial.println(hidden->dims->data[1]);

  // --------------------------------------------------
  // Zero input test
  // --------------------------------------------------

  Serial.println();
  Serial.println("=== Zero Input Inference ===");

  // INT8 zero-point values.
  // These represent real-valued zero.
  int8_t feature_zero_point =
      features->params.zero_point;

  int8_t hidden_zero_point =
      hidden->params.zero_point;

  Serial.print("Feature scale: ");
  Serial.println(features->params.scale, 9);

  Serial.print("Feature zero-point: ");
  Serial.println(feature_zero_point);

  Serial.print("Hidden zero-point: ");
  Serial.println(hidden_zero_point);

  int8_t test_features[44] = {
  -7, -16, -41, 9, -20, -37, 1, -57,
  -22, -21, -19, -17, -16, -14, -12, -11,
  -9, -26, -27, -29, -31, -32, -34, -36,
  -6, -4, -2, -1, 1, 3, 4, 6,
  -42, -44, -46, -47, -49, -51, -52, -54,
  -23, -25, -21, -27
};

for (int i = 0; i < 44; i++) {
  features->data.int8[i] = test_features[i];
}

// Start hidden state at zero
for (int i = 0; i < 64; i++) {
  hidden->data.int8[i] = hidden->params.zero_point;
}
  // --------------------------------------------------
  // Run inference
  // --------------------------------------------------

  unsigned long start_us = micros();

  status = interpreter.Invoke();

  unsigned long elapsed_us = micros() - start_us;

  if (status != kTfLiteOk) {
    Serial.println("ERROR: Invoke() failed");

    while (true) {
      delay(1000);
    }
  }

  Serial.println("Invoke() OK.");

  Serial.print("Inference time: ");
  Serial.print(elapsed_us);
  Serial.println(" us");

  Serial.print("Inference time: ");
  Serial.print(elapsed_us / 1000.0f, 3);
  Serial.println(" ms");

  // --------------------------------------------------
  // Print output quantization
  // --------------------------------------------------

  Serial.println();
  Serial.println("=== Output Quantization ===");

  Serial.print("Mask scale: ");
  Serial.println(mask_output->params.scale, 9);

  Serial.print("Mask zero-point: ");
  Serial.println(mask_output->params.zero_point);

  Serial.print("Hidden scale: ");
  Serial.println(hidden_output->params.scale, 9);

  Serial.print("Hidden zero-point: ");
  Serial.println(hidden_output->params.zero_point);

  // --------------------------------------------------
  // Print mask output
  // --------------------------------------------------

  Serial.println();
  Serial.println("=== 44 Mask Outputs ===");

  for (int i = 0; i < 44; i++) {

    int8_t q = mask_output->data.int8[i];

    float value =
        (q - mask_output->params.zero_point)
        * mask_output->params.scale;

    Serial.print(i);
    Serial.print(": ");
    Serial.println(value, 6);
  }

  // --------------------------------------------------
  // Print hidden state
  // --------------------------------------------------

  Serial.println();
  Serial.println("=== First 10 Hidden State Values ===");

  for (int i = 0; i < 10; i++) {

    int8_t q = hidden_output->data.int8[i];

    float value =
        (q - hidden_output->params.zero_point)
        * hidden_output->params.scale;

    Serial.print(i);
    Serial.print(": ");
    Serial.println(value, 6);
  }

  Serial.println();
  Serial.println("=== V1 ESP32 GRU TEST PASSED ===");
}

void loop() {
  delay(1000);
}