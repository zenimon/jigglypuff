#ifndef GRU_INFERENCE_H
#define GRU_INFERENCE_H

#include <stdint.h>

constexpr int GRU_INPUT_SIZE = 44;
constexpr int GRU_HIDDEN_SIZE = 64;
constexpr int GRU_OUTPUT_SIZE = 44;

void gru_init();

bool gru_infer(
    const float* features,
    float* mask_output
);

uint32_t gru_get_last_latency_us();

#endif
