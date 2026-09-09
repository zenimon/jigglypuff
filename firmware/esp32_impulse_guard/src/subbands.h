#ifndef SUBBANDS_H
#define SUBBANDS_H

#include <stdint.h>

constexpr int NUM_SUBBANDS = 22;
constexpr int NUM_FEATURES = 44;

void subbands_init();

void extract_subband_features(
    const float* real,
    const float* imag,
    float* features
);

#endif
