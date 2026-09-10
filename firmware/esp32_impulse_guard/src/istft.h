#ifndef ISTFT_H
#define ISTFT_H

#include <stdint.h>

void istft_init();

void istft_process(
    const float* enhanced_real,
    const float* enhanced_imag,
    int16_t* output
);

#endif