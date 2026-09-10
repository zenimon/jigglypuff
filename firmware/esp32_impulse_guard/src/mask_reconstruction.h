#ifndef MASK_RECONSTRUCTION_H
#define MASK_RECONSTRUCTION_H

#include <stdint.h>

constexpr int MASK_SUBBANDS = 22;
constexpr int MASK_FREQ_BINS = 257;

// Initialize the 22 Bark-band center frequencies.
void mask_reconstruction_init();

// Convert 44 GRU outputs into a 257-bin complex mask.
//
// GRU output layout:
//   [0..21]  = real mask
//   [22..43] = imaginary mask
//
// Output:
//   real_mask[257]
//   imag_mask[257]
void expand_mask_to_fft(
    const float* gru_output,
    float* real_mask,
    float* imag_mask
);

// Apply the 257-bin complex mask to the input STFT.
//
// enhanced = input × mask
void apply_complex_mask(
    const float* input_real,
    const float* input_imag,
    const float* mask_real,
    const float* mask_imag,
    float* output_real,
    float* output_imag
);

#endif