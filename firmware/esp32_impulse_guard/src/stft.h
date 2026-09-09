#ifndef STFT_H
#define STFT_H

#include <stdint.h>

constexpr int SAMPLE_RATE = 16000;
constexpr int FRAME_SIZE = 320;
constexpr int HOP_SIZE = 160;
constexpr int FFT_SIZE = 512;
constexpr int NUM_FREQ_BINS = 257;

// Initialize the Hann window and FFT tables.
void stft_init();

// Compute one 320-sample Hann-windowed frame.
// Input: 320 PCM samples.
// Output: 257 complex FFT bins.
void stft_compute(
    const int16_t* input,
    float* real,
    float* imag
);

#endif
