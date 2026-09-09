#include "stft.h"

#include <math.h>
#include <string.h>

#include "dsps_fft2r.h"

static float hann_window[FRAME_SIZE];
static float fft_buffer[2 * FFT_SIZE];

void stft_init() {
    // Hann window matching librosa/scipy:
    // w[n] = 0.5 * (1 - cos(2*pi*n/N))
    for (int n = 0; n < FRAME_SIZE; n++) {
        hann_window[n] =
            0.5f * (1.0f - cosf(
                2.0f * M_PI * n / FRAME_SIZE
            ));
    }

    // Initialize ESP-DSP FFT coefficient table.
    dsps_fft2r_init_fc32(NULL, FFT_SIZE);
}

void stft_compute(
    const int16_t* input,
    float* real,
    float* imag
) {
    // Clear the complete 512-point FFT buffer.
    memset(fft_buffer, 0, sizeof(fft_buffer));

    // Apply the 320-sample Hann window.
    // Zero-pad samples 320..511.
    for (int n = 0; n < FRAME_SIZE; n++) {
        fft_buffer[2 * n] =
    ((float)input[n] / 32768.0f) * hann_window[n];
    }

    // 512-point complex FFT.
    dsps_fft2r_fc32(fft_buffer, FFT_SIZE);

    // Bit-reverse the FFT output.
    dsps_bit_rev_fc32_ansi(fft_buffer, FFT_SIZE);

    // Keep the 257 unique positive-frequency bins.
    for (int k = 0; k < NUM_FREQ_BINS; k++) {
        real[k] = fft_buffer[2 * k];
        imag[k] = fft_buffer[2 * k + 1];
    }
}
