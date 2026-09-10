#include "istft.h"

#include <Arduino.h>
#include <math.h>
#include <string.h>

#include "dsps_fft2r.h"
#include "stft.h"

static float hann_window[FRAME_SIZE];

static float fft_buffer[2 * FFT_SIZE];

static float overlap_buffer[HOP_SIZE];

static float window_norm[HOP_SIZE];

void istft_init()
{
    // Match the STFT window EXACTLY.
    for (int n = 0; n < FRAME_SIZE; n++) {
        hann_window[n] =
            0.5f *
            (1.0f -
             cosf(
                 2.0f * M_PI * n / FRAME_SIZE
             ));
    }

    // ---------------------------------------------------------
    // Window-overlap normalization
    //
    // Two 50%-overlapped frames contribute to each output
    // sample. The normalization compensates for w[n]^2.
    // ---------------------------------------------------------

    for (int n = 0; n < HOP_SIZE; n++) {

        float w1 = hann_window[n];
        float w2 = hann_window[n + HOP_SIZE];

        window_norm[n] =
            w1 * w1 +
            w2 * w2;

        if (window_norm[n] < 1e-8f) {
            window_norm[n] = 1.0f;
        }
    }

    memset(
        fft_buffer,
        0,
        sizeof(fft_buffer)
    );

    memset(
        overlap_buffer,
        0,
        sizeof(overlap_buffer)
    );
}

void istft_process(
    const float* enhanced_real,
    const float* enhanced_imag,
    int16_t* output
)
{
    // ---------------------------------------------------------
    // Reconstruct full complex spectrum
    // ---------------------------------------------------------

    memset(
        fft_buffer,
        0,
        sizeof(fft_buffer)
    );

    // DC
    fft_buffer[0] =
        enhanced_real[0];

    fft_buffer[1] = 0.0f;

    // Positive frequencies
    for (int k = 1; k < NUM_FREQ_BINS - 1; k++) {

        fft_buffer[2 * k] =
            enhanced_real[k];

        fft_buffer[2 * k + 1] =
            enhanced_imag[k];

        // Conjugate mirror
        int mirror =
            FFT_SIZE - k;

        fft_buffer[2 * mirror] =
            enhanced_real[k];

        fft_buffer[2 * mirror + 1] =
            -enhanced_imag[k];
    }

    // Nyquist
    fft_buffer[
        2 * (NUM_FREQ_BINS - 1)
    ] =
        enhanced_real[
            NUM_FREQ_BINS - 1
        ];

    fft_buffer[
        2 * (NUM_FREQ_BINS - 1) + 1
    ] = 0.0f;

    // ---------------------------------------------------------
    // IFFT
    //
    // ESP-DSP does not need a separate inverse FFT routine.
    // Conjugate -> FFT -> conjugate -> scale.
    // ---------------------------------------------------------

    for (int k = 0; k < FFT_SIZE; k++) {
        fft_buffer[2 * k + 1] =
            -fft_buffer[2 * k + 1];
    }

    dsps_fft2r_fc32(
        fft_buffer,
        FFT_SIZE
    );

    dsps_bit_rev_fc32_ansi(
        fft_buffer,
        FFT_SIZE
    );

    // ---------------------------------------------------------
    // Convert IFFT result back to time domain
    // ---------------------------------------------------------

    for (int n = 0; n < FRAME_SIZE; n++) {

        float sample =
            fft_buffer[2 * n] /
            FFT_SIZE;

        // Synthesis window
        sample *= hann_window[n];

        // -----------------------------------------------------
        // Overlap-add
        // -----------------------------------------------------

        if (n < HOP_SIZE) {

            sample +=
                overlap_buffer[n];

            // Normalize overlapping windows.
            sample /=
                window_norm[n];

            float value =
                sample * 32767.0f;

            if (value > 32767.0f)
                value = 32767.0f;

            if (value < -32768.0f)
                value = -32768.0f;

            output[n] =
                (int16_t)value;
        }
        else {

            overlap_buffer[
                n - HOP_SIZE
            ] = sample;
        }
    }
}