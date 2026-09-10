#include "mask_reconstruction.h"
#include <Arduino.h>
#include <math.h>

// ============================================================
// PROJECT PARAMETERS
// ============================================================

constexpr float SAMPLE_RATE = 16000.0f;
constexpr float FFT_SIZE = 512.0f;
constexpr float NYQUIST = 8000.0f;

// ============================================================
// BARK CENTER FREQUENCIES
// ============================================================

static float center_frequencies[MASK_SUBBANDS];

// ============================================================
// PYTHON-EQUIVALENT BARK CONVERSION
// ============================================================

static float hz_to_bark(float frequency)
{
    return
        13.0f * atanf(0.00076f * frequency)
        +
        3.5f * atanf(
            powf(frequency / 7500.0f, 2.0f)
        );
}


// ============================================================
// PYTHON-EQUIVALENT BARK → Hz
//
// Python version:
//
// frequencies = np.linspace(0, 8000, 10000)
// barks = hz_to_bark(frequencies)
// np.interp(bark, barks, frequencies)
//
// We reproduce that calculation during initialization only.
// ============================================================

static float bark_to_hz(float target_bark)
{
    const int TABLE_SIZE = 10000;

    float previous_frequency = 0.0f;
    float previous_bark =
        hz_to_bark(previous_frequency);

    for (int i = 1; i < TABLE_SIZE; i++)
    {
        float frequency =
            NYQUIST *
            ((float)i / (float)(TABLE_SIZE - 1));

        float bark =
            hz_to_bark(frequency);

        if (bark >= target_bark)
        {
            float denominator =
                bark - previous_bark;

            if (denominator <= 0.0f)
                return frequency;

            float fraction =
                (target_bark - previous_bark)
                / denominator;

            return
                previous_frequency
                +
                fraction *
                (frequency - previous_frequency);
        }

        previous_frequency = frequency;
        previous_bark = bark;
    }

    return NYQUIST;
}


// ============================================================
// INITIALIZE CENTER FREQUENCIES
//
// Exact equivalent of:
//
// min_bark = hz_to_bark(0)
// max_bark = hz_to_bark(8000)
//
// bark_points = np.linspace(
//     min_bark,
//     max_bark,
//     24
// )
//
// center_frequencies = bark_points[1:-1]
// ============================================================

void mask_reconstruction_init()
{
    float min_bark =
        hz_to_bark(0.0f);

    float max_bark =
        hz_to_bark(NYQUIST);

    for (int band = 0; band < MASK_SUBBANDS; band++)
    {
        float position =
            (float)(band + 1)
            / (float)(MASK_SUBBANDS + 1);

        float target_bark =
            min_bark
            +
            position *
            (max_bark - min_bark);

        center_frequencies[band] =
            bark_to_hz(target_bark);
    }

    Serial.println();
    Serial.println(
        "=== BARK MASK CENTER FREQUENCIES ==="
    );

    for (int i = 0; i < MASK_SUBBANDS; i++)
    {
        Serial.printf(
            "%d: %.3f Hz\n",
            i,
            center_frequencies[i]
        );
    }

    Serial.println(
        "====================================="
    );
}


// ============================================================
// LINEAR INTERPOLATION
//
// Equivalent to:
//
// np.interp(
//     frequencies,
//     center_frequencies,
//     mask_values
// )
// ============================================================

static float interpolate_mask(
    float frequency,
    const float* values
)
{
    // Below first center:
    // np.interp() uses first value.
    if (frequency <= center_frequencies[0])
    {
        return values[0];
    }

    // Above last center:
    // np.interp() uses last value.
    if (
        frequency >=
        center_frequencies[MASK_SUBBANDS - 1]
    )
    {
        return values[MASK_SUBBANDS - 1];
    }

    // Find surrounding center frequencies.
    for (int band = 0;
         band < MASK_SUBBANDS - 1;
         band++)
    {
        float f0 =
            center_frequencies[band];

        float f1 =
            center_frequencies[band + 1];

        if (
            frequency >= f0 &&
            frequency <= f1
        )
        {
            float fraction =
                (frequency - f0)
                /
                (f1 - f0);

            return
                values[band]
                +
                fraction *
                (
                    values[band + 1]
                    - values[band]
                );
        }
    }

    return values[MASK_SUBBANDS - 1];
}


// ============================================================
// EXPAND 22-BAND MASK → 257 FFT BINS
// ============================================================

void expand_mask_to_fft(
    const float* gru_output,
    float* real_mask,
    float* imag_mask
)
{
    // --------------------------------------------------------
    // Split GRU output
    //
    // 0..21   = real
    // 22..43  = imaginary
    // --------------------------------------------------------

    const float* real_subband =
        gru_output;

    const float* imag_subband =
        gru_output + MASK_SUBBANDS;


    // --------------------------------------------------------
    // Interpolate onto 257 FFT bins
    // --------------------------------------------------------

    for (int k = 0;
         k < MASK_FREQ_BINS;
         k++)
    {
        float frequency =
            SAMPLE_RATE *
            (float)k /
            FFT_SIZE;

        real_mask[k] =
            interpolate_mask(
                frequency,
                real_subband
            );

        imag_mask[k] =
            interpolate_mask(
                frequency,
                imag_subband
            );
    }
}


// ============================================================
// COMPLEX SPECTRAL MASK
//
// (Xr + jXi)(Mr + jMi)
//
// Real:
// Xr*Mr - Xi*Mi
//
// Imag:
// Xr*Mi + Xi*Mr
// ============================================================

void apply_complex_mask(
    const float* input_real,
    const float* input_imag,
    const float* mask_real,
    const float* mask_imag,
    float* output_real,
    float* output_imag
)
{
    for (int k = 0;
         k < MASK_FREQ_BINS;
         k++)
    {
        float xr =
            input_real[k];

        float xi =
            input_imag[k];

        float mr =
            mask_real[k];

        float mi =
            mask_imag[k];

        output_real[k] =
            xr * mr
            -
            xi * mi;

        output_imag[k] =
            xr * mi
            +
            xi * mr;
    }
}