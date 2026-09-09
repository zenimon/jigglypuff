#include "subbands.h"

#include <math.h>
#include <string.h>

float filters[NUM_SUBBANDS][257];

static float hz_to_bark(float frequency) {
    return (
        13.0f * atanf(0.00076f * frequency)
        + 3.5f * atanf(
            powf(frequency / 7500.0f, 2.0f)
        )
    );
}

static float bark_to_hz(float bark) {
    const int N = 10000;

    float previous_bark = hz_to_bark(0.0f);
    float previous_hz = 0.0f;

    for (int i = 1; i < N; i++) {

        float hz =
            (8000.0f * i) / (N - 1);

        float current_bark =
            hz_to_bark(hz);

        if (current_bark >= bark) {

            float ratio =
                (bark - previous_bark) /
                (current_bark - previous_bark);

            return previous_hz +
                   ratio * (hz - previous_hz);
        }

        previous_bark = current_bark;
        previous_hz = hz;
    }

    return 8000.0f;
}

void subbands_init() {

    memset(
        filters,
        0,
        sizeof(filters)
    );

    const float min_bark =
        hz_to_bark(0.0f);

    const float max_bark =
        hz_to_bark(8000.0f);

    float bark_points[NUM_SUBBANDS + 2];
    int bin_points[NUM_SUBBANDS + 2];

    for (int i = 0;
         i < NUM_SUBBANDS + 2;
         i++) {

        bark_points[i] =
            min_bark +
            (max_bark - min_bark) *
            ((float)i /
             (NUM_SUBBANDS + 1));

        float hz =
            bark_to_hz(
                bark_points[i]
            );

        int bin =
            (int)floorf(
                513.0f * hz / 16000.0f
            );

        if (bin < 0)
            bin = 0;

        if (bin > 256)
            bin = 256;

        bin_points[i] = bin;
    }

    for (int band = 0;
         band < NUM_SUBBANDS;
         band++) {

        int left =
            bin_points[band];

        int center =
            bin_points[band + 1];

        int right =
            bin_points[band + 2];

        if (center > left) {

            for (int k = left;
                 k < center;
                 k++) {

                filters[band][k] =
                    ((float)(k - left)) /
                    ((float)(center - left));
            }
        }

        if (right > center) {

            for (int k = center;
                 k < right;
                 k++) {

                filters[band][k] =
                    ((float)(right - k)) /
                    ((float)(right - center));
            }
        }
    }
}


// ============================================================
// STREAMING FEATURE EXTRACTION
// ============================================================

void extract_subband_features(
    const float* real,
    const float* imag,
    float* features
) {
    static float previous_log_band_power[
        NUM_SUBBANDS
    ];

    static bool first_frame = true;

    float log_band_power[
        NUM_SUBBANDS
    ];


    // --------------------------------------------------------
    // Calculate 22 Bark-band log energies
    // --------------------------------------------------------

    for (int band = 0;
         band < NUM_SUBBANDS;
         band++) {

        float band_power = 0.0f;

        for (int k = 0;
             k < 257;
             k++) {

            float magnitude =
                sqrtf(
                    real[k] * real[k] +
                    imag[k] * imag[k]
                );

            float power =
                magnitude * magnitude;

            band_power +=
                filters[band][k] *
                power;
        }

        log_band_power[band] =
            10.0f *
            log10f(
                band_power + 1e-10f
            );
    }


    // --------------------------------------------------------
    // Features 0..21
    // Current Bark log energies
    // --------------------------------------------------------

    for (int band = 0;
         band < NUM_SUBBANDS;
         band++) {

        features[band] =
            log_band_power[band];
    }


    // --------------------------------------------------------
    // Features 22..43
    // Temporal differences
    // --------------------------------------------------------

    if (first_frame) {

        for (int band = 0;
             band < NUM_SUBBANDS;
             band++) {

            features[
                NUM_SUBBANDS + band
            ] = 0.0f;

            previous_log_band_power[band] =
                log_band_power[band];
        }

        first_frame = false;

    } else {

        for (int band = 0;
             band < NUM_SUBBANDS;
             band++) {

            features[
                NUM_SUBBANDS + band
            ] =
                log_band_power[band]
                -
                previous_log_band_power[band];

            previous_log_band_power[band] =
                log_band_power[band];
        }
    }
}