
#include <Arduino.h>
#include "driver/i2s_std.h"

#include "src/stft.h"
#include "src/subbands.h"
#include "src/gru_inference.h"
#include "src/mask_reconstruction.h"
#include "src/istft.h"

// ============================================================
// HARDWARE
// ============================================================

#define SHARED_BCLK 15
#define SHARED_WS   16
#define MIC_SD      17

#define HOP_SIZE 160

// ============================================================
// I2S
// ============================================================

i2s_chan_handle_t rx_handle;

// ============================================================
// AUDIO BUFFERS
// ============================================================

static int16_t frame_buffer[FRAME_SIZE];
static int16_t hop_buffer[HOP_SIZE];

// ISTFT output: 160 samples = 10 ms
static int16_t output_buffer[HOP_SIZE];

// ============================================================
// DSP BUFFERS
// ============================================================

static float fft_real[NUM_FREQ_BINS];
static float fft_imag[NUM_FREQ_BINS];

static float features[NUM_FEATURES];

// GRU output:
// 44 values = 22 complex sub-band mask values
static float mask_output[GRU_OUTPUT_SIZE];

// Expanded 257-bin complex mask
static float mask_real[NUM_FREQ_BINS];
static float mask_imag[NUM_FREQ_BINS];

// Enhanced complex spectrum
static float enhanced_real[NUM_FREQ_BINS];
static float enhanced_imag[NUM_FREQ_BINS];

// ============================================================
// INITIALIZE MICROPHONE
// ============================================================

void startMicrophone()
{
    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(
            I2S_NUM_0,
            I2S_ROLE_MASTER
        );

    esp_err_t err =
        i2s_new_channel(
            &chan_cfg,
            NULL,
            &rx_handle
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "I2S channel creation failed: %d\n",
            err
        );

        while (true)
            delay(1000);
    }

    i2s_std_config_t std_cfg =
    {
        .clk_cfg =
            I2S_STD_CLK_DEFAULT_CONFIG(
                SAMPLE_RATE
            ),

        .slot_cfg =
            I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                I2S_DATA_BIT_WIDTH_32BIT,
                I2S_SLOT_MODE_MONO
            ),

        .gpio_cfg =
        {
            .mclk = I2S_GPIO_UNUSED,

            .bclk =
                (gpio_num_t)SHARED_BCLK,

            .ws =
                (gpio_num_t)SHARED_WS,

            .dout =
                I2S_GPIO_UNUSED,

            .din =
                (gpio_num_t)MIC_SD,

            .invert_flags =
            {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false
            }
        }
    };

    // INMP441 L/R pin is connected to GND.
    std_cfg.slot_cfg.slot_mask =
        I2S_STD_SLOT_LEFT;

    err =
        i2s_channel_init_std_mode(
            rx_handle,
            &std_cfg
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "I2S init failed: %d\n",
            err
        );

        while (true)
            delay(1000);
    }

    err =
        i2s_channel_enable(
            rx_handle
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "I2S enable failed: %d\n",
            err
        );

        while (true)
            delay(1000);
    }

    Serial.println(
        "I2S MICROPHONE INITIALIZED"
    );
}

// ============================================================
// READ 160 NEW SAMPLES
// ============================================================

void readHop(int16_t* output)
{
    static int32_t raw_buffer[HOP_SIZE];

    size_t bytes_read = 0;

    esp_err_t err =
        i2s_channel_read(
            rx_handle,
            raw_buffer,
            sizeof(raw_buffer),
            &bytes_read,
            portMAX_DELAY
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "I2S read error: %d\n",
            err
        );

        return;
    }

    int samples =
        bytes_read /
        sizeof(int32_t);

    for (int i = 0; i < samples; i++)
    {
        // Validated INMP441 conversion.
        output[i] =
            (int16_t)(
                raw_buffer[i] >> 14
            );
    }
}

// ============================================================
// PROCESS ONE 320-SAMPLE FRAME
// ============================================================

void processFrame()
{
    uint32_t start_time =
        micros();

    // --------------------------------------------------------
    // STFT
    // --------------------------------------------------------

    uint32_t stft_start =
        micros();

    stft_compute(
        frame_buffer,
        fft_real,
        fft_imag
    );

    uint32_t stft_time =
        micros() - stft_start;

    // --------------------------------------------------------
    // BARK FEATURES
    // --------------------------------------------------------

    uint32_t bark_start =
        micros();

    extract_subband_features(
        fft_real,
        fft_imag,
        features
    );

    uint32_t bark_time =
        micros() - bark_start;

    // --------------------------------------------------------
    // GRU INFERENCE
    // --------------------------------------------------------

    uint32_t gru_start =
        micros();

    bool gru_ok =
        gru_infer(
            features,
            mask_output
        );

    uint32_t gru_time =
        micros() - gru_start;

    // --------------------------------------------------------
    // COMPLEX MASK RECONSTRUCTION
    // --------------------------------------------------------

    uint32_t mask_start =
        micros();

    if (gru_ok)
    {
        // Expand 22 complex sub-band masks
        // into 257 FFT-bin masks.
        expand_mask_to_fft(
            mask_output,
            mask_real,
            mask_imag
        );

        // Apply complex spectral mask.
        apply_complex_mask(
            fft_real,
            fft_imag,
            mask_real,
            mask_imag,
            enhanced_real,
            enhanced_imag
        );
    }
    else
    {
        // Safety fallback:
        // pass the original spectrum through.
        memcpy(
            enhanced_real,
            fft_real,
            sizeof(fft_real)
        );

        memcpy(
            enhanced_imag,
            fft_imag,
            sizeof(fft_imag)
        );
    }

    uint32_t mask_time =
        micros() - mask_start;

    // --------------------------------------------------------
    // ISTFT
    // --------------------------------------------------------

    uint32_t istft_start =
        micros();

    istft_process(
        enhanced_real,
        enhanced_imag,
        output_buffer
    );

    uint32_t istft_time =
        micros() - istft_start;

    // --------------------------------------------------------
    // TOTAL PROCESSING TIME
    // --------------------------------------------------------

    uint32_t total_time =
        micros() - start_time;

    // --------------------------------------------------------
    // SERIAL OUTPUT
    // --------------------------------------------------------

    static uint32_t frame_count = 0;

    frame_count++;

    // Print timing every 10 frames.
    if (frame_count % 10 == 0)
    {
        Serial.printf(
            "FRAME %lu | "
            "STFT: %lu us | "
            "Bark: %lu us | "
            "GRU: %lu us | "
            "MASK: %lu us | "
            "ISTFT: %lu us | "
            "TOTAL: %lu us | %s\n",

            (unsigned long)frame_count,

            (unsigned long)stft_time,

            (unsigned long)bark_time,

            (unsigned long)gru_time,

            (unsigned long)mask_time,

            (unsigned long)istft_time,

            (unsigned long)total_time,

            gru_ok
                ? "GRU OK"
                : "GRU FAIL"
        );
    }

    // --------------------------------------------------------
    // PRINT FEATURES + MASK ONCE PER SECOND
    // --------------------------------------------------------

    if (frame_count % 100 == 0)
    {
        Serial.println();

        Serial.println(
            "========== 44 FEATURES =========="
        );

        for (int i = 0;
             i < NUM_FEATURES;
             i++)
        {
            Serial.printf(
                "%d: %.5f\n",
                i,
                features[i]
            );
        }

        Serial.println(
            "================================="
        );

        Serial.println();

        Serial.println(
            "========== 44 GRU MASK OUTPUTS =========="
        );

        for (int i = 0;
             i < GRU_OUTPUT_SIZE;
             i++)
        {
            Serial.printf(
                "%d: %.6f\n",
                i,
                mask_output[i]
            );
        }

        Serial.println(
            "=========================================="
        );

        Serial.println();

        // ----------------------------------------------------
        // SELECTED FFT MASK VALUES
        // ----------------------------------------------------

        Serial.println(
            "========== SELECTED FFT MASK VALUES =========="
        );

        for (int k = 0;
             k < NUM_FREQ_BINS;
             k += 32)
        {
            Serial.printf(
                "Bin %d | %.1f Hz | "
                "Real: %.5f | Imag: %.5f\n",

                k,

                SAMPLE_RATE *
                (float)k /
                FFT_SIZE,

                mask_real[k],

                mask_imag[k]
            );
        }

        Serial.println(
            "==============================================="
        );

        Serial.println();
    }
}

// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(115200);

    delay(2000);

    Serial.println();

    Serial.println(
        "========================================"
    );

    Serial.println(
        " IMPULSE GUARD - REAL-TIME V1 PIPELINE"
    );

    Serial.println(
        "========================================"
    );

    Serial.printf(
        "Sample rate : %d Hz\n",
        SAMPLE_RATE
    );

    Serial.printf(
        "Frame size  : %d samples\n",
        FRAME_SIZE
    );

    Serial.printf(
        "Hop size    : %d samples\n",
        HOP_SIZE
    );

    Serial.printf(
        "FFT size    : %d\n",
        FFT_SIZE
    );

    Serial.printf(
        "Frame time  : %.2f ms\n",

        1000.0f *
        FRAME_SIZE /
        SAMPLE_RATE
    );

    Serial.printf(
        "Hop time    : %.2f ms\n",

        1000.0f *
        HOP_SIZE /
        SAMPLE_RATE
    );

    Serial.println();

    // --------------------------------------------------------
    // INITIALIZE DSP
    // --------------------------------------------------------

    stft_init();

    subbands_init();

    mask_reconstruction_init();

    istft_init();

    // --------------------------------------------------------
    // INITIALIZE GRU
    // --------------------------------------------------------

    gru_init();

    // --------------------------------------------------------
    // INITIALIZE MICROPHONE
    // --------------------------------------------------------

    startMicrophone();

    // --------------------------------------------------------
    // FILL FIRST 320-SAMPLE ANALYSIS FRAME
    // --------------------------------------------------------

    Serial.println(
        "Filling initial frame..."
    );

    readHop(
        frame_buffer
    );

    readHop(
        frame_buffer + HOP_SIZE
    );

    Serial.println(
        "Initial frame ready."
    );

    // --------------------------------------------------------
    // PROCESS FIRST FRAME
    // --------------------------------------------------------

    processFrame();

    Serial.println(
        "Starting streaming..."
    );
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop()
{
    // --------------------------------------------------------
    // Shift previous 160 samples out.
    //
    // Before:
    //
    // [ OLD 160 ][ OLD 160 ]
    //
    // After:
    //
    // [ OLD 160 ][ ???????? ]
    // --------------------------------------------------------

    memmove(
        frame_buffer,
        frame_buffer + HOP_SIZE,
        HOP_SIZE * sizeof(int16_t)
    );

    // --------------------------------------------------------
    // Capture 160 new samples
    // --------------------------------------------------------

    readHop(
        hop_buffer
    );

    // --------------------------------------------------------
    // Append new samples.
    //
    // Result:
    //
    // [ OLD 160 ][ NEW 160 ]
    //
    // = 320-sample analysis frame
    // --------------------------------------------------------

    memcpy(
        frame_buffer + HOP_SIZE,
        hop_buffer,
        HOP_SIZE * sizeof(int16_t)
    );

    // --------------------------------------------------------
    // Run complete V1 DSP pipeline
    // --------------------------------------------------------

    processFrame();
}