#include <Arduino.h>
#include "driver/i2s_std.h"

#include "src/stft.h"
#include "src/subbands.h"
#include "src/gru_inference.h"

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

// ============================================================
// DSP BUFFERS
// ============================================================

static float fft_real[NUM_FREQ_BINS];
static float fft_imag[NUM_FREQ_BINS];

static float features[NUM_FEATURES];

// GRU output: 44 values = 22 complex sub-band mask values
static float mask_output[GRU_OUTPUT_SIZE];

// ============================================================
// INITIALIZE MICROPHONE
// ============================================================

void startMicrophone() {

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

    if (err != ESP_OK) {
        Serial.printf(
            "I2S channel creation failed: %d\n",
            err
        );
        while (1);
    }

    i2s_std_config_t std_cfg = {

        .clk_cfg =
            I2S_STD_CLK_DEFAULT_CONFIG(
                SAMPLE_RATE
            ),

        .slot_cfg =
            I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                I2S_DATA_BIT_WIDTH_32BIT,
                I2S_SLOT_MODE_MONO
            ),

        .gpio_cfg = {

            .mclk = I2S_GPIO_UNUSED,

            .bclk =
                (gpio_num_t)SHARED_BCLK,

            .ws =
                (gpio_num_t)SHARED_WS,

            .dout =
                I2S_GPIO_UNUSED,

            .din =
                (gpio_num_t)MIC_SD,

            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false
            }
        }
    };

    // INMP441 L/R pin is connected to GND
    std_cfg.slot_cfg.slot_mask =
        I2S_STD_SLOT_LEFT;

    err =
        i2s_channel_init_std_mode(
            rx_handle,
            &std_cfg
        );

    if (err != ESP_OK) {
        Serial.printf(
            "I2S init failed: %d\n",
            err
        );
        while (1);
    }

    err =
        i2s_channel_enable(rx_handle);

    if (err != ESP_OK) {
        Serial.printf(
            "I2S enable failed: %d\n",
            err
        );
        while (1);
    }

    Serial.println(
        "I2S MICROPHONE INITIALIZED"
    );
}


// ============================================================
// READ 160 NEW SAMPLES
// ============================================================

void readHop(int16_t* output) {

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

    if (err != ESP_OK) {
        Serial.printf(
            "I2S read error: %d\n",
            err
        );
        return;
    }

    int samples =
        bytes_read / sizeof(int32_t);

    for (int i = 0; i < samples; i++) {

        // Same conversion as the previously
        // validated INMP441 microphone test.
        output[i] =
            (int16_t)(
                raw_buffer[i] >> 14
            );
    }
}


// ============================================================
// PROCESS ONE 320-SAMPLE FRAME
// ============================================================

void processFrame() {

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
    // TOTAL PROCESSING TIME
    // --------------------------------------------------------

    uint32_t total_time =
        micros() - start_time;


    // --------------------------------------------------------
    // SERIAL OUTPUT
    // --------------------------------------------------------

    static uint32_t frame_count = 0;

    frame_count++;

    // Print timing every 10 frames
    if (frame_count % 10 == 0) {

        Serial.printf(
            "FRAME %lu | STFT: %lu us | Bark: %lu us | GRU: %lu us | TOTAL: %lu us | %s\n",
            (unsigned long)frame_count,
            (unsigned long)stft_time,
            (unsigned long)bark_time,
            (unsigned long)gru_time,
            (unsigned long)total_time,
            gru_ok ? "GRU OK" : "GRU FAIL"
        );
    }


    // --------------------------------------------------------
    // PRINT FEATURES + MASK ONCE PER SECOND
    // --------------------------------------------------------

    if (frame_count % 100 == 0) {

        Serial.println();
        Serial.println(
            "========== 44 FEATURES =========="
        );

        for (int i = 0;
             i < NUM_FEATURES;
             i++) {

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
             i++) {

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
    }
}


// ============================================================
// SETUP
// ============================================================

void setup() {

    Serial.begin(115200);

    delay(2000);

    Serial.println();
    Serial.println(
        "========================================"
    );
    Serial.println(
        " IMPULSE GUARD - REAL-TIME GRU PIPELINE"
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
    // Initialize DSP
    // --------------------------------------------------------

    stft_init();

    subbands_init();


    // --------------------------------------------------------
    // Initialize GRU
    // --------------------------------------------------------

    gru_init();


    // --------------------------------------------------------
    // Initialize microphone
    // --------------------------------------------------------

    startMicrophone();


    // --------------------------------------------------------
    // Fill first 320-sample analysis frame
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
    // Process first frame
    // --------------------------------------------------------

    processFrame();

    Serial.println(
        "Starting streaming..."
    );
}


// ============================================================
// MAIN LOOP
// ============================================================

void loop() {

    // --------------------------------------------------------
    // Shift the previous 160 samples out.
    //
    // Before:
    //
    // [ OLD 160 ][ OLD 160 ]
    //
    // After shift:
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
    // Add new samples to the end of the frame
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
    // Run STFT + Bark + GRU
    // --------------------------------------------------------

    processFrame();
}