
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

// INMP441
#define MIC_SD      17

// MAX98357A
#define AMP_DIN     5

// ============================================================
// AUDIO
// ============================================================

#define SAMPLE_RATE 16000

#define FRAME_SIZE 320
#define HOP_SIZE   160

#define RECORD_SECONDS 5
#define PAUSE_SECONDS  2

#define NUM_OUTPUT_SAMPLES \
    (SAMPLE_RATE * RECORD_SECONDS)

// ============================================================
// I2S
// ============================================================

i2s_chan_handle_t tx_handle;
i2s_chan_handle_t rx_handle;

// ============================================================
// AUDIO BUFFERS
// ============================================================

#if CONFIG_SPIRAM

static int16_t* recorded_output = nullptr;

#else

static int16_t recorded_output[NUM_OUTPUT_SAMPLES];

#endif

static int16_t frame_buffer[FRAME_SIZE];
static int16_t hop_buffer[HOP_SIZE];
static int16_t output_buffer[HOP_SIZE];

// ============================================================
// DSP BUFFERS
// ============================================================

static float fft_real[NUM_FREQ_BINS];
static float fft_imag[NUM_FREQ_BINS];

static float features[NUM_FEATURES];

static float mask_output[GRU_OUTPUT_SIZE];

static float mask_real[NUM_FREQ_BINS];
static float mask_imag[NUM_FREQ_BINS];

static float enhanced_real[NUM_FREQ_BINS];
static float enhanced_imag[NUM_FREQ_BINS];

// ============================================================
// I2S TX BUFFER
// Same format as previous working mic -> speaker code
// ============================================================

static int32_t tx_buffer[HOP_SIZE];

// ============================================================
// GLOBALS
// ============================================================

static uint32_t frame_count = 0;

float VOLUME_GAIN = 1.0f;

// ============================================================
// I2S INITIALIZATION
// ============================================================

void startFullDuplexChannel()
{
    Serial.println("Initializing I2S...");

    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(
            I2S_NUM_0,
            I2S_ROLE_MASTER
        );

    esp_err_t err =
        i2s_new_channel(
            &chan_cfg,
            &tx_handle,
            &rx_handle
        );

    if (err != ESP_OK)
    {
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
                (gpio_num_t)AMP_DIN,

            .din =
                (gpio_num_t)MIC_SD,

            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false
            }
        }
    };

    // INMP441 L/R = GND
    // LEFT channel
    std_cfg.slot_cfg.slot_mask =
        I2S_STD_SLOT_LEFT;

    // --------------------------------------------------------
    // TX
    // --------------------------------------------------------

    err =
        i2s_channel_init_std_mode(
            tx_handle,
            &std_cfg
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "I2S TX initialization failed: %d\n",
            err
        );

        while (1);
    }

    // --------------------------------------------------------
    // RX
    // --------------------------------------------------------

    err =
        i2s_channel_init_std_mode(
            rx_handle,
            &std_cfg
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "I2S RX initialization failed: %d\n",
            err
        );

        while (1);
    }

    // --------------------------------------------------------
    // Enable
    // --------------------------------------------------------

    err =
        i2s_channel_enable(tx_handle);

    if (err != ESP_OK)
    {
        Serial.printf(
            "I2S TX enable failed: %d\n",
            err
        );

        while (1);
    }

    err =
        i2s_channel_enable(rx_handle);

    if (err != ESP_OK)
    {
        Serial.printf(
            "I2S RX enable failed: %d\n",
            err
        );

        while (1);
    }

    delay(50);

    // --------------------------------------------------------
    // Speaker warm-up
    // Same approach as previous working code
    // --------------------------------------------------------

    int32_t silence[HOP_SIZE] = {0};

    size_t written;

    int warmupWrites =
        (int)((SAMPLE_RATE * 0.6f) / HOP_SIZE);

    for (int i = 0; i < warmupWrites; i++)
    {
        i2s_channel_write(
            tx_handle,
            silence,
            sizeof(silence),
            &written,
            portMAX_DELAY
        );
    }

    Serial.println(
        "I2S FULL-DUPLEX INITIALIZED"
    );
}

// ============================================================
// SILENCE
// ============================================================

void writeSilenceBlock()
{
    static int32_t silence[HOP_SIZE] = {0};

    size_t written;

    i2s_channel_write(
        tx_handle,
        silence,
        sizeof(silence),
        &written,
        portMAX_DELAY
    );
}

// ============================================================
// READ MICROPHONE
// ============================================================

bool readHop(int16_t* output)
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
            "I2S READ ERROR: %d\n",
            err
        );

        return false;
    }

    int samples =
        bytes_read /
        sizeof(int32_t);

    if (samples != HOP_SIZE)
    {
        Serial.printf(
            "WARNING: Expected %d samples, got %d\n",
            HOP_SIZE,
            samples
        );

        return false;
    }

    // IMPORTANT:
    // This is the same conversion used in
    // your previously working mic -> speaker code.

    for (int i = 0; i < HOP_SIZE; i++)
    {
        output[i] =
            (int16_t)(
                raw_buffer[i] >> 14
            );
    }

    return true;
}

// ============================================================
// MICROPHONE TEST
// ============================================================

void microphoneTest()
{
    Serial.println();
    Serial.println(
        "=========================================="
    );

    Serial.println(
        "MICROPHONE TEST"
    );

    Serial.println(
        "Speak / clap near the INMP441..."
    );

    Serial.println(
        "=========================================="
    );

    int16_t test_buffer[HOP_SIZE];

    int32_t peak = 0;

    double sum_sq = 0.0;

    // 1 second
    for (int block = 0; block < 100; block++)
    {
        if (!readHop(test_buffer))
            continue;

        for (int i = 0; i < HOP_SIZE; i++)
        {
            int32_t x =
                test_buffer[i];

            int32_t magnitude =
                abs(x);

            if (magnitude > peak)
                peak = magnitude;

            sum_sq +=
                (double)x *
                (double)x;
        }
    }

    float rms =
        sqrtf(
            sum_sq /
            (100.0 * HOP_SIZE)
        );

    Serial.printf(
        "MIC TEST | RMS: %.1f | PEAK: %ld\n",
        rms,
        (long)peak
    );

    if (peak > 100)
    {
        Serial.println(
            "MIC STATUS: WORKING"
        );
    }
    else
    {
        Serial.println(
            "MIC STATUS: CHECK WIRING"
        );
    }

    Serial.println();
}

// ============================================================
// SPEAKER TEST
// ============================================================
//
// This is the same I2S TX path as your known-working
// mic -> speaker implementation.
// ============================================================

void speakerTest()
{
    Serial.println();
    Serial.println(
        "=========================================="
    );

    Serial.println(
        "SPEAKER TEST"
    );

    Serial.println(
        "Playing 1 kHz test tone..."
    );

    Serial.println(
        "=========================================="
    );

    const float frequency =
        1000.0f;

    const float amplitude =
        12000.0f;

    const int total_samples =
        SAMPLE_RATE;

    int32_t sampleBuf[HOP_SIZE];

    size_t bytesWritten;

    for (
        int start = 0;
        start < total_samples;
        start += HOP_SIZE
    )
    {
        int chunkLen =
            min(
                HOP_SIZE,
                total_samples - start
            );

        for (int j = 0; j < chunkLen; j++)
        {
            int index =
                start + j;

            float t =
                (float)index /
                SAMPLE_RATE;

            float sample =
                amplitude *
                sinf(
                    2.0f *
                    M_PI *
                    frequency *
                    t
                );

            // EXACT SAME TX FORMAT
            // AS THE PREVIOUS WORKING CODE

            sampleBuf[j] =
                ((int32_t)sample)
                << 16;
        }

        i2s_channel_write(
            tx_handle,
            sampleBuf,
            chunkLen * sizeof(int32_t),
            &bytesWritten,
            portMAX_DELAY
        );

        // Drain RX
        int32_t dummyRead;

        size_t bytesRead;

        i2s_channel_read(
            rx_handle,
            &dummyRead,
            sizeof(dummyRead),
            &bytesRead,
            0
        );
    }

    // Silence after tone

    for (int i = 0; i < 5; i++)
    {
        writeSilenceBlock();
    }

    Serial.println(
        "SPEAKER TEST COMPLETE"
    );

    Serial.println();
}

// ============================================================
// PROCESS ONE FRAME
// ============================================================

bool processFrame()
{
    uint32_t total_start =
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
    // BARK
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
    // GRU
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
    // MASK
    // --------------------------------------------------------

    uint32_t mask_start =
        micros();

    expand_mask_to_fft(
        mask_output,
        mask_real,
        mask_imag
    );

    apply_complex_mask(
        fft_real,
        fft_imag,

        mask_real,
        mask_imag,

        enhanced_real,
        enhanced_imag
    );

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
    // TOTAL
    // --------------------------------------------------------

    uint32_t total_time =
        micros() - total_start;

    frame_count++;

    // --------------------------------------------------------
    // Timing print
    // --------------------------------------------------------

    if (frame_count % 10 == 0)
    {
        Serial.printf(
            "FRAME %lu | "
            "STFT: %lu us | "
            "Bark: %lu us | "
            "GRU: %lu us | "
            "MASK: %lu us | "
            "ISTFT: %lu us | "
            "TOTAL: %lu us | "
            "%s\n",

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

    return gru_ok;
}

// ============================================================
// RECORD + AI PROCESSING
// ============================================================

void recordAndProcess()
{
    Serial.println();
    Serial.println(
        "=========================================="
    );

    Serial.println(
        "RECORDING + AI MASKING"
    );

    Serial.println(
        "Duration: 5 seconds"
    );

    Serial.println(
        "Speak now..."
    );

    Serial.println(
        "=========================================="
    );

    frame_count = 0;

    // ========================================================
    // FIRST 320-SAMPLE FRAME
    // ========================================================

    if (!readHop(frame_buffer))
        return;

    if (!readHop(
            frame_buffer + HOP_SIZE
        ))
        return;

    // ========================================================
    // PROCESS FIRST FRAME
    // ========================================================

    processFrame();

    // ========================================================
    // STORE FIRST 160 OUTPUT SAMPLES
    // ========================================================

    for (int i = 0; i < HOP_SIZE; i++)
    {
        int32_t sample =
            (int32_t)(
                output_buffer[i] *
                VOLUME_GAIN
            );

        if (sample > 32767)
            sample = 32767;

        if (sample < -32768)
            sample = -32768;

        recorded_output[i] =
            (int16_t)sample;
    }

    int output_position =
        HOP_SIZE;

    // ========================================================
    // PROCESS REMAINING HOPS
    // ========================================================

    while (
        output_position <
        NUM_OUTPUT_SAMPLES
    )
    {
        // ----------------------------------------------------
        // Shift 160 samples
        // ----------------------------------------------------

        memmove(
            frame_buffer,
            frame_buffer + HOP_SIZE,
            HOP_SIZE * sizeof(int16_t)
        );

        // ----------------------------------------------------
        // Read new 160 samples
        // ----------------------------------------------------

        if (!readHop(hop_buffer))
            break;

        // ----------------------------------------------------
        // Add new samples
        // ----------------------------------------------------

        memcpy(
            frame_buffer + HOP_SIZE,
            hop_buffer,
            HOP_SIZE * sizeof(int16_t)
        );

        // ----------------------------------------------------
        // MIC LEVEL
        // ----------------------------------------------------

        if (frame_count % 50 == 0)
        {
            int32_t peak = 0;

            double sum_sq = 0.0;

            for (int i = 0; i < HOP_SIZE; i++)
            {
                int32_t x =
                    hop_buffer[i];

                int32_t magnitude =
                    abs(x);

                if (magnitude > peak)
                    peak = magnitude;

                sum_sq +=
                    (double)x *
                    (double)x;
            }

            float rms =
                sqrtf(
                    sum_sq /
                    HOP_SIZE
                );

            Serial.printf(
                "MIC LIVE | RMS: %.1f | PEAK: %ld\n",
                rms,
                (long)peak
            );
        }

        // ----------------------------------------------------
        // AI PROCESSING
        // ----------------------------------------------------

        processFrame();

        // ----------------------------------------------------
        // STORE MASKED OUTPUT
        // ----------------------------------------------------

        int samples_to_copy =
            min(
                HOP_SIZE,
                NUM_OUTPUT_SAMPLES -
                output_position
            );

        for (
            int i = 0;
            i < samples_to_copy;
            i++
        )
        {
            int32_t sample =
                (int32_t)(
                    output_buffer[i] *
                    VOLUME_GAIN
                );

            if (sample > 32767)
                sample = 32767;

            if (sample < -32768)
                sample = -32768;

            recorded_output[
                output_position + i
            ] =
                (int16_t)sample;
        }

        output_position +=
            samples_to_copy;
    }

    // ========================================================
    // FILL REMAINING SAMPLES WITH ZERO
    // ========================================================

    while (
        output_position <
        NUM_OUTPUT_SAMPLES
    )
    {
        recorded_output[
            output_position++
        ] = 0;
    }

    // ========================================================
    // COMPLETE
    // ========================================================

    Serial.println();
    Serial.println(
        "=========================================="
    );

    Serial.println(
        "RECORDING COMPLETE"
    );

    Serial.printf(
        "Processed samples: %d\n",
        NUM_OUTPUT_SAMPLES
    );

    Serial.println(
        "AI masked output stored in PSRAM."
    );

    Serial.println(
        "=========================================="
    );
}

// ============================================================
// MASKED OUTPUT DIAGNOSTIC
// ============================================================

void printMaskedOutputLevel()
{
    int16_t peak = 0;

    double sum_sq = 0.0;

    for (
        int i = 0;
        i < NUM_OUTPUT_SAMPLES-320;
        i++
    )
    {
        int32_t x =
            recorded_output[i];

        int32_t magnitude =
            abs(x);

        if (magnitude > peak)
            peak = magnitude;

        sum_sq +=
            (double)x *
            (double)x;
    }

    float rms =
        sqrt(
            sum_sq /
            NUM_OUTPUT_SAMPLES
        );

    Serial.println();
    Serial.println(
        "=========================================="
    );

    Serial.println(
        "MASKED OUTPUT DIAGNOSTIC"
    );

    Serial.printf(
        "RMS  : %.2f\n",
        rms
    );

    Serial.printf(
        "PEAK : %d\n",
        peak
    );

    Serial.println(
        "=========================================="
    );

    if (peak < 10)
    {
        Serial.println(
            "WARNING: MASKED OUTPUT IS NEAR SILENT"
        );
    }
    else
    {
        Serial.println(
            "MASKED OUTPUT CONTAINS AUDIO"
        );
    }

    Serial.println();
}

// ============================================================
// 2 SECOND PAUSE
// ============================================================

void pauseBeforePlayback()
{
    Serial.println();
    Serial.println(
        "=========================================="
    );

    Serial.println(
        "PAUSE"
    );

    Serial.println(
        "2 seconds..."
    );

    Serial.println(
        "=========================================="
    );

    int32_t silence[HOP_SIZE] = {0};

    size_t bytesWritten;

    int32_t dummyRead;

    size_t bytesRead;

    int total_samples =
        SAMPLE_RATE *
        PAUSE_SECONDS;

    for (
        int i = 0;
        i < total_samples;
        i += HOP_SIZE
    )
    {
        // Speaker silence

        i2s_channel_write(
            tx_handle,
            silence,
            sizeof(silence),
            &bytesWritten,
            portMAX_DELAY
        );

        // Drain microphone

        i2s_channel_read(
            rx_handle,
            &dummyRead,
            sizeof(dummyRead),
            &bytesRead,
            0
        );
    }
}

// ============================================================
// PLAY MASKED OUTPUT
// ============================================================
//
// IMPORTANT:
// Uses the same TX conversion as the previously working
// mic -> speaker code:
//
// int32_t sampleBuf[j] = ((int32_t)audioBuffer[i+j]) << 16;
// ============================================================

void playMaskedOutput()
{
    Serial.println();
    Serial.println(
        "=========================================="
    );

    Serial.println(
        "PLAYING MASKED OUTPUT"
    );

    Serial.println(
        "MAX98357A SPEAKER ACTIVE"
    );

    Serial.println(
        "=========================================="
    );

    int32_t sampleBuf[HOP_SIZE];

    size_t bytesWritten;

    for (
        int i = 0;
        i < NUM_OUTPUT_SAMPLES;
        i += HOP_SIZE
    )
    {
        int chunkLen =
            min(
                HOP_SIZE,
                NUM_OUTPUT_SAMPLES - i
            );

        // ----------------------------------------------------
        // EXACT SAME FORMAT AS WORKING PLAYBACK
        // ----------------------------------------------------

        for (int j = 0; j < chunkLen; j++)
        {
            sampleBuf[j] =
                ((int32_t)
                recorded_output[i + j])
                << 16;
        }

        // ----------------------------------------------------
        // WRITE TO MAX98357A
        // ----------------------------------------------------

        i2s_channel_write(
            tx_handle,
            sampleBuf,
            chunkLen * sizeof(int32_t),
            &bytesWritten,
            portMAX_DELAY
        );

        // ----------------------------------------------------
        // Drain RX
        // ----------------------------------------------------

        int32_t dummyRead;

        size_t bytesRead;

        i2s_channel_read(
            rx_handle,
            &dummyRead,
            sizeof(dummyRead),
            &bytesRead,
            0
        );
    }

    // --------------------------------------------------------
    // Silence at end
    // --------------------------------------------------------

    for (int i = 0; i < 5; i++)
    {
        writeSilenceBlock();
    }

    Serial.println(
        "PLAYBACK COMPLETE"
    );

    Serial.println();
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
        "=========================================="
    );

    Serial.println(
        "        IMPULSE GUARD V1"
    );

    Serial.println(
        "  MASKED AUDIO RECORD / PLAYBACK TEST"
    );

    Serial.println(
        "=========================================="
    );

    Serial.println();

    // ========================================================
    // PSRAM BUFFER
    // ========================================================

#if CONFIG_SPIRAM

    recorded_output =
        (int16_t*)ps_malloc(
            NUM_OUTPUT_SAMPLES *
            sizeof(int16_t)
        );

    if (recorded_output == nullptr)
    {
        Serial.println(
            "ERROR: PSRAM allocation failed!"
        );

        while (1);
    }

    Serial.printf(
        "Masked output buffer: %d bytes in PSRAM\n",
        NUM_OUTPUT_SAMPLES *
        sizeof(int16_t)
    );

#else

    Serial.printf(
        "Masked output buffer: %d bytes\n",
        NUM_OUTPUT_SAMPLES *
        sizeof(int16_t)
    );

#endif

    // ========================================================
    // PARAMETERS
    // ========================================================

    Serial.printf(
        "Sample rate : %d Hz\n",
        SAMPLE_RATE
    );

    Serial.printf(
        "Frame size  : %d samples\n",
        FRAME_SIZE
    );

    Serial.printf(
        "Record time : %d seconds\n",
        RECORD_SECONDS
    );

    Serial.printf(
        "Pause time  : %d seconds\n",
        PAUSE_SECONDS
    );

    Serial.printf(
        "Volume gain : %.2f\n",
        VOLUME_GAIN
    );

    Serial.println();

    // ========================================================
    // DSP INITIALIZATION
    // ========================================================

    Serial.println(
        "Initializing STFT..."
    );

    stft_init();

    Serial.println(
        "Initializing Bark..."
    );

    subbands_init();

    Serial.println(
        "Initializing mask reconstruction..."
    );

    mask_reconstruction_init();

    Serial.println(
        "Initializing ISTFT..."
    );

    istft_init();

    Serial.println(
        "Initializing GRU..."
    );

    gru_init();

    // ========================================================
    // I2S
    // ========================================================

    startFullDuplexChannel();

    delay(500);

    // ========================================================
    // MICROPHONE TEST
    // ========================================================

    microphoneTest();

    // ========================================================
    // SPEAKER TEST
    // ========================================================

    speakerTest();

    // ========================================================
    // READY
    // ========================================================

    Serial.println(
        "=========================================="
    );

    Serial.println(
        "SYSTEM READY"
    );

    Serial.println(
        "Starting:"
    );

    Serial.println(
        "5 sec RECORD + AI MASK"
    );

    Serial.println(
        "2 sec PAUSE"
    );

    Serial.println(
        "5 sec MASKED PLAYBACK"
    );

    Serial.println(
        "=========================================="
    );

    delay(1000);
}

// ============================================================
// LOOP
// ============================================================

void loop()
{
    // ========================================================
    // 1. RECORD + AI PROCESS
    // ========================================================

    recordAndProcess();

    // ========================================================
    // 2. CHECK STORED MASKED AUDIO
    // ========================================================

    printMaskedOutputLevel();

    // ========================================================
    // 3. PAUSE
    // ========================================================

    pauseBeforePlayback();

    // ========================================================
    // 4. PLAY MASKED AUDIO
    // ========================================================

    playMaskedOutput();

    // ========================================================
    // 5. COMPLETE
    // ========================================================

    Serial.println(
        "=========================================="
    );

    Serial.println(
        "CYCLE COMPLETE"
    );

    Serial.println(
        "Restarting in 1 second..."
    );

    Serial.println(
        "=========================================="
    );

    delay(1000);
}