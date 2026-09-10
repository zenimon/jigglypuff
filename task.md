# TASK: Fully Analyze ImpulseGuard Codebase and Generate Complete Documentation

You are working on the **ImpulseGuard** project, an edge-AI speech enhancement system designed for noisy and impulsive environments.

Your task is to **deeply inspect and understand the ENTIRE existing codebase first**, and then create/update three documentation files:

1. `README.md`
2. `STUDY.md`
3. `DOCS.md`

The documentation must be based on the **actual implementation in the repository**, not assumptions or generic explanations.

---

# 1. FIRST: UNDERSTAND THE ENTIRE CODEBASE

Before writing any documentation, inspect the repository recursively.

You must examine:

* Every folder
* Every Python file
* Every C/C++/Arduino file
* Every configuration file
* Every JSON/JSONL/YAML/YML file
* Every notebook
* Every script
* Every model-related file
* Dataset-generation code
* Audio-processing code
* STFT/ISTFT implementation
* Filterbank implementation
* Feature extraction
* Model architecture
* Training pipeline
* Evaluation pipeline
* Inference pipeline
* Impulse detection
* Attack/hold/release logic
* ESP32 code
* I2S configuration
* Microphone interface
* Speaker/audio output
* TFLite/TFLite Micro/deployment-related code
* Requirements/dependency files
* Git-related configuration where relevant
* Existing documentation

Also inspect:

* Function definitions
* Classes
* Important variables
* Constants
* Configuration values
* Input/output formats
* Data flow between modules
* Model inputs and outputs
* Tensor shapes
* Audio dimensions
* Sampling rates
* Frame sizes
* Hop sizes
* FFT sizes
* Number of frequency bins
* Number of subbands
* GRU dimensions
* Mask dimensions
* Feature dimensions
* Training parameters
* Evaluation metrics
* Hardware pin configurations
* Latency measurements
* Any comments/docstrings explaining design decisions

Do NOT modify the code during this analysis phase.

---

# 2. TRACE THE PROJECT END-TO-END

After inspecting the files, reconstruct the complete pipeline.

You should understand the project as a complete system:

Dataset
→ Clean Speech
→ Noise / Impulsive Noise
→ Mixing
→ Dataset Metadata
→ STFT
→ Spectral Representation
→ Bark/ERB Filterbank
→ Feature Extraction
→ GRU
→ Subband Mask
→ Mask Processing
→ Reconstruction
→ ISTFT
→ Enhanced Speech

And for V2:

Input Audio
→ STFT
→ Feature Extraction
→ GRU Enhancement
→ Impulse Detection
→ Impulse State
→ Attack/Hold/Release Control
→ Mask / Spectral Control
→ ISTFT
→ Enhanced Audio

Also understand the deployment pipeline:

Python / TensorFlow
→ Trained Model
→ Conversion / Optimization
→ Embedded Deployment
→ ESP32-S3
→ INMP441
→ Audio Processing
→ MAX98357A
→ Speaker

IMPORTANT:

Do not assume that every stage above exists exactly as described.

Verify each stage against the actual code.

If something is planned but not implemented, explicitly distinguish:

* IMPLEMENTED
* PARTIALLY IMPLEMENTED
* EXPERIMENTAL
* PLANNED
* NOT IMPLEMENTED

Do not present planned functionality as completed functionality.

---

# 3. GENERATE README.md

Create or completely rewrite:

`README.md`

The README should be suitable for someone who has never seen this project before.

It should include:

## 3.1 Project Overview

Explain:

* What ImpulseGuard is
* What problem it solves
* Why impulsive noise is difficult
* Target environment
* Why edge processing is important
* Why local processing is useful for sensitive audio

Keep this engineering-focused.

---

## 3.2 Key Features

List only features that are actually implemented or clearly marked as planned.

For example, if supported by the code:

* Neural speech enhancement
* GRU-based processing
* Bark/ERB subband representation
* Complex/real spectral masking
* Impulsive-noise handling
* Impulse detector
* Attack/hold/release control
* Streaming audio processing
* ESP32-S3 deployment
* I2S microphone
* I2S audio output

Verify everything before including it.

---

## 3.3 System Architecture

Create a clear Mermaid diagram showing the complete system.

Include both:

### Training pipeline

Dataset
→ preprocessing
→ mixing
→ STFT
→ feature extraction
→ model
→ loss
→ training
→ validation
→ checkpoint
→ evaluation

### Runtime pipeline

Microphone
→ I2S
→ framing
→ STFT
→ filterbank/features
→ GRU
→ impulse detector/control
→ spectral reconstruction
→ ISTFT
→ audio output

Make the diagram reflect the actual implementation.

---

## 3.4 Repository Structure

Create a complete repository tree.

For example:

```text
ImpulseGuard/
├── data/
├── notebooks/
├── scripts/
├── src/
├── models/
├── results/
├── ...
```

BUT DO NOT INVENT FILES.

Use the actual repository structure.

For **EVERY folder**, explain:

* Purpose
* What type of files it contains
* How it participates in the project

For **EVERY important file**, explain:

* Filename
* Purpose
* Main responsibility
* Inputs
* Outputs
* Important functions/classes
* Dependencies
* Where it fits in the pipeline

The goal is that a new developer can understand the repository without opening every file.

---

## 3.5 Installation

Document the actual environment required.

Include:

* Python version
* Virtual environment setup
* Required Python packages
* TensorFlow version
* NumPy
* SciPy
* Librosa if actually used
* OpenCV if actually used
* Any other dependency actually required

Do not list packages merely because they are commonly used.

Use the project's actual dependency files where available.

---

## 3.6 Running the Project

Document actual commands for:

* Environment setup
* Dataset preparation
* Dataset generation
* Training
* Evaluation
* Inference
* Testing
* ESP32 build/upload

For each command explain what it does.

Do not invent commands.

If a workflow is incomplete, explicitly say so.

---

## 3.7 Model

Document:

* Architecture
* Input
* Output
* Feature dimensions
* Number of subbands
* GRU hidden size
* Dense/output layer
* Parameter count
* Trainable parameter count
* Activation functions
* Loss function
* Optimizer
* Learning rate
* Batch size
* Epochs
* Validation method
* Checkpoint selection

Extract these values from the actual code.

---

## 3.8 Audio Processing

Explain:

* Sampling rate
* Frame length
* Hop length
* FFT size
* Frequency bins
* Window function
* STFT
* Filterbank
* Feature extraction
* Mask generation
* Reconstruction
* ISTFT

Explain the dimensions at each stage.

---

## 3.9 V2 Architecture

If V2 exists in the codebase, document:

* Impulse detection
* Detection features
* Detector output
* State machine
* Attack
* Hold
* Release
* Interaction with the neural model
* Why V2 does not necessarily require retraining the V1 model

Again, distinguish implementation from planned design.

---

## 3.10 Hardware

Document actual hardware:

* ESP32-S3
* INMP441
* MAX98357A
* Speaker
* I2S configuration
* GPIO mapping
* Sample rate
* Buffer configuration
* Audio data format

Create a Mermaid hardware/data-flow diagram.

---

## 3.11 Performance

Document actual measured results found in the repository.

Include metrics such as:

* SI-SNR
* SI-SNR improvement
* PESQ
* STOI
* SI-SDR
* DNSMOS
* Impulse attenuation
* IISRT
* RSDD
* Processing latency
* Memory usage
* Model size

Only include metrics that actually exist.

Clearly distinguish:

* Training results
* Offline evaluation
* PC inference
* Embedded measurements

Do not mix them.

---

## 3.12 Limitations

Document real limitations found from the code.

Examples:

* Unsupported hardware
* Conversion problems
* Memory limitations
* Real-time limitations
* Dataset limitations
* Missing deployment stages
* Audio quality issues

Do not hide incomplete implementation.

---

# 4. GENERATE STUDY.md

Create:

`STUDY.md`

This document should be a **deep technical learning guide for the entire project**.

The goal is:

> If a student reads STUDY.md from beginning to end, they should understand the theory behind every major concept used in ImpulseGuard and understand exactly how that theory maps to the code.

Organize it logically from fundamentals to advanced implementation.

---

# 5. STUDY.md CONTENT

Include sections such as:

## 5.1 Digital Audio Fundamentals

Explain:

* Analog vs digital audio
* Sampling
* Sampling frequency
* Nyquist theorem
* Quantization
* PCM
* Audio amplitude
* RMS
* Peak amplitude
* Clipping
* Dynamic range

Connect each concept to the project's code.

---

## 5.2 Speech and Noise

Explain:

* Speech signal characteristics
* Stationary noise
* Non-stationary noise
* Impulsive noise
* Transient noise
* Why impulsive noise is difficult for conventional speech enhancement

Explain why ImpulseGuard specifically focuses on impulsive events.

---

## 5.3 SNR

Explain:

* Signal power
* Noise power
* SNR
* Input SNR
* Output SNR
* SNR improvement

Give mathematical equations.

Explain how SNR is used in this project.

---

## 5.4 STFT

Explain deeply:

* Why normal FFT is insufficient for changing audio
* Windowing
* Frames
* Hop length
* Overlap
* FFT
* Magnitude
* Phase
* Complex spectrum
* STFT matrix
* Frequency bins
* Time-frequency representation

Use equations.

Then explain exactly how the project's STFT implementation works.

---

## 5.5 ISTFT

Explain:

* Inverse STFT
* Overlap-add
* Window reconstruction
* Why phase matters
* How enhanced spectra become audio again

Map the explanation to the code.

---

## 5.6 Frequency Bins

Explain:

* FFT size
* Number of bins
* Positive-frequency bins
* Why NFFT=512 results in 257 positive-frequency bins for real audio

Explain the project's dimensions.

---

## 5.7 Bark / ERB Filterbank

Explain:

* Human auditory perception
* Bark scale
* ERB scale
* Filterbanks
* Why compressing frequency information can reduce model complexity
* Subband representation

Explain how the project's filterbank maps:

257 FFT bins
→ 22 subbands
→ feature representation

Use equations or diagrams where useful.

---

## 5.8 Feature Extraction

Explain every feature actually used by the project.

For example, if implemented:

* Subband energy
* Magnitude
* Log energy
* Spectral flux
* Energy change
* Crest factor
* High-frequency ratio
* Spectral flatness

For every feature explain:

1. Mathematical definition
2. Physical meaning
3. Why it is useful
4. Where it appears in the code
5. What its output dimension is

---

## 5.9 GRU

Teach:

* RNN
* Vanishing gradients
* Why GRU exists
* Update gate
* Reset gate
* Hidden state
* Temporal context

Include the GRU equations.

Then explain the exact GRU architecture used by ImpulseGuard.

---

## 5.10 Neural Speech Enhancement

Explain:

* Denoising
* Spectral masking
* Neural mask estimation
* Magnitude masks
* Complex masks
* Ideal ratio masks
* Complex ratio masks

Explain exactly which mask approach this project uses.

---

## 5.11 Model Training

Explain the complete training process:

Dataset
→ input/target generation
→ preprocessing
→ feature extraction
→ batching
→ forward pass
→ prediction
→ loss
→ backpropagation
→ optimizer
→ validation
→ checkpoint
→ best model

Explain:

* Inputs
* Targets
* Tensor shapes
* Loss
* Optimizer
* Learning rate
* Epochs
* Batch size
* Validation
* Checkpointing

Explain the actual implementation line-by-line conceptually where useful.

---

## 5.12 Dataset Generation

Explain:

* Clean speech
* Noise
* Impulsive noise
* Mixing
* SNR levels
* Clipping protection
* Metadata
* Train/validation/test split

Explain how the dataset-generation scripts work.

---

## 5.13 Impulse Detection

Explain the theory behind the detector.

For every detector feature explain:

* Formula
* Meaning
* Why it detects impulsive events
* Strengths
* Weaknesses

Then explain the actual detector implementation.

---

## 5.14 State Machine

Explain:

```text
IDLE
 ↓
ONSET
 ↓
HOLD / ACTIVE
 ↓
RECOVERY
 ↓
IDLE
```

Only use states that actually exist.

Explain:

* Transition conditions
* Thresholds
* Timers
* Hysteresis
* Why a state machine is useful

---

## 5.15 Attack / Hold / Release

Explain:

* Attack
* Hold
* Release
* Envelope control
* Temporal smoothing
* Why instant changes can cause artifacts
* Why recovery must be controlled

Explain how the project's parameters affect behavior.

---

## 5.16 Evaluation Metrics

Explain every metric actually used:

* SI-SNR
* SI-SNR improvement
* SI-SDR
* PESQ
* STOI
* DNSMOS
* Impulse attenuation
* IISRT
* RSDD

For every metric explain:

* What it measures
* Formula where appropriate
* Higher/lower is better
* Strength
* Weakness
* Why it matters for ImpulseGuard

Clearly explain why average speech-quality metrics alone may not fully characterize recovery from impulsive events.

---

## 5.17 Latency

Explain:

* Algorithmic latency
* Frame latency
* Buffer latency
* Model inference latency
* End-to-end latency
* Streaming constraints

Explain the actual measured latency in this project.

---

## 5.18 Edge AI

Explain:

* Cloud inference
* Edge inference
* Why military/sensitive audio may require local processing
* Memory constraints
* CPU constraints
* Model size
* Quantization
* TFLite
* TFLite Micro

Explain the project's intended deployment architecture.

---

## 5.19 ESP32-S3

Explain:

* MCU architecture
* CPU
* RAM
* Flash
* DSP capabilities
* Why ESP32-S3 is being used
* Constraints for neural audio processing

---

## 5.20 I2S

Explain:

* BCLK
* LRCLK/WS
* Data line
* Master/slave
* Sample rate
* Bits per sample
* Mono/stereo
* DMA
* Buffering

Then explain the project's actual I2S configuration.

---

## 5.21 INMP441

Explain:

* MEMS microphone
* Digital output
* I2S
* Wiring
* Clock signals
* Data signal

Map it to the project's GPIO configuration.

---

## 5.22 MAX98357A

Explain:

* Digital audio amplifier
* I2S input
* Power
* Speaker output
* Audio chain

Map it to the project's implementation.

---

## 5.23 Complete Runtime Execution

Provide a detailed step-by-step explanation of what happens to **one audio frame** from microphone input to speaker output.

For example:

1. Microphone captures samples
2. DMA receives samples
3. Frame is constructed
4. Window applied
5. FFT performed
6. Frequency spectrum generated
7. Filterbank applied
8. Features generated
9. GRU processes features
10. Mask generated
11. Impulse detector runs
12. Attack/hold/release modifies control
13. Spectrum reconstructed
14. ISTFT performed
15. Audio buffer generated
16. Speaker output

Adjust this to match the actual implementation.

---

# 6. GENERATE DOCS.md

Create:

`DOCS.md`

This should be the project's **technical reference/manual**.

It should be more implementation-oriented than STUDY.md.

Include:

## 6.1 Project Configuration

Document all important configuration parameters.

Create tables like:

| Parameter       | Value | File | Purpose              |
| --------------- | ----: | ---- | -------------------- |
| Sample rate     |   ... | ...  | Audio sampling       |
| Frame size      |   ... | ...  | STFT frame           |
| Hop size        |   ... | ...  | Frame overlap        |
| FFT size        |   ... | ...  | Frequency resolution |
| Subbands        |   ... | ...  | Filterbank           |
| GRU hidden size |   ... | ...  | Model capacity       |

Populate with actual values.

---

## 6.2 File-by-File API Reference

For every important Python module:

* Module
* Functions
* Classes
* Parameters
* Return values
* Side effects
* Dependencies
* Example usage if appropriate

---

## 6.3 Data Formats

Document:

* WAV format
* Sample rate
* Bit depth
* Tensor shapes
* NumPy arrays
* JSON/JSONL metadata
* Model input/output
* Embedded buffers

---

## 6.4 Dataset Metadata

Explain every field in metadata files.

For example:

```json
{
  "id": "...",
  "speech": "...",
  "noise": "...",
  "snr": "...",
  "split": "..."
}
```

Only document fields that actually exist.

---

## 6.5 Model Specification

Provide a formal model specification.

Include:

```text
Input shape
↓
Layer
↓
Output shape
↓
Layer
↓
Output shape
```

Include parameter counts.

---

## 6.6 Training Documentation

Document exact training commands and configuration.

Include:

* Dataset location
* Preprocessing
* Training script
* Configuration
* Checkpoints
* Output directory
* Logs
* Evaluation

---

## 6.7 Evaluation Documentation

Document:

* Evaluation scripts
* Required inputs
* Generated outputs
* Metrics
* Result files
* How to reproduce results

---

## 6.8 Embedded Documentation

Document:

* Arduino/ESP-IDF setup
* Board configuration
* Port
* Flash settings
* PSRAM settings
* I2S pins
* Microphone configuration
* Speaker configuration
* Build/upload commands
* Serial monitor
* Expected output

---

## 6.9 Troubleshooting

Create a troubleshooting section based on actual problems visible in the project/code.

Include issues such as:

* No microphone data
* Zero-valued I2S data
* Incorrect GPIO
* Speaker not producing sound
* Distortion
* Clipping
* Chirping
* Model conversion failure
* Memory problems
* Latency problems

Only include problems relevant to the actual repository/history if evidence exists.

For every issue provide:

Problem
→ Possible cause
→ Diagnostic command/test
→ Fix

---

## 6.10 Reproducibility Guide

Explain how another developer can reproduce:

1. Environment
2. Dataset
3. Training
4. Evaluation
5. Model
6. Embedded deployment
7. Hardware testing

---

# 7. IMPORTANT: CODE-TO-DOCUMENTATION TRACEABILITY

Every important technical claim must be traceable to the code.

When documenting a parameter or behavior, mention the relevant file/function.

Example:

```text
Sample rate: 16 kHz
Defined in: src/audio/config.py
Used by: STFT preprocessing and audio I/O
```

Do this wherever useful.

Do not make documentation generic.

---

# 8. NEVER HALLUCINATE

This is extremely important.

If you cannot find something in the codebase:

DO NOT INVENT IT.

Instead write:

> Not found in the current implementation.

If something is planned:

> Planned / not currently implemented.

If something is experimental:

> Experimental implementation.

If something is partially implemented:

> Partially implemented.

If documentation conflicts with code:

> Treat the code as the source of truth and mention the discrepancy.

---

# 9. KEEP VERSIONS CLEAR

The project contains multiple stages such as V1 and V2.

Do not mix them.

Clearly label:

* V1
* V2
* Experimental
* Planned
* Deployment

If the code contains older implementations, explain their relationship to the current implementation.

---

# 10. DOCUMENT ACTUAL NUMBERS

Extract actual values from the repository.

Important values to verify include:

* 16 kHz sample rate
* 20 ms frame
* 10 ms hop
* 512 FFT
* 257 frequency bins
* 22 subbands
* 44 features/output dimensions where applicable
* GRU hidden size
* Model parameter count
* Training configuration
* Dataset size
* Evaluation dataset size
* SNR levels
* Latency
* Model file sizes
* Memory usage
* Hardware GPIOs

Do not assume these values. Verify them.

---

# 11. DIAGRAMS

Use Mermaid diagrams wherever they improve understanding.

Include diagrams for:

1. Overall system
2. Dataset generation
3. Training pipeline
4. V1 architecture
5. V2 architecture
6. Runtime inference
7. Hardware architecture
8. One-frame processing
9. State machine
10. Deployment pipeline

Keep diagrams readable.

---

# 12. WRITING STYLE

Documentation should be:

* Technical
* Clear
* Structured
* Beginner-friendly but not oversimplified
* Engineering-focused
* Reproducible
* Honest about limitations

Avoid marketing language.

Do not write vague statements such as:

> "AI intelligently enhances audio."

Instead explain the actual mechanism.

---

# 13. FINAL VALIDATION

After generating the three files, perform a final verification pass.

Check:

### README.md

* Does every folder appear?
* Are important files explained?
* Are commands correct?
* Are architecture diagrams correct?
* Are model specifications correct?
* Are hardware details correct?

### STUDY.md

* Does every major concept used in the codebase have an explanation?
* Are mathematical concepts explained?
* Is theory connected to implementation?
* Is training explained?
* Is inference explained?
* Is embedded deployment explained?

### DOCS.md

* Can another developer reproduce the project?
* Are configuration values documented?
* Are functions/modules documented?
* Are data formats documented?
* Are hardware instructions documented?
* Are troubleshooting steps included?

---

# 14. IMPORTANT FINAL REQUIREMENT

Do NOT simply generate three generic documentation files.

Your workflow must be:

```text
INSPECT ENTIRE CODEBASE
        ↓
UNDERSTAND ARCHITECTURE
        ↓
TRACE DATA FLOW
        ↓
TRACE TRAINING FLOW
        ↓
TRACE INFERENCE FLOW
        ↓
TRACE EMBEDDED FLOW
        ↓
VERIFY PARAMETERS
        ↓
IDENTIFY IMPLEMENTED VS PLANNED
        ↓
GENERATE README.md
        ↓
GENERATE STUDY.md
        ↓
GENERATE DOCS.md
        ↓
CROSS-CHECK DOCUMENTATION AGAINST CODE
        ↓
FIX INCONSISTENCIES
```

The **codebase is the source of truth**.

Do not modify project source code unless absolutely necessary for documentation-related tooling.

Your final response should briefly report:

* Files inspected
* Files created/updated
* Major architecture understood
* Any inconsistencies discovered
* Any undocumented/planned features that were identified
* Whether README.md, STUDY.md, and DOCS.md passed the final consistency check
