# Dataset Architecture & Index

## 1. Overview
The Impulse Guard dataset (`ImpulseGuard_dataset`) comprises **21,220 audio clips** (~79.5 hours) standardized to 16 kHz mono PCM WAV format for speech denoising and impulse noise suppression tasks, including 257 recordings from the University of Glasgow Drone Authentication dataset.

- **Audio Format**: 16,000 Hz, 1-channel (mono), 16-bit PCM WAV
- **Total Duration**: ~79.51 hours
- **Total Audio Files**: 21,220 files

---

## 2. Dataset Hierarchy & Layout

```
data/raw/Impulse_Guard/ImpulseGuard_dataset/
├── Speech/
│   ├── Clean_Speech/English/librispeech_train-clean-100_wav/   (17,345 clips, 60.72 hrs)
│   └── hindi_speech/Audio/                                     (Hindi CommonVoice clips)
└── Noise/
    ├── Impulsive_noise/Gunshot_Audio/                         (374 gunshot clips from UrbanSound8K)
    ├── Stationary/Engine_Idling_Audio/                         (1,000 engine idling clips)
    ├── Non_stationary/
    │   ├── Siren_Audio/                                       (929 siren clips)
    │   ├── Wind_noise/                                        (378 wind clips)
    │   ├── Drone/                                             (257 University of Glasgow drone recordings, 9.70 hrs)
    │   └── Drone_rotor/                                       (7 legacy drone rotor clips)
    └── musan/noise/                                           (930 general noise clips from MUSAN)
```

Aliases are symlinked under `data/raw/` for flexible access:
- `data/raw/speech/` -> `data/raw/Impulse_Guard/ImpulseGuard_dataset/Speech`
- `data/raw/noise/`  -> `data/raw/Impulse_Guard/ImpulseGuard_dataset/Noise`
- `data/raw/ImpulseGuard_dataset/` -> `data/raw/Impulse_Guard/ImpulseGuard_dataset/`

---

## 3. Class & Source Breakdown

### A. Clean Speech
| Category | Source | Clips | Duration (hrs) | Sample Rate |
| :--- | :--- | :--- | :--- | :--- |
| English | LibriSpeech (`train-clean-100`) | 17,345 | 60.72 | 16,000 Hz |
| Hindi | CommonVoice | Sub-corpus | - | 16,000 Hz |
| **Total Speech** | | **17,345** | **60.72** | |

### B. Noise Classes
| Category | Noise Type | Source | Clips | Duration (hrs) |
| :--- | :--- | :--- | :--- | :--- |
| Gunshot | Impulsive | UrbanSound8K | 374 | ~0.25 |
| Engine Idling | Stationary | UrbanSound8K | 1,000 | ~2.50 |
| Siren | Non-stationary | UrbanSound8K | 929 | ~2.32 |
| Wind | Non-stationary | Environmental | 378 | ~0.95 |
| Drone | Non-stationary | Glasgow Drone Auth + Rotor | 264 | ~9.72 |
| General Noise | Mixed / Non-stationary | MUSAN | 930 | ~3.05 |
| **Total Noise** | | | **3,875** | **18.79** |

---

## 4. Train / Validation / Test Splits

Splits follow a 70% / 15% / 15% partition preserved in `data/splits/`:

| Split | Speech Clips | Noise Clips | Total Clips | File List |
| :--- | :--- | :--- | :--- | :--- |
| **Train** (~70%) | 12,141 | 2,707 | 14,848 | `data/splits/speech_train.txt`, `noise_train.txt`, `train.txt`, `drone_train.txt` |
| **Validation** (~15%) | 2,602 | 582 | 3,184 | `data/splits/speech_validation.txt`, `noise_validation.txt`, `validation.txt`, `drone_validation.txt` |
| **Test** (~15%) | 2,602 | 586 | 3,188 | `data/splits/speech_test.txt`, `noise_test.txt`, `test.txt`, `drone_test.txt` |
| **Total** | **17,345** | **3,875** | **21,220** | |

### Drone-Specific Grouped Splits (Leakage-Free by Drone ID)
- **Train (d1..d16 + ambient)**: 177 files (25,099.9s / 6.97 hrs, 71.85% duration, 68.87% files)
- **Validation (d17..d20)**: 40 files (4,934.0s / 1.37 hrs, 14.12% duration, 15.56% files)
- **Test (d21..d24 - Unseen Drones)**: 40 files (4,901.8s / 1.36 hrs, 14.03% duration, 15.56% files)

---

## 5. Metadata Index
- [data/metadata/dataset.json](file:///home/agniva/impuse-guard/data/metadata/dataset.json): Machine-readable overall dataset summary and statistics.
- [data/metadata/drone_metadata.jsonl](file:///home/agniva/impuse-guard/data/metadata/drone_metadata.jsonl): Per-file drone recording attributes (drone ID, date, distance, duration, channels, sample rate).
- [data/metadata/drone_split_metadata.json](file:///home/agniva/impuse-guard/data/metadata/drone_split_metadata.json): Detailed split counts, duration, drone IDs, distance/session distributions, and leakage validation.
- [data/metadata/speech_metadata.jsonl](file:///home/agniva/impuse-guard/data/metadata/speech_metadata.jsonl): Per-file audio duration, channels, sample rate, language.
- [data/metadata/noise_metadata.jsonl](file:///home/agniva/impuse-guard/data/metadata/noise_metadata.jsonl): Per-file category, type (impulsive/stationary/non-stationary), duration, source.
- [data/metadata/samples.jsonl](file:///home/agniva/impuse-guard/data/metadata/samples.jsonl): Synthetic speech-noise mixture pairs and test evaluations.

