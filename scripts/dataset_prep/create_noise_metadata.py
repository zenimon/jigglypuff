import json
from pathlib import Path
import soundfile as sf


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path("D:/Impulse_Guard")

DATASET_DIR = PROJECT_ROOT / "ImpulseGuard_dataset"

NOISE_DIR = DATASET_DIR / "Noise"

METADATA_DIR = PROJECT_ROOT / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = METADATA_DIR / "noise_metadata.jsonl"


# =========================================================
# NOISE SOURCES
# =========================================================

NOISE_SOURCES = {

    # -----------------------------------------------------
    # 1. GUNSHOT
    # -----------------------------------------------------

    "gunshot": {
        "folder": NOISE_DIR / "Impulsive_noise" / "Gunshot_Audio",
        "category": "gunshot",
        "type": "impulsive",
        "source": "UrbanSound8K"
    },


    # -----------------------------------------------------
    # 2. ENGINE IDLING
    # -----------------------------------------------------

    "engine_idling": {
        "folder": NOISE_DIR / "Stationary" / "Engine_Idling_Audio",
        "category": "engine_idling",
        "type": "stationary",
        "source": "UrbanSound8K"
    },


    # -----------------------------------------------------
    # 3. SIREN
    # -----------------------------------------------------

    "siren": {
        "folder": NOISE_DIR / "Non_stationary" / "Siren_Audio",
        "category": "siren",
        "type": "non_stationary",
        "source": "UrbanSound8K"
    },


    # -----------------------------------------------------
    # 4. DRONE - TEST
    # -----------------------------------------------------

    "drone_test": {
        "folder": NOISE_DIR / "Non_stationary" / "Drone_rotor" / "noises-test-drones",
        "category": "drone",
        "type": "non_stationary",
        "source": "Drone Dataset"
    },


    # -----------------------------------------------------
    # 5. DRONE - TRAIN
    # -----------------------------------------------------

    "drone_train": {
        "folder": NOISE_DIR / "Non_stationary" / "Drone_rotor" / "noises-train-drones",
        "category": "drone",
        "type": "non_stationary",
        "source": "Drone Dataset"
    },


    # -----------------------------------------------------
    # 6. NORMAL WIND
    # -----------------------------------------------------

    "normal_wind": {
        "folder": NOISE_DIR / "Non_stationary" / "Wind_noise" / "normal_wind",
        "category": "wind",
        "type": "non_stationary",
        "source": "Wind Dataset"
    },


    # -----------------------------------------------------
    # 7. STRONG WIND
    # -----------------------------------------------------

    "strong_wind": {
        "folder": NOISE_DIR / "Non_stationary" / "Wind_noise" / "strong_wind",
        "category": "wind",
        "type": "non_stationary",
        "source": "Wind Dataset"
    },


    # -----------------------------------------------------
    # 8. MUSAN - FREE SOUND
    # -----------------------------------------------------

    "musan_free_sound": {
        "folder": NOISE_DIR / "musan" / "noise" / "free-sound",
        "category": "musan",
        "type": "musan",
        "source": "MUSAN"
    },


    # -----------------------------------------------------
    # 9. MUSAN - SOUND BIBLE
    # -----------------------------------------------------

    "musan_sound_bible": {
        "folder": NOISE_DIR / "musan" / "noise" / "sound-bible",
        "category": "musan",
        "type": "musan",
        "source": "MUSAN"
    }
}


# =========================================================
# CREATE METADATA
# =========================================================

total_count = 0

category_counts = {}


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for noise_name, info in NOISE_SOURCES.items():

        folder = info["folder"]

        print("\n-----------------------------------------")
        print("Scanning:", noise_name)
        print("Folder:", folder)
        print("-----------------------------------------")

        # Check folder
        if not folder.exists():

            print("WARNING: Folder not found!")
            continue


        file_count = 0


        # Search WAV files recursively
        for audio_file in sorted(folder.rglob("*.wav")):

            try:

                # -----------------------------------------
                # Read audio information
                # -----------------------------------------

                audio_info = sf.info(audio_file)


                # -----------------------------------------
                # Path relative to ImpulseGuard_dataset
                # -----------------------------------------

                relative_path = audio_file.relative_to(DATASET_DIR)


                # -----------------------------------------
                # Create metadata record
                # -----------------------------------------

                metadata = {

                    "file": str(relative_path).replace("\\", "/"),

                    "source": info["source"],

                    "category": info["category"],

                    "type": info["type"],

                    "duration_sec": round(
                        audio_info.duration, 3
                    ),

                    "sample_rate": audio_info.samplerate,

                    "channels": audio_info.channels

                }


                # -----------------------------------------
                # Write JSON object
                # -----------------------------------------

                f.write(
                    json.dumps(metadata)
                    + "\n"
                )


                total_count += 1
                file_count += 1


            except Exception as e:

                print("\nCould not read:")
                print(audio_file)

                print("Error:")
                print(e)


        print("Files found:", file_count)


        # Save category count
        category = info["category"]

        if category not in category_counts:
            category_counts[category] = 0

        category_counts[category] += file_count


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("============================================")
print("       NOISE METADATA CREATION COMPLETE")
print("============================================")

print("\nTotal noise recordings:", total_count)

print("\nRecordings by category:")

for category, count in category_counts.items():

    print(f"  {category}: {count}")

print("\nMetadata file created at:")

print(OUTPUT_FILE)

print("\n============================================")