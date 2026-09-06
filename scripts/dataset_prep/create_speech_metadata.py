import json
from pathlib import Path
import soundfile as sf

# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path("D:/Impulse_Guard")

# Your raw speech folder
SPEECH_DIR = PROJECT_ROOT / "ImpulseGuard_dataset" / "Speech"

# Your metadata folder
METADATA_DIR = PROJECT_ROOT / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Output metadata file
OUTPUT_FILE = METADATA_DIR / "speech_metadata.jsonl"


# ==========================================
# SPEECH FOLDERS
# ==========================================

SPEECH_SOURCES = {
    "english": SPEECH_DIR / "Clean_Speech",
    "hindi": SPEECH_DIR / "hindi_speech"
}


# ==========================================
# CREATE METADATA
# ==========================================

count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for language, folder in SPEECH_SOURCES.items():

        print(f"\nScanning {language}: {folder}")

        if not folder.exists():
            print("WARNING: Folder not found:", folder)
            continue

        # Search all WAV files inside the folder
        for audio_file in sorted(folder.rglob("*.wav")):

            try:
                # Get audio information
                audio_info = sf.info(audio_file)

                # Path relative to ImpulseGuard_dataset
                relative_path = audio_file.relative_to(
                    PROJECT_ROOT / "ImpulseGuard_dataset"
                )

                metadata = {
                    "file": str(relative_path).replace("\\", "/"),
                    "language": language,
                    "source": "speech_dataset",
                    "duration_sec": round(audio_info.duration, 3),
                    "sample_rate": audio_info.samplerate,
                    "channels": audio_info.channels
                }

                # Write one JSON object per line
                f.write(json.dumps(metadata) + "\n")

                count += 1

            except Exception as e:

                print("Could not read:", audio_file)
                print("Error:", e)


# ==========================================
# FINISHED
# ==========================================

print("\n===================================")
print("SPEECH METADATA CREATION COMPLETE")
print("===================================")
print("Total speech recordings:", count)
print("Metadata file:", OUTPUT_FILE)