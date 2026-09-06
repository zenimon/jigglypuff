#!/usr/bin/env python3
"""
extract_and_convert_drone.py
----------------------------
Extracts the University of Glasgow ACSAC 2022 Drone Authentication dataset
from ~/Downloads/ACSAC_2022_Drone_Authentication.7z, resamples all recordings
to 16 kHz mono 16-bit PCM WAV, and saves them to:
  data/raw/Impulse_Guard/ImpulseGuard_dataset/Noise/Non_stationary/Drone/
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import soundfile as sf
import soxr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE_PATH = Path.home() / "Downloads" / "ACSAC_2022_Drone_Authentication.7z"
DEST_DIR = PROJECT_ROOT / "data" / "raw" / "Impulse_Guard" / "ImpulseGuard_dataset" / "Noise" / "Non_stationary" / "Drone"
TEMP_DIR = Path("/home/agniva/.gemini/antigravity-cli/brain/72987316-d363-4427-a806-094e7d5b3b28/scratch/temp_drone_raw")
TARGET_SR = 16000

SUBFOLDERS = ["1to8", "9to16", "17to24", "noise"]

def main():
    if not ARCHIVE_PATH.exists():
        print(f"ERROR: Archive not found at {ARCHIVE_PATH}")
        sys.exit(1)

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print(" Drone Acoustic Dataset Extraction & Conversion")
    print(f" Source: {ARCHIVE_PATH}")
    print(f" Destination: {DEST_DIR}")
    print(f" Target sample rate: {TARGET_SR} Hz (mono 16-bit PCM)")
    print("==================================================")

    total_converted = 0

    for subfolder in SUBFOLDERS:
        print(f"\n---> Extracting subfolder: {subfolder}...")
        extract_cmd = [
            "7z", "x", str(ARCHIVE_PATH),
            f"-i!{subfolder}",
            f"-o{TEMP_DIR}",
            "-y"
        ]
        res = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"ERROR extracting {subfolder}: {res.stderr}")
            sys.exit(1)

        extracted_subfolder = TEMP_DIR / subfolder
        wav_files = sorted(list(extracted_subfolder.glob("*.[wW][aA][vV]")))
        print(f"Found {len(wav_files)} WAV files in {subfolder}. Converting to {TARGET_SR} Hz mono...")

        for idx, wav_path in enumerate(wav_files, start=1):
            out_name = wav_path.stem + ".wav"
            out_path = DEST_DIR / out_name

            # Read source audio
            data, sr = sf.read(wav_path)
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Resample to TARGET_SR if necessary
            if sr != TARGET_SR:
                resampled = soxr.resample(data, sr, TARGET_SR)
            else:
                resampled = data

            # Save as 16-bit PCM WAV
            sf.write(out_path, resampled, TARGET_SR, subtype="PCM_16")
            total_converted += 1

            if idx % 20 == 0 or idx == len(wav_files):
                print(f"  [{idx}/{len(wav_files)}] Converted {out_name}")

        # Clean up temporary raw files for this subfolder to save disk space
        shutil.rmtree(extracted_subfolder, ignore_errors=True)
        print(f"Cleaned up temp raw files for {subfolder}.")

    # Clean up top-level temp dir
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    print("\n==================================================")
    print(f" SUCCESS: Converted {total_converted} recordings to {DEST_DIR}")
    print("==================================================")

if __name__ == "__main__":
    main()
