from pathlib import Path
import librosa
import soundfile as sf

# Change this to wherever your extracted Hindi audio is located
INPUT_FOLDER = Path("D:/Impulse_Guard/ImpulseGuard_dataset/test/audio")

# Output folder
OUTPUT_FOLDER = Path("processed_data/hindi_speech")

# Your project's required sample rate
TARGET_SR = 16000

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Find all WAV files
wav_files = list(INPUT_FOLDER.rglob("*.wav"))

print(f"Found {len(wav_files)} WAV files.")

for i, input_file in enumerate(wav_files, start=1):

    try:
        # Load audio as mono
        audio, sr = librosa.load(
            input_file,
            sr=TARGET_SR,
            mono=True
        )

        # Create a unique output filename
        output_file = OUTPUT_FOLDER / f"hindi_{i:05d}.wav"

        # Save as 16 kHz WAV
        sf.write(
            output_file,
            audio,
            TARGET_SR,
            subtype="PCM_16"
        )

        print(f"[{i}/{len(wav_files)}] {output_file.name}")

    except Exception as e:
        print(f"ERROR: {input_file}")
        print(e)

print("\nConversion complete!")