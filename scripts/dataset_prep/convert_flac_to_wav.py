import soundfile as sf
from pathlib import Path

# Location of your original LibriSpeech FLAC dataset
input_folder = Path(
    r"D:/Impulse_Guard/ImpulseGuard_dataset/Speech/Clean_Speech/English/librispeech_train-clean-100"
)

# Location where the converted WAV files will be saved
output_folder = Path(
    r"D:/Impulse_Guard/ImpulseGuard_dataset/Speech/Clean_Speech/English/librispeech_train-clean-100_wav"
)

# Create output folder
output_folder.mkdir(parents=True, exist_ok=True)

# Find every FLAC file inside all subfolders
flac_files = list(input_folder.rglob("*.flac"))

print(f"Found {len(flac_files)} FLAC files.")

if len(flac_files) == 0:
    print("ERROR: No FLAC files were found.")
    print("Please check the input folder path.")
else:
    for i, flac_file in enumerate(flac_files, start=1):

        # Keep the same folder structure
        relative_path = flac_file.relative_to(input_folder)

        # Change .flac to .wav
        wav_file = output_folder / relative_path.with_suffix(".wav")

        # Create required subfolders
        wav_file.parent.mkdir(parents=True, exist_ok=True)

        # Read FLAC
        audio, sample_rate = sf.read(flac_file)

        # Write WAV
        sf.write(wav_file, audio, sample_rate)

        print(f"[{i}/{len(flac_files)}] {flac_file.name} -> {wav_file.name}")

    print("\n================================")
    print("CONVERSION COMPLETE!")
    print(f"Total files converted: {len(flac_files)}")
    print(f"WAV files saved in:")
    print(output_folder)
    print("================================")