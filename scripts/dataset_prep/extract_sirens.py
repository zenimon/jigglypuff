import pandas as pd
from pathlib import Path
import shutil

# Paths
csv_path = Path(
    "D:/Impulse_Guard/ImpulseGuard_dataset/UrbanSound8K/metadata/UrbanSound8K.csv"
)

audio_folder = Path(
    "D:/Impulse_Guard/ImpulseGuard_dataset/UrbanSound8K/audio"
)

output_folder = Path(
    "D:/Impulse_Guard/ImpulseGuard_dataset/Siren_Audio"
)

# Create output folder
output_folder.mkdir(parents=True, exist_ok=True)

# Read metadata
df = pd.read_csv(csv_path)

# Select only siren
sirens = df[df["class"] == "siren"]

print("Total siren files found:", len(sirens))

# Copy siren files
copied = 0
missing = 0

for _, row in sirens.iterrows():

    source_file = (
        audio_folder
        / f"fold{row['fold']}"
        / row["slice_file_name"]
    )

    destination_file = output_folder / row["slice_file_name"]

    if source_file.exists():
        shutil.copy2(source_file, destination_file)
        copied += 1
    else:
        print("Missing:", source_file)
        missing += 1

print("\n--- EXTRACTION COMPLETE ---")
print("Siren files in CSV:", len(sirens))
print("Files copied:", copied)
print("Missing files:", missing)
print("Output folder:", output_folder)