import pandas as pd
from pathlib import Path
import shutil

# Paths
csv_path = Path("D:/Impulse_Guard/ImpulseGuard_dataset/UrbanSound8K/metadata/UrbanSound8K.csv")

audio_folder = Path("D:/Impulse_Guard/ImpulseGuard_dataset/UrbanSound8K/audio")

output_folder = Path("D:/Impulse_Guard/ImpulseGuard_dataset/Gunshot_Audio")

# Create output folder
output_folder.mkdir(parents=True, exist_ok=True)

# Read metadata
df = pd.read_csv(csv_path)

# Select only gun_shot
gunshots = df[df["class"] == "gun_shot"]

print("Total gunshot files found:", len(gunshots))

# Copy gunshot files
copied = 0
missing = 0

for _, row in gunshots.iterrows():
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
print("Gunshot files in CSV:", len(gunshots))
print("Files copied:", copied)
print("Missing files:", missing)
print("Output folder:", output_folder)