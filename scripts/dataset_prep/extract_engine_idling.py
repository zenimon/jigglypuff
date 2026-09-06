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
    "D:/Impulse_Guard/ImpulseGuard_dataset/Engine_Idling_Audio"
)

# Create output folder
output_folder.mkdir(parents=True, exist_ok=True)

# Read metadata
df = pd.read_csv(csv_path)

# Select only engine idling
engine_idling = df[df["class"] == "engine_idling"]

print("Total engine idling files found:", len(engine_idling))

# Copy engine idling files
copied = 0
missing = 0

for _, row in engine_idling.iterrows():

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
print("Engine idling files in CSV:", len(engine_idling))
print("Files copied:", copied)
print("Missing files:", missing)
print("Output folder:", output_folder)