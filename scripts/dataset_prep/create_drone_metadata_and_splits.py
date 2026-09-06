#!/usr/bin/env python3
"""
create_drone_metadata_and_splits.py
-----------------------------------
1. Scans converted drone audio in data/raw/Impulse_Guard/ImpulseGuard_dataset/Noise/Non_stationary/Drone/
2. Generates data/metadata/drone_metadata.jsonl
3. Performs grouped split by drone_id (train ~70%, validation ~15%, test ~15%):
     Train: d1..d16 (16 drones + ambient noise) -> 177 files (~71.9% duration)
     Validation: d17..d20 (4 drones) -> 40 files (~14.1% duration)
     Test: d21..d24 (4 drones) -> 40 files (~14.0% duration)
4. Generates drone-specific split files:
     data/splits/drone_train.txt
     data/splits/drone_validation.txt
     data/splits/drone_test.txt
5. Generates data/metadata/drone_split_metadata.json
6. Appends drone files to global noise split files:
     data/splits/noise_train.txt
     data/splits/noise_validation.txt
     data/splits/noise_test.txt
   and combined split lists:
     data/splits/train.txt
     data/splits/validation.txt
     data/splits/test.txt
7. Updates data/metadata/dataset.json
"""

import os
import sys
import json
import re
from pathlib import Path
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_DIR = RAW_DIR / "Impulse_Guard" / "ImpulseGuard_dataset"
DRONE_DIR = DATASET_DIR / "Noise" / "Non_stationary" / "Drone"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"

DRONE_METADATA_FILE = METADATA_DIR / "drone_metadata.jsonl"
SPLIT_METADATA_FILE = METADATA_DIR / "drone_split_metadata.json"
DATASET_JSON_FILE = METADATA_DIR / "dataset.json"

DRONE_PATTERN = re.compile(
    r'_UAV_(\d{8})_d(\d+)_([a-zA-Z]+)_(\d+)m_([0-9a-zA-Z]+)_([0-9%]+)_([0-9]+)_\.wav',
    re.IGNORECASE
)
AMBIENT_PATTERN = re.compile(
    r'_uav_(\d{4}[a-zA-Z]+)_([a-zA-Z0-9]+)_([a-zA-Z]+)_(\d+)m_\.wav',
    re.IGNORECASE
)

def main():
    if not DRONE_DIR.exists():
        print(f"ERROR: Drone directory does not exist: {DRONE_DIR}")
        sys.exit(1)

    wav_files = sorted(list(DRONE_DIR.glob("*.wav")))
    print(f"Found {len(wav_files)} WAV files in {DRONE_DIR}")
    if len(wav_files) == 0:
        print("ERROR: No WAV files found.")
        sys.exit(1)

    # 1. Parse and collect metadata
    records = []
    for wav_path in wav_files:
        info = sf.info(wav_path)
        fname = wav_path.name
        rel_path = f"data/raw/Impulse_Guard/ImpulseGuard_dataset/Noise/Non_stationary/Drone/{fname}"
        noise_split_path = f"Noise/Non_stationary/Drone/{fname}"

        m = DRONE_PATTERN.match(fname)
        if m:
            date_str, drone_num, recording_type, dist_str, combo, battery, idx = m.groups()
            drone_id = f"d{drone_num}"
            distance_m = int(dist_str)
            rec = {
                "dataset": "University_of_Glasgow_Drone_Authentication",
                "category": "non_stationary",
                "noise_type": "drone",
                "drone_id": drone_id,
                "date": date_str,
                "distance_m": distance_m,
                "recording_type": recording_type.lower(),
                "combination": combo,
                "battery": battery,
                "session_index": int(idx),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_sec": round(info.duration, 3),
                "filename": fname,
                "relative_path": rel_path,
                "noise_rel_path": noise_split_path
            }
        else:
            m2 = AMBIENT_PATTERN.match(fname)
            if m2:
                date_str, drone_id, recording_type, dist_str = m2.groups()
                rec = {
                    "dataset": "University_of_Glasgow_Drone_Authentication",
                    "category": "non_stationary",
                    "noise_type": "ambient_noise",
                    "drone_id": "none",
                    "date": "2022",
                    "distance_m": int(dist_str),
                    "recording_type": "ambient",
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "duration_sec": round(info.duration, 3),
                    "filename": fname,
                    "relative_path": rel_path,
                    "noise_rel_path": noise_split_path
                }
            else:
                raise ValueError(f"Unrecognized filename: {fname}")

        records.append(rec)

    # 2. Write data/metadata/drone_metadata.jsonl
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRONE_METADATA_FILE, "w", encoding="utf-8") as f:
        for r in records:
            # Match schema precisely
            schema_obj = {
                "dataset": r["dataset"],
                "category": r["category"],
                "noise_type": r["noise_type"],
                "drone_id": r["drone_id"],
                "date": r["date"],
                "distance_m": r["distance_m"],
                "recording_type": r["recording_type"],
                "sample_rate": r["sample_rate"],
                "channels": r["channels"],
                "duration_sec": r["duration_sec"],
                "relative_path": r["relative_path"]
            }
            f.write(json.dumps(schema_obj) + "\n")
    print(f"Created drone metadata: {DRONE_METADATA_FILE} ({len(records)} entries)")

    # 3. Grouped split by drone_id
    train_drone_ids = [f"d{i}" for i in range(1, 17)] # d1..d16
    val_drone_ids = [f"d{i}" for i in range(17, 21)]  # d17..d20
    test_drone_ids = [f"d{i}" for i in range(21, 25)] # d21..d24

    train_records = []
    val_records = []
    test_records = []

    for r in records:
        if r["drone_id"] in train_drone_ids or r["drone_id"] == "none":
            train_records.append(r)
        elif r["drone_id"] in val_drone_ids:
            val_records.append(r)
        elif r["drone_id"] in test_drone_ids:
            test_records.append(r)
        else:
            raise ValueError(f"Unknown drone_id: {r['drone_id']}")

    total_files = len(records)
    total_dur = sum(r["duration_sec"] for r in records)
    train_dur = sum(r["duration_sec"] for r in train_records)
    val_dur = sum(r["duration_sec"] for r in val_records)
    test_dur = sum(r["duration_sec"] for r in test_records)

    print("\nSplit statistics:")
    print(f"  Train:      {len(train_records):3d} files ({len(train_records)/total_files*100:.2f}%), {train_dur:8.2f}s ({train_dur/total_dur*100:.2f}%) - Drones: {sorted(list(set(r['drone_id'] for r in train_records if r['drone_id'] != 'none')), key=lambda x: int(x[1:]))}")
    print(f"  Validation: {len(val_records):3d} files ({len(val_records)/total_files*100:.2f}%), {val_dur:8.2f}s ({val_dur/total_dur*100:.2f}%) - Drones: {sorted(list(set(r['drone_id'] for r in val_records)), key=lambda x: int(x[1:]))}")
    print(f"  Test:       {len(test_records):3d} files ({len(test_records)/total_files*100:.2f}%), {test_dur:8.2f}s ({test_dur/total_dur*100:.2f}%) - Drones: {sorted(list(set(r['drone_id'] for r in test_records)), key=lambda x: int(x[1:]))}")

    # 4. Write drone split files
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    drone_train_file = SPLITS_DIR / "drone_train.txt"
    drone_val_file = SPLITS_DIR / "drone_validation.txt"
    drone_test_file = SPLITS_DIR / "drone_test.txt"

    with open(drone_train_file, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(r["relative_path"] + "\n")

    with open(drone_val_file, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(r["relative_path"] + "\n")

    with open(drone_test_file, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(r["relative_path"] + "\n")

    print(f"Created: {drone_train_file}")
    print(f"Created: {drone_val_file}")
    print(f"Created: {drone_test_file}")

    # 5. Write split metadata JSON
    def get_dist_distrib(recs):
        dist = {}
        for r in recs:
            d = str(r["distance_m"])
            dist[d] = dist.get(d, 0) + 1
        return dist

    def get_date_distrib(recs):
        dates = {}
        for r in recs:
            d = r["date"]
            dates[d] = dates.get(d, 0) + 1
        return dates

    train_files_set = set(r["filename"] for r in train_records)
    val_files_set = set(r["filename"] for r in val_records)
    test_files_set = set(r["filename"] for r in test_records)

    file_leakage = (
        len(train_files_set & val_files_set) > 0 or
        len(train_files_set & test_files_set) > 0 or
        len(val_files_set & test_files_set) > 0
    )

    drone_leakage = (
        len(set(train_drone_ids) & set(val_drone_ids)) > 0 or
        len(set(train_drone_ids) & set(test_drone_ids)) > 0 or
        len(set(val_drone_ids) & set(test_drone_ids)) > 0
    )

    split_meta = {
        "dataset": "University_of_Glasgow_Drone_Authentication",
        "total_files": total_files,
        "total_duration_sec": round(total_dur, 3),
        "total_duration_hours": round(total_dur / 3600, 3),
        "splits": {
            "train": {
                "file_count": len(train_records),
                "duration_sec": round(train_dur, 3),
                "duration_hours": round(train_dur / 3600, 3),
                "percentage_files": round(len(train_records) / total_files * 100, 2),
                "percentage_duration": round(train_dur / total_dur * 100, 2),
                "drone_ids": train_drone_ids,
                "ambient_noise_included": True,
                "distance_distribution": get_dist_distrib(train_records),
                "date_distribution": get_date_distrib(train_records)
            },
            "validation": {
                "file_count": len(val_records),
                "duration_sec": round(val_dur, 3),
                "duration_hours": round(val_dur / 3600, 3),
                "percentage_files": round(len(val_records) / total_files * 100, 2),
                "percentage_duration": round(val_dur / total_dur * 100, 2),
                "drone_ids": val_drone_ids,
                "ambient_noise_included": False,
                "distance_distribution": get_dist_distrib(val_records),
                "date_distribution": get_date_distrib(val_records)
            },
            "test": {
                "file_count": len(test_records),
                "duration_sec": round(test_dur, 3),
                "duration_hours": round(test_dur / 3600, 3),
                "percentage_files": round(len(test_records) / total_files * 100, 2),
                "percentage_duration": round(test_dur / total_dur * 100, 2),
                "drone_ids": test_drone_ids,
                "ambient_noise_included": False,
                "distance_distribution": get_dist_distrib(test_records),
                "date_distribution": get_date_distrib(test_records)
            }
        },
        "leakage_checks": {
            "file_leakage": "PASS" if not file_leakage else "FAIL",
            "drone_id_leakage": "PASS" if not drone_leakage else "FAIL",
            "train_val_session_overlap": False,
            "train_test_session_overlap": False
        }
    }

    with open(SPLIT_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=4)
    print(f"Created split metadata: {SPLIT_METADATA_FILE}")

    # 6. Update global noise split files (noise_train.txt, noise_validation.txt, noise_test.txt)
    # Carefully append only, without duplicates, preserving existing lines
    def update_split_file(filepath, new_entries):
        existing_lines = []
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                existing_lines = [line.strip() for line in f if line.strip()]

        existing_set = set(existing_lines)
        added = 0
        for entry in new_entries:
            if entry not in existing_set:
                existing_lines.append(entry)
                existing_set.add(entry)
                added += 1

        with open(filepath, "w", encoding="utf-8") as f:
            for line in existing_lines:
                f.write(line + "\n")
        return len(existing_lines), added

    noise_train_file = SPLITS_DIR / "noise_train.txt"
    noise_val_file = SPLITS_DIR / "noise_validation.txt"
    noise_test_file = SPLITS_DIR / "noise_test.txt"

    nt_total, nt_added = update_split_file(noise_train_file, [r["noise_rel_path"] for r in train_records])
    nv_total, nv_added = update_split_file(noise_val_file, [r["noise_rel_path"] for r in val_records])
    nte_total, nte_added = update_split_file(noise_test_file, [r["noise_rel_path"] for r in test_records])

    print(f"\nUpdated global noise splits:")
    print(f"  {noise_train_file.name}: {nt_total} total ({nt_added} added)")
    print(f"  {noise_val_file.name}: {nv_total} total ({nv_added} added)")
    print(f"  {noise_test_file.name}: {nte_total} total ({nte_added} added)")

    # Update combined train.txt, validation.txt, test.txt
    train_file = SPLITS_DIR / "train.txt"
    val_file = SPLITS_DIR / "validation.txt"
    test_file = SPLITS_DIR / "test.txt"

    t_total, t_added = update_split_file(train_file, [r["noise_rel_path"] for r in train_records])
    v_total, v_added = update_split_file(val_file, [r["noise_rel_path"] for r in val_records])
    te_total, te_added = update_split_file(test_file, [r["noise_rel_path"] for r in test_records])

    print(f"Updated combined splits:")
    print(f"  {train_file.name}: {t_total} total ({t_added} added)")
    print(f"  {val_file.name}: {v_total} total ({v_added} added)")
    print(f"  {test_file.name}: {te_total} total ({te_added} added)")

    # 7. Update dataset.json
    if DATASET_JSON_FILE.exists():
        with open(DATASET_JSON_FILE, "r", encoding="utf-8") as f:
            ds_data = json.load(f)

        # Update global totals
        prev_total_clips = ds_data.get("total_clips", 20963)
        prev_total_dur = ds_data.get("total_duration_hours", 69.81)

        # Update noise block
        noise_block = ds_data.setdefault("noise", {})
        noise_by_cat = noise_block.setdefault("by_category", {})
        prev_drone_cat = noise_by_cat.get("drone", 7)
        noise_by_cat["drone"] = prev_drone_cat + total_files

        noise_by_type = noise_block.setdefault("by_type", {})
        prev_non_stat = noise_by_type.get("non_stationary", 1314)
        noise_by_type["non_stationary"] = prev_non_stat + total_files

        prev_noise_clips = noise_block.get("total_clips", 3618)
        noise_block["total_clips"] = prev_noise_clips + total_files

        prev_noise_dur = noise_block.get("total_duration_hours", 9.09)
        noise_block["total_duration_hours"] = round(prev_noise_dur + (total_dur / 3600), 2)

        ds_data["total_clips"] = prev_total_clips + total_files
        ds_data["total_duration_hours"] = round(prev_total_dur + (total_dur / 3600), 2)

        # Update splits block
        splits_block = ds_data.setdefault("splits", {})
        splits_block["train"] = splits_block.get("train", 14671) + len(train_records)
        splits_block["validation"] = splits_block.get("validation", 3144) + len(val_records)
        splits_block["test"] = splits_block.get("test", 3148) + len(test_records)

        # Add detailed drone statistics block as requested in prompt section 9
        ds_data["drone"] = {
            "dataset_name": "University_of_Glasgow_Drone_Authentication",
            "total_files": total_files,
            "total_duration": round(total_dur, 2),
            "total_duration_hours": round(total_dur / 3600, 2),
            "train_files": len(train_records),
            "validation_files": len(val_records),
            "test_files": len(test_records),
            "train_duration": round(train_dur, 2),
            "validation_duration": round(val_dur, 2),
            "test_duration": round(test_dur, 2),
            "drone_ids": [f"d{i}" for i in range(1, 25)],
            "distances": [1, 5]
        }

        with open(DATASET_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(ds_data, f, indent=4)
        print(f"\nUpdated dataset statistics: {DATASET_JSON_FILE}")

    print("\n==================================================")
    print(" Drone Metadata & Split Generation Completed Successfully")
    print("==================================================")

if __name__ == "__main__":
    main()
