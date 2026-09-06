#!/usr/bin/env python3
"""
validate_dataset.py
-------------------
Comprehensive dataset validation script for Impulse Guard.
Performs:
  1. Path checks (all split paths exist, no broken paths, no duplicate paths)
  2. Split leakage checks (no file overlap, no drone ID overlap, session leakage analysis)
  3. Audio checks (readable WAV, sample rate == 16000, mono, finite samples, duration > 0)
  4. Metadata checks (every drone WAV has metadata, every metadata path exists,
                      split files have corresponding metadata)
Prints a clear, formatted summary report.
"""

import sys
import os
import json
import re
from pathlib import Path
import soundfile as sf
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

def resolve_path(p_str: str) -> Path:
    p_str = p_str.strip()
    p = Path(p_str)
    if p.is_file():
        return p
    if (PROJECT_ROOT / p_str).is_file():
        return PROJECT_ROOT / p_str
    if (RAW_DIR / p_str).is_file():
        return RAW_DIR / p_str
    if (DATASET_DIR / p_str).is_file():
        return DATASET_DIR / p_str
    return None

def extract_drone_id(filename: str):
    m = DRONE_PATTERN.match(filename)
    if m:
        return f"d{m.group(2)}"
    m2 = AMBIENT_PATTERN.match(filename)
    if m2:
        return "ambient"
    return None

def extract_session_date(filename: str):
    m = DRONE_PATTERN.match(filename)
    if m:
        return m.group(1)
    m2 = AMBIENT_PATTERN.match(filename)
    if m2:
        return m2.group(1)
    return None

def main():
    print("=" * 60)
    print("           IMPULSE GUARD - DATASET VALIDATION")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Inspect on-disk Drone files
    # -------------------------------------------------------------
    if not DRONE_DIR.exists():
        print(f"[FAIL] Drone directory does not exist: {DRONE_DIR}")
        sys.exit(1)

    drone_files_on_disk = sorted(list(DRONE_DIR.glob("*.wav")))
    drone_filenames_on_disk = {f.name for f in drone_files_on_disk}
    print(f"\nDiscovered on-disk drone files: {len(drone_files_on_disk)}")

    # -------------------------------------------------------------
    # 2. Metadata validation
    # -------------------------------------------------------------
    metadata_records = []
    metadata_by_filename = {}
    missing_metadata_paths = 0

    if not DRONE_METADATA_FILE.exists():
        print(f"[FAIL] Metadata file missing: {DRONE_METADATA_FILE}")
        sys.exit(1)

    with open(DRONE_METADATA_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            metadata_records.append(rec)
            resolved = resolve_path(rec["relative_path"])
            if resolved is None:
                missing_metadata_paths += 1
            else:
                metadata_by_filename[resolved.name] = rec

    missing_metadata_files = drone_filenames_on_disk - set(metadata_by_filename.keys())

    # -------------------------------------------------------------
    # 3. Split file loading and path checks
    # -------------------------------------------------------------
    split_names = ["drone_train", "drone_validation", "drone_test"]
    split_files = {}
    split_resolved = {}
    split_drone_ids = {}
    split_dates = {}
    split_durations = {}

    missing_files_count = 0
    duplicate_paths_count = 0

    for name in split_names:
        filepath = SPLITS_DIR / f"{name}.txt"
        if not filepath.exists():
            print(f"[FAIL] Split file missing: {filepath}")
            sys.exit(1)

        with open(filepath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        split_files[name] = lines
        resolved_list = []
        d_ids = set()
        dates = set()
        dur = 0.0

        seen_in_split = set()
        for p_str in lines:
            if p_str in seen_in_split:
                duplicate_paths_count += 1
            seen_in_split.add(p_str)

            res = resolve_path(p_str)
            if res is None:
                missing_files_count += 1
            else:
                resolved_list.append(res)
                did = extract_drone_id(res.name)
                if did:
                    d_ids.add(did)
                d = extract_session_date(res.name)
                if d:
                    dates.add(d)
                if res.name in metadata_by_filename:
                    dur += metadata_by_filename[res.name]["duration_sec"]

        split_resolved[name] = resolved_list
        split_drone_ids[name] = sorted(list(d_ids), key=lambda x: int(x[1:]) if x.startswith('d') else 999)
        split_dates[name] = sorted(list(dates))
        split_durations[name] = dur

    # Check that split files match on-disk drone files
    all_split_filenames = set()
    for name in split_names:
        all_split_filenames.update(f.name for f in split_resolved[name])

    unaccounted_files = drone_filenames_on_disk - all_split_filenames
    orphan_split_files = all_split_filenames - drone_filenames_on_disk

    # -------------------------------------------------------------
    # 4. Leakage checks
    # -------------------------------------------------------------
    train_files = {f.name for f in split_resolved["drone_train"]}
    val_files = {f.name for f in split_resolved["drone_validation"]}
    test_files = {f.name for f in split_resolved["drone_test"]}

    file_leak_train_val = train_files & val_files
    file_leak_train_test = train_files & test_files
    file_leak_val_test = val_files & test_files
    total_file_leakage = len(file_leak_train_val) + len(file_leak_train_test) + len(file_leak_val_test)

    train_drones = {d for d in split_drone_ids["drone_train"] if d != "ambient"}
    val_drones = {d for d in split_drone_ids["drone_validation"] if d != "ambient"}
    test_drones = {d for d in split_drone_ids["drone_test"] if d != "ambient"}

    drone_leak_train_val = train_drones & val_drones
    drone_leak_train_test = train_drones & test_drones
    drone_leak_val_test = val_drones & test_drones
    total_drone_leakage = len(drone_leak_train_val) + len(drone_leak_train_test) + len(drone_leak_val_test)

    # Date / Session overlap checks
    train_dates = set(split_dates["drone_train"])
    val_dates = set(split_dates["drone_validation"])
    test_dates = set(split_dates["drone_test"])

    train_val_date_leak = train_dates & val_dates
    train_test_date_leak = train_dates & test_dates

    # Check global noise splits leakage
    global_noise_splits = {}
    global_noise_duplicates = 0
    global_missing_files = 0
    for split_key in ["noise_train", "noise_validation", "noise_test"]:
        g_path = SPLITS_DIR / f"{split_key}.txt"
        if g_path.exists():
            with open(g_path, "r", encoding="utf-8") as f:
                g_lines = [l.strip() for l in f if l.strip()]
            global_noise_splits[split_key] = set(g_lines)
            for l in g_lines:
                if resolve_path(l) is None:
                    global_missing_files += 1

    noise_cross_leakage = 0
    if len(global_noise_splits) == 3:
        n_tr = global_noise_splits["noise_train"]
        n_va = global_noise_splits["noise_validation"]
        n_te = global_noise_splits["noise_test"]
        noise_cross_leakage = len(n_tr & n_va) + len(n_tr & n_te) + len(n_va & n_te)

    # -------------------------------------------------------------
    # 5. Audio sanity checks
    # -------------------------------------------------------------
    corrupt_files = 0
    sr_violations = 0
    channel_violations = 0
    duration_violations = 0
    non_finite_count = 0

    for f in drone_files_on_disk:
        try:
            info = sf.info(f)
            if info.samplerate != 16000:
                sr_violations += 1
            if info.channels != 1:
                channel_violations += 1
            if info.duration <= 0:
                duration_violations += 1

            data, sr = sf.read(f)
            if not np.isfinite(data).all():
                non_finite_count += 1
        except Exception as e:
            corrupt_files += 1

    # -------------------------------------------------------------
    # Print formatted report
    # -------------------------------------------------------------
    total_drone_files = len(drone_files_on_disk)
    n_train = len(split_resolved["drone_train"])
    n_val = len(split_resolved["drone_validation"])
    n_test = len(split_resolved["drone_test"])

    dur_train_hr = split_durations["drone_train"] / 3600
    dur_val_hr = split_durations["drone_validation"] / 3600
    dur_test_hr = split_durations["drone_test"] / 3600
    total_dur_hr = dur_train_hr + dur_val_hr + dur_test_hr

    print("\nDATASET VALIDATION REPORT")
    print("-" * 35)
    print(f"Drone files:           {total_drone_files}")
    print(f"Train:                 {n_train} ({n_train/total_drone_files*100:.1f}%)")
    print(f"Validation:            {n_val} ({n_val/total_drone_files*100:.1f}%)")
    print(f"Test:                  {n_test} ({n_test/total_drone_files*100:.1f}%)")
    print()
    print(f"Train duration:        {dur_train_hr:.2f} hours ({dur_train_hr/total_dur_hr*100:.1f}%)")
    print(f"Validation duration:   {dur_val_hr:.2f} hours ({dur_val_hr/total_dur_hr*100:.1f}%)")
    print(f"Test duration:         {dur_test_hr:.2f} hours ({dur_test_hr/total_dur_hr*100:.1f}%)")
    print(f"Total duration:        {total_dur_hr:.2f} hours")
    print()
    print(f"Drone IDs train:       {split_drone_ids['drone_train']}")
    print(f"Drone IDs validation:  {split_drone_ids['drone_validation']}")
    print(f"Drone IDs test:        {split_drone_ids['drone_test']}")
    print()
    print(f"File leakage:          {'PASS' if total_file_leakage == 0 else 'FAIL'}")
    print(f"Drone-ID leakage:      {'PASS' if total_drone_leakage == 0 else 'FAIL'}")
    print(f"Session leakage:       {'PASS' if len(train_val_date_leak) == 0 and len(train_test_date_leak) == 0 else 'FAIL'}")
    print(f"Global noise leakage:  {'PASS' if noise_cross_leakage == 0 else 'FAIL'}")
    print()
    print(f"Missing files:         {missing_files_count + global_missing_files}")
    print(f"Corrupt files:         {corrupt_files}")
    print(f"Duplicate paths:       {duplicate_paths_count}")
    print(f"Sample-rate violations:{sr_violations}")
    print(f"Channel violations:    {channel_violations}")
    print(f"Non-finite samples:    {non_finite_count}")
    print(f"Duration violations:   {duration_violations}")
    print(f"Metadata coverage:     {'PASS' if len(missing_metadata_files) == 0 and missing_metadata_paths == 0 else 'FAIL'}")
    print("-" * 35)

    all_pass = (
        total_file_leakage == 0 and
        total_drone_leakage == 0 and
        missing_files_count == 0 and
        global_missing_files == 0 and
        corrupt_files == 0 and
        duplicate_paths_count == 0 and
        sr_violations == 0 and
        channel_violations == 0 and
        non_finite_count == 0 and
        duration_violations == 0 and
        len(missing_metadata_files) == 0 and
        missing_metadata_paths == 0
    )

    if all_pass:
        print("\nOVERALL VALIDATION STATUS: PASSED ALL CHECKS")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\nOVERALL VALIDATION STATUS: FAILED CHECKS DETECTED")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
