import os
import json
import pandas as pd
import soundfile as sf
import numpy as np

from .metrics import compute_all_standard_metrics
from .impulse_metrics import calculate_impulse_metrics
from .recovery_time import calculate_iisrt_and_rsdd


def _load_audio_mono(filepath: str):
    """Safely loads an audio file and ensures 1D float32 mono output."""
    data, sr = sf.read(filepath, dtype='float32')
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sr


def process_single_dict(meta: dict, data_root: str, system_label: str = "Impulse Guard") -> dict:
    if not isinstance(meta, dict) or "audio" not in meta or not isinstance(meta["audio"], dict):
        return None

    audio_info = meta["audio"]

    # Extract relative audio paths safely with fallback checks
    clean_rel = audio_info.get("clean_file") or audio_info.get("clean") or ""
    noisy_rel = audio_info.get("noisy_file") or audio_info.get("noisy") or ""
    enhanced_rel = audio_info.get("enhanced_file") or audio_info.get("enhanced") or ""

    if not clean_rel or not noisy_rel or not enhanced_rel:
        print(f"[Skipping] Sample {meta.get('sample_id', 'unknown')}: Missing file paths in audio dictionary.")
        return None

    root_dir = data_root if data_root else ""
    clean_path = os.path.normpath(os.path.join(root_dir, str(clean_rel)))
    noisy_path = os.path.normpath(os.path.join(root_dir, str(noisy_rel)))
    enhanced_path = os.path.normpath(os.path.join(root_dir, str(enhanced_rel)))

    if not (os.path.exists(clean_path) and os.path.exists(noisy_path) and os.path.exists(enhanced_path)):
        print(f"[Missing WAV] Sample {meta.get('sample_id', 'unknown')}: Could not find audio files at:\n  - Clean: {clean_path}\n  - Noisy: {noisy_path}\n  - Enhanced: {enhanced_path}")
        return None

    clean, sr = _load_audio_mono(clean_path)
    noisy, _ = _load_audio_mono(noisy_path)
    enhanced, _ = _load_audio_mono(enhanced_path)

    noise_category = meta.get("noise", {}).get("category", "unknown").lower()
    noise_type = meta.get("noise", {}).get("type", "unknown").lower()
    is_impulsive = "impulse" in noise_category or "impulsive" in noise_category or "gunshot" in noise_type or "explosion" in noise_type

    record = {
        "sample_id": meta.get("sample_id"),
        "system": system_label,
        "language": meta.get("speech", {}).get("language", "unknown"),
        "noise_type": noise_type,
        "noise_category": noise_category,
        "is_impulsive": is_impulsive,
        "target_snr": meta.get("mixing", {}).get("target_snr_db", 0),
        "actual_snr": meta.get("mixing", {}).get("actual_snr_db", 0)
    }
    
    # 1. Standard Audio Metrics
    std_metrics = compute_all_standard_metrics(clean, noisy, enhanced, sr=sr)
    record.update(std_metrics)

    # 2. Impulse Suppression and Recovery Metrics
    impulses = meta.get("impulses", [])
    iisrt_list, rsdd_list = [], []
    attenuation_list, energy_list = [], []
    
    for imp in impulses:
        imp_stats = calculate_impulse_metrics(
            noisy, enhanced,
            impulse_start_sec=imp["start_sec"],
            impulse_end_sec=imp["end_sec"],
            sr=sr
        )
        attenuation_list.append(imp_stats["peak_attenuation_db"])
        energy_list.append(imp_stats["residual_impulse_energy_ratio"])

        rec_stats = calculate_iisrt_and_rsdd(
            clean, enhanced, 
            impulse_start_sec=imp["start_sec"], 
            impulse_end_sec=imp["end_sec"], 
            sr=sr
        )
        if not np.isnan(rec_stats["iisrt_ms"]):
            iisrt_list.append(rec_stats["iisrt_ms"])
        if not np.isnan(rec_stats["rsdd_ms"]):
            rsdd_list.append(rec_stats["rsdd_ms"])

    record["peak_attenuation_db_mean"] = float(np.mean(attenuation_list)) if attenuation_list else float('nan')
    record["residual_energy_ratio_mean"] = float(np.mean(energy_list)) if energy_list else float('nan')
    record["iisrt_mean_ms"] = float(np.mean(iisrt_list)) if iisrt_list else float('nan')
    record["rsdd_mean_ms"] = float(np.mean(rsdd_list)) if rsdd_list else float('nan')

    return record


def run_evaluation_suite(metadata_dir: str, data_root: str, output_csv: str, system_label: str = "Impulse Guard"):
    results = []
    
    for root, _, files in os.walk(metadata_dir):
        for file in files:
            full_path = os.path.join(root, file)
            
            # Handle .jsonl format
            if file.endswith('.jsonl'):
                with open(full_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            meta = json.loads(line)
                            res = process_single_dict(meta, data_root, system_label=system_label)
                            if res is not None:
                                results.append(res)
                        except Exception as e:
                            print(f"Error parsing line {line_num} in {full_path}: {e}")
                            
            # Handle standard single .json format
            elif file.endswith('.json'):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        res = process_single_dict(meta, data_root, system_label=system_label)
                        if res is not None:
                            results.append(res)
                except Exception as e:
                    pass

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print("\n================ EVALUATION SUMMARY ================")
    if not df.empty:
        summary_cols = [col for col in ["si_snr_improvement", "si_sdr_improvement", "stoi_improvement", "pesq_improvement", "iisrt_mean_ms"] if col in df.columns]
        print("Continuous Background Noise Metrics:")
        non_imp = df[df["is_impulsive"] == False] if "is_impulsive" in df.columns else df
        if not non_imp.empty:
            print(non_imp[summary_cols].mean())
        else:
            print(df[summary_cols].mean())
            
        print("\nImpulsive Noise Events Metrics (Kept Separate):")
        imp = df[df["is_impulsive"] == True] if "is_impulsive" in df.columns else df
        if not imp.empty:
            imp_cols = [c for c in ["peak_attenuation_db_mean", "residual_energy_ratio_mean", "iisrt_mean_ms", "rsdd_mean_ms"] if c in imp.columns]
            print(imp[imp_cols].mean())
    else:
        print("No valid evaluation samples were processed. Check the printed warnings above to locate missing WAV files.")
    print("====================================================\n")
