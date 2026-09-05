import numpy as np
import pandas as pd
from scipy import stats


def compute_confidence_intervals(data: np.ndarray, confidence: float = 0.95) -> dict:
    """Calculates mean, std, and 95% confidence intervals."""
    clean_data = data[~np.isnan(data)]
    if len(clean_data) == 0:
        return {"mean": np.nan, "std": np.nan, "ci_95": np.nan}
    
    mean = np.mean(clean_data)
    std = np.std(clean_data, ddof=1) if len(clean_data) > 1 else 0.0
    sem = stats.sem(clean_data) if len(clean_data) > 1 else 0.0
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(clean_data) - 1) if len(clean_data) > 1 else 0.0
    
    return {
        "mean": float(mean),
        "std": float(std),
        "ci_95": float(ci)
    }


def format_ppt_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Groups metrics by system & noise_type with Mean ± Std formatting."""
    if df.empty:
        return pd.DataFrame()
        
    summary_rows = []
    
    group_cols = [c for c in ["system", "noise_type"] if c in df.columns]
    if not group_cols:
        group_cols = ["system"] if "system" in df.columns else []
        
    if group_cols:
        grouped = df.groupby(group_cols)
    else:
        grouped = [("All", df)]
    
    for name, group in grouped:
        system = name[0] if isinstance(name, tuple) else (name if group_cols else "Impulse Guard")
        noise_type = name[1] if isinstance(name, tuple) and len(name) > 1 else "all"
        
        si_snr_stats = compute_confidence_intervals(group["si_snr_improvement"].values) if "si_snr_improvement" in group.columns else {"mean": np.nan, "std": np.nan}
        pesq_stats = compute_confidence_intervals(group["pesq_improvement"].values) if "pesq_improvement" in group.columns else {"mean": np.nan, "std": np.nan}
        stoi_stats = compute_confidence_intervals(group["stoi_improvement"].values) if "stoi_improvement" in group.columns else {"mean": np.nan, "std": np.nan}
        iisrt_stats = compute_confidence_intervals(group["iisrt_mean_ms"].values) if "iisrt_mean_ms" in group.columns else {"mean": np.nan, "std": np.nan}
        
        summary_rows.append({
            "System": system,
            "Noise Type": noise_type,
            "SI-SNR Imp (dB)": f"{si_snr_stats['mean']:.2f} +/- {si_snr_stats['std']:.2f}" if not np.isnan(si_snr_stats['mean']) else "N/A",
            "PESQ Imp": f"{pesq_stats['mean']:.2f} +/- {pesq_stats['std']:.2f}" if not np.isnan(pesq_stats['mean']) else "N/A",
            "STOI Imp": f"{stoi_stats['mean']:.2f} +/- {stoi_stats['std']:.2f}" if not np.isnan(stoi_stats['mean']) else "N/A",
            "IISRT (ms)": f"{iisrt_stats['mean']:.1f} +/- {iisrt_stats['std']:.1f}" if not np.isnan(iisrt_stats['mean']) else "N/A"
        })
        
    return pd.DataFrame(summary_rows)