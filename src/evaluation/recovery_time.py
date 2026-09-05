import numpy as np
from .metrics import calculate_si_sdr, _ensure_mono_1d


def calculate_iisrt_and_rsdd(
    clean: np.ndarray, 
    enhanced: np.ndarray, 
    impulse_start_sec: float, 
    impulse_end_sec: float, 
    sr: int = 16000, 
    win_len_ms: float = 40.0, 
    hop_len_ms: float = 10.0,
    required_stable_ms: float = 50.0,
    threshold_drop_db: float = 2.0
) -> dict:
    """
    Calculates IISRT (Impulse-Induced SDR Recovery Time) and RSDD (Post-Recovery SDR Dip Duration).
    """
    clean = _ensure_mono_1d(clean)
    enhanced = _ensure_mono_1d(enhanced)
    
    min_len = min(len(clean), len(enhanced))
    if min_len == 0:
        return {"iisrt_ms": float('nan'), "rsdd_ms": float('nan'), "baseline_sdr_db": float('nan')}
        
    win_samples = int(sr * (win_len_ms / 1000.0))
    hop_samples = int(sr * (hop_len_ms / 1000.0))
    
    imp_start_sample = max(0, min(min_len, int(impulse_start_sec * sr)))
    imp_end_sample = max(0, min(min_len, int(impulse_end_sec * sr)))
    
    if imp_start_sample >= imp_end_sample:
        return {"iisrt_ms": float('nan'), "rsdd_ms": float('nan'), "baseline_sdr_db": float('nan')}
    
    # 1. Pre-impulse baseline (500ms window before impulse onset)
    pre_start = max(0, imp_start_sample - int(sr * 0.5))
    if imp_start_sample - pre_start > win_samples:
        baseline_sdr = calculate_si_sdr(
            clean[pre_start:imp_start_sample], 
            enhanced[pre_start:imp_start_sample]
        )
    else:
        baseline_sdr = 10.0  # Fallback baseline
        
    target_threshold = baseline_sdr - threshold_drop_db
    
    # 2. Windowed analysis post impulse end
    stable_frames_needed = max(1, int(required_stable_ms / hop_len_ms))
    consecutive_stable = 0
    
    recovery_sample = None
    rsdd_samples = 0
    
    curr_ptr = imp_end_sample
    max_eval_ptr = min(min_len, imp_end_sample + int(sr * 2.0))
    
    while curr_ptr + win_samples <= max_eval_ptr:
        c_win = clean[curr_ptr : curr_ptr + win_samples]
        e_win = enhanced[curr_ptr : curr_ptr + win_samples]
        
        frame_sdr = calculate_si_sdr(c_win, e_win)
        
        if frame_sdr >= target_threshold:
            consecutive_stable += 1
            if consecutive_stable == stable_frames_needed and recovery_sample is None:
                recovery_sample = curr_ptr + (win_samples // 2)
        else:
            if recovery_sample is not None:
                rsdd_samples += hop_samples
            consecutive_stable = 0
            
        curr_ptr += hop_samples

    if recovery_sample is not None:
        iisrt_ms = ((recovery_sample - imp_end_sample) / sr) * 1000.0
    else:
        iisrt_ms = float('nan') # Flagged as 'Not Recovered'
        
    rsdd_ms = (rsdd_samples / sr) * 1000.0 if recovery_sample is not None else float('nan')

    return {
        "iisrt_ms": float(iisrt_ms),
        "rsdd_ms": float(rsdd_ms),
        "baseline_sdr_db": float(baseline_sdr)
    }