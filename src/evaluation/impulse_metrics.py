import numpy as np


def calculate_impulse_metrics(
    noisy: np.ndarray, 
    enhanced: np.ndarray, 
    impulse_start_sec: float, 
    impulse_end_sec: float, 
    detected_start_sec: float = None,
    sr: int = 16000
) -> dict:
    """
    Evaluates impulse peak suppression, remaining energy, and detection lag.
    """
    noisy = np.asarray(noisy, dtype=np.float32)
    enhanced = np.asarray(enhanced, dtype=np.float32)
    if noisy.ndim > 1:
        noisy = np.mean(noisy, axis=-1)
    if enhanced.ndim > 1:
        enhanced = np.mean(enhanced, axis=-1)

    min_len = min(len(noisy), len(enhanced))
    start_idx = max(0, int(impulse_start_sec * sr))
    end_idx = min(min_len, int(impulse_end_sec * sr))
    
    if start_idx >= end_idx or min_len == 0:
        return {
            "peak_attenuation_db": float('nan'),
            "residual_impulse_energy_ratio": float('nan'),
            "detection_delay_ms": float('nan')
        }

    noisy_segment = noisy[start_idx:end_idx]
    enhanced_segment = enhanced[start_idx:end_idx]
    
    # 1. Peak Attenuation (dB)
    peak_noisy = np.max(np.abs(noisy_segment)) + 1e-8
    peak_enhanced = np.max(np.abs(enhanced_segment)) + 1e-8
    peak_attenuation_db = 20 * np.log10(peak_noisy / peak_enhanced)
    
    # 2. Residual Impulse Energy Ratio
    energy_noisy = np.sum(noisy_segment ** 2) + 1e-8
    energy_enhanced = np.sum(enhanced_segment ** 2) + 1e-8
    residual_energy_ratio = energy_enhanced / energy_noisy
    
    # 3. Detection Delay (ms)
    if detected_start_sec is not None:
        detection_delay_ms = (detected_start_sec - impulse_start_sec) * 1000.0
    else:
        detection_delay_ms = float('nan')

    return {
        "peak_attenuation_db": float(peak_attenuation_db),
        "residual_impulse_energy_ratio": float(residual_energy_ratio),
        "detection_delay_ms": float(detection_delay_ms)
    }