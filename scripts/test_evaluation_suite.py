import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import (
    calculate_si_snr,
    calculate_si_sdr,
    calculate_stoi,
    calculate_pesq,
    compute_dnsmos,
    compute_all_standard_metrics
)
from src.evaluation.impulse_metrics import calculate_impulse_metrics
from src.evaluation.recovery_time import calculate_iisrt_and_rsdd
from src.evaluation.ablation import compute_confidence_intervals, format_ppt_summary_table
import pandas as pd


def test_evaluation_suite():
    print("=" * 60)
    print("RUNNING UNIT TESTS FOR EVALUATION SUITE")
    print("=" * 60)

    sr = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Generate clean reference (sine wave speech proxy)
    clean = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # 2. Generate noisy audio (clean + gaussian noise + impulse peak)
    noise = 0.1 * np.random.randn(len(t)).astype(np.float32)
    noisy = clean + noise
    # Insert sharp impulse between 0.4s and 0.45s
    noisy[int(0.4 * sr):int(0.45 * sr)] += 2.0

    # 3. Generate enhanced audio (simulated noise reduction)
    enhanced = clean + 0.02 * np.random.randn(len(t)).astype(np.float32)

    # Test standard metrics
    print("\n[1/4] Testing Standard Metrics...")
    si_snr = calculate_si_snr(clean, enhanced)
    si_sdr = calculate_si_sdr(clean, enhanced)
    stoi_val = calculate_stoi(clean, enhanced, sr=sr)
    estoi_val = calculate_stoi(clean, enhanced, sr=sr, extended=True)
    pesq_val = calculate_pesq(clean, enhanced, sr=sr)
    dnsmos_res = compute_dnsmos(noisy, enhanced, sr=sr)

    print(f"  - SI-SNR:  {si_snr:.2f} dB")
    print(f"  - SI-SDR:  {si_sdr:.2f} dB")
    print(f"  - STOI:    {stoi_val:.3f}")
    print(f"  - ESTOI:   {estoi_val:.3f}")
    print(f"  - PESQ:    {pesq_val}")
    print(f"  - DNSMOS:  SIG={dnsmos_res['dnsmos_sig_enhanced']:.2f}, BAK={dnsmos_res['dnsmos_bak_enhanced']:.2f}, OVRL={dnsmos_res['dnsmos_ovrl_enhanced']:.2f}")

    assert not np.isnan(si_snr), "SI-SNR calculation failed"
    assert not np.isnan(si_sdr), "SI-SDR calculation failed"
    assert not np.isnan(stoi_val), "STOI calculation failed"
    assert 1.0 <= dnsmos_res['dnsmos_ovrl_enhanced'] <= 5.0, "DNSMOS overall score out of range"

    # Test impulse metrics
    print("\n[2/4] Testing Impulse Metrics...")
    imp_stats = calculate_impulse_metrics(
        noisy, enhanced,
        impulse_start_sec=0.4,
        impulse_end_sec=0.45,
        sr=sr
    )
    print(f"  - Peak Attenuation:     {imp_stats['peak_attenuation_db']:.2f} dB")
    print(f"  - Residual Energy Ratio: {imp_stats['residual_impulse_energy_ratio']:.4f}")

    assert imp_stats['peak_attenuation_db'] > 0, "Peak attenuation should be positive for suppressed impulse"

    # Test recovery time
    print("\n[3/4] Testing Recovery Time Metrics...")
    rec_stats = calculate_iisrt_and_rsdd(
        clean, enhanced,
        impulse_start_sec=0.4,
        impulse_end_sec=0.45,
        sr=sr
    )
    print(f"  - IISRT: {rec_stats['iisrt_ms']} ms")
    print(f"  - RSDD:  {rec_stats['rsdd_ms']} ms")

    # Test summary table formatting & confidence intervals
    print("\n[4/4] Testing Statistical Confidence Intervals & Summary Formatting...")
    dummy_df = pd.DataFrame([{
        "system": "Impulse Guard",
        "noise_type": "gunshot",
        "si_snr_improvement": 4.5,
        "pesq_improvement": 0.45,
        "stoi_improvement": 0.08,
        "iisrt_mean_ms": 12.0
    }, {
        "system": "Impulse Guard",
        "noise_type": "gunshot",
        "si_snr_improvement": 5.1,
        "pesq_improvement": 0.52,
        "stoi_improvement": 0.10,
        "iisrt_mean_ms": 14.5
    }])

    summary_tbl = format_ppt_summary_table(dummy_df)
    print(summary_tbl.to_string())
    assert not summary_tbl.empty, "Summary table formatting failed"

    print("\n[SUCCESS] ALL EVALUATION SUITE TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_evaluation_suite()
