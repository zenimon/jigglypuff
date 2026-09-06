import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.benchmarking.model_benchmark import compute_model_statistics, measure_real_time_factor
from src.benchmarking.power_measurement import log_esp32_power_draw
from src.benchmarking.serial_dashboard import format_serial_payload, send_test_results_to_esp32


def test_benchmarking_suite():
    print("=" * 60)
    print("RUNNING UNIT TESTS FOR BENCHMARKING SUITE")
    print("=" * 60)

    # 1. Test model statistics
    print("\n[1/3] Testing Model Parameter & FLOPS Computation...")
    stats = compute_model_statistics()
    print(f"  - Total Parameters:      {stats['total_parameters']}")
    print(f"  - Float32 Flash Size:    {stats['float32_flash_kb']:.2f} KB")
    print(f"  - INT8 Flash Size:       {stats['int8_flash_kb']:.2f} KB")
    print(f"  - FLOPS per 10ms frame:  {stats['flops_per_frame']}")

    assert stats['total_parameters'] > 0, "Model parameter count should be > 0"
    assert stats['int8_flash_kb'] < stats['float32_flash_kb'], "INT8 size should be ~1/4 of Float32 size"

    # 2. Test Real-Time Factor (RTF)
    print("\n[2/3] Testing Real-Time Factor (RTF) & Latency...")
    rtf_res = measure_real_time_factor(num_frames=50)
    print(f"  - Audio Duration:        {rtf_res['audio_duration_sec']:.2f} s")
    print(f"  - Execution Time:        {rtf_res['elapsed_sec']:.4f} s")
    print(f"  - Real-Time Factor (RTF): {rtf_res['real_time_factor']:.4f}")
    print(f"  - Latency per frame:     {rtf_res['latency_per_frame_ms']:.3f} ms")
    print(f"  - Real-time capable:     {rtf_res['is_realtime_capable']}")

    assert rtf_res['real_time_factor'] < 1.0, "Model RTF should be < 1.0 for real-time capability"

    # 3. Test Power logger simulation and Serial output formatting
    print("\n[3/3] Testing Hardware Telemetry Mock & Serial Formatting...")
    power_res = log_esp32_power_draw(mock=True)
    print(f"  - Simulated Current:     {power_res['avg_current_ma']:.2f} mA")
    print(f"  - Simulated Power:       {power_res['avg_power_mw']:.2f} mW")

    mock_metrics = {
        "si_snr_enhanced": 8.52,
        "si_sdr_enhanced": 7.91,
        "stoi_enhanced": 0.89,
        "pesq_enhanced": 2.45,
        "iisrt_mean_ms": 11.2
    }
    payload_str = format_serial_payload(mock_metrics)
    print(f"  - Serial Packet:         {payload_str.strip()}")

    sent_ok = send_test_results_to_esp32(results_payload=mock_metrics, mock=True)
    assert sent_ok, "Mock serial send failed"

    print("\n[SUCCESS] ALL BENCHMARKING SUITE TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_benchmarking_suite()
