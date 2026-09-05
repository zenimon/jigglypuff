import serial
import time


def format_serial_payload(results_payload: dict) -> str:
    """Formats PC evaluation results into a compact serial packet string."""
    snr = float(results_payload.get('si_snr_enhanced', 0.0) or 0.0)
    sdr = float(results_payload.get('si_sdr_enhanced', 0.0) or 0.0)
    stoi = float(results_payload.get('stoi_enhanced', 0.0) or 0.0)
    pesq_val = float(results_payload.get('pesq_enhanced', 0.0) or 0.0)
    iisrt = float(results_payload.get('iisrt_mean_ms', 0.0) or 0.0)

    return f"RESULT:SNR={snr:.2f},SDR={sdr:.2f},STOI={stoi:.2f},PESQ={pesq_val:.2f},IISRT={iisrt:.1f}\n"


def send_test_results_to_esp32(port: str = None, baudrate: int = 115200, results_payload: dict = None, mock: bool = False) -> bool:
    """
    Pushes PC-evaluated metrics over USB serial to render on ESP32 Serial Monitor.
    Supports a mock dry-run mode for testing without hardware.
    """
    if results_payload is None:
        results_payload = {}

    payload_str = format_serial_payload(results_payload)

    if mock or port is None:
        print(f"[Mock Serial Output]: {payload_str.strip()}")
        return True

    try:
        ser = serial.Serial(port, baudrate, timeout=2)
        time.sleep(0.5) # Connection warm-up
        ser.write(payload_str.encode('utf-8'))
        print(f"Pushed to ESP32 ({port}): {payload_str.strip()}")
        ser.close()
        return True
    except Exception as e:
        print(f"Serial communications unavailable ({e}). Fallback to mock mode.")
        print(f"[Mock Serial Output]: {payload_str.strip()}")
        return False