import time
import serial
import numpy as np


def log_esp32_power_draw(serial_port: str = None, duration_sec: int = 2, supply_voltage: float = 3.3, mock: bool = False) -> dict:
    """
    Reads hardware current telemetry from an inline sensor over serial.
    Supports a mock mode for offline testing without serial hardware attached.
    """
    if mock or serial_port is None:
        # Simulate realistic 3.3V, ~80-120mA active WiFi/DSP draw on ESP32
        simulated_currents = np.random.uniform(85.0, 115.0, size=10)
        avg_current = float(np.mean(simulated_currents))
        avg_power = float(avg_current * supply_voltage)
        return {
            "avg_current_ma": avg_current,
            "avg_power_mw": avg_power,
            "voltage_v": supply_voltage,
            "mode": "mock"
        }

    try:
        ser = serial.Serial(serial_port, 115200, timeout=1)
        currents_ma = []
        
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "I_mA:" in line:
                try:
                    val = float(line.split("I_mA:")[1].split()[0])
                    currents_ma.append(val)
                except ValueError:
                    continue
                    
        ser.close()
        avg_current = np.mean(currents_ma) if currents_ma else 0.0
        avg_power = avg_current * supply_voltage
        
        return {
            "avg_current_ma": float(avg_current),
            "avg_power_mw": float(avg_power),
            "voltage_v": supply_voltage,
            "mode": "live"
        }
    except Exception as e:
        print(f"Power logger serial interface unavailable ({e}). Falling back to mock simulation.")
        return log_esp32_power_draw(serial_port=None, duration_sec=duration_sec, supply_voltage=supply_voltage, mock=True)