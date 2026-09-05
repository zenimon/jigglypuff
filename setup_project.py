import os

DIRECTORIES = [
    "src/audio",
    "src/evaluation",
    "src/benchmarking",
    "scripts",
    "data/clean",
    "data/noisy",
    "data/enhanced",
    "data/metadata",
    "results/metrics",
    "results/tables",
    "results/benchmarks",
    "results/demo_clips",
    "firmware/ESP32",
]

FILES = [
    "src/__init__.py",
    "src/audio/__init__.py",
    "src/audio/stft.py",
    "src/audio/istft.py",
    "src/evaluation/__init__.py",
    "src/evaluation/metrics.py",
    "src/evaluation/impulse_metrics.py",
    "src/evaluation/recovery_time.py",
    "src/evaluation/ablation.py",
    "src/evaluation/evaluate.py",
    "src/benchmarking/__init__.py",
    "src/benchmarking/serial_dashboard.py",
    "src/benchmarking/power_measurement.py",
    "scripts/run_evaluation.py",
    "scripts/export_demo_audio.py",
    "requirements.txt",
    "README.md",
]

def create_structure():
    print("Creating project structure inside 'jigglypuff'...\n")

    for folder in DIRECTORIES:
        os.makedirs(folder, exist_ok=True)
        print(f"  [Directory] {folder}/")

    for file_path in FILES:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                pass
            print(f"  [File]      {file_path}")

    req_path = "requirements.txt"
    with open(req_path, "w", encoding="utf-8") as f:
        f.write("numpy\nscipy\nsoundfile\nlibrosa\npystoi\npesq\npandas\npyserial\n")

    print("\n✅ Project directory and files successfully created inside jigglypuff!")

if __name__ == "__main__":
    create_structure()
