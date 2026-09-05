import os

# Define directory layout
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

# Define initial files to create
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


def create_structure(base_dir="impulse-guard"):
    print(f"Creating project structure under '{base_dir}/'...\n")

    # 1. Create directories
    for folder in DIRECTORIES:
        path = os.path.join(base_dir, folder)
        os.makedirs(path, exist_ok=True)
        print(f"  [Directory] {path}/")

    # 2. Create empty files
    for file_path in FILES:
        path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                pass  # create empty file
            print(f"  [File]      {path}")

    # 3. Create sample requirements.txt
    req_path = os.path.join(base_dir, "requirements.txt")
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(
            "numpy\nscipy\nsoundfile\nlibrosa\npystoi\npesq\npandas\npyserial\n"
        )

    print("\n✅ Project directory and files successfully created!")


if __name__ == "__main__":
    create_structure()