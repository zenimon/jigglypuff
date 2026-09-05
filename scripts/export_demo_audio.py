import os
import json
import soundfile as sf
import numpy as np

def generate_ab_demo_clip(metadata_file: str, data_root: str, output_dir: str):
    """
    Stitches Noisy -> Enhanced -> Difference into a single audio clip for demos.
    """
    with open(metadata_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    noisy_path = os.path.join(data_root, meta["audio"]["noisy_file"])
    enhanced_path = os.path.join(data_root, meta["audio"]["enhanced_file"])

    noisy, sr = sf.read(noisy_path)
    enhanced, _ = sf.read(enhanced_path)
    
    removed_noise = noisy - enhanced
    
    pad = np.zeros(int(sr * 0.5))
    stitched = np.concatenate([noisy, pad, enhanced, pad, removed_noise])
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"demo_ab_{meta['sample_id']}.wav")
    sf.write(out_path, stitched, sr)
    print(f"Demo A/B file saved to: {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_ab_demo_clip(sys.argv[1], "data", "results/demo_clips")
    else:
        print("Usage: python scripts/export_demo_audio.py <path_to_metadata.json>")