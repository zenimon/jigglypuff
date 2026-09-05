import sys
import os
import argparse

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.evaluation.evaluate import run_evaluation_suite
from src.evaluation.ablation import format_ppt_summary_table

def main():
    parser = argparse.ArgumentParser(description="Jigglypuff - Evaluation CLI")
    parser.add_argument("--meta_dir", type=str, default="data/metadata", help="Path to JSON metadata directory")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory containing audio folders")
    parser.add_argument("--out_csv", type=str, default="results/metrics/evaluation_results.csv", help="Output CSV path")
    parser.add_argument("--system", type=str, default="Impulse Guard", help="System model label")
    
    args = parser.parse_args()
    
    print(f"Starting evaluation suite for [{args.system}]...")
    run_evaluation_suite(args.meta_dir, args.data_root, args.out_csv, system_label=args.system)
    
    if os.path.exists(args.out_csv):
        df = pd.read_csv(args.out_csv)
        if not df.empty:
            summary_df = format_ppt_summary_table(df)
            summary_out = "results/tables/ppt_summary_table.csv"
            summary_df.to_csv(summary_out, index=False)
            print(f"Formatted PPT Summary Table saved to: {summary_out}")
            print(summary_df.to_string())

if __name__ == "__main__":
    main()
