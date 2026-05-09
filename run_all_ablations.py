import os
import subprocess
import time
import glob
import json
import pandas as pd

# Safe models for 2x T4 Kaggle environment (4-bit by default in eka-eval unless patched)
MODELS_TO_RUN = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/phi-2"
]

LANGUAGES = "hi,bn,en"
MAX_NEW_TOKENS = 5

def run_ablation_sweep(model_id):
    print("\n" + "="*80)
    print(f"🚀 LAUNCHING ABLATION SWEEP FOR: {model_id}")
    print("="*80)
    
    # We call the dual-T4 runner as a separate subprocess.
    # This is CRITICAL for Kaggle: when the subprocess dies, the OS immediately
    # reclaims all VRAM. This prevents OOM crashes between heavy model loads.
    cmd = [
        "python", "run_interpretability_dual_t4.py",
        "--model", model_id,
        "--languages", LANGUAGES,
        "--max-new-tokens", str(MAX_NEW_TOKENS)
    ]
    
    try:
        # Run and stream output to console
        process = subprocess.run(cmd, check=True)
        print(f"\n✅ SUCCESS: Ablation sweep completed for {model_id}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Ablation sweep FAILED for {model_id}. Exit code: {e.returncode}")
    
    # Pause to ensure CUDA contexts are fully destroyed before the next load
    time.sleep(5)

def aggregate_all_reports():
    print("\n" + "="*80)
    print("📊 AGGREGATING FINAL ABLATION REPORTS")
    print("="*80)
    
    # Find all generated summary.json files
    summary_files = glob.glob("results_output/interpretability/dual_t4_*/summary.json")
    
    if not summary_files:
        print("No summary files found to aggregate.")
        return
        
    for summary_file in summary_files:
        with open(summary_file, "r") as f:
            data = json.load(f)
            model_name = data.get("model", "unknown").replace("/", "_")
            
        md_file = f"ablation_heatmap_{model_name}.md"
        csv_file = f"ablation_heatmap_{model_name}.csv"
        
        cmd = [
            "python", "summarize_interpretability_results.py",
            "--summary-file", summary_file,
            "--output-md", md_file,
            "--output-csv", csv_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Generated report for {model_name}: {md_file}")
        except subprocess.CalledProcessError:
            print(f"Failed to generate report for {summary_file}")

if __name__ == "__main__":
    print("Starting Multi-Model Causal Ablation Sweep...")
    for model in MODELS_TO_RUN:
        run_ablation_sweep(model)
        
    aggregate_all_reports()
    print("\n🎉 ALL SWEEPS FINISHED. Artifacts are ready in the root directory.")
