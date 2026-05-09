import os
import subprocess
import sys
import shutil
import glob
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it"
]
PRECISIONS = [8, 4]  # 8-bit and 4-bit
EKA_REPO = "https://github.com/lingo-iitgn/eka-eval.git"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

def log(msg):
    print(f"\n{'='*20} {msg} {'='*20}")

def run_cmd(cmd, input_str=None, cwd=None):
    """Executes a shell command and optionally feeds stdin."""
    result = subprocess.run(
        cmd, 
        input=input_str, 
        shell=True, 
        text=True, 
        capture_output=True,
        cwd=cwd
    )
    if result.returncode != 0:
        print(f"Error in command: {cmd}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
    return result

# 1. Pre-flight Hardware Check
log("HARDWARE VERIFICATION")
gpu_info = run_cmd("nvidia-smi -L").stdout
print(gpu_info)
if "T4" not in gpu_info:
    print("FATAL ERROR: This script requires T4 GPUs for NF4 support. P100 detected.")
    sys.exit(1)

# 2. Clean Installation
log("CLEAN ROOM SETUP")
if os.path.exists("eka-eval"):
    shutil.rmtree("eka-eval")

run_cmd(f"git clone {EKA_REPO}")
run_cmd("pip install -q transformers bitsandbytes accelerate peft datasets numpy scipy kneed scikit-image tqdm evaluate rouge_score")
run_cmd("pip install -e .", cwd="eka-eval")

# Path constants
LOADER_PATH = "eka-eval/eka_eval/core/model_loader.py"
CONFIG_PATH = "eka-eval/eka_eval/config/benchmark_config.py"

# 3. Patch MMLU-IN Typo (Permanent for this run)
log("PATCHING BENCHMARK CONFIG")
run_cmd(f"sed -i 's/indic\\.mmlu_in\\.evaluate_mmlu_in/multilingual.mmlu_in.evaluate_mmlu_in/g' {CONFIG_PATH}")

def run_single_eval(model_id, precision):
    model_name = model_id.split("/")[-1]
    tag = f"{precision}bit_{model_name}"
    target_dir = f"results_sweep_{RUN_ID}/{tag}"
    os.makedirs(target_dir, exist_ok=True)
    
    log(f"STARTING EVAL: {tag}")
    
    # Patch precision in source
    if precision == 8:
        run_cmd(f"sed -i 's/load_in_4bit *= *True/load_in_8bit=True/g' {LOADER_PATH}")
        run_cmd(f"sed -i 's/load_in_8bit *= *True/load_in_8bit=True/g' {LOADER_PATH}") # No-op if already set
    else:
        run_cmd(f"sed -i 's/load_in_8bit *= *True/load_in_4bit=True/g' {LOADER_PATH}")
        run_cmd(f"sed -i 's/load_in_4bit *= *True/load_in_4bit=True/g' {LOADER_PATH}") # No-op if already set

    # Clear hardcoded folder
    if os.path.exists("eka-eval/results_output"):
        shutil.rmtree("eka-eval/results_output")

    # Prompt Sequence: 1(Local) -> 1(HF) -> {ID} -> no(BM) -> 9(INDIC) -> 1(MMLU) -> no(Viz)
    wizard_input = f"1\n1\n{model_id}\nno\n9\n1\nno\n"
    
    print(f"Running benchmarking script for {model_id}...")
    run_cmd("python eka-eval/scripts/run_benchmarks.py", input_str=wizard_input)
    
    # Isolate results
    if os.path.exists("eka-eval/results_output"):
        for item in os.listdir("eka-eval/results_output"):
            s = os.path.join("eka-eval/results_output", item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print(f"SUCCESS: Results moved to {target_dir}")
    else:
        print(f"ERROR: No results found in eka-eval/results_output for {tag}")

# --- MAIN EXECUTION LOOP ---
os.makedirs(f"results_sweep_{RUN_ID}", exist_ok=True)

for model in MODELS:
    for prec in PRECISIONS:
        run_single_eval(model, prec)

# 4. Final Aggregation
log("FINAL RESULTS SUMMARY")
all_results = []
csv_files = glob.glob(f"results_sweep_{RUN_ID}/**/calculated.csv", recursive=True)

for f in csv_files:
    # Try to extract precision from the folder name
    path_parts = f.split(os.sep)
    prec_tag = next((p for p in path_parts if "bit" in p), "Unknown")
    
    try:
        df = pd.read_csv(f)
        df["Quantization"] = prec_tag
        all_results.append(df)
    except:
        pass

if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    summary_path = f"final_summary_{RUN_ID}.csv"
    final_df.to_csv(summary_path, index=False)
    print(final_df.to_markdown())
    print(f"\nFinal summary saved to: {summary_path}")

# Zip it all up
zip_file = f"complete_results_{RUN_ID}"
shutil.make_archive(zip_file, 'zip', f"results_sweep_{RUN_ID}")
print(f"All artifacts zipped in: {zip_file}.zip")
