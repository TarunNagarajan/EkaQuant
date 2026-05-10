import os
import subprocess
import sys
import shutil
import glob
from datetime import datetime

# ==============================================================================
# CONFIGURATION: Models and Precisions to Sweep
# ==============================================================================
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
]
PRECISIONS = [8, 4]  # Runs both 8-bit and 4-bit for each model
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
MASTER_DIR = f"/kaggle/working/master_sweep_{RUN_ID}"


def print_banner(msg):
    print(f"\n\n{'#' * 80}\n# {msg.center(76)} #\n{'#' * 80}\n")


# Force Python to be unbuffered globally
os.environ["PYTHONUNBUFFERED"] = "1"


def run_realtime_cmd(cmd, input_str=None, cwd=None):
    """Executes a command and streams every character directly to stdout for zero latency."""
    print(f"--- [DEBUG] Launching Subprocess: {cmd} ---")
    sys.stdout.flush()

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        text=True,
        cwd=cwd,
        bufsize=0,  # Completely unbuffered
    )

    if input_str:
        process.stdin.write(input_str)
        process.stdin.close()

    # Read character by character
    while True:
        char = process.stdout.read(1)
        if not char and process.poll() is not None:
            break
        if char:
            sys.stdout.write(char)
            sys.stdout.flush()

    print(f"\n--- [DEBUG] Subprocess Finished (RC: {process.returncode}) ---")
    sys.stdout.flush()
    return process.returncode


# --- 1. Pre-flight Hardware Check ---
print_banner("STEP 1: HARDWARE VERIFICATION")
gpu_info = subprocess.check_output("nvidia-smi -L", shell=True).decode()
print(gpu_info)
if "T4" not in gpu_info:
    print("FATAL ERROR: This script requires T4 GPUs for NF4 support. P100 detected.")
    sys.exit(1)

# --- 2. Clean Installation ---
print_banner("STEP 2: CLEAN ROOM INSTALLATION")
# Wipe any existing clones to ensure zero contamination
for repo in ["eka-eval", "EkaQuant"]:
    if os.path.exists(repo):
        shutil.rmtree(repo)

print("Installing dependencies...")
run_realtime_cmd(
    "pip install -q transformers bitsandbytes accelerate peft datasets numpy scipy kneed scikit-image tqdm evaluate rouge_score pandas tabulate"
)

print("Cloning eka-eval...")
run_realtime_cmd("git clone https://github.com/lingo-iitgn/eka-eval.git")
run_realtime_cmd("pip install -e .", cwd="eka-eval")

# Path constants for patching
LOADER_PATH = "eka-eval/eka_eval/core/model_loader.py"
CONFIG_PATH = "eka-eval/eka_eval/config/benchmark_config.py"

# --- 3. Patching eka-eval Source ---
print_banner("STEP 3: PATCHING EKA-EVAL SOURCE")
# Fix the "indic." vs "multilingual." module path typo
print("Patching benchmark_config.py typo...")
run_realtime_cmd(
    f"sed -i 's/indic\\.mmlu_in\\.evaluate_mmlu_in/multilingual.mmlu_in.evaluate_mmlu_in/g' {CONFIG_PATH}"
)


def run_single_eval(model_id, precision):
    model_name = model_id.split("/")[-1]
    tag = f"{precision}bit_{model_name}"
    target_folder = os.path.join(MASTER_DIR, tag)
    os.makedirs(target_folder, exist_ok=True)

    print_banner(f"EVALUATING: {model_id} ({precision}-bit)")

    # Surgical precision toggle in eka_eval/core/model_loader.py
    if precision == 8:
        print("Switching model_loader to 8-bit...")
        run_realtime_cmd(
            f"sed -i 's/load_in_4bit *= *True/load_in_8bit=True/g' {LOADER_PATH}"
        )
        run_realtime_cmd(
            f"sed -i 's/load_in_8bit *= *True/load_in_8bit=True/g' {LOADER_PATH}"
        )
    else:
        print("Switching model_loader to 4-bit...")
        run_realtime_cmd(
            f"sed -i 's/load_in_8bit *= *True/load_in_4bit=True/g' {LOADER_PATH}"
        )
        run_realtime_cmd(
            f"sed -i 's/load_in_4bit *= *True/load_in_4bit=True/g' {LOADER_PATH}"
        )

    # Explicitly clear the hardcoded results folder to prevent contamination/appending
    if os.path.exists("eka-eval/results_output"):
        shutil.rmtree("eka-eval/results_output")

    # Automated interactive wizard input:
    # 1. Local Model
    # 1. Hugging Face
    # {model_id}
    # no (Do not add custom benchmarks)
    # 9 (INDIC BENCHMARKS Group)
    # 1 (MMLU-IN Task)
    # no (Do not create visualizations)
    wizard_input = f"1\n1\n{model_id}\nno\n9\n1\nno\n"

    # Run the benchmark script
    rc = run_realtime_cmd(
        "python eka-eval/scripts/run_benchmarks.py", input_str=wizard_input
    )

    if rc != 0:
        print(f"\n[!] WARNING: Benchmark script exited with code {rc}")

    # Move results from hardcoded folder to our isolated MASTER_DIR
    if os.path.exists("eka-eval/results_output"):
        for item in os.listdir("eka-eval/results_output"):
            src = os.path.join("eka-eval/results_output", item)
            dst = os.path.join(target_folder, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"\n[SUCCESS] All results for {tag} isolated in {target_folder}")
    else:
        print(f"\n[ERROR] No output files found for {tag}!")


# --- 4. Main Execution Loop ---
os.makedirs(MASTER_DIR, exist_ok=True)
for model in MODELS:
    for prec in PRECISIONS:
        try:
            run_single_eval(model, prec)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed during {model} {prec}-bit: {e}")

# --- 5. Final Aggregation and Summary ---
print_banner("STEP 4: FINAL SUMMARY GENERATION")
import pandas as pd

summary_list = []
csv_files = glob.glob(f"{MASTER_DIR}/**/calculated.csv", recursive=True)

for f in csv_files:
    # Extract meta info from folder path
    folder_name = os.path.basename(os.path.dirname(f))
    try:
        df = pd.read_csv(f)
        df.insert(0, "Config", folder_name)
        summary_list.append(df)
    except Exception as e:
        print(f"Could not read {f}: {e}")

if summary_list:
    final_df = pd.concat(summary_list, ignore_index=True)
    summary_csv = f"/kaggle/working/master_sweep_summary_{RUN_ID}.csv"
    final_df.to_csv(summary_csv, index=False)

    print("\n--- AGGREGATED BENCHMARK SCORES ---")
    print(final_df.to_markdown(index=False))
    print(f"\nAggregated summary saved to: {summary_csv}")
else:
    print("\n[!] No calculated.csv files were found to aggregate.")

# --- 6. Zipping Everything ---
zip_path = f"/kaggle/working/full_results_{RUN_ID}"
print(f"\nZipping all artifacts into {zip_path}.zip...")
shutil.make_archive(zip_path, "zip", MASTER_DIR)

print_banner("SWEEP COMPLETE")
print(f"Find your isolated results in: {MASTER_DIR}")
print(f"Download the full package: {zip_path}.zip")
