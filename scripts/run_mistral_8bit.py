import os
import sys
import subprocess
import glob
import json
import pandas as pd
import shutil

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
PRECISION = 8
FOLDER_NAME = f"eval_{PRECISION}bit_Mistral-7B"

def setup_pristine_environment():
    print("="*80)
    print(f"INITIALIZING PRISTINE {PRECISION}-BIT ENVIRONMENT FOR {MODEL_ID}")
    print("="*80)
    
    os.system("rm -rf eka-eval results_output results calculated.csv")
    print("Cloning fresh eka-eval...")
    os.system("git clone -q https://github.com/lingo-iitgn/eka-eval.git")
    os.system(f"{sys.executable} -m pip install -q -e eka-eval")
    os.system(f"{sys.executable} -m pip install -q tabulate")
    
    # 1. Patch Configs
    config_path = "eka-eval/eka_eval/config/benchmark_config.py"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_content = f.read()
        config_content = config_content.replace("indic.mmlu_in.evaluate_mmlu_in", "multilingual.mmlu_in.evaluate_mmlu_in")
        config_content = config_content.replace("indic.arc_c_in.evaluate_arc_c_in", "multilingual.arc_c_in.evaluate_arc_c_in")
        config_content = config_content.replace('"save_detailed": False', '"save_detailed": True')
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        
    # 2. Patch Model Loader for Multi-GPU and Precision
    loader_path = "eka-eval/eka_eval/core/model_loader.py"
    if os.path.exists(loader_path):
        with open(loader_path, "r", encoding="utf-8") as f:
            loader_content = f.read()
            
        loader_content = loader_content.replace("device_map_arg = {'': f'cuda:{target_device_id}'}", "device_map_arg = 'auto'")
        
        old_bnb_block = """    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=target_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )"""
            
        new_bnb_block = """    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )"""
        
        loader_content = loader_content.replace(old_bnb_block, new_bnb_block)
        with open(loader_path, "w", encoding="utf-8") as f:
            f.write(loader_content)

def run_evaluation():
    print(f"\n⏳ Running {PRECISION}-bit benchmark for {MODEL_ID} on 2x T4...")
    # 9 = INDIC BENCHMARKS, 6 = ARC-Challenge-Indic
    input_seq = f"1\n1\n{MODEL_ID}\nno\n9\n6\nno\n"
    
    log_filename = f"{FOLDER_NAME}.log"
    with open(log_filename, "w") as log_file:
        subprocess.run(
            [sys.executable, "eka-eval/scripts/run_benchmarks.py"],
            input=input_seq,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
    print("\n✅ BENCHMARK SCRIPT COMPLETED.")
    
    source_dir = None
    for root, dirs, files in os.walk("."):
        if "calculated.csv" in files and "eval_" not in root:
            source_dir = root
            break
            
    if source_dir:
        os.makedirs(FOLDER_NAME, exist_ok=True)
        abs_folder = os.path.abspath(FOLDER_NAME)
        shutil.copy2(os.path.join(source_dir, "calculated.csv"), os.path.join(abs_folder, "calculated.csv"))
        
        if os.path.exists(os.path.join(source_dir, "detailed_results")):
            dest_detailed = os.path.join(abs_folder, "detailed_results")
            if os.path.exists(dest_detailed):
                shutil.rmtree(dest_detailed)
            shutil.copytree(os.path.join(source_dir, "detailed_results"), dest_detailed)
            
        csv_path = os.path.join(abs_folder, "calculated.csv")
        df = pd.read_csv(csv_path)
        print("\n--- RESULTS ---")
        print(df.to_markdown(index=False))
        
        print("\n--- SNEAK PEEK (FROM LOG) ---")
        os.system(f"tail -n 50 {log_filename}")
    else:
        print("\n❌ FATAL ERROR: No results found. Dumping log:")
        os.system(f"tail -n 50 {log_filename}")

if __name__ == "__main__":
    setup_pristine_environment()
    run_evaluation()
