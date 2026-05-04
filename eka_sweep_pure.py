import os
import sys
import subprocess
import glob
import json
import re
import pandas as pd

def patch_file(file_path, old_str, new_str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(old_str, new_str)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def set_precision(loader_path, precision):
    with open(loader_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Reset both values first to avoid overlap issues
    content = re.sub(r"load_in_4bit\s*=\s*(True|False)", "load_in_4bit=False", content)
    content = re.sub(r"load_in_8bit\s*=\s*(True|False)", "load_in_8bit=False", content)
    
    # Set the target precision to True
    content = re.sub(f"load_in_{precision}bit=False", f"load_in_{precision}bit=True", content)
    
    with open(loader_path, "w", encoding="utf-8") as f:
        f.write(content)

def check_hardware():
    print("="*60)
    print("1. HARDWARE VERIFICATION")
    print("="*60)
    try:
        gpu_info = subprocess.check_output("nvidia-smi -L", shell=True).decode()
        print(gpu_info.strip())
        if "T4" not in gpu_info:
            print("ERROR: T4 GPU not detected. Please switch your Kaggle accelerator to T4 x2.")
            sys.exit(1)
        print("Hardware check passed.")
    except Exception as e:
        print(f"Error checking hardware: {e}")
        sys.exit(1)

def setup_environment():
    print("\n" + "="*60)
    print("2. ENVIRONMENT SETUP")
    print("="*60)
    try:
        from kaggle_secrets import UserSecretsClient
        from huggingface_hub import login
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            print("HF authentication successful.")
    except Exception:
        print("Note: Could not authenticate with HF. Ensure HF_TOKEN is in Kaggle Secrets if you hit rate limits.")
        
    print("Installing dependencies...")
    os.system("rm -rf eka-eval results_output results")
    os.system("pip install -q transformers bitsandbytes accelerate peft datasets numpy scipy kneed scikit-image tqdm evaluate rouge_score pandas tabulate")
    
    print("Cloning eka-eval...")
    os.system("git clone -q https://github.com/lingo-iitgn/eka-eval.git")
    print("Installing eka-eval...")
    os.system("cd eka-eval && pip install -q -e .")
    
    print("Patching config...")
    config_path = "eka-eval/eka_eval/config/benchmark_config.py"
    patch_file(config_path, "indic.mmlu_in.evaluate_mmlu_in", "multilingual.mmlu_in.evaluate_mmlu_in")
    print("Setup complete.\n")

def evaluate_model(model_id, precision):
    model_name = model_id.split("/")[-1]
    folder_name = f"eval_{precision}bit_{model_name}"
    
    print("\n" + "="*80)
    print(f"🚀 STARTING {precision}-BIT EVALUATION: {model_id}")
    print("="*80)
    
    loader_path = "eka-eval/eka_eval/core/model_loader.py"
    set_precision(loader_path, precision)

    # Force a clean slate
    os.system("rm -rf results_output results")
    
    print(f"⏳ Running benchmarks on 2x T4... (Verbose output piped to eval_{precision}bit_{model_name}.log)")
    input_seq = f"1\n1\n{model_id}\nno\n9\n1\nno\n"
    
    log_filename = f"eval_{precision}bit_{model_name}.log"
    with open(log_filename, "w") as log_file:
        subprocess.run(
            ["python", "eka-eval/scripts/run_benchmarks.py"],
            input=input_seq,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
    if os.path.exists("results_output"):
        os.makedirs(folder_name, exist_ok=True)
        os.system(f"cp -r results_output/* {folder_name}/")
        
        print("\n✅ EVALUATION COMPLETE!\n")
        csv_path = f"{folder_name}/calculated.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(df.to_markdown(index=False))
        else:
            print("calculated.csv not found.")
            
        detailed_jsons = sorted(glob.glob(f"{folder_name}/detailed_results/*.json"))
        if detailed_jsons:
            latest_json = detailed_jsons[-1]
            try:
                with open(latest_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    questions = data.get("detailed_results", [])
                    print(f"\n🔍 --- SNEAK PEEK: LAST 3 MODEL RESPONSES FOR {model_name} ({precision}-bit) ---")
                    for idx, q in enumerate(questions[-3:]):
                        print(f"\n❓ QUESTION {idx+1}:\n{q.get('question')}\n")
                        print(f"🤖 RAW OUTPUT:\n{q.get('raw_response')}\n")
                        print(f"🎯 CORRECT? {q.get('is_correct')}")
                        print("-" * 80)
            except Exception as e:
                print(f"Could not parse detailed JSON: {e}")
        else:
            print("No detailed JSON results found.")
    else:
        print(f"\n❌ ERROR: No results generated for {model_id} {precision}-bit. Check {log_filename}.")

def main():
    check_hardware()
    setup_environment()
    
    models = [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "google/gemma-2-9b-it"
    ]
    precisions = [8, 4]
    
    for model in models:
        for prec in precisions:
            evaluate_model(model, prec)
            
    print("\n" + "="*80)
    print("🏆 FINAL EKAQUANT SWEEP SUMMARY")
    print("="*80)
    
    all_df = []
    csv_files = sorted(glob.glob("eval_*/calculated.csv"))
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df.insert(0, "Config", f.split("/")[0])
            all_df.append(df)
        except Exception:
            pass
            
    if all_df:
        res = pd.concat(all_df, ignore_index=True)
        print(res.to_markdown(index=False))
        print("\n📦 Zipping artifacts...")
        os.system("zip -q -r all_sweep_results.zip eval_*")
        print("✅ all_sweep_results.zip is ready for download!")
    else:
        print("⚠️ No final results to summarize.")

if __name__ == "__main__":
    main()
