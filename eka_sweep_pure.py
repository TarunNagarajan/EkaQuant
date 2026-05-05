import os
import sys
import subprocess
import glob
import json
import pandas as pd
import shutil

def patch_file(file_path, old_str, new_str):
    if not os.path.exists(file_path):
        return
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(old_str, new_str)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def set_precision(loader_path, precision):
    with open(loader_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    new_lines = []
    in_block = False
    
    for line in lines:
        if "quantization_config = BitsAndBytesConfig(" in line:
            in_block = True
            if precision == 8:
                new_lines.append("    quantization_config = BitsAndBytesConfig(load_in_8bit=True)\n")
            else:
                new_lines.append("    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=target_dtype, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)\n")
            continue
        if in_block:
            if ")" in line:
                in_block = False
            continue
        new_lines.append(line)
        
    with open(loader_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def check_hardware():
    print("="*60)
    print("1. HARDWARE VERIFICATION")
    print("="*60)
    try:
        gpu_info = subprocess.check_output("nvidia-smi -L", shell=True).decode()
        print(gpu_info.strip())
        if "T4" not in gpu_info:
            print("ERROR: T4 GPU not detected.")
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
        pass
        
    print("Installing dependencies...")
    if not os.path.exists("eka-eval"):
        os.system("git clone -q https://github.com/lingo-iitgn/eka-eval.git")
        os.system("cd eka-eval && pip install -q -e .")
    
    print("Patching config...")
    config_path = "eka-eval/eka_eval/config/benchmark_config.py"
    patch_file(config_path, "indic.mmlu_in.evaluate_mmlu_in", "multilingual.mmlu_in.evaluate_mmlu_in")
    patch_file(config_path, "indic.arc_c_in.evaluate_arc_c_in", "multilingual.arc_c_in.evaluate_arc_c_in")
    
    print("Patching model_loader for multi-GPU...")
    loader_path = "eka-eval/eka_eval/core/model_loader.py"
    if os.path.exists(loader_path):
        with open(loader_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("device_map_arg = {'': f'cuda:{target_device_id}'}", "device_map_arg = 'auto'")
        with open(loader_path, "w", encoding="utf-8") as f:
            f.write(content)

    print("Injecting direct logging into ARC-C-IN...")
    arc_path = "eka-eval/eka_eval/benchmarks/tasks/multilingual/arc_c_in.py"
    if os.path.exists(arc_path):
        with open(arc_path, "r", encoding="utf-8") as f:
            arc_content = f.read()
        
        # Inject print statements so we can see the model's raw reasoning in the logs
        arc_patch = """
                predicted_letter = _parse_predicted_answer(generated_text, lang_code, all_mappings)                print("\n" + "-"*80)
                print(f"QUESTION [{lang_code.upper()}]:\n{question}")
                print(f"CHOICES:\n{choices_dict}")
                print(f"RAW OUTPUT:\n{generated_text}")
                print(f"EXTRACTED: {predicted_letter} | TRUE: {answer_key}")
                print("-" * 80 + "\n", flush=True)
                
                # Convert to indices for metric computation"""
        
        # Only replace if not already patched
        if "RAW OUTPUT:" not in arc_content:
            arc_content = arc_content.replace("                predicted_letter = _parse_predicted_answer(generated_text, lang_code, all_mappings)\n                \n                # Convert to indices for metric computation", arc_patch)
            with open(arc_path, "w", encoding="utf-8") as f:
                f.write(arc_content)
        
    print("Setup complete.\n")

def clean_old_results():
    """Recursively delete all old calculated.csv and detailed_results to prevent ghost data."""
    for root, dirs, files in os.walk("."):
        for f in files:
            if f == "calculated.csv":
                try:
                    os.remove(os.path.join(root, f))
                except:
                    pass
        if "detailed_results" in dirs:
            try:
                shutil.rmtree(os.path.join(root, "detailed_results"))
            except:
                pass

def evaluate_model(model_id, precision):
    model_name = model_id.split("/")[-1]
    folder_name = f"eval_{precision}bit_{model_name}"
    
    print("\n" + "="*80)
    print(f"STARTING {precision}-BIT EVALUATION: {model_id} (ARC-Challenge-Indic)")
    print("="*80)
    
    loader_path = "eka-eval/eka_eval/core/model_loader.py"
    set_precision(loader_path, precision)

    clean_old_results()
    
    log_filename = f"eval_{precision}bit_{model_name}.log"
    print(f"Running benchmarks... (Log: {log_filename})")
    
    # 9 = INDIC BENCHMARKS, 6 = ARC-Challenge-Indic
    input_seq = f"1\n1\n{model_id}\nno\n9\n6\nno\n"
    
    with open(log_filename, "w") as log_file:
        subprocess.run(
            ["python", "eka-eval/scripts/run_benchmarks.py"],
            input=input_seq,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        
    # Find the newly generated calculated.csv natively with Python
    source_dir = None
    for root, dirs, files in os.walk("."):
        if "calculated.csv" in files and "eval_" not in root:
            source_dir = root
            break
            
    if source_dir:
        os.makedirs(folder_name, exist_ok=True)
        abs_folder = os.path.abspath(folder_name)
        
        # Native Python copy to avoid 'cp' shell errors
        shutil.copy2(os.path.join(source_dir, "calculated.csv"), os.path.join(abs_folder, "calculated.csv"))
        
        if os.path.exists(os.path.join(source_dir, "detailed_results")):
            dest_detailed = os.path.join(abs_folder, "detailed_results")
            if os.path.exists(dest_detailed):
                shutil.rmtree(dest_detailed)
            shutil.copytree(os.path.join(source_dir, "detailed_results"), dest_detailed)
        
        print("\nEVALUATION COMPLETE!\n")
        csv_path = os.path.join(abs_folder, "calculated.csv")
        df = pd.read_csv(csv_path)
        print(df.to_markdown(index=False))
            
        print("\nSNEAK PEEK: LAST 3 MODEL RESPONSES (from log)")
        try:
            with open(log_filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
                print("".join(lines[-60:]))
        except Exception as e:
            print(f"Could not read log for sneak peek: {e}")

    else:
        print(f"\nERROR: No results generated for {model_id} {precision}-bit.")
        if os.path.exists(log_filename):
            with open(log_filename, "r", encoding="utf-8") as f:
                print("\n--- LAST 2000 CHARACTERS OF LOG ---")
                print(f.read()[-2000:])
        else:
            print(f"Log file {log_filename} does not exist.")

def main():
    check_hardware()
    setup_environment()
    
    models = ["mistralai/Mistral-7B-Instruct-v0.3"]
    precisions = [8, 4]
    
    for model in models:
        for prec in precisions:
            evaluate_model(model, prec)
            
    print("\n" + "="*80)
    print("FINAL EKAQUANT SWEEP SUMMARY (ARC-Challenge-Indic)")
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
        try:
            import zipfile
            with zipfile.ZipFile('all_sweep_results.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk('.'):
                    if 'eval_' in root:
                        for file in files:
                            zipf.write(os.path.join(root, file))
            print("\nArchived all results into all_sweep_results.zip")
        except Exception as e:
            print(f"Failed to zip results natively: {e}")
    else:
        print("No final results to summarize.")

if __name__ == "__main__":
    main()
