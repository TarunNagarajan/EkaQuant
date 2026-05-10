import argparse
import gc
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from ekaquant.quantization import TaskAwareQuantizer

def add_eka_eval_to_path(eka_eval_path: str):
    project_root = os.path.abspath(eka_eval_path)
    if not os.path.isdir(project_root):
        raise FileNotFoundError(f"eka-eval path not found: {project_root}")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

def load_sensitivity_map(summary_path):
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sensitivity_map = {}
    for exp in data['merged']['experiments']:
        delta = exp['overall_delta']
        score = abs(delta) if delta < 0 else 0.0
        for mod_name in exp['module_names']:
            if mod_name.endswith('.mlp'):
                sensitivity_map[mod_name + '.gate_proj'] = score
                sensitivity_map[mod_name + '.up_proj'] = score
                sensitivity_map[mod_name + '.down_proj'] = score
            elif mod_name.endswith('.self_attn'):
                sensitivity_map[mod_name + '.q_proj'] = score
                sensitivity_map[mod_name + '.k_proj'] = score
                sensitivity_map[mod_name + '.v_proj'] = score
                sensitivity_map[mod_name + '.o_proj'] = score
            else:
                sensitivity_map[mod_name] = score
    return sensitivity_map

def evaluate_model(model, tokenizer, args):
    from eka_eval.benchmarks.tasks.multilingual.arc_c_in import evaluate_arc_c_in
    
    device = next(model.parameters()).device
    
    # We do not specify device_map="auto" in pipeline because model is already distributed
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    scores = evaluate_arc_c_in(
        pipe=pipe,
        tokenizer=tokenizer,
        model_name_for_logging=args.model_id,
        device=device,
        dataset_name="sarvamai/arc-challenge-indic",
        target_languages=args.languages.split(","),
        dataset_split=args.dataset_split,
        max_new_tokens=args.max_new_tokens,
        save_detailed=False,
        use_checkpoints=False,
        prompt_template_name_zeroshot="arc_c_in_0shot",
        prompt_file_benchmark_key="arc_c_in",
        prompt_file_category="indic",
        use_fwe_kv_eviction=False,
    )
    return scores

def print_memory_usage(prefix):
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"[{prefix}] VRAM Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--summary-json", type=str, default="data/sweep_summary.json")
    parser.add_argument("--eka-eval-path", type=str, default="eka-eval")
    parser.add_argument("--languages", type=str, default="hi,bn")
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--budget-mb", type=float, default=150.0)
    args = parser.parse_args()

    print("Importing eka-eval dependencies...")
    add_eka_eval_to_path(args.eka_eval_path)
    
    print(f"Loading Tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    results = {}

    # 1. 8-bit Baseline
    print("\n" + "="*50)
    print("1. EVALUATING 8-BIT BASELINE")
    print("="*50)
    bnb_config_8 = BitsAndBytesConfig(load_in_8bit=True)
    model_8bit = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_config_8, device_map="auto")
    print_memory_usage("8-bit Baseline")
    scores_8bit = evaluate_model(model_8bit, tokenizer, args)
    results["8-bit"] = scores_8bit
    del model_8bit
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Uniform 4-bit
    print("\n" + "="*50)
    print("2. EVALUATING UNIFORM 4-BIT (BITSANDBYTES)")
    print("="*50)
    bnb_config_4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model_4bit = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_config_4, device_map="auto")
    print_memory_usage("Uniform 4-bit")
    scores_4bit = evaluate_model(model_4bit, tokenizer, args)
    results["Uniform 4-bit"] = scores_4bit
    del model_4bit
    gc.collect()
    torch.cuda.empty_cache()

    # 3. EkaQuant Task-Aware 4-bit
    print("\n" + "="*50)
    print("3. EVALUATING EKAQUANT TASK-AWARE 4-BIT")
    print("="*50)
    model_eka = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    quantizer = TaskAwareQuantizer(model_eka, tokenizer)
    quantizer.sensitivity_map = load_sensitivity_map(args.summary_json)
    print(f"Applying Knapsack Selection (Budget: {args.budget_mb} MB)...")
    model_eka = quantizer.quantize(calibration_texts=[], selection_method="knapsack", budget_mb=args.budget_mb)
    print_memory_usage("EkaQuant 4-bit")
    scores_eka = evaluate_model(model_eka, tokenizer, args)
    results["EkaQuant 4-bit"] = scores_eka
    del model_eka
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("FINAL DOWNSTREAM RESULTS (ARC-Challenge-Indic)")
    print("="*50)
    print(json.dumps(results, indent=2))
    
    with open("downstream_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to downstream_results.json")

if __name__ == "__main__":
    main()
