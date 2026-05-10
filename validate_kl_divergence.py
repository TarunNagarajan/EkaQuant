import argparse
import gc
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ekaquant.quantization import TaskAwareQuantizer

def load_sensitivity_map(summary_path):
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sensitivity_map = {}
    for exp in data['merged']['experiments']:
        # 'overall_delta' is negative if the score dropped (higher drop = higher sensitivity)
        delta = exp['overall_delta']
        # If delta < 0, we use its absolute value as sensitivity.
        score = abs(delta) if delta < 0 else 0.0
        
        for mod_name in exp['module_names']:
            sensitivity_map[mod_name] = score
            
    return sensitivity_map

def compute_kl_divergence(logits_p, logits_q):
    """
    Computes KL(P || Q) where P is the ground truth (e.g. bfloat16) and Q is the quantized model.
    """
    p = F.log_softmax(logits_p, dim=-1)
    q = F.log_softmax(logits_q, dim=-1)
    
    # log_target=True means target (p) is already in log space.
    kl_div = F.kl_div(q, p, reduction='batchmean', log_target=True)
    return kl_div.item()

def evaluate_model_logits(model, tokenizer, texts, max_length=128):
    all_logits = []
    device = next(model.parameters()).device
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # Store on CPU to save VRAM
            all_logits.append(outputs.logits.cpu())
    return all_logits

def main():
    parser = argparse.ArgumentParser(description="Validate KL Divergence of EkaQuant vs Uniform 4-bit")
    parser.add_argument("--model-id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--summary-json", type=str, required=True, help="Path to sweep summary.json")
    parser.add_argument("--budget-mb", type=float, default=500.0, help="VRAM budget to keep layers in FP16 (in MB)")
    args = parser.parse_args()

    # We will test on a couple of short examples representing our target languages
    test_texts = [
        "भारत की राजधानी क्या है? दिल्ली एक बहुत बड़ा शहर है।", # Hindi
        "আমার নাম কি? আমি বাংলায় কথা বলতে পারি।", # Bengali
        "The quick brown fox jumps over the lazy dog.", # English
    ]
    
    print(f"Loading Ground Truth Model (bfloat16): {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 1. Ground Truth (FP16/BF16)
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    gt_logits = evaluate_model_logits(model_fp16, tokenizer, test_texts)
    
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Uniform 4-bit (Baseline)
    print("\nLoading Uniform 4-bit Model (BitsAndBytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    model_4bit = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    uniform_logits = evaluate_model_logits(model_4bit, tokenizer, test_texts)
    
    uniform_kls = []
    for p, q in zip(gt_logits, uniform_logits):
        uniform_kls.append(compute_kl_divergence(p, q))
    
    del model_4bit
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. EkaQuant Task-Aware 4-bit
    print("\nLoading EkaQuant Task-Aware Model...")
    model_eka = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    quantizer = TaskAwareQuantizer(model_eka, tokenizer)
    
    # Inject the empirical sensitivity map directly
    print(f"Applying Sensitivity Map from {args.summary_json}")
    quantizer.sensitivity_map = load_sensitivity_map(args.summary_json)
    
    # Quantize using knapsack. We allocate a budget to keep the most sensitive layers in bfloat16.
    print(f"Running Knapsack Selection (Budget: {args.budget_mb} MB)...")
    model_eka = quantizer.quantize(
        calibration_texts=[], # Skipped since sensitivity_map is pre-loaded
        selection_method="knapsack",
        budget_mb=args.budget_mb
    )
    
    eka_logits = evaluate_model_logits(model_eka, tokenizer, test_texts)
    
    eka_kls = []
    for p, q in zip(gt_logits, eka_logits):
        eka_kls.append(compute_kl_divergence(p, q))
        
    print("\n" + "="*50)
    print(f"RESULTS (Lower KL is better, meaning closer to FP16 Ground Truth)")
    print("="*50)
    for i, text in enumerate(test_texts):
        print(f"Test Text {i+1}: '{text[:30]}...'")
        print(f"  Uniform 4-bit KL: {uniform_kls[i]:.6f}")
        print(f"  EkaQuant 4-bit KL: {eka_kls[i]:.6f}")
        
    avg_uniform = sum(uniform_kls) / len(uniform_kls)
    avg_eka = sum(eka_kls) / len(eka_kls)
    print("-"*50)
    print(f"Average Uniform 4-bit KL: {avg_uniform:.6f}")
    print(f"Average EkaQuant KL:      {avg_eka:.6f}")
    
    if avg_eka < avg_uniform:
        print(f"\nSUCCESS: EkaQuant is {(avg_uniform - avg_eka) / avg_uniform * 100:.2f}% closer to Ground Truth than Uniform 4-bit!")

if __name__ == "__main__":
    main()
