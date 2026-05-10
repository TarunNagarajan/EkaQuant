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

def compute_kl_divergence(logits_p, logits_q):
    p = F.log_softmax(logits_p, dim=-1)
    q = F.log_softmax(logits_q, dim=-1)
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
            all_logits.append(outputs.logits.cpu())
    return all_logits

def main():
    parser = argparse.ArgumentParser(description="Validate KL Divergence of EkaQuant vs Uniform 4-bit")
    parser.add_argument("--model-id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--budget-mb", type=float, default=150.0)
    args = parser.parse_args()

    test_texts = [
        "भारत की राजधानी क्या है? दिल्ली एक बहुत बड़ा शहर है।",
        "আমার নাম কি? আমি বাংলায় কথা বলতে পারি।",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    gt_logits = evaluate_model_logits(model_fp16, tokenizer, test_texts)
    
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()
    
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
    
    model_eka = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    quantizer = TaskAwareQuantizer(model_eka, tokenizer)
    quantizer.sensitivity_map = load_sensitivity_map(args.summary_json)
    
    model_eka = quantizer.quantize(
        calibration_texts=[],
        selection_method="knapsack",
        budget_mb=args.budget_mb
    )
    
    eka_logits = evaluate_model_logits(model_eka, tokenizer, test_texts)
    
    eka_kls = []
    for p, q in zip(gt_logits, eka_logits):
        eka_kls.append(compute_kl_divergence(p, q))
        
    for i, text in enumerate(test_texts):
        print(f"Text {i+1}: Uniform KL: {uniform_kls[i]:.6f} | EkaQuant KL: {eka_kls[i]:.6f}")
        
    avg_uniform = sum(uniform_kls) / len(uniform_kls)
    avg_eka = sum(eka_kls) / len(eka_kls)
    
    print(f"Average Uniform KL: {avg_uniform:.6f}")
    print(f"Average EkaQuant KL: {avg_eka:.6f}")
    print(f"Improvement: {(avg_uniform - avg_eka) / avg_uniform * 100:.2f}%")

if __name__ == "__main__":
    main()
