import argparse
import json
import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from peft import PeftModel


def add_eka_eval_to_path(eka_eval_path: str):
    project_root = os.path.abspath(eka_eval_path)
    if not os.path.isdir(project_root):
        raise FileNotFoundError(f"eka-eval path not found: {project_root}")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def print_memory_usage(prefix):
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(
        f"[{prefix}] VRAM Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB"
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the trained SR-LoRA adapter",
    )
    parser.add_argument("--eka-eval-path", type=str, default="eka-eval")
    parser.add_argument("--languages", type=str, default="hi,bn")
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    args = parser.parse_args()

    print("Importing eka-eval dependencies...")
    add_eka_eval_to_path(args.eka_eval_path)

    print(f"Loading Tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load the Base Model in Uniform 4-bit
    print("\n" + "=" * 50)
    print("1. LOADING BASE MODEL (UNIFORM 4-BIT)")
    print("=" * 50)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb_config, device_map="auto"
    )

    # 2. Inject the SR-LoRA Adapter
    print("\n" + "=" * 50)
    print(f"2. INJECTING SR-LORA ADAPTER ({args.adapter_path})")
    print("=" * 50)
    model = PeftModel.from_pretrained(model, args.adapter_path)

    # Merge the adapter weights into the base 4-bit weights for fastest inference
    # (Note: merging 4-bit is tricky in peft, so we just run it with the adapter active)
    model.eval()

    print_memory_usage("SR-LoRA 4-bit")

    # 3. Evaluate
    print("\n" + "=" * 50)
    print("3. EVALUATING DOWNSTREAM BENCHMARK (ARC-Challenge-Indic)")
    print("=" * 50)

    scores = evaluate_model(model, tokenizer, args)

    print("\n" + "=" * 50)
    print("FINAL DOWNSTREAM RESULTS (SR-LoRA)")
    print("=" * 50)
    print(json.dumps(scores, indent=2))

    with open("sr_lora_results.json", "w") as f:
        json.dump(scores, f, indent=2)
    print("Saved results to sr_lora_results.json")


if __name__ == "__main__":
    main()
