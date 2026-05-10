import argparse
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from ekaquant.quantization import TaskAwareQuantizer


def load_sensitivity_map(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sensitivity_map = {}
    for exp in data["merged"]["experiments"]:
        delta = exp["overall_delta"]
        score = abs(delta) if delta < 0 else 0.0
        for mod_name in exp["module_names"]:
            sensitivity_map[mod_name] = score
    return sensitivity_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3"
    )
    parser.add_argument("--summary-json", type=str, default="data/sweep_summary.json")
    parser.add_argument("--budget-mb", type=float, default=50.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Base Model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print("Injecting Surgical Recovery LoRA (SR-LoRA)...")
    quantizer = TaskAwareQuantizer(model, tokenizer)
    quantizer.sensitivity_map = load_sensitivity_map(args.summary_json)

    # We use mode="sr_lora" which quantizes everything to 4-bit,
    # but injects LoRA adapters onto the targeted bottlenecks.
    model = quantizer.quantize(
        calibration_texts=[],
        selection_method="knapsack",
        mode="sr_lora",
        budget_mb=args.budget_mb,
        lora_rank=8,
        lora_alpha=16,
    )

    model.gradient_checkpointing_enable()
    print("Trainable Parameters:")
    model.print_trainable_parameters()

    # Load a small sample dataset for calibration/recovery training
    # For demonstration, we use a tiny subset of Hindi/Bengali Wikipedia or similar.
    # In practice, use a targeted alignment dataset.
    print("Loading calibration dataset...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train[:100]")

    def format_prompts(examples):
        texts = []
        for text in examples["text"]:
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(format_prompts, batched=True)

    from trl import SFTConfig

    sft_config = SFTConfig(
        output_dir="./sr_lora_output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=50,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        max_seq_length=512,
    )

    print("Starting SR-LoRA fine-tuning...")
    trainer.train()

    print("Saving SR-LoRA adapter...")
    model.save_pretrained("ekaquant_sr_lora_adapter")
    print("Done! The surgical adapter is saved and ready for inference.")


if __name__ == "__main__":
    main()
