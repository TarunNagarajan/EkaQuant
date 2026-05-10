import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import argparse


def run_probe(model_id, languages="hi"):
    print(f"🚀 Initializing Cross-Lingual Invariance Probe for: {model_id}")

    # 1. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda", load_in_4bit=True
    )

    # 2. Parallel Premises (English vs Hindi)
    # These are basic logical premises that should converge in latent space
    premises = [
        ("The sun rises in the east.", "सूर्य पूर्व में उगता है।"),
        (
            "Gravity pulls objects toward the earth.",
            "गुरुत्वाकर्षण वस्तुओं को पृथ्वी की ओर खींचता है।",
        ),
        ("Water boils at 100 degrees Celsius.", "पानी 100 डिग्री सेल्सियस पर उबलता है।"),
        ("A triangle has three sides.", "एक त्रिभुज की तीन भुजाएं होती हैं।"),
        ("Oxygen is necessary for human life.", "जीवन के लिए ऑक्सीजन आवश्यक है।"),
        ("The capital of France is Paris.", "फ्रांस की राजधानी पेरिस है।"),
        ("Mammals feed their young with milk.", "स्तनधारी अपने बच्चों को दूध पिलाते हैं।"),
        ("The earth revolves around the sun.", "पृथ्वी सूर्य के चारों ओर घूमती है।"),
    ]

    layer_similarities = []

    print(f"Probing {model_id} layer-by-layer...")

    with torch.no_grad():
        for en_text, hi_text in premises:
            # Get hidden states for English
            en_inputs = tokenizer(en_text, return_tensors="pt").to("cuda")
            en_outputs = model(**en_inputs, output_hidden_states=True)
            en_hidden = en_outputs.hidden_states  # Tuple of (batch, seq, dim)

            # Get hidden states for Hindi
            hi_inputs = tokenizer(hi_text, return_tensors="pt").to("cuda")
            hi_outputs = model(**hi_inputs, output_hidden_states=True)
            hi_hidden = hi_outputs.hidden_states

            # Calculate Cosine Similarity per layer (using the mean of the sequence to capture semantic center)
            sims = []
            for l_idx in range(len(en_hidden)):
                v_en = en_hidden[l_idx][0].mean(dim=0).float()
                v_hi = hi_hidden[l_idx][0].mean(dim=0).float()
                similarity = torch.nn.functional.cosine_similarity(v_en, v_hi, dim=0)
                sims.append(similarity.item())
            layer_similarities.append(sims)

    # 3. Aggregate and Plot
    avg_sims = np.mean(layer_similarities, axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(avg_sims, marker="o", linestyle="-", color="b", linewidth=2)
    plt.axhline(y=0.9, color="r", linestyle="--", label="High Convergence (>0.9)")
    plt.title(f"Cross-Lingual Invariance (EN-HI): {model_id}")
    plt.xlabel("Layer Depth")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_img = f"invariance_plot_{model_id.replace('/', '_')}.png"
    plt.savefig(output_img)
    print(f"📊 Plot saved as: {output_img}")

    print("\n--- INVARIANCE DATA (Cosine Similarity) ---")
    for i, sim in enumerate(avg_sims):
        marker = "🔥 [CONVERGED]" if sim > 0.9 else ""
        print(f"Layer {i:02d}: {sim:.4f} {marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    run_probe(args.model)
