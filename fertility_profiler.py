from transformers import AutoTokenizer
import pandas as pd
import warnings
import logging

warnings.filtertransformers = logging.CRITICAL
warnings.filterwarnings("ignore")

# The 3 major architectures
models = {
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen": "Qwen/Qwen2.5-7B-Instruct",
    "Sarvam": "sarvamai/sarvam-1"
}

# Sample parallel text (English vs Hindi)
texts = {
    "English": "The quick brown fox jumps over the lazy dog. Quantum mechanics is the study of physics at the microscopic level.",
    "Hindi": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है। क्वांटम यांत्रिकी सूक्ष्म स्तर पर भौतिकी का अध्ययन है।"
}

results = []

for name, model_id in models.items():
    print(f"Loading tokenizer for {name}...")
    try:
        # Load tokenizer (uses CPU/RAM only)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        en_tokens = len(tokenizer.encode(texts["English"]))
        hi_tokens = len(tokenizer.encode(texts["Hindi"]))
        
        ratio = hi_tokens / en_tokens
        
        results.append({
            "Model": name,
            "English Tokens": en_tokens,
            "Hindi Tokens": hi_tokens,
            "Fertility Penalty": f"{ratio:.2f}x"
        })
    except Exception as e:
        print(f"Could not load {name}: {e}")

df = pd.DataFrame(results)
print("\n--- TOKEN FERTILITY CRISIS ---")
print(df.to_markdown(index=False))
