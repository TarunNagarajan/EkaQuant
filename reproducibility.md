# EkaQuant Reproducibility Guide

This document provides step-by-step instructions for independently reproducing the 4-hour interpretability sweep, the KL-divergence validation, and the generation of visual artifacts for the EkaQuant project.

## 1. Environment Setup

It is highly recommended to run these scripts in an environment with at least 2x NVIDIA T4 GPUs (e.g., Kaggle, Google Colab Pro, or an equivalent cloud instance).

```bash
# Clone the repository
git clone https://github.com/TarunNagarajan/EkaQuant.git
cd EkaQuant

# Install required dependencies
pip install -q transformers accelerate bitsandbytes datasets kneed scikit-image tabulate matplotlib seaborn pandas scipy

# Install EkaQuant locally in editable mode
pip install -q -e .
```

## 2. Generating the Sensitivity Sweep Data

The core of EkaQuant relies on a language-aware sensitivity map. To generate this map, run the dual-GPU interpretability sweep. This will systematically ablate Transformer blocks across the specified target languages and measure the performance delta.

*Note: This process takes approximately 4-5 hours on a 2x T4 setup for Mistral-7B.*

```bash
python scripts/run_interpretability_dual_t4.py \
    --model "mistralai/Mistral-7B-Instruct-v0.3" \
    --languages "hi,bn,en" \
    --max-new-tokens 5
```

**Expected Output:**
Upon completion, the script will generate a directory under `results_output/interpretability/dual_t4_<timestamp>/` containing a `summary.json` file. This JSON file is the empirical sensitivity map.

*(For immediate validation testing without running the sweep, you can use the pre-computed `summary.json` provided in the `data/` directory of this repository).*

## 3. Validating the KL-Divergence Recovery

Once the sensitivity map is generated (or using the provided one), you can run the validation script to prove the mathematical recovery of EkaQuant over uniform 4-bit quantization.

The script compares the probability distributions (logits) of:
1. The unquantized `bfloat16` model (Ground Truth).
2. The standard `nf4` quantized model (Uniform 4-bit Baseline).
3. The EkaQuant Task-Aware selectively quantized model.

```bash
python scripts/validate.py \
    --model-id "mistralai/Mistral-7B-Instruct-v0.3" \
    --summary-json "data/sweep_summary.json" \
    --budget-mb 150.0
```

**Understanding the Budget:**
The `--budget-mb` flag determines how much VRAM (in Megabytes) the Knapsack algorithm is allowed to use to protect critical layers in `bfloat16`. 
- 150 MB is the recommended baseline (protecting ~1.5% of the model).
- You can iterate this value (e.g., 50.0, 300.0, 500.0) to observe the tradeoff between VRAM usage and KL recovery.

## 4. Generating Visual Artifacts

To generate the comprehensive suite of 20 visual charts and rankings analyzing the network's fragility and the algorithm's allocation:

```bash
python scripts/generate_visuals.py
```

**Expected Output:**
This script will read `data/sweep_summary.json` and generate a `plots/` directory containing structured subfolders. Each subfolder will contain a high-resolution `.png` chart and an `explanation.md` detailing the analytical significance of the visualization.
