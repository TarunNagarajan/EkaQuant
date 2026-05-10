# EkaQuant: Task-Aware Selective Quantization

EkaQuant is a selective quantization library designed to recover the performance lost during standard uniform quantization (like 4-bit BitsAndBytes), specifically targeting the fragile representations of low-resource languages (e.g., Hindi, Bengali) in Small Language Models (SLMs).

By empirically identifying the "Language Bottlenecks" and allocating a tiny high-precision memory budget using a Knapsack algorithm, EkaQuant mathematically restores the model's intelligence without the massive VRAM overhead of an 8-bit or 16-bit model.

## 🚀 Key Achievement: 54.16% Mathematical Recovery

In our latest empirical validation using **Mistral-7B-Instruct-v0.3**, standard Uniform 4-bit quantization severely degraded the model's internal probability distributions (KL-Divergence) for Hindi and Bengali.

By allocating a microscopic **150 MB budget** (less than 1.5% of the model's total parameters) to keep the 5 most critical language-specific layers in `bfloat16`, **EkaQuant closed the gap to the unquantized Ground Truth by 54.16%**.

### KL Divergence Recovery (Lower is better)
| Language | Uniform 4-bit KL | EkaQuant 4-bit KL | Improvement |
| :--- | :--- | :--- | :--- |
| **Hindi** | 17.75 | **6.93** | 🟢 **60.9%** |
| **Bengali** | 17.37 | **9.00** | 🟢 **48.1%** |
| **English** | 0.12 | 0.22 | 🟡 (Negligible impact) |
| **Average** | 11.74 | **5.38** | 🟢 **54.16% Overall** |

---

## 📊 The Proof is in the Data

We ran a rigorous 4-hour interpretability sweep across Dual-T4 GPUs to measure the performance drop ("Sensitivity Delta") when individual Transformer blocks were ablated. We have generated **20 distinct visualizations** mapping the exact fragility of the model.

You can view the full suite of visualizations in the [`plots/`](plots/) directory. Highlights include:

1.  **[Language Bottlenecks (Bengali vs English)](plots/04_bn_vs_en_scatter.png):** Demonstrates how specific layers (`layer.13.k_proj`, `layer.27.up_proj`) uniquely bottleneck Bengali while having almost no impact on English.
2.  **[The Anchor (Layer 0)](plots/10_layer_0_anchor.png):** Visualizes the catastrophic collapse (Delta: -45.27) that occurs if the very first layer is perturbed, proving it must be protected at all costs.
3.  **[KL Divergence Comparison](plots/13_kl_divergence_comparison.png):** A stark visual of EkaQuant's massive error reduction for Indic languages compared to naive Uniform 4-bit.
4.  **[VRAM Tradeoff Curve](plots/15_vram_tradeoff_curve.png):** Showcasing our massive ROI: 54% recovery for only 150 MB of VRAM.
5.  **[Knapsack Allocation Map](plots/16_knapsack_allocation.png):** Shows exactly which layers the algorithm mathematically deemed "mission-critical" enough to protect with the 150 MB budget.

*Also check out the [Artifacts Directory](artifacts/) for raw CSV rankings and a Markdown Heatmap.*

---

## MMLU-IN Baseline Results
*(Note: Qwen-3B struggles with 4-bit, making it a prime future target for EkaQuant)*

| Model | Precision | Score |
| :--- | :--- | :--- |
| Qwen/Qwen2.5-3B-Instruct | 8-bit | 35.4386% |
| Qwen/Qwen2.5-3B-Instruct | 4-bit | 30.7018% |
| Qwen/Qwen2.5-7B-Instruct | 8-bit | 38.5965% |
| Qwen/Qwen2.5-7B-Instruct | 4-bit | 40.0000% |
| Mistral-7B-Instruct-v0.3 | 8-bit | 29.82% |

## Integration
The integration with `eka-eval` is fully operational with support for:
- Automated multi-model sweep (Mistral 7B).
- Multi-GPU (2x T4) sharding via `device_map="auto"`.
- KL-Divergence validation scripts (`validate_kl_divergence.py`).
- Automated visual artifact generation (`generate_extended_artifacts.py`).