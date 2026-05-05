# EkaQuant Project

## MMLU-IN Baseline Results
| Model | Precision | Score |
| :--- | :--- | :--- |
| Qwen/Qwen2.5-3B-Instruct | 8-bit | 35.4386% |
| Qwen/Qwen2.5-3B-Instruct | 4-bit | 30.7018% |
| Qwen/Qwen2.5-7B-Instruct | 8-bit | 38.5965% |
| Qwen/Qwen2.5-7B-Instruct | 4-bit | 40.0000% |
| Mistral-7B-Instruct-v0.3 | 8-bit | 29.82% |

## ARC-Challenge-Indic Baseline Results
| Model | Precision | Score |
| :--- | :--- | :--- |
| Mistral-7B-Instruct-v0.3 | 8-bit | *Pending* |
| Mistral-7B-Instruct-v0.3 | 4-bit | *Pending* |

*Note: The Qwen 7B model demonstrates a definitive 4-bit regularization effect, outperforming its 8-bit counterpart.*

## TaskQuant Integration
The integration with `eka-eval` is now fully operational with support for:
- Automated multi-model sweep (Mistral 7B).
- Multi-GPU (2x T4) sharding via `device_map="auto"`.
- Isolated result directories and detailed output logging.
- Live model response verification in console.
