# EkaQuant

EkaQuant is a focused extraction of TaskQuant for integration with eka-eval.

It targets one gap in eka-eval’s current model loading path: uniform 4-bit quantization can hurt some tasks and languages more than others. EkaQuant adds task-aware selective quantization so only lower-sensitivity layers are quantized and higher-sensitivity layers remain in higher precision.

## Why this fits eka-eval

- eka-eval is a multilingual benchmark pipeline for English and low-resource languages.
- eka-eval already supports 4-bit and 8-bit inference.
- language-specific evaluation workloads naturally provide calibration text for task-aware layer sensitivity.
- eka-eval’s modular benchmark/task registry can pass calibration records and task args to a quantization step before evaluation.

## Included modules

- `ekaquant.sensitivity`
  - Fisher sensitivity
  - Magnitude sensitivity
  - Perturbation sensitivity
- `ekaquant.selection`
  - Percentile, Otsu, elbow, gradient, cumulative, knapsack selection
- `ekaquant.quantization`
  - Selective replacement of `nn.Linear` with `bitsandbytes` `Linear4bit`
- `ekaquant.integrations.eka_eval`
  - Thin integration helpers for eka-eval worker/model flow

## Install

```bash
pip install -e .
```

## Minimal usage

```python
from ekaquant.integrations.eka_eval import apply_task_aware_quantization

model, metadata = apply_task_aware_quantization(
    model=model,
    tokenizer=tokenizer,
    calibration_texts=calibration_texts,
    sensitivity_method="fisher",
    selection_method="pct",
    percentile=0.2,
)
```

## Suggested eka-eval integration point

Use this in the local-model branch of `eka_eval/core/model_loader.py` after model load and before pipeline creation. Feed calibration text from selected benchmark/task records in `evaluation_worker.py`.
