from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from ..quantization import TaskAwareQuantizer


def records_to_calibration_texts(
    records: Sequence[Dict],
    text_fields: Sequence[str] = ("question", "prompt", "text", "input"),
    max_items: int = 64,
) -> List[str]:
    texts: List[str] = []
    for row in records[:max_items]:
        for key in text_fields:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
                break
    if not texts:
        raise ValueError("No calibration texts extracted from records.")
    return texts


def apply_task_aware_quantization(
    model,
    tokenizer,
    calibration_texts: Iterable[str],
    sensitivity_method: str = "fisher",
    selection_method: str = "pct",
    percentile: float = 0.2,
    sensitivity_ratio: float = 0.05,
    budget: float = 0.95,
    budget_mb: float = 4096,
    invert_selection: bool = False,
    reduction: str = "mean",
    fisher_clip_percentile: float | None = 99.0,
    fisher_clip_samples: int = 32,
    max_length: int = 2048,
) -> Tuple[object, Dict]:
    quantizer = TaskAwareQuantizer(model, tokenizer)
    quantized_model = quantizer.quantize(
        calibration_texts=calibration_texts,
        sensitivity_method=sensitivity_method,
        selection_method=selection_method,
        percentile=percentile,
        sensitivity_ratio=sensitivity_ratio,
        budget=budget,
        budget_mb=budget_mb,
        invert_selection=invert_selection,
        reduction=reduction,
        fisher_clip_percentile=fisher_clip_percentile,
        fisher_clip_samples=fisher_clip_samples,
        max_length=max_length,
    )
    metadata = {
        "sensitivity_method": sensitivity_method,
        "selection_method": selection_method,
        "layers_scored": len(quantizer.sensitivity_map or {}),
    }
    return quantized_model, metadata
