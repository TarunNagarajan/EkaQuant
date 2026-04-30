from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch.nn as nn
from kneed import KneeLocator
from skimage.filters import threshold_otsu


def _valid_scores(sensitivity_map: Dict[str, float], arg_name: str = "sensitivity_map") -> np.ndarray:
    if not isinstance(sensitivity_map, dict) or not sensitivity_map:
        raise ValueError(f"{arg_name} must be a non-empty dictionary.")
    scores = np.array(list(sensitivity_map.values()), dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{arg_name} contains non-finite values.")
    return scores


def threshold_pct(sensitivity_map: Dict[str, float], percentile: float) -> float:
    if not (0 < percentile <= 1):
        raise ValueError(f"percentile must be in (0, 1], got {percentile}")
    scores = _valid_scores(sensitivity_map)
    return float(np.quantile(scores, 1 - percentile))


def threshold_otsu_method(sensitivity_map: Dict[str, float]) -> float:
    scores = _valid_scores(sensitivity_map)
    if np.all(scores == scores[0]):
        return float(scores[0])
    return float(threshold_otsu(scores))


def threshold_elbow(sensitivity_map: Dict[str, float]) -> float:
    scores = sorted(_valid_scores(sensitivity_map), reverse=True)
    if len(scores) < 3:
        return float(scores[-1])
    kneedle = KneeLocator(range(len(scores)), scores, curve="convex", direction="decreasing")
    if kneedle.knee is None:
        return float(np.quantile(scores, 0.20))
    return float(scores[kneedle.knee])


def threshold_gradient(sensitivity_map: Dict[str, float], sensitivity_ratio: float = 0.01) -> float:
    if sensitivity_ratio <= 0:
        raise ValueError(f"sensitivity_ratio must be > 0, got {sensitivity_ratio}")
    scores = np.sort(_valid_scores(sensitivity_map))[::-1]
    if scores.size < 2:
        return float(scores[0])
    grad = np.abs(np.diff(np.log10(scores + 1e-10)))
    if grad.size == 0:
        return float(scores[-1])
    max_grad = np.max(grad)
    if not np.isfinite(max_grad) or max_grad <= 0:
        return float(scores[-1])
    grad_norm = grad / max_grad
    idxs = np.where(grad_norm < sensitivity_ratio)[0]
    idx = int(idxs[0]) if idxs.size else len(scores) - 1
    return float(scores[idx])


def threshold_cumulative(sensitivity_map: Dict[str, float], budget: float = 0.95) -> float:
    if not (0 < budget <= 1):
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    scores = np.sort(_valid_scores(sensitivity_map))[::-1]
    total = np.sum(scores)
    if not np.isfinite(total) or total <= 0:
        return float(scores[-1])
    cumsum = np.cumsum(scores) / total
    idx = int(np.argmax(cumsum >= budget))
    return float(scores[idx])


def get_param_size_mb(param) -> float:
    return param.numel() * param.element_size() / (1024 * 1024)


def get_module_cost_mb(module, precision: str) -> float:
    cost = 0.0
    if precision == "fp16":
        for param in module.parameters():
            cost += get_param_size_mb(param)
    elif precision == "int4":
        for param in module.parameters():
            cost += param.numel() / 2 / (1024 * 1024)
    else:
        raise ValueError(f"Unknown precision: {precision}")
    return float(cost)


def knapsack_keep_layers(model, sensitivity_map: Dict[str, float], budget_mb: float) -> List[str]:
    items = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in sensitivity_map:
            continue
        cost_keep = get_module_cost_mb(module, "fp16") - get_module_cost_mb(module, "int4")
        if not np.isfinite(cost_keep) or cost_keep <= 0:
            continue
        density = sensitivity_map[name] / cost_keep
        items.append((name, cost_keep, density))
    items.sort(key=lambda x: x[2], reverse=True)
    keep = []
    used = 0.0
    for name, cost_keep, _ in items:
        if used + cost_keep <= budget_mb:
            keep.append(name)
            used += cost_keep
    return keep


def select_layers(
    model,
    sensitivity_map: Dict[str, float],
    method: str = "pct",
    percentile: float = 0.2,
    sensitivity_ratio: float = 0.05,
    budget: float = 0.95,
    budget_mb: float = 4096,
    invert_selection: bool = False,
) -> List[str]:
    linear_layers = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    filtered = {k: v for k, v in sensitivity_map.items() if k in linear_layers}
    if not filtered:
        raise ValueError("No linear-layer sensitivity scores match model modules.")

    method = method.lower()
    if method == "knapsack":
        return knapsack_keep_layers(model, filtered, budget_mb)
    if method == "pct":
        t = threshold_pct(filtered, percentile)
    elif method == "otsu":
        t = threshold_otsu_method(filtered)
    elif method == "elb":
        t = threshold_elbow(filtered)
    elif method == "gradient":
        t = threshold_gradient(filtered, sensitivity_ratio)
    elif method == "cumulative":
        t = threshold_cumulative(filtered, budget)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    if invert_selection:
        return [name for name, score in filtered.items() if score <= t]
    return [name for name, score in filtered.items() if score >= t]
