from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _to_list(texts: Iterable[str]) -> List[str]:
    values = list(texts)
    if not values:
        raise ValueError("calibration_texts must be non-empty")
    return values


def _tokenize(tokenizer, text: str, max_length: int):
    return tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def compute_magnitude(model, normalize_by_size: bool = True) -> Dict[str, float]:
    sensitivity_map: Dict[str, float] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad or "weight" not in name:
                continue
            score = param.pow(2).sum().item()
            if normalize_by_size:
                score /= max(param.numel(), 1)
            sensitivity_map[name.replace(".weight", "")] = float(score)
    if not sensitivity_map:
        raise ValueError("No trainable weight parameters found for magnitude sensitivity.")
    return sensitivity_map


def compute_fisher(
    model,
    tokenizer,
    calibration_texts: Iterable[str],
    reduction: str = "mean",
    clip_percentile: float | None = 99.0,
    clip_samples: int = 32,
    max_length: int = 2048,
) -> Dict[str, float]:
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
    texts = _to_list(calibration_texts)
    model.eval()
    model_device = next(model.parameters()).device
    sensitivity_map: Dict[str, float] = {}
    layer_counts: Dict[str, int] = {}
    max_grad_norm = None

    if clip_percentile is not None:
        grad_norms: List[float] = []
        probe_texts = texts[: min(clip_samples, len(texts))]
        for text in tqdm(probe_texts, desc="Fisher clip probe", leave=False):
            try:
                inputs = _tokenize(tokenizer, text, max_length)
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                model.zero_grad(set_to_none=True)
                loss = model(**inputs, labels=inputs["input_ids"]).loss
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if "weight" in name and param.grad is not None:
                            value = param.grad.norm().item()
                            if np.isfinite(value) and value > 0:
                                grad_norms.append(value)
            except Exception:
                continue
            finally:
                model.zero_grad(set_to_none=True)
        if grad_norms:
            max_grad_norm = float(np.percentile(np.array(grad_norms), clip_percentile))

    processed = 0
    for text in tqdm(texts, desc="Fisher", leave=False):
        try:
            inputs = _tokenize(tokenizer, text, max_length)
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            model.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=inputs["input_ids"]).loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "weight" not in name or param.grad is None:
                        continue
                    grad = param.grad.detach()
                    if not torch.isfinite(grad).all():
                        continue
                    grad_norm = grad.norm().item()
                    if max_grad_norm is not None and grad_norm > max_grad_norm:
                        grad = grad * (max_grad_norm / (grad_norm + 1e-8))
                    grad_sq = grad.float().square().sum().item()
                    module_name = name.replace(".weight", "")
                    fisher_density = grad_sq / max(param.numel(), 1)
                    sensitivity_map[module_name] = sensitivity_map.get(module_name, 0.0) + fisher_density
                    layer_counts[module_name] = layer_counts.get(module_name, 0) + 1
            processed += 1
        except Exception:
            continue
        finally:
            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if processed == 0 or not sensitivity_map:
        raise RuntimeError("No samples were successfully processed for Fisher sensitivity.")

    if reduction == "mean":
        for name in list(sensitivity_map.keys()):
            count = layer_counts.get(name, 0)
            if count > 0:
                sensitivity_map[name] /= count

    return sensitivity_map


def fake_quantize_int4(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.abs().amax(dim=1, keepdim=True) / 7.0
    scale = scale.clamp(min=1e-5)
    w_int4 = (weight / scale).round().clamp(-8, 7)
    return (w_int4 * scale).detach()


def compute_perturbation_sensitivity(
    model,
    tokenizer,
    calibration_texts: Iterable[str],
    max_length: int = 2048,
) -> Dict[str, float]:
    texts = _to_list(calibration_texts)
    model.eval()
    model_device = next(model.parameters()).device
    sensitivity_map: Dict[str, float] = {}
    layer_counts: Dict[str, int] = {}
    layer_inputs: Dict[str, torch.Tensor] = {}
    hooks = []
    target_modules = []
    module_map = dict(model.named_modules())

    def make_hook(layer_name: str):
        def hook(_module, inputs, _output):
            if isinstance(inputs, tuple) and len(inputs) > 0:
                layer_inputs[layer_name] = inputs[0].detach()
            elif isinstance(inputs, torch.Tensor):
                layer_inputs[layer_name] = inputs.detach()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            target_modules.append(name)
            hooks.append(module.register_forward_hook(make_hook(name)))

    processed = 0
    try:
        for text in tqdm(texts, desc="Perturbation", leave=False):
            sample_scored = False
            try:
                inputs = _tokenize(tokenizer, text, max_length).to(model_device)
                with torch.no_grad():
                    model(**inputs)
                for name in target_modules:
                    if name not in layer_inputs:
                        continue
                    module = module_map[name]
                    inp = layer_inputs[name].to(module.weight.device)
                    with torch.no_grad():
                        out_gt = module(inp)
                        out_q = F.linear(inp, fake_quantize_int4(module.weight.data), module.bias)
                        mse = (out_gt - out_q).pow(2).mean().item()
                        norm = out_gt.pow(2).mean().item() + 1e-6
                        score = mse / norm
                    sensitivity_map[name] = sensitivity_map.get(name, 0.0) + score
                    layer_counts[name] = layer_counts.get(name, 0) + 1
                    sample_scored = True
            except Exception:
                pass
            finally:
                layer_inputs.clear()
                if sample_scored:
                    processed += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        for hook in hooks:
            hook.remove()

    if processed == 0 or not sensitivity_map:
        raise RuntimeError("No samples were successfully processed for perturbation sensitivity.")

    for name in list(sensitivity_map.keys()):
        count = layer_counts.get(name, 0)
        if count > 0:
            sensitivity_map[name] /= count

    return sensitivity_map
