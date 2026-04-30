from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import bitsandbytes as bnb
import torch
import torch.nn as nn
from bitsandbytes.nn import Params4bit

from .selection import select_layers
from .sensitivity import compute_fisher, compute_magnitude, compute_perturbation_sensitivity


class TaskAwareQuantizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sensitivity_map: Optional[Dict[str, float]] = None

    def compute_sensitivity(
        self,
        method: str,
        calibration_texts: Iterable[str],
        reduction: str = "mean",
        fisher_clip_percentile: float | None = 99.0,
        fisher_clip_samples: int = 32,
        max_length: int = 2048,
    ) -> Dict[str, float]:
        method = method.lower()
        if method == "fisher":
            self.sensitivity_map = compute_fisher(
                self.model,
                self.tokenizer,
                calibration_texts,
                reduction=reduction,
                clip_percentile=fisher_clip_percentile,
                clip_samples=fisher_clip_samples,
                max_length=max_length,
            )
        elif method == "magnitude":
            self.sensitivity_map = compute_magnitude(self.model)
        elif method == "perturbation":
            self.sensitivity_map = compute_perturbation_sensitivity(
                self.model,
                self.tokenizer,
                calibration_texts,
                max_length=max_length,
            )
        else:
            raise ValueError(f"Unknown sensitivity method: {method}")
        return self.sensitivity_map

    def _replace_linear_with_bnb(self, full_name: str, layer: nn.Linear, target_device: torch.device):
        parent = self.model
        child_name = full_name
        if "." in full_name:
            parent_name, child_name = full_name.rsplit(".", 1)
            parent = self.model.get_submodule(parent_name)

        new_layer = bnb.nn.Linear4bit(
            input_features=layer.in_features,
            output_features=layer.out_features,
            bias=layer.bias is not None,
            compute_dtype=layer.weight.dtype,
            quant_type="nf4",
        )

        with torch.no_grad():
            weight_data = layer.weight.data.to("cpu", copy=True)
            quantized_weight = Params4bit(weight_data, requires_grad=False, quant_type="nf4")
            new_layer.weight = quantized_weight
            if layer.bias is not None:
                bias_data = layer.bias.data.to(dtype=layer.weight.dtype, device="cpu", copy=True)
                new_layer.bias = nn.Parameter(bias_data, requires_grad=False)

        setattr(parent, child_name, new_layer.to(target_device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def quantize(
        self,
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
    ):
        if self.sensitivity_map is None:
            self.compute_sensitivity(
                method=sensitivity_method,
                calibration_texts=calibration_texts,
                reduction=reduction,
                fisher_clip_percentile=fisher_clip_percentile,
                fisher_clip_samples=fisher_clip_samples,
                max_length=max_length,
            )

        layers_to_keep = set(
            select_layers(
                model=self.model,
                sensitivity_map=self.sensitivity_map,
                method=selection_method,
                percentile=percentile,
                sensitivity_ratio=sensitivity_ratio,
                budget=budget,
                budget_mb=budget_mb,
                invert_selection=invert_selection,
            )
        )

        target_device = next(self.model.parameters()).device
        layers_to_quantize: List[str] = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name not in layers_to_keep:
                layers_to_quantize.append(name)

        for layer_name in layers_to_quantize:
            module = dict(self.model.named_modules())[layer_name]
            self._replace_linear_with_bnb(layer_name, module, target_device)

        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.model
