from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Union

import bitsandbytes as bnb
import torch
import torch.nn as nn
from bitsandbytes.nn import Params4bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .selection import expand_sensitivity_map, select_layers
from .sensitivity import (
    compute_fisher,
    compute_magnitude,
    compute_perturbation_sensitivity,
)


class TaskAwareQuantizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sensitivity_map: Optional[Dict[str, float]] = None

    def compute_sensitivity(
        self,
        method: Union[str, Callable],
        calibration_texts: Iterable[str],
        reduction: str = "mean",
        fisher_clip_percentile: float | None = 99.0,
        fisher_clip_samples: int = 32,
        max_length: int = 2048,
        **kwargs,
    ) -> Dict[str, float]:
        if callable(method):
            self.sensitivity_map = method(
                self.model, self.tokenizer, calibration_texts, **kwargs
            )
            return self.sensitivity_map

        method_str = method.lower()
        if method_str == "fisher":
            self.sensitivity_map = compute_fisher(
                self.model,
                self.tokenizer,
                calibration_texts,
                reduction=reduction,
                clip_percentile=fisher_clip_percentile,
                clip_samples=fisher_clip_samples,
                max_length=max_length,
            )
        elif method_str == "magnitude":
            self.sensitivity_map = compute_magnitude(self.model)
        elif method_str == "perturbation":
            self.sensitivity_map = compute_perturbation_sensitivity(
                self.model,
                self.tokenizer,
                calibration_texts,
                max_length=max_length,
            )
        else:
            raise ValueError(f"Unknown sensitivity method: {method}")
        return self.sensitivity_map

    def _replace_linear_with_bnb(
        self, full_name: str, layer: nn.Linear, target_device: torch.device
    ):
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
            quantized_weight = Params4bit(
                weight_data, requires_grad=False, quant_type="nf4"
            )
            new_layer.weight = quantized_weight
            if layer.bias is not None:
                bias_data = layer.bias.data.to(
                    dtype=layer.weight.dtype, device="cpu", copy=True
                )
                new_layer.bias = nn.Parameter(bias_data, requires_grad=False)

        # Free the original fp16 weights
        layer.weight = None
        layer.bias = None

        setattr(parent, child_name, new_layer.to(target_device))
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def quantize(
        self,
        calibration_texts: Iterable[str],
        sensitivity_method: Union[str, Callable] = "fisher",
        selection_method: Union[str, Callable] = "pct",
        mode: str = "mixed_precision",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        percentile: float = 0.2,
        sensitivity_ratio: float = 0.05,
        budget: float = 0.95,
        budget_mb: float = 4096,
        invert_selection: bool = False,
        reduction: str = "mean",
        fisher_clip_percentile: float | None = 99.0,
        fisher_clip_samples: int = 32,
        max_length: int = 2048,
        **kwargs,
    ):
        if self.sensitivity_map is None:
            self.compute_sensitivity(
                method=sensitivity_method,
                calibration_texts=calibration_texts,
                reduction=reduction,
                fisher_clip_percentile=fisher_clip_percentile,
                fisher_clip_samples=fisher_clip_samples,
                max_length=max_length,
                **kwargs,
            )

        # Automatically broadcast block-level sensitivities to linear modules
        self.sensitivity_map = expand_sensitivity_map(self.model, self.sensitivity_map)

        selected_layers = set(
            select_layers(
                model=self.model,
                sensitivity_map=self.sensitivity_map,
                method=selection_method,
                percentile=percentile,
                sensitivity_ratio=sensitivity_ratio,
                budget=budget,
                budget_mb=budget_mb,
                invert_selection=invert_selection,
                **kwargs,
            )
        )

        mode_str = mode.lower()
        if mode_str == "mixed_precision":
            print(
                f"EkaQuant [Mixed Precision]: Keeping {len(selected_layers)} layers in high precision."
            )
            layers_to_quantize: List[tuple[str, torch.device]] = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and name not in selected_layers:
                    layers_to_quantize.append((name, module.weight.device))

            for layer_name, target_device in layers_to_quantize:
                module = dict(self.model.named_modules())[layer_name]
                self._replace_linear_with_bnb(layer_name, module, target_device)

            self.model.eval()

        elif mode_str == "sr_lora":
            print(
                f"EkaQuant [SR-LoRA]: Quantizing all layers. Injecting LoRA adapters into {len(selected_layers)} critical layers."
            )
            layers_to_quantize: List[tuple[str, torch.device]] = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    layers_to_quantize.append((name, module.weight.device))

            for layer_name, target_device in layers_to_quantize:
                module = dict(self.model.named_modules())[layer_name]
                self._replace_linear_with_bnb(layer_name, module, target_device)

            if not selected_layers:
                print(
                    "Warning: No layers selected for SR-LoRA injection based on budget/threshold."
                )
            else:
                self.model = prepare_model_for_kbit_training(self.model)
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=list(selected_layers),
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(self.model, lora_config)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.model
