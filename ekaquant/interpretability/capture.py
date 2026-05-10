from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass(frozen=True)
class ModuleCaptureSpec:
    module_name: str
    capture_input: bool = False
    capture_output: bool = True


class ActivationCaptureSession:
    def __init__(
        self,
        model: torch.nn.Module,
        specs: List[ModuleCaptureSpec],
        move_to_cpu: bool = True,
    ):
        self.model = model
        self.specs = specs
        self.move_to_cpu = move_to_cpu
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.captured: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    def __enter__(self) -> "ActivationCaptureSession":
        spec_map = {spec.module_name: spec for spec in self.specs}
        for module_name, module in self.model.named_modules():
            spec = spec_map.get(module_name)
            if spec is None:
                continue
            self.captured[module_name] = {"inputs": [], "outputs": []}
            handle = module.register_forward_hook(self._make_hook(module_name, spec))
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        for module_name in list(self.captured.keys()):
            self.captured[module_name]["inputs"].clear()
            self.captured[module_name]["outputs"].clear()

    def _make_hook(self, module_name: str, spec: ModuleCaptureSpec):
        def hook(_module, inputs, output):
            if spec.capture_input:
                self.captured[module_name]["inputs"].append(
                    self._normalize_tensor(inputs)
                )
            if spec.capture_output:
                self.captured[module_name]["outputs"].append(
                    self._normalize_tensor(output)
                )

        return hook

    def _normalize_tensor(self, value):
        if isinstance(value, tuple):
            if not value:
                return torch.empty(0)
            value = value[0]
        if isinstance(value, list):
            if not value:
                return torch.empty(0)
            value = value[0]
        if not torch.is_tensor(value):
            return torch.empty(0)
        tensor = value.detach()
        if self.move_to_cpu:
            tensor = tensor.cpu()
        return tensor
