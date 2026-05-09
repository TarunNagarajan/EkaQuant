from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable

import torch


def _apply_to_tensor_tree(value, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_apply_to_tensor_tree(item, fn) for item in value)
    if isinstance(value, list):
        return [_apply_to_tensor_tree(item, fn) for item in value]
    return value


@contextmanager
def ablate_modules(model: torch.nn.Module, module_names: Iterable[str], scale: float = 0.0):
    names = set(module_names)
    handles = []

    def make_hook():
        def hook(_module, _inputs, output):
            return _apply_to_tensor_tree(output, lambda tensor: tensor * scale)

        return hook

    for module_name, module in model.named_modules():
        if module_name in names:
            handles.append(module.register_forward_hook(make_hook()))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_modules(model: torch.nn.Module, patch_tensors: Dict[str, torch.Tensor]):
    handles = []

    def make_hook(module_name: str):
        def hook(_module, _inputs, output):
            patch = patch_tensors[module_name]
            return _apply_to_tensor_tree(output, lambda tensor: patch.to(tensor.device, dtype=tensor.dtype))

        return hook

    for module_name, module in model.named_modules():
        if module_name in patch_tensors:
            handles.append(module.register_forward_hook(make_hook(module_name)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
