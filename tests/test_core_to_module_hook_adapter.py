"""Tests for core HookPoint -> ModuleHookPoint adapter on HookedRootModule."""

import torch
import torch.nn as nn

from world_model_lens.core import hooks as core_hooks
from world_model_lens.core.hooked_root import HookedRootModule


class SimpleModel(HookedRootModule):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.lin(x)


def test_add_and_remove_core_hook_adapter():
    model = SimpleModel(4, 4)
    # must call setup_hooks to attach forward hooks to layers
    model.setup_hooks()

    calls: list[str] = []

    def core_hook_fn(tensor: torch.Tensor, ctx: core_hooks.HookContext) -> torch.Tensor:
        calls.append(f"core:{ctx.component}")
        return tensor

    # The linear layer registered under name 'lin' will have hook name 'lin.hook_linear'
    hook = core_hooks.HookPoint(name="lin.hook_linear", fn=core_hook_fn)

    model.add_core_hook(hook)

    x = torch.randn(1, 4)
    _ = model(x)
    assert calls == ["core:lin"]

    # remove and ensure it no longer fires
    calls.clear()
    model.remove_core_hook(hook)
    _ = model(x)
    assert calls == []
