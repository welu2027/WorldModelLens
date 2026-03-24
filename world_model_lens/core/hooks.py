"""Hook system for intercepting and modifying computations."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple
import torch


@dataclass
class HookPoint:
    """A named hook function registered at a specific component.

    Attributes:
        name: Component name (e.g., 'z_posterior', 'h', 'reward_pred').
        fn: The hook function, takes (tensor, context) and returns modified tensor.
        stage: When to apply the hook ('pre', 'post'). 'post' fires after
               computation, before downstream use.
        timestep: Optional specific timestep to hook. None means all timesteps.
    """

    name: str
    fn: Callable[[torch.Tensor, "HookContext"], torch.Tensor]
    stage: str = "post"
    timestep: Optional[int] = None

    def __post_init__(self):
        if self.stage not in ("pre", "post"):
            raise ValueError(f"stage must be 'pre' or 'post', got {self.stage}")


@dataclass
class HookContext:
    """Runtime metadata passed to hook functions.

    Attributes:
        timestep: Current timestep in the sequence.
        component: Component name that was just computed.
        trajectory_so_far: List of LatentState objects computed up to this point.
        metadata: Additional context dictionary.
    """

    timestep: int
    component: str
    trajectory_so_far: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HookRegistry:
    """Registry for managing hook functions on world model components.

    Hooks are stored as a list per (component, timestep) combination and
    applied in registration order.
    """

    def __init__(self):
        self._hooks: Dict[Tuple[str, Optional[int]], List[HookPoint]] = {}
        self._global_hooks: Dict[str, List[HookPoint]] = {}

    def register(self, hook: HookPoint) -> None:
        """Register a hook.

        Args:
            hook: The HookPoint to register.
        """
        if hook.timestep is not None:
            key = (hook.name, hook.timestep)
        else:
            key = (hook.name, None)
            self._global_hooks.setdefault(hook.name, []).append(hook)
            return

        self._hooks.setdefault(key, []).append(hook)

    def clear(self) -> None:
        """Remove all registered hooks."""
        self._hooks.clear()
        self._global_hooks.clear()

    def get_hooks_for(self, component: str, timestep: Optional[int] = None) -> List[HookPoint]:
        """Get all hooks matching a component and timestep.

        Args:
            component: Component name.
            timestep: Optional specific timestep.

        Returns:
            List of matching HookPoints.
        """
        hooks = list(self._global_hooks.get(component, []))

        if timestep is not None:
            specific = self._hooks.get((component, timestep), [])
            hooks.extend(specific)

        return hooks

    def apply(
        self,
        component: str,
        timestep: int,
        tensor: torch.Tensor,
        context: HookContext,
    ) -> torch.Tensor:
        """Apply all matching hooks to a tensor.

        Hooks fire after computation but before downstream use.

        Args:
            component: Component name.
            timestep: Current timestep.
            tensor: The activation tensor.
            context: Hook context with runtime metadata.

        Returns:
            The tensor after all hooks have been applied.
        """
        hooks = self.get_hooks_for(component, timestep)
        for hook in hooks:
            tensor = hook.fn(tensor, context)
        return tensor

    def __len__(self) -> int:
        """Total number of registered hooks."""
        return sum(len(h) for h in self._hooks.values()) + sum(
            len(h) for h in self._global_hooks.values()
        )
