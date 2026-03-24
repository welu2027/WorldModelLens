"""Hook system for intercepting and modifying world-model computation.

Three objects form the hook system:

* :class:`HookPoint` — a *registered* hook: a named callable attached to a
  specific ``(component, stage)`` pair, optionally scoped to a single
  timestep.
* :class:`HookContext` — *runtime metadata* passed to every hook function,
  giving hooks access to the current timestep, component name, and the
  trajectory built so far.
* :class:`HookRegistry` — a mutable collection of :class:`HookPoint` objects
  with methods to register, clear, query, and *apply* hooks.

Hook function signature
-----------------------
Every :attr:`HookPoint.fn` must conform to::

    def my_hook(tensor: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        ...

Hooks may modify the tensor (patching), inspect it (probing), or both.
Returning the tensor unchanged is always valid.

Example
-------
>>> registry = HookRegistry()
>>> def log_hook(t, ctx):
...     print(f"[t={ctx.timestep}] {ctx.component}: {t.shape}")
...     return t
>>> registry.register(HookPoint(name="rnn.h", stage="post", fn=log_hook))
>>> # later, inside the forward pass:
>>> ctx = HookContext(timestep=0, component="rnn.h")
>>> h = registry.apply("rnn.h", 0, h_tensor, ctx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

# Signature of a hook function.
HookFn = Callable[["torch.Tensor", "HookContext"], "torch.Tensor"]


# ---------------------------------------------------------------------------
# HookContext
# ---------------------------------------------------------------------------

@dataclass
class HookContext:
    """Runtime metadata supplied to every hook function.

    Parameters
    ----------
    timestep:
        Current step index (0-indexed).
    component:
        Name of the component being processed (e.g. ``"rnn.h"``).
    trajectory_so_far:
        The partial :class:`~world_model_lens.core.LatentTrajectory`
        accumulated *before* this step.  ``None`` at the very first step
        or when running outside of a trajectory loop.
    metadata:
        Arbitrary extra information supplied by the caller.

    Examples
    --------
    >>> ctx = HookContext(timestep=3, component="z_posterior")
    >>> ctx.timestep
    3
    """

    timestep: int
    """Current step index."""

    component: str
    """Name of the component whose activation is being intercepted."""

    trajectory_so_far: Optional[Any] = None
    """Partial LatentTrajectory up to (but not including) this timestep."""

    metadata: Optional[Dict[str, Any]] = field(default=None)
    """Arbitrary extra context supplied by the calling code."""


# ---------------------------------------------------------------------------
# HookPoint
# ---------------------------------------------------------------------------

@dataclass
class HookPoint:
    """A single registered hook: component + stage + callable.

    Parameters
    ----------
    name:
        The component name this hook is attached to (e.g. ``"encoder.out"``,
        ``"rnn.h"``, ``"z_posterior"``).  Must match the ``component``
        argument passed to :meth:`HookRegistry.apply`.
    stage:
        Lifecycle stage at which the hook fires.  Common values:
        ``"pre"`` (before the component runs), ``"post"`` (after),
        ``"encoder"``, ``"dynamics"``, ``"reward"``, ``"value"``.
        The stage is stored for documentation and filtering purposes;
        :class:`HookRegistry` does *not* filter on stage by default — the
        caller is responsible for calling :meth:`HookRegistry.apply` at the
        right moment.
    fn:
        Hook function with signature
        ``(tensor: Tensor, ctx: HookContext) -> Tensor``.
        Must return a tensor of the *same shape and dtype* as the input
        (patching hooks may alter *values* but should not change structure
        without careful coordination with downstream code).
    timestep:
        If ``None`` (default), the hook fires at every timestep.
        If an integer, the hook only fires when the current timestep
        equals this value.

    Examples
    --------
    >>> def zero_out(t, ctx):
    ...     return torch.zeros_like(t)
    >>> hp = HookPoint(name="z_posterior", stage="post", fn=zero_out, timestep=5)
    >>> hp.timestep
    5
    """

    name: str
    """Component name this hook is attached to."""

    stage: str
    """Lifecycle stage label."""

    fn: HookFn
    """Hook function: ``(tensor, context) -> tensor``."""

    timestep: Optional[int] = None
    """Specific timestep to fire on, or ``None`` for all timesteps."""

    def matches(self, component: str, timestep: int) -> bool:
        """Return ``True`` if this hook should fire for *component* at *timestep*.

        Parameters
        ----------
        component:
            The component name from the caller.
        timestep:
            The current step index.

        Returns
        -------
        bool
        """
        if self.name != component:
            return False
        if self.timestep is not None and self.timestep != timestep:
            return False
        return True


# ---------------------------------------------------------------------------
# HookRegistry
# ---------------------------------------------------------------------------

class HookRegistry:
    """Mutable registry of :class:`HookPoint` objects.

    Hooks are stored in registration order and applied in that order by
    :meth:`apply`.  Hooks registered later run after earlier ones.

    Examples
    --------
    >>> registry = HookRegistry()
    >>> registry.register(HookPoint("rnn.h", "post", my_fn))
    >>> len(registry)
    1
    >>> registry.get_hooks_for("rnn.h", 0)
    [HookPoint(name='rnn.h', ...)]
    >>> patched = registry.apply("rnn.h", 0, tensor, ctx)
    """

    def __init__(self) -> None:
        self._hooks: List[HookPoint] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, hook: HookPoint) -> None:
        """Add *hook* to the registry.

        Parameters
        ----------
        hook:
            :class:`HookPoint` to register.  Duplicates are allowed; if the
            same callable needs to be removed later use :meth:`clear` with the
            component name.

        Examples
        --------
        >>> registry.register(HookPoint("encoder.out", "post", my_fn))
        """
        if not isinstance(hook, HookPoint):
            raise TypeError(
                f"register() expects a HookPoint, got {type(hook).__name__}."
            )
        self._hooks.append(hook)

    def clear(self, name: Optional[str] = None) -> None:
        """Remove hooks from the registry.

        Parameters
        ----------
        name:
            If provided, remove only hooks whose :attr:`HookPoint.name`
            equals *name*.  If ``None``, remove *all* hooks.

        Examples
        --------
        >>> registry.clear("rnn.h")   # remove hooks on rnn.h only
        >>> registry.clear()          # remove all hooks
        """
        if name is None:
            self._hooks.clear()
        else:
            self._hooks = [h for h in self._hooks if h.name != name]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_hooks_for(self, component: str, timestep: int) -> List[HookPoint]:
        """Return all hooks that should fire for *component* at *timestep*.

        Hooks are returned in registration order.

        Parameters
        ----------
        component:
            Component name (must match :attr:`HookPoint.name`).
        timestep:
            Current step index.

        Returns
        -------
        list of HookPoint
            May be empty if no hooks are registered for this combination.

        Examples
        --------
        >>> hooks = registry.get_hooks_for("rnn.h", 0)
        >>> len(hooks)
        2
        """
        return [h for h in self._hooks if h.matches(component, timestep)]

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(
        self,
        component: str,
        timestep: int,
        tensor: torch.Tensor,
        context: HookContext,
    ) -> torch.Tensor:
        """Apply all matching hooks to *tensor* in registration order.

        Each hook receives the *output* of the previous hook, enabling
        hook chaining (e.g. first probe, then patch).

        Parameters
        ----------
        component:
            Component name — used to find matching :class:`HookPoint` objects.
        timestep:
            Current step index.
        tensor:
            The activation tensor to pass through the hook chain.
        context:
            :class:`HookContext` with runtime metadata for the hooks.

        Returns
        -------
        torch.Tensor
            Tensor after all matching hooks have been applied.  Equals the
            input tensor if no hooks match (i.e. this is a no-op when the
            registry is empty or has no hooks for this component/timestep).

        Raises
        ------
        RuntimeError
            If a hook function raises, the exception is re-raised with
            additional context (hook name, component, timestep).

        Examples
        --------
        >>> out = registry.apply("z_posterior", 3, z_tensor, ctx)
        """
        hooks = self.get_hooks_for(component, timestep)
        for hook in hooks:
            try:
                result = hook.fn(tensor, context)
            except Exception as exc:
                raise RuntimeError(
                    f"Hook {hook.name!r} (stage={hook.stage!r}, "
                    f"component={component!r}, t={timestep}) raised an error: {exc}"
                ) from exc
            if not isinstance(result, torch.Tensor):
                raise TypeError(
                    f"Hook {hook.name!r} returned {type(result).__name__!r} "
                    f"instead of torch.Tensor."
                )
            tensor = result
        return tensor

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._hooks)

    def __iter__(self):
        return iter(self._hooks)

    def __repr__(self) -> str:
        names = [h.name for h in self._hooks]
        return f"HookRegistry({len(self._hooks)} hooks: {names})"
