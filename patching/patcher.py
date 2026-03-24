"""Temporal activation patching for world models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
from tqdm import tqdm

from world_model_lens.core.hooks import HookContext, HookPoint

if TYPE_CHECKING:
    from world_model_lens.core.activation_cache import ActivationCache
    from world_model_lens.hooked_world_model import HookedWorldModel
    from torch import Tensor


@dataclass
class PatchResult:
    """Result of a single activation patch experiment.
    
    Attributes
    ----------
    metric_name:
        Name of the metric computed.
    metric_clean:
        Metric value on the clean trajectory.
    metric_corrupted:
        Metric value on the corrupted trajectory.
    metric_patched:
        Metric value after patching the activation.
    recovery_rate:
        Recovery rate: (patched - corrupted) / (clean - corrupted).
        Clipped to [-1, 2] to handle edge cases.
    patch_timestep:
        Timestep at which the patch was applied.
    patch_activation:
        Name of the activation that was patched (e.g. "rnn.h").
    """

    metric_name: str
    metric_clean: float
    metric_corrupted: float
    metric_patched: float
    recovery_rate: float
    patch_timestep: int
    patch_activation: str

    def summary(self) -> str:
        """Return a one-line summary string."""
        return (
            f"[t={self.patch_timestep}, {self.patch_activation}] "
            f"clean={self.metric_clean:.4f}, corrupted={self.metric_corrupted:.4f}, "
            f"patched={self.metric_patched:.4f}, recovery={self.recovery_rate:.4f}"
        )


class TemporalPatcher:
    """Run activation patching experiments on world model trajectories.

    Activation patching: replace an activation at timestep t in the "corrupted"
    run with the corresponding activation from the "clean" run, then measure how
    much of the clean behavior is recovered.

    KEY PROPERTY: Because world models are recurrent, patching h_t at timestep t
    causes the change to propagate through all subsequent GRU steps automatically.
    The patch is applied via a hook that fires once at step t, after which the
    model's own dynamics carry the effect forward.

    Parameters
    ----------
    wm:
        The :class:`HookedWorldModel` to patch on.

    Examples
    --------
    >>> patcher = TemporalPatcher(wm)
    >>> result = patcher.patch_state(
    ...     clean_cache, corrupted_obs_seq, corrupted_action_seq,
    ...     patch_timestep=5, patch_activation="rnn.h",
    ...     metric_fn=my_metric_fn, metric_name="reward_prediction"
    ... )
    """

    def __init__(self, wm: HookedWorldModel) -> None:
        self.wm = wm

    def patch_state(
        self,
        clean_cache: ActivationCache,
        corrupted_obs_seq: Tensor,
        corrupted_action_seq: Tensor,
        patch_timestep: int,
        patch_activation: str,
        metric_fn: Callable[[ActivationCache], float],
        metric_name: str = "metric",
    ) -> PatchResult:
        """Patch a single activation at a single timestep.

        Implementation strategy:
        1. Run corrupted sequence to get corrupted_cache and compute metric_corrupted
        2. Compute metric_clean from the provided clean_cache
        3. Re-run corrupted sequence with a hook that replaces patch_activation at
           patch_timestep with the value from clean_cache
        4. Compute metric_patched from the patched cache
        5. Compute recovery_rate = (metric_patched - metric_corrupted) / (metric_clean - metric_corrupted)

        The patch is injected via a single-timestep hook that fires exactly once
        at the correct timestep. This allows downstream dynamics to propagate the
        change naturally through the recurrent model.

        Parameters
        ----------
        clean_cache:
            Activation cache from a clean run.
        corrupted_obs_seq:
            Observation sequence for the corrupted run.
        corrupted_action_seq:
            Action sequence for the corrupted run.
        patch_timestep:
            Timestep at which to apply the patch (0-indexed).
        patch_activation:
            Name of the activation to patch (e.g. "rnn.h", "z_posterior").
        metric_fn:
            Function that takes an ActivationCache and returns a scalar metric.
        metric_name:
            Name of the metric for logging/reporting.

        Returns
        -------
        PatchResult
            Contains metrics and recovery_rate.

        Raises
        ------
        KeyError
            If the patch_activation is not found in clean_cache at patch_timestep.
        """
        # Step 1: Compute clean metric
        metric_clean = metric_fn(clean_cache)

        # Step 2: Run corrupted sequence to get baseline
        _, corrupted_cache = self.wm.run_with_cache(
            corrupted_obs_seq, corrupted_action_seq
        )
        metric_corrupted = metric_fn(corrupted_cache)

        # Step 3: Create a patch hook that replaces the activation at patch_timestep
        clean_activation = clean_cache[patch_activation, patch_timestep]

        def patch_hook(tensor: Tensor, ctx: HookContext) -> Tensor:
            """Replace tensor with clean activation if at the right timestep."""
            if ctx.timestep == patch_timestep:
                return clean_activation
            return tensor

        patch_hp = HookPoint(
            name=patch_activation,
            stage="post",
            fn=patch_hook,
            timestep=None,  # Will check timestep in the hook function
        )

        # Step 4: Run corrupted sequence with the patch hook
        _, patched_cache = self.wm.run_with_hooks(
            corrupted_obs_seq, corrupted_action_seq,
            fwd_hooks=[patch_hp],
            return_cache=True,
        )
        metric_patched = metric_fn(patched_cache)

        # Step 5: Compute recovery rate
        denom = metric_clean - metric_corrupted
        if abs(denom) < 1e-8:
            # If clean and corrupted are nearly equal, recovery is undefined
            recovery_rate = 0.0
        else:
            recovery_rate = (metric_patched - metric_corrupted) / denom
            # Clip to reasonable range: full corrupted (-1), full clean (1), or overcorrection (up to 2)
            recovery_rate = max(-1.0, min(2.0, recovery_rate))

        return PatchResult(
            metric_name=metric_name,
            metric_clean=metric_clean,
            metric_corrupted=metric_corrupted,
            metric_patched=metric_patched,
            recovery_rate=recovery_rate,
            patch_timestep=patch_timestep,
            patch_activation=patch_activation,
        )

    def full_sweep(
        self,
        clean_cache: ActivationCache,
        corrupted_obs_seq: Tensor,
        corrupted_action_seq: Tensor,
        metric_fn: Callable[[ActivationCache], float],
        activation_names: Optional[List[str]] = None,
        timesteps: Optional[List[int]] = None,
        metric_name: str = "metric",
        show_progress: bool = True,
    ) -> PatchingSweepResult:
        """Run patch_state for all (timestep, activation) pairs.

        This runs a full grid of patching experiments, one for each combination
        of timestep and activation name, and returns a 2D array of recovery rates.

        Parameters
        ----------
        clean_cache:
            Activation cache from a clean run.
        corrupted_obs_seq:
            Observation sequence for corrupted runs.
        corrupted_action_seq:
            Action sequence for corrupted runs.
        metric_fn:
            Function that takes a cache and returns a scalar metric.
        activation_names:
            List of activation names to patch. If None, uses all components
            from clean_cache.
        timesteps:
            List of timesteps to patch. If None, uses all timesteps from clean_cache.
        metric_name:
            Name of the metric.
        show_progress:
            If True, show a tqdm progress bar.

        Returns
        -------
        PatchingSweepResult
            Contains a recovery_matrix of shape (n_timesteps, n_activations).
        """
        import numpy as np

        if activation_names is None:
            activation_names = clean_cache.component_names
        if timesteps is None:
            timesteps = clean_cache.timesteps

        n_timesteps = len(timesteps)
        n_activations = len(activation_names)

        # Initialize recovery matrix
        recovery_matrix = np.zeros((n_timesteps, n_activations), dtype=np.float32)

        # Iterate over all (timestep, activation) pairs
        total = n_timesteps * n_activations
        iterator = tqdm(
            total=total,
            desc="Patching sweep",
            disable=not show_progress,
        )

        for i, t in enumerate(timesteps):
            for j, act_name in enumerate(activation_names):
                try:
                    result = self.patch_state(
                        clean_cache=clean_cache,
                        corrupted_obs_seq=corrupted_obs_seq,
                        corrupted_action_seq=corrupted_action_seq,
                        patch_timestep=t,
                        patch_activation=act_name,
                        metric_fn=metric_fn,
                        metric_name=metric_name,
                    )
                    recovery_matrix[i, j] = result.recovery_rate
                except KeyError:
                    # Activation not available at this timestep
                    recovery_matrix[i, j] = np.nan
                iterator.update(1)

        iterator.close()

        # Import here to avoid circular imports
        from world_model_lens.patching.sweep_result import PatchingSweepResult

        return PatchingSweepResult(
            metric_name=metric_name,
            recovery_matrix=recovery_matrix,
            activation_names=activation_names,
            timesteps=timesteps,
        )


# Import at end to avoid circular dependency
from world_model_lens.patching.sweep_result import PatchingSweepResult  # noqa: E402
