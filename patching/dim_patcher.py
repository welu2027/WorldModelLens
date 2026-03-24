"""Dimension-level activation patching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from world_model_lens.core.hooks import HookContext, HookPoint

if TYPE_CHECKING:
    from world_model_lens.core.activation_cache import ActivationCache
    from world_model_lens.patching.patcher import TemporalPatcher
    from torch import Tensor


@dataclass
class DimPatchResult:
    """Result of dimension-level patching.

    Attributes
    ----------
    activation_name:
        Name of the activation that was patched.
    timestep:
        Timestep at which patching occurred.
    dim_recovery_rates:
        1D array of shape (d,) containing recovery rate per dimension.
    concept_alignment:
        Optional Spearman correlation between dim_recovery_rates and
        external probe_weights (e.g., features learned by a classifier).
        None if probe_weights was not provided.
    """

    activation_name: str
    timestep: int
    dim_recovery_rates: np.ndarray  # (d,)
    concept_alignment: Optional[float] = None

    def plot(self, ax=None, top_k: int = 20):
        """Plot the top_k dimensions by |recovery_rate| as a bar chart.

        Parameters
        ----------
        ax:
            Matplotlib axes. If None, creates a new figure.
        top_k:
            Number of top dimensions to show.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            ) from e

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Get top_k dimensions by absolute recovery rate
        abs_rates = np.abs(self.dim_recovery_rates)
        top_indices = np.argsort(-abs_rates)[:top_k]
        top_rates = self.dim_recovery_rates[top_indices]

        # Create bar chart
        x = np.arange(len(top_indices))
        colors = ["green" if r > 0 else "red" for r in top_rates]
        ax.bar(x, top_rates, color=colors, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"d{i}" for i in top_indices], rotation=45)
        ax.set_ylabel("Recovery Rate", fontsize=11)
        ax.set_xlabel("Dimension", fontsize=11)
        ax.set_title(
            f"Top {top_k} Dimensions ({self.activation_name}, t={self.timestep})",
            fontsize=12,
            fontweight="bold",
        )
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        return ax

    def summary(self) -> str:
        """Return a text summary."""
        abs_rates = np.abs(self.dim_recovery_rates)
        top_indices = np.argsort(-abs_rates)[:5]

        lines = [
            f"DimPatchResult ({self.activation_name}, t={self.timestep})",
            f"  Dimensions: {len(self.dim_recovery_rates)}",
        ]

        if self.concept_alignment is not None:
            lines.append(f"  Concept alignment (Spearman r): {self.concept_alignment:.4f}")

        lines.append("  Top 5 dimensions:")
        for rank, dim_idx in enumerate(top_indices, 1):
            rate = self.dim_recovery_rates[dim_idx]
            lines.append(f"    {rank}. dim {dim_idx}: {rate:+.4f}")

        return "\n".join(lines)


class DimensionPatcher:
    """Patch individual dimensions of an activation to identify important ones.

    This class extends :class:`TemporalPatcher` to work at the dimension level.
    Three modes are supported:

    - **single**: Patch one dimension at a time. Slowest but most accurate;
      each dimension is independently replaced with the clean value.
    - **cumulative**: Patch top-k dimensions cumulatively. Start with the
      dimension with highest recovery, then add the next-highest, etc.
    - **ablation**: Ablate (zero out) one dimension at a time to measure
      the importance of each dimension.

    Parameters
    ----------
    patcher:
        A :class:`TemporalPatcher` instance used for full-activation patching.

    Examples
    --------
    >>> dim_patcher = DimensionPatcher(patcher)
    >>> result = dim_patcher.patch_by_dimension(
    ...     clean_cache, corrupted_obs_seq, corrupted_action_seq,
    ...     patch_timestep=5, patch_activation="rnn.h",
    ...     metric_fn=my_metric_fn, mode="single"
    ... )
    """

    def __init__(self, patcher: TemporalPatcher) -> None:
        self.patcher = patcher
        self.wm = patcher.wm

    def patch_by_dimension(
        self,
        clean_cache: ActivationCache,
        corrupted_obs_seq: Tensor,
        corrupted_action_seq: Tensor,
        patch_timestep: int,
        patch_activation: str,
        metric_fn: Callable[[ActivationCache], float],
        mode: str = "single",
        probe_weights: Optional[np.ndarray] = None,
    ) -> DimPatchResult:
        """Run dimension-level patching experiment.

        Parameters
        ----------
        clean_cache:
            Activation cache from clean run.
        corrupted_obs_seq:
            Observation sequence for corrupted runs.
        corrupted_action_seq:
            Action sequence for corrupted runs.
        patch_timestep:
            Timestep to patch.
        patch_activation:
            Name of the activation to patch dimensionally.
        metric_fn:
            Function that takes a cache and returns a scalar metric.
        mode:
            One of "single" (patch dimensions independently), "cumulative"
            (patch top dimensions cumulatively), or "ablation" (zero out
            dimensions one at a time).
        probe_weights:
            Optional array of shape (d,) containing external weights
            (e.g., from a classifier probe). If provided, concept_alignment
            is computed as Spearman correlation with dim_recovery_rates.

        Returns
        -------
        DimPatchResult
            Contains dim_recovery_rates and optional concept_alignment.

        Raises
        ------
        ValueError
            If mode is not one of the supported options.
        KeyError
            If the activation is not in clean_cache at patch_timestep.
        """
        if mode not in ("single", "cumulative", "ablation"):
            raise ValueError(
                f"mode must be 'single', 'cumulative', or 'ablation', got {mode!r}"
            )

        clean_activation = clean_cache[patch_activation, patch_timestep]
        d = clean_activation.shape[0] if len(clean_activation.shape) > 0 else 1

        # Compute baseline metrics
        metric_clean = metric_fn(clean_cache)
        _, corrupted_cache = self.wm.run_with_cache(
            corrupted_obs_seq, corrupted_action_seq
        )
        metric_corrupted = metric_fn(corrupted_cache)

        # Initialize dimension-level recovery rates
        dim_recovery_rates = np.zeros(d, dtype=np.float32)

        if mode == "single":
            dim_recovery_rates = self._patch_single(
                clean_cache,
                corrupted_obs_seq,
                corrupted_action_seq,
                patch_timestep,
                patch_activation,
                metric_fn,
                metric_clean,
                metric_corrupted,
            )
        elif mode == "cumulative":
            dim_recovery_rates = self._patch_cumulative(
                clean_cache,
                corrupted_obs_seq,
                corrupted_action_seq,
                patch_timestep,
                patch_activation,
                metric_fn,
                metric_clean,
                metric_corrupted,
            )
        elif mode == "ablation":
            dim_recovery_rates = self._patch_ablation(
                clean_cache,
                corrupted_obs_seq,
                corrupted_action_seq,
                patch_timestep,
                patch_activation,
                metric_fn,
                metric_clean,
                metric_corrupted,
            )

        # Compute concept alignment if probe weights provided
        concept_alignment = None
        if probe_weights is not None:
            if len(probe_weights) == len(dim_recovery_rates):
                # Spearman correlation between recovery rates and probe weights
                r, _ = spearmanr(dim_recovery_rates, probe_weights)
                concept_alignment = float(r) if not np.isnan(r) else None

        return DimPatchResult(
            activation_name=patch_activation,
            timestep=patch_timestep,
            dim_recovery_rates=dim_recovery_rates,
            concept_alignment=concept_alignment,
        )

    def _patch_single(
        self,
        clean_cache: ActivationCache,
        corrupted_obs_seq: Tensor,
        corrupted_action_seq: Tensor,
        patch_timestep: int,
        patch_activation: str,
        metric_fn: Callable[[ActivationCache], float],
        metric_clean: float,
        metric_corrupted: float,
    ) -> np.ndarray:
        """Patch dimensions independently."""
        clean_activation = clean_cache[patch_activation, patch_timestep]
        d = clean_activation.shape[0] if len(clean_activation.shape) > 0 else 1

        dim_recovery_rates = np.zeros(d, dtype=np.float32)

        for dim in tqdm(range(d), desc=f"Patching {patch_activation} dims (single)"):
            def dim_patch_hook(tensor: Tensor, ctx: HookContext) -> Tensor:
                """Patch only one dimension."""
                if ctx.timestep == patch_timestep:
                    patched = tensor.clone()
                    patched[dim] = clean_activation[dim]
                    return patched
                return tensor

            patch_hp = HookPoint(
                name=patch_activation,
                stage="post",
                fn=dim_patch_hook,
            )

            _, patched_cache = self.wm.run_with_hooks(
                corrupted_obs_seq,
                corrupted_action_seq,
                fwd_hooks=[patch_hp],
                return_cache=True,
            )
            metric_patched = metric_fn(patched_cache)

            # Compute recovery rate for this dimension
            denom = metric_clean - metric_corrupted
            if abs(denom) < 1e-8:
                dim_recovery_rates[dim] = 0.0
            else:
                recovery = (metric_patched - metric_corrupted) / denom
                dim_recovery_rates[dim] = float(np.clip(recovery, -1.0, 2.0))

        return dim_recovery_rates

    def _patch_cumulative(
        self,
        clean_cache: ActivationCache,
        corrupted_obs_seq: Tensor,
        corrupted_action_seq: Tensor,
        patch_timestep: int,
        patch_activation: str,
        metric_fn: Callable[[ActivationCache], float],
        metric_clean: float,
        metric_corrupted: float,
    ) -> np.ndarray:
        """Patch dimensions cumulatively, starting with highest-impact ones."""
        clean_activation = clean_cache[patch_activation, patch_timestep]
        d = clean_activation.shape[0] if len(clean_activation.shape) > 0 else 1

        # First pass: evaluate each dimension independently
        single_rates = self._patch_single(
            clean_cache,
            corrupted_obs_seq,
            corrupted_action_seq,
            patch_timestep,
            patch_activation,
            metric_fn,
            metric_clean,
            metric_corrupted,
        )

        # Sort dimensions by absolute recovery rate
        dim_order = np.argsort(-np.abs(single_rates))

        dim_recovery_rates = np.zeros(d, dtype=np.float32)
        patched_dims = set()

        for dim in tqdm(dim_order, desc=f"Patching {patch_activation} dims (cumulative)"):
            patched_dims.add(int(dim))

            def cumul_patch_hook(tensor: Tensor, ctx: HookContext) -> Tensor:
                """Patch all dimensions in patched_dims."""
                if ctx.timestep == patch_timestep:
                    patched = tensor.clone()
                    for d_idx in patched_dims:
                        patched[d_idx] = clean_activation[d_idx]
                    return patched
                return tensor

            patch_hp = HookPoint(
                name=patch_activation,
                stage="post",
                fn=cumul_patch_hook,
            )

            _, patched_cache = self.wm.run_with_hooks(
                corrupted_obs_seq,
                corrupted_action_seq,
                fwd_hooks=[patch_hp],
                return_cache=True,
            )
            metric_patched = metric_fn(patched_cache)

            denom = metric_clean - metric_corrupted
            if abs(denom) < 1e-8:
                dim_recovery_rates[dim] = 0.0
            else:
                recovery = (metric_patched - metric_corrupted) / denom
                dim_recovery_rates[dim] = float(np.clip(recovery, -1.0, 2.0))

        return dim_recovery_rates

    def _patch_ablation(
        self,
        clean_cache: ActivationCache,
        corrupted_obs_seq: Tensor,
        corrupted_action_seq: Tensor,
        patch_timestep: int,
        patch_activation: str,
        metric_fn: Callable[[ActivationCache], float],
        metric_clean: float,
        metric_corrupted: float,
    ) -> np.ndarray:
        """Ablate (zero out) dimensions one at a time."""
        clean_activation = clean_cache[patch_activation, patch_timestep]
        d = clean_activation.shape[0] if len(clean_activation.shape) > 0 else 1

        dim_recovery_rates = np.zeros(d, dtype=np.float32)

        for dim in tqdm(range(d), desc=f"Patching {patch_activation} dims (ablation)"):
            def ablation_hook(tensor: Tensor, ctx: HookContext) -> Tensor:
                """Zero out one dimension."""
                if ctx.timestep == patch_timestep:
                    ablated = tensor.clone()
                    ablated[dim] = 0.0
                    return ablated
                return tensor

            patch_hp = HookPoint(
                name=patch_activation,
                stage="post",
                fn=ablation_hook,
            )

            _, patched_cache = self.wm.run_with_hooks(
                corrupted_obs_seq,
                corrupted_action_seq,
                fwd_hooks=[patch_hp],
                return_cache=True,
            )
            metric_patched = metric_fn(patched_cache)

            # For ablation, negative recovery means ablating the dimension
            # was harmful (dimension is important)
            denom = metric_clean - metric_corrupted
            if abs(denom) < 1e-8:
                dim_recovery_rates[dim] = 0.0
            else:
                # Note: we use metric_patched directly, not as "recovery"
                # Negative values indicate dimension importance
                recovery = (metric_patched - metric_clean) / denom
                dim_recovery_rates[dim] = float(np.clip(recovery, -2.0, 1.0))

        return dim_recovery_rates
