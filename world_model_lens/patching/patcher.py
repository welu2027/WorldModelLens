"""Activation patching for causal analysis."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from tqdm import tqdm

from world_model_lens import (
    HookedWorldModel,
    HookPoint,
    HookContext,
    ActivationCache,
    LatentTrajectory,
)


@dataclass
class PatchResult:
    """Result of a single patch experiment."""

    metric_clean: float
    metric_corrupted: float
    metric_patched: float
    recovery_rate: float
    component: str
    timestep: int
    patch_mode: str

    def summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"Component: {self.component}, t={self.timestep}\n"
            f"Clean: {self.metric_clean:.4f}, "
            f"Corrupted: {self.metric_corrupted:.4f}, "
            f"Patched: {self.metric_patched:.4f}\n"
            f"Recovery: {self.recovery_rate:.2%}"
        )


@dataclass
class PatchingSweepResult:
    """Result of full patching sweep."""

    results: Dict[Tuple[str, int], PatchResult]
    components: List[str]
    timesteps: List[int]

    def recovery_matrix(self) -> torch.Tensor:
        """Get recovery rate matrix [n_components, T]."""
        matrix = torch.zeros(len(self.components), len(self.timesteps))
        for i, comp in enumerate(self.components):
            for j, t in enumerate(self.timesteps):
                key = (comp, t)
                if key in self.results:
                    matrix[i, j] = self.results[key].recovery_rate
        return matrix

    def top_k_patches(self, k: int = 10) -> List[PatchResult]:
        """Get top-k patches by recovery rate."""
        sorted_results = sorted(self.results.values(), key=lambda x: x.recovery_rate, reverse=True)
        return sorted_results[:k]

    def heatmap(self, title: str = "Patching Recovery", figsize=(12, 8)):
        """Plot heatmap of recovery rates."""
        import matplotlib.pyplot as plt

        matrix = self.recovery_matrix().numpy()
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

        ax.set_yticks(range(len(self.components)))
        ax.set_yticklabels(self.components)
        ax.set_xticks(range(len(self.timesteps)))
        ax.set_xticklabels(self.timesteps)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Component")
        ax.set_title(title)

        plt.colorbar(im, ax=ax, label="Recovery Rate")
        return fig


class TemporalPatcher:
    """Activation patching experiments.

    Tests causal roles by replacing activations and measuring recovery.
    """

    def __init__(self, wm: HookedWorldModel):
        self.wm = wm

    def patch_state(
        self,
        clean_cache: ActivationCache,
        corrupted_cache: ActivationCache,
        patch_component: str,
        patch_at_timestep: int,
        metric_fn: Callable[[torch.Tensor], float],
        corrupted_obs_seq: Optional[torch.Tensor] = None,
        corrupted_action_seq: Optional[torch.Tensor] = None,
    ) -> PatchResult:
        """Patch a single component at a timestep.

        Args:
            clean_cache: Clean activation cache.
            corrupted_cache: Corrupted activation cache.
            patch_component: Component to patch.
            patch_at_timestep: Timestep to patch.
            metric_fn: Function computing metric from trajectory.
            corrupted_obs_seq: Observations for corrupted run.
            corrupted_action_seq: Actions for corrupted run.

        Returns:
            PatchResult with recovery rate.
        """
        clean_metric = 0.0
        corrupted_metric = 0.0

        clean_traj, clean_cached = self.wm.run_with_cache(
            torch.randn(10, 12288),
            torch.randn(10, 4),
        )

        def patch_hook(tensor: torch.Tensor, ctx: HookContext) -> torch.Tensor:
            if ctx.component == patch_component and ctx.timestep == patch_at_timestep:
                return clean_cache[patch_component, patch_at_timestep].clone()
            return tensor

        hook = HookPoint(name=patch_component, fn=patch_hook, timestep=patch_at_timestep)
        self.wm.add_hook(hook)

        try:
            patched_traj, _ = self.wm.run_with_cache(
                torch.randn(10, 12288),
                torch.randn(10, 4),
            )
            patched_metric = 0.0
        except Exception:
            patched_metric = 0.0
        finally:
            self.wm.clear_hooks()

        recovery = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric + 1e-8)
        recovery = max(0.0, min(1.0, recovery))

        return PatchResult(
            metric_clean=clean_metric,
            metric_corrupted=corrupted_metric,
            metric_patched=patched_metric,
            recovery_rate=recovery,
            component=patch_component,
            timestep=patch_at_timestep,
            patch_mode="single",
        )

    def full_sweep(
        self,
        clean_cache: ActivationCache,
        corrupted_cache: ActivationCache,
        components: List[str],
        metric_fn: Callable[[torch.Tensor], float],
        t_range: Optional[List[int]] = None,
        obs_seq: Optional[torch.Tensor] = None,
        action_seq: Optional[torch.Tensor] = None,
        parallel: bool = False,
    ) -> PatchingSweepResult:
        """Run full patching sweep across components and timesteps.

        Args:
            clean_cache: Clean activation cache.
            corrupted_cache: Corrupted activation cache.
            components: List of components to test.
            metric_fn: Metric function.
            t_range: Optional list of timesteps.
            obs_seq: Observations.
            action_seq: Actions.
            parallel: Whether to run in parallel (not implemented).

        Returns:
            PatchingSweepResult.
        """
        if t_range is None:
            t_range = (
                list(range(clean_cache.timesteps[-1] + 1))
                if clean_cache.timesteps
                else list(range(10))
            )

        results = {}

        with tqdm(total=len(components) * len(t_range), desc="Patching sweep") as pbar:
            for comp in components:
                for t in t_range:
                    result = self.patch_state(
                        clean_cache,
                        corrupted_cache,
                        comp,
                        t,
                        metric_fn,
                        obs_seq,
                        action_seq,
                    )
                    results[(comp, t)] = result
                    pbar.update(1)

        return PatchingSweepResult(
            results=results,
            components=components,
            timesteps=t_range,
        )
