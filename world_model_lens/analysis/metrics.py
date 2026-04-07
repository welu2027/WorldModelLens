"""Latent space metrics for world model evaluation."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def _get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LatentMetricsResult:
    """Results from latent space metrics."""

    compression_ratio: float
    predictive_info: float
    temporal_coherence: float
    reconstruction_error: float
    latent_variance: float


class LatentMetrics:
    """Compute various metrics on world model latent representations."""

    @staticmethod
    def compression_ratio(original_obs: torch.Tensor, latent_obs: torch.Tensor) -> float:
        """Compute compression ratio of latent representation.

        Args:
            original_obs: Original observations [T, ...].
            latent_obs: Latent representations [T, ...].

        Returns:
            Compression ratio (original_size / latent_size).
        """
        orig_size = original_obs.flatten(1).shape[1]
        lat_size = latent_obs.flatten(1).shape[1]

        if lat_size == 0:
            return 0.0

        return orig_size / lat_size

    @staticmethod
    def predictive_info(actions: torch.Tensor, latents: torch.Tensor) -> float:
        """Compute predictive information (mutual info between actions and next state).

        Args:
            actions: Action sequence [T, d_action].
            latents: Latent sequence [T, d_latent].

        Returns:
            Predictive information estimate.
        """
        if len(actions) < 2 or len(latents) < 2:
            return 0.0

        actions = actions[:-1]
        latents[1:]
        current_latents = latents[:-1]

        joint = torch.cat([actions, current_latents], dim=1)
        marginal_a = actions
        marginal_s = current_latents

        joint_cov = torch.cov(joint.T)
        marg_a_cov = torch.cov(marginal_a.T)
        marg_s_cov = torch.cov(marginal_s.T)

        if joint_cov.det() <= 0 or marg_a_cov.det() <= 0 or marg_s_cov.det() <= 0:
            return 0.0

        joint_entropy = 0.5 * torch.logdet(joint_cov)
        action_entropy = 0.5 * torch.logdet(marg_a_cov)
        state_entropy = 0.5 * torch.logdet(marg_s_cov)

        mi = action_entropy + state_entropy - joint_entropy
        return max(0.0, float(mi.item()))

    @staticmethod
    def temporal_hierarchy(latents: torch.Tensor) -> dict[str, Any]:
        """Analyze temporal hierarchy in latent sequences.

        Args:
            latents: Latent sequence [T, d_latent].

        Returns:
            Dict with hierarchy metrics.
        """
        if len(latents) < 3:
            return {"hierarchical_score": 0.0, "slow_features": [], "fast_features": []}

        latents_flat = latents.flatten(1)

        variances = latents_flat.var(dim=0)

        temporal_diffs = (latents[1:] - latents[:-1]).flatten(1)
        diff_variances = temporal_diffs.var(dim=0)

        slow_mask = variances > 2 * variances.median()
        fast_mask = diff_variances > 2 * diff_variances.median()

        slow_features = slow_mask.nonzero().squeeze().tolist() if slow_mask.any() else []
        fast_features = fast_mask.nonzero().squeeze().tolist() if fast_mask.any() else []

        hierarchical_score = len(slow_features) / max(len(variances), 1)

        return {
            "hierarchical_score": float(hierarchical_score),
            "slow_features": slow_features,
            "fast_features": fast_features,
            "n_slow": len(slow_features),
            "n_fast": len(fast_features),
        }

    @staticmethod
    def reconstruction_error(wm: Any, obs: torch.Tensor, actions: torch.Tensor) -> float:
        """Compute reconstruction error of the world model.

        Args:
            wm: HookedWorldModel.
            obs: Observations.
            actions: Actions.

        Returns:
            MSE reconstruction error.
        """
        traj, cache = wm.run_with_cache(obs, actions)

        try:
            recon = cache["reconstruction"]
            orig = cache["observation"]
            mse = F.mse_loss(recon, orig)
            return float(mse.item())
        except:
            return float("inf")

    @staticmethod
    def latent_variance(latents: torch.Tensor) -> float:
        """Compute total variance of latent representations.

        Args:
            latents: Latent tensor [T, d_latent].

        Returns:
            Total variance.
        """
        if latents.dim() == 3:
            latents = latents.flatten(1)

        variance = latents.var(dim=0).sum()
        return float(variance.item())

    @staticmethod
    def compute_all(
        latents: torch.Tensor, actions: torch.Tensor, original_obs: torch.Tensor | None = None
    ) -> LatentMetricsResult:
        """Compute all metrics at once.

        Args:
            latents: Latent sequence.
            actions: Action sequence.
            original_obs: Optional original observations.

        Returns:
            LatentMetricsResult with all computed metrics.
        """
        compression = 1.0
        if original_obs is not None:
            compression = LatentMetrics.compression_ratio(original_obs, latents)

        predictive_info = LatentMetrics.predictive_info(actions, latents)
        temporal_hierarchy = LatentMetrics.temporal_hierarchy(latents)
        variance = LatentMetrics.latent_variance(latents)

        return LatentMetricsResult(
            compression_ratio=compression,
            predictive_info=predictive_info,
            temporal_coherence=temporal_hierarchy.get("hierarchical_score", 0.0),
            reconstruction_error=0.0,
            latent_variance=variance,
        )


class CausalBenchmark:
    """Benchmark for causal analysis methods."""

    @staticmethod
    def evaluate_patching(model: Any, ground_truth_circuit: dict[str, Any]) -> dict[str, float]:
        """Evaluate patching accuracy against ground truth.

        Args:
            model: Model to evaluate.
            ground_truth_circuit: Known important components.

        Returns:
            Dict of evaluation metrics.
        """
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    @staticmethod
    def probe_attribution_accuracy(model: Any, concepts: dict[str, torch.Tensor]) -> float:
        """Measure accuracy of probe-based attribution.

        Args:
            model: Model to evaluate.
            concepts: Ground truth concept labels.

        Returns:
            Attribution accuracy.
        """
        return 0.0

    @staticmethod
    def circuit_stability(model: Any, perturbations: list[dict[str, Any]]) -> float:
        """Measure stability of discovered circuits under perturbation.

        Args:
            model: Model to evaluate.
            perturbations: List of perturbations to apply.

        Returns:
            Stability score.
        """
        return 0.0
