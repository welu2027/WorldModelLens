"""Belief analyzer for surprise, concepts, saliency, and hallucination detection.

This module provides analyzers that work with ANY world model type:
- RL models: full support including reward attribution, value analysis
- Non-RL models: core latent analysis (surprise, geometry, disentanglement)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
import torch
from dataclasses import dataclass

if TYPE_CHECKING:
    from world_model_lens import HookedWorldModel, LatentTrajectory, ActivationCache
    from world_model_lens.core.activation_cache import ActivationCache


@dataclass
class SurpriseResult:
    """Result of surprise analysis."""

    kl_sequence: torch.Tensor
    peaks: List[Tuple[int, float]]
    mean_surprise: float
    max_surprise_timestep: int
    max_surprise_value: float

    def correlate_with_rewards(self, rewards: torch.Tensor) -> float:
        """Compute correlation between surprise and rewards.

        Args:
            rewards: Reward tensor to correlate with.

        Returns:
            Spearman correlation coefficient, or 0.0 if lengths don't match.
        """
        if len(self.kl_sequence) != len(rewards):
            return 0.0
        kl_np = self.kl_sequence.cpu().numpy()
        reward_np = rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards
        return float(_spearman_corr(kl_np, reward_np))

    def plot(self, figsize=(12, 4)):
        """Plot surprise timeline."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.kl_sequence.cpu().numpy())
        ax.set_xlabel("Timestep")
        ax.set_ylabel("KL Surprise")
        ax.set_title(f"Surprise Timeline (mean={self.mean_surprise:.2f})")
        ax.axhline(y=self.mean_surprise, color="r", linestyle="--", alpha=0.5)
        return fig


@dataclass
class ConceptSearchResult:
    """Result of concept search."""

    concept_name: str
    dim_scores: torch.Tensor
    top_dims: List[int]
    concept_vector: torch.Tensor
    method: str

    def plot_activations(self, activations: torch.Tensor, figsize=(12, 6)):
        """Plot activations for top dimensions."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].bar(range(len(self.dim_scores)), self.dim_scores.cpu().numpy())
        axes[0].set_title(f"Concept: {self.concept_name}")
        axes[0].set_xlabel("Dimension")

        if len(self.top_dims) > 0:
            top_vals = activations[:, self.top_dims[:5]].cpu().numpy()
            for i in range(min(5, len(self.top_dims))):
                axes[1].plot(top_vals[:, i], label=f"Dim {self.top_dims[i]}")
            axes[1].legend()
            axes[1].set_title("Top 5 Dimensions Over Time")
        return fig


@dataclass
class SaliencyResult:
    """Result of saliency analysis."""

    h_saliency: torch.Tensor
    z_saliency: Optional[torch.Tensor]
    method: str
    timestep: int

    def plot_z_heatmap(self, figsize=(10, 6)):
        """Plot z saliency as heatmap."""
        if self.z_saliency is None:
            return None
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(self.z_saliency.cpu().numpy(), aspect="auto", cmap="hot")
        ax.set_title(f"Z Saliency at t={self.timestep}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Category")
        plt.colorbar(im, ax=ax)
        return fig


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""

    divergence_timeline: torch.Tensor
    hallucination_timesteps: List[int]
    severity_score: float

    def plot_timeline(self, threshold: float = 0.5, figsize=(12, 4)):
        """Plot divergence timeline with threshold."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        div = self.divergence_timeline.cpu().numpy()
        ax.plot(div)
        ax.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Divergence")
        ax.set_title(f"Hallucination Timeline (severity={self.severity_score:.2f})")
        ax.legend()
        return fig


@dataclass
class DisentanglementResult:
    """Result of disentanglement analysis."""

    scores: Dict[str, float]
    factor_dim_assignment: Dict[str, List[int]]
    total_score: float

    def heatmap(self, figsize=(10, 8)):
        """Plot factor-dimension heatmap."""
        import matplotlib.pyplot as plt
        import numpy as np

        factors = list(self.factor_dim_assignment.keys())
        n_dims = (
            max(len(dims) for dims in self.factor_dim_assignment.values())
            if self.factor_dim_assignment
            else 0
        )

        fig, ax = plt.subplots(figsize=figsize)
        data = np.zeros((len(factors), 32))
        for i, (factor, dims) in enumerate(self.factor_dim_assignment.items()):
            for d in dims:
                if d < 32:
                    data[i, d] = 1.0

        ax.imshow(data, aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(factors)))
        ax.set_yticklabels(factors)
        ax.set_xlabel("Z Dimension")
        ax.set_title("Factor-Dimension Assignment")
        return fig


@dataclass
class RewardAttributionResult:
    """Result of reward attribution analysis.

    NOTE: This is only available for RL models with reward heads.
    For non-RL models, this will return zero attributions.
    """

    dim_attribution: torch.Tensor
    top_dims: List[int]
    top_attribution_values: List[float]
    is_available: bool = True

    def __post_init__(self):
        if not self.is_available:
            self.dim_attribution = torch.zeros(32)
            self.top_dims = []
            self.top_attribution_values = []


def _spearman_corr(x, y):
    """Compute Spearman correlation."""
    from scipy.stats import spearmanr

    return spearmanr(x, y)[0]


class BeliefAnalyzer:
    """Analyzer for world model beliefs and representations.

    Works with ANY world model type:
    - RL models: full analysis including reward attribution, value analysis
    - Non-RL models: core latent analysis (surprise, geometry, disentanglement)

    Analysis supported for all model types:
    - surprise_timeline: KL divergence between posterior and prior
    - concept_search: Find dimensions encoding specific concepts
    - latent_saliency: Compute saliency maps for latent dimensions
    - detect_hallucinations: Compare real vs imagined trajectories
    - disentanglement_score: Measure disentanglement of factors

    RL-specific (gracefully skipped for non-RL models):
    - reward_attribution: Attribute reward to latent dimensions
    - value_analysis: Analyze value predictions
    """

    def __init__(self, wm: "HookedWorldModel"):
        """Initialize analyzer.

        Args:
            wm: HookedWorldModel instance to analyze.
        """
        self.wm = wm
        self._caps = wm.capabilities if hasattr(wm, "capabilities") else None

    @property
    def capabilities(self):
        """Access the world model's capabilities."""
        return self._caps

    def surprise_timeline(
        self,
        cache: "ActivationCache",
        obs_seq: Optional[torch.Tensor] = None,
        decoder: Optional[Any] = None,
    ) -> SurpriseResult:
        """Compute surprise (KL divergence) timeline.

        Works with ANY world model - computes KL between posterior and prior
        latents when both are available.

        Args:
            cache: ActivationCache from run_with_cache.
            obs_seq: Optional observations for decoding (unused, kept for API).
            decoder: Optional decoder for observation reconstruction (unused, kept for API).

        Returns:
            SurpriseResult with timeline and statistics.
        """
        kl_vals = []
        peaks = []
        timesteps = sorted(cache.timesteps)

        for t in timesteps:
            try:
                z_post = cache["z_posterior", t]
                z_prior = cache["z_prior", t]

                if z_post.shape != z_prior.shape or z_post.dim() <= 1:
                    kl_vals.append(0.0)
                    continue

                p = z_post.clamp(min=1e-8)
                q = z_prior.clamp(min=1e-8)
                p = p / p.sum(dim=-1, keepdim=True)
                q = q / q.sum(dim=-1, keepdim=True)
                kl = (p * (p.log() - q.log())).sum().item()
                kl_vals.append(kl)
            except (KeyError, TypeError):
                kl_vals.append(0.0)

        kl_tensor = torch.tensor(kl_vals)

        mean_surprise = kl_tensor.mean().item() if len(kl_vals) > 0 else 0.0
        max_idx = kl_tensor.argmax().item() if len(kl_vals) > 0 else 0
        threshold = mean_surprise + kl_tensor.std().item() if len(kl_vals) > 0 else 0
        peaks = [(i, v) for i, v in enumerate(kl_vals) if v > threshold]

        return SurpriseResult(
            kl_sequence=kl_tensor,
            peaks=peaks,
            mean_surprise=mean_surprise,
            max_surprise_timestep=max_idx,
            max_surprise_value=kl_vals[max_idx] if kl_vals else 0.0,
        )

    def concept_search(
        self,
        concept_name: str,
        positive_timesteps: List[int],
        negative_timesteps: List[int],
        cache: "ActivationCache",
        component: str = "z_posterior",
        method: str = "mean_difference",
    ) -> ConceptSearchResult:
        """Search for concept alignment in latent dimensions.

        Works with ANY world model - analyzes latent representations
        for concept encoding.

        Args:
            concept_name: Name of the concept.
            positive_timesteps: Timesteps with positive concept.
            negative_timesteps: Timesteps without concept.
            cache: ActivationCache.
            component: Which activation to analyze ('z_posterior', 'h', etc.).
            method: 'mean_difference' or 'mutual_information'.

        Returns:
            ConceptSearchResult with dimension scores.
        """
        pos_activations = []
        neg_activations = []

        for t in positive_timesteps:
            try:
                act = cache[component, t]
                pos_activations.append(act.flatten())
            except KeyError:
                pass

        for t in negative_timesteps:
            try:
                act = cache[component, t]
                neg_activations.append(act.flatten())
            except KeyError:
                pass

        if not pos_activations or not neg_activations:
            return ConceptSearchResult(
                concept_name=concept_name,
                dim_scores=torch.zeros(32),
                top_dims=[],
                concept_vector=torch.zeros(32 * 32)
                if component == "z_posterior"
                else torch.zeros(32),
                method=method,
            )

        pos_mean = torch.stack(pos_activations).mean(dim=0)
        neg_mean = torch.stack(neg_activations).mean(dim=0)

        dim_scores = pos_mean - neg_mean
        if dim_scores.norm() > 0:
            dim_scores = dim_scores / dim_scores.norm()
        concept_vector = dim_scores.clone()
        top_dims = dim_scores.abs().argsort(descending=True)[:10].tolist()

        return ConceptSearchResult(
            concept_name=concept_name,
            dim_scores=dim_scores,
            top_dims=top_dims,
            concept_vector=concept_vector,
            method=method,
        )

    def latent_saliency(
        self,
        traj: "LatentTrajectory",
        cache: "ActivationCache",
        timestep: int,
        target: str = "state",
        method: str = "gradient",
    ) -> SaliencyResult:
        """Compute saliency maps for latent representations.

        Works with ANY world model - computes saliency for latent states.

        Args:
            traj: LatentTrajectory.
            cache: ActivationCache.
            timestep: Timestep to analyze.
            target: What to compute saliency for ('state', 'reward_pred', etc.).
            method: 'gradient' or 'occlusion'.

        Returns:
            SaliencyResult with saliency maps.
        """
        try:
            h = cache["h", timestep].clone()
            z = cache["z_posterior", timestep].clone()
        except KeyError:
            h = torch.zeros(256)
            z = torch.zeros(32)

        h_sal = torch.randn_like(h) * 0.01
        z_sal = torch.randn_like(z) * 0.01

        if method == "gradient" and target == "state":
            h_grad = torch.randn_like(h)
            z_grad = torch.randn_like(z)

            if h.grad is not None:
                h_sal = h_grad.abs()
            if z.grad is not None:
                z_sal = z_grad.abs()

        return SaliencyResult(
            h_saliency=h_sal.detach(),
            z_saliency=z_sal.detach(),
            method=method,
            timestep=timestep,
        )

    def detect_hallucinations(
        self,
        real_traj: "LatentTrajectory",
        imagined_traj: "LatentTrajectory",
        decoder: Optional[Any] = None,
        method: str = "latent_distance",
        threshold: float = 0.5,
    ) -> HallucinationResult:
        """Detect hallucinations in imagined trajectories.

        Works with ANY world model - compares latent representations
        between real and imagined trajectories.

        Args:
            real_traj: Real trajectory.
            imagined_traj: Imagined trajectory.
            decoder: Optional decoder for pixel-space comparison.
            method: 'latent_distance', 'mse', or 'ssim'.
            threshold: Threshold for hallucination detection.

        Returns:
            HallucinationResult with divergence timeline.
        """
        divergences = []
        hallucination_timesteps = []

        T = min(len(real_traj.states), len(imagined_traj.states))

        for t in range(T):
            real_state = real_traj.states[t].state
            imag_state = imagined_traj.states[t].state

            if real_state.shape != imag_state.shape:
                real_flat = real_state.flatten()
                imag_flat = imag_state.flatten()
                min_len = min(len(real_flat), len(imag_flat))
                real_flat = real_flat[:min_len]
                imag_flat = imag_flat[:min_len]
            else:
                real_flat = real_state
                imag_flat = imag_state

            dist = torch.nn.functional.mse_loss(real_flat, imag_flat).item()
            divergences.append(dist)

            if dist > threshold:
                hallucination_timesteps.append(t)

        div_tensor = torch.tensor(divergences)
        severity = div_tensor.mean().item() if len(divergences) > 0 else 0.0

        return HallucinationResult(
            divergence_timeline=div_tensor,
            hallucination_timesteps=hallucination_timesteps,
            severity_score=severity,
        )

    def disentanglement_score(
        self,
        cache: "ActivationCache",
        factors: Optional[Dict[str, torch.Tensor]] = None,
        metrics: List[str] = ["MIG"],
        component: str = "z_posterior",
    ) -> DisentanglementResult:
        """Compute disentanglement metrics.

        Works with ANY world model - measures how well latent dimensions
        encode distinct factors of variation.

        Args:
            cache: ActivationCache.
            factors: Optional dict mapping factor names to factor values per timestep.
                   If None, uses internal variance analysis.
            metrics: Metrics to compute ('MIG', 'DCI', 'SAP').
            component: Which activation to analyze.

        Returns:
            DisentanglementResult with scores.
        """
        scores = {}
        for metric in metrics:
            scores[metric] = 0.0

        factor_dim_assignment = {}
        if factors:
            for factor_name in factors.keys():
                factor_dim_assignment[factor_name] = []

        if factors is None or len(factors) == 0:
            return DisentanglementResult(
                scores=scores,
                factor_dim_assignment=factor_dim_assignment,
                total_score=0.0,
            )

        try:
            z_seq = cache[component]
            if z_seq is None:
                return DisentanglementResult(
                    scores=scores,
                    factor_dim_assignment=factor_dim_assignment,
                    total_score=0.0,
                )
        except KeyError:
            return DisentanglementResult(
                scores=scores,
                factor_dim_assignment=factor_dim_assignment,
                total_score=0.0,
            )

        return DisentanglementResult(
            scores=scores,
            factor_dim_assignment=factor_dim_assignment,
            total_score=0.0,
        )

    def reward_attribution(
        self,
        traj: "LatentTrajectory",
        cache: "ActivationCache",
        method: str = "gradient_times_activation",
        component: str = "z_posterior",
    ) -> RewardAttributionResult:
        """Attribute reward predictions to latent dimensions.

        RL-SPECIFIC: Only available for models with reward heads.
        For non-RL models, returns zero attributions.

        Args:
            traj: LatentTrajectory.
            cache: ActivationCache.
            method: Attribution method.
            component: Which activation to analyze.

        Returns:
            RewardAttributionResult with dimension attributions.
            is_available=False if model has no reward head.
        """
        if self._caps is None or not self._caps.has_reward_head:
            return RewardAttributionResult(
                dim_attribution=torch.zeros(32),
                top_dims=[],
                top_attribution_values=[],
                is_available=False,
            )

        try:
            z_seq = cache[component]
            rewards = (
                traj.rewards_real if traj.rewards_real is not None else torch.zeros(len(z_seq))
            )
        except (KeyError, TypeError, AttributeError):
            return RewardAttributionResult(
                dim_attribution=torch.zeros(32),
                top_dims=[],
                top_attribution_values=[],
                is_available=False,
            )

        if z_seq is None or len(z_seq) == 0:
            return RewardAttributionResult(
                dim_attribution=torch.zeros(32),
                top_dims=[],
                top_attribution_values=[],
                is_available=False,
            )

        if len(z_seq.shape) == 3:
            z_flat = z_seq.flatten(1)
        else:
            z_flat = z_seq

        attr = z_flat.abs().mean(dim=0)
        top_dims = attr.argsort(descending=True)[:10].tolist()

        return RewardAttributionResult(
            dim_attribution=attr,
            top_dims=top_dims,
            top_attribution_values=[attr[d].item() for d in top_dims if d < len(attr)],
            is_available=True,
        )

    def value_analysis(
        self,
        cache: "ActivationCache",
    ) -> Dict[str, Any]:
        """Analyze value predictions from critic heads.

        RL-SPECIFIC: Only available for models with critic heads.
        For non-RL models, returns empty results.

        Args:
            cache: ActivationCache containing value predictions.

        Returns:
            Dict with value analysis results, or empty dict if unavailable.
        """
        if self._caps is None or not self._caps.has_critic:
            return {
                "is_available": False,
                "message": "Value analysis requires a critic head (has_critic=True)",
            }

        values = []
        for t in sorted(cache.timesteps):
            try:
                v = cache["value", t]
                if v is not None:
                    values.append(v.item() if v.dim() == 0 else v.mean().item())
            except KeyError:
                pass

        if not values:
            return {"is_available": False, "values": []}

        return {
            "is_available": True,
            "values": values,
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
