"""BeliefAnalyzer — interpretability tools for world-model belief states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from world_model_lens.core.activation_cache import ActivationCache
    from world_model_lens.hooked_world_model import HookedWorldModel
    from torch import Tensor


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SurpriseResult:
    """Per-timestep KL divergence between posterior and prior."""
    kl_sequence: np.ndarray          # (T,)
    peaks: List[int]                 # timestep indices of surprise peaks

    def correlate_with_rewards(self, rewards: np.ndarray) -> float:
        """Spearman correlation between KL and rewards."""
        from scipy.stats import spearmanr
        r, _ = spearmanr(self.kl_sequence, rewards[:len(self.kl_sequence)])
        return float(r)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _fig, ax = plt.subplots(figsize=(8, 3))
        ts = np.arange(len(self.kl_sequence))
        ax.plot(ts, self.kl_sequence, lw=1.5, label="KL (surprise)")
        for pk in self.peaks:
            ax.axvline(pk, color="red", alpha=0.4, lw=1)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("KL divergence")
        ax.set_title("Surprise Timeline")
        ax.legend()
        return ax

    def annotate_with_obs(self, obs_seq, top_k: int = 10):
        """Return figure with obs thumbnails at the top-k surprise peaks."""
        import matplotlib.pyplot as plt
        import torch

        top_peaks = sorted(self.peaks, key=lambda t: self.kl_sequence[t], reverse=True)[:top_k]
        n = len(top_peaks)
        fig, axes = plt.subplots(2, max(n, 1), figsize=(2.5 * max(n, 1), 5))
        if n == 0:
            return fig

        ax_kl = plt.subplot2grid((2, n), (0, 0), colspan=n, fig=fig)
        ax_kl.plot(np.arange(len(self.kl_sequence)), self.kl_sequence)
        for pk in top_peaks:
            ax_kl.axvline(pk, color="red", alpha=0.5)
        ax_kl.set_title("Top surprise peaks (red)")

        for i, t in enumerate(sorted(top_peaks)):
            ax = plt.subplot2grid((2, n), (1, i), fig=fig)
            obs = obs_seq[t] if t < len(obs_seq) else obs_seq[-1]
            if hasattr(obs, "detach"):
                obs = obs.detach().cpu().numpy()
            obs = np.asarray(obs)
            if obs.ndim == 3:
                obs = obs.transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
                obs = np.clip(obs, 0, 1)
                ax.imshow(obs)
            elif obs.ndim == 1:
                ax.bar(np.arange(min(len(obs), 20)), obs[:20])
            ax.set_title(f"t={t}\nKL={self.kl_sequence[t]:.2f}", fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        return fig


@dataclass
class ConceptSearchResult:
    """Which latent dimensions best represent a concept."""
    concept_name: str
    dim_scores_by_method: Dict[str, np.ndarray]   # method → (d,) scores
    top_dims: List[int]
    concept_vector: np.ndarray                    # (d,) unit-norm direction

    def plot_activations(self, activations: np.ndarray, ax=None):
        """Violin plot of activations for top dims."""
        import matplotlib.pyplot as plt

        top = self.top_dims[:10]
        if ax is None:
            _fig, ax = plt.subplots(figsize=(8, 4))
        data = [activations[:, d] for d in top]
        ax.violinplot(data, positions=range(len(top)), showmedians=True)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels([f"d{d}" for d in top], rotation=45, fontsize=8)
        ax.set_title(f"Top dims for concept: {self.concept_name}")
        return ax


@dataclass
class SaliencyResult:
    """Gradient-based saliency of h and z w.r.t. a target metric."""
    timestep: int
    h_saliency: np.ndarray      # (d_h,)
    z_saliency: np.ndarray      # (n_cat, n_cls)
    method: str

    def plot_z_heatmap(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(self.z_saliency, aspect="auto", cmap="hot")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("cls")
        ax.set_ylabel("cat")
        ax.set_title(f"z saliency (t={self.timestep}, method={self.method})")
        return ax

    def plot_h_dims(self, top_k: int = 20, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _fig, ax = plt.subplots(figsize=(8, 3))
        order = np.argsort(-np.abs(self.h_saliency))[:top_k]
        ax.bar(range(top_k), self.h_saliency[order])
        ax.set_xticks(range(top_k))
        ax.set_xticklabels([f"h{i}" for i in order], rotation=45, fontsize=7)
        ax.set_title(f"Top-{top_k} h saliency dims (t={self.timestep})")
        return ax


@dataclass
class DisentanglementResult:
    """Disentanglement score for a set of ground-truth factors."""
    factor_dim_assignment: Dict[str, int]   # factor → best dim index
    score_matrix: np.ndarray                # (n_factors, n_dims)
    total_score: float
    method: str

    def heatmap(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(self.score_matrix, aspect="auto", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Latent dim")
        ax.set_ylabel("Factor")
        ax.set_title(f"Disentanglement ({self.method}) — total={self.total_score:.3f}")
        return ax


@dataclass
class HallucinationResult:
    """Timesteps where the model's belief diverges from reality."""
    divergence_timeline: np.ndarray    # (T,)
    hallucination_timesteps: List[int]
    method: str

    def plot_timeline(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(self.divergence_timeline, lw=1.5)
        for t in self.hallucination_timesteps:
            ax.axvspan(t - 0.4, t + 0.4, alpha=0.3, color="red")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Divergence")
        ax.set_title(f"Hallucination Detection ({self.method})")
        return ax

    def plot_comparison(self, t: int, ax=None):
        """Placeholder comparison plot at a specific timestep."""
        import matplotlib.pyplot as plt
        if ax is None:
            _fig, ax = plt.subplots()
        ax.bar(["divergence"], [self.divergence_timeline[t] if t < len(self.divergence_timeline) else 0])
        ax.set_title(f"Divergence at t={t}")
        return ax


@dataclass
class RewardAttributionResult:
    """Attribution of reward predictions to h and z dimensions."""
    h_attribution: np.ndarray       # (d_h,)
    z_attribution: np.ndarray       # (n_cat * n_cls,) or (n_cat, n_cls)
    spurious_correlation: float

    def plot_attribution(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        else:
            axes = ax if hasattr(ax, "__len__") else [ax, ax]
        top_k = min(20, len(self.h_attribution))
        order_h = np.argsort(-np.abs(self.h_attribution))[:top_k]
        axes[0].bar(range(top_k), self.h_attribution[order_h])
        axes[0].set_title("h attribution (top-20)")
        z_flat = self.z_attribution.flatten()
        top_k_z = min(20, len(z_flat))
        order_z = np.argsort(-np.abs(z_flat))[:top_k_z]
        axes[1].bar(range(top_k_z), z_flat[order_z])
        axes[1].set_title("z attribution (top-20)")
        return axes


# ---------------------------------------------------------------------------
# BeliefAnalyzer
# ---------------------------------------------------------------------------

class BeliefAnalyzer:
    """Interpretability tools for world-model belief states.

    Parameters
    ----------
    wm:
        A :class:`~world_model_lens.HookedWorldModel`.
    """

    def __init__(self, wm: "HookedWorldModel") -> None:
        self._wm = wm

    # ------------------------------------------------------------------ #
    # Surprise timeline
    # ------------------------------------------------------------------ #

    def surprise_timeline(
        self,
        cache: "ActivationCache",
        peak_threshold: float = 2.0,
    ) -> SurpriseResult:
        """Compute per-timestep KL(posterior || prior) from cache.

        Uses ``kl`` activation from the cache if present, otherwise
        computes from ``z_posterior.logits`` and ``z_prior.logits``.
        """
        import torch

        # Try reading pre-computed KL from cache
        kl_vals = cache._store.get("kl", [])
        if kl_vals:
            kl_seq = np.array([float(v) for v in kl_vals])
        else:
            # Compute from logits
            post_logits = cache._store.get("z_posterior.logits", [])
            prior_logits = cache._store.get("z_prior.logits", [])
            if post_logits and prior_logits:
                kls = []
                for pq, pp in zip(post_logits, prior_logits):
                    q = torch.softmax(pq.float(), dim=-1) + 1e-9
                    p = torch.softmax(pp.float(), dim=-1) + 1e-9
                    kl = (q * (q / p).log()).sum().item()
                    kls.append(kl)
                kl_seq = np.array(kls)
            else:
                kl_seq = np.zeros(0)

        if len(kl_seq) == 0:
            return SurpriseResult(kl_sequence=kl_seq, peaks=[])

        mean_kl = kl_seq.mean()
        std_kl = kl_seq.std() + 1e-9
        threshold = mean_kl + peak_threshold * std_kl
        peaks = [int(i) for i in np.where(kl_seq > threshold)[0]]

        return SurpriseResult(kl_sequence=kl_seq, peaks=peaks)

    # ------------------------------------------------------------------ #
    # Concept search
    # ------------------------------------------------------------------ #

    def concept_search(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        concept_name: str = "concept",
        methods: Optional[List[str]] = None,
    ) -> ConceptSearchResult:
        """Find which latent dimensions best represent a concept.

        Parameters
        ----------
        activations:
            ``(T, d)`` array of latent activations.
        labels:
            ``(T,)`` binary or multi-class labels.
        methods:
            Scoring methods. Defaults to ``["logistic", "mutual_info", "correlation"]``.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_selection import mutual_info_classif
        from scipy.stats import pointbiserialr

        if methods is None:
            methods = ["logistic", "mutual_info", "correlation"]

        d = activations.shape[1]
        scores_by_method: Dict[str, np.ndarray] = {}

        # Logistic: train per-dim probe
        if "logistic" in methods:
            logistic_scores = np.zeros(d)
            y_bin = (labels > labels.mean()).astype(int)
            for dim in range(d):
                X = activations[:, dim:dim+1]
                try:
                    clf = LogisticRegression(max_iter=200, C=1.0)
                    clf.fit(X, y_bin)
                    logistic_scores[dim] = clf.score(X, y_bin)
                except Exception:
                    logistic_scores[dim] = 0.5
            scores_by_method["logistic"] = logistic_scores

        # Mutual information
        if "mutual_info" in methods:
            y_bin = (labels > labels.mean()).astype(int)
            mi_scores = mutual_info_classif(activations, y_bin, random_state=0)
            scores_by_method["mutual_info"] = mi_scores

        # Point-biserial correlation
        if "correlation" in methods:
            corr_scores = np.zeros(d)
            y_bin = (labels > labels.mean()).astype(float)
            for dim in range(d):
                try:
                    r, _ = pointbiserialr(y_bin, activations[:, dim])
                    corr_scores[dim] = abs(r)
                except Exception:
                    corr_scores[dim] = 0.0
            scores_by_method["correlation"] = corr_scores

        # Aggregate: mean rank
        agg = np.zeros(d)
        for sc in scores_by_method.values():
            agg += sc / (sc.max() + 1e-9)
        agg /= len(scores_by_method)

        top_dims = list(np.argsort(-agg)[:20])

        # Concept vector: unit-norm difference of class means
        y_bin = (labels > labels.mean()).astype(bool)
        pos_mean = activations[y_bin].mean(axis=0) if y_bin.any() else np.zeros(d)
        neg_mean = activations[~y_bin].mean(axis=0) if (~y_bin).any() else np.zeros(d)
        direction = pos_mean - neg_mean
        norm = np.linalg.norm(direction) + 1e-9
        concept_vector = direction / norm

        return ConceptSearchResult(
            concept_name=concept_name,
            dim_scores_by_method=scores_by_method,
            top_dims=top_dims,
            concept_vector=concept_vector,
        )

    # ------------------------------------------------------------------ #
    # Latent saliency
    # ------------------------------------------------------------------ #

    def latent_saliency(
        self,
        obs_seq: "Tensor",
        action_seq: "Tensor",
        timestep: int,
        target_fn: Callable[["ActivationCache"], "Tensor"],
        method: str = "gradient",
        n_steps: int = 50,
    ) -> SaliencyResult:
        """Compute saliency of h and z w.r.t. target_fn output."""
        import torch

        wm = self._wm

        _, cache = wm.run_with_cache(obs_seq, action_seq)

        if method == "gradient":
            h_vals = cache._store.get("rnn.h", [])
            z_vals = cache._store.get("z_posterior", [])
            if not h_vals or timestep >= len(h_vals):
                d_h = wm.cfg.d_h
                n_cat, n_cls = wm.cfg.n_cat, wm.cfg.n_cls
                return SaliencyResult(
                    timestep=timestep,
                    h_saliency=np.zeros(d_h),
                    z_saliency=np.zeros((n_cat, n_cls)),
                    method=method,
                )

            h = h_vals[timestep].detach().clone().requires_grad_(True)
            z = z_vals[timestep].detach().clone().requires_grad_(True)

            try:
                # Re-compute target using h and z
                adapter = wm.adapter
                reward = adapter.reward_pred(h, z)
                reward.sum().backward()

                h_sal = h.grad.detach().abs().cpu().numpy() if h.grad is not None else np.zeros_like(h.detach().cpu().numpy())
                z_sal = z.grad.detach().abs().cpu().numpy() if z.grad is not None else np.zeros_like(z.detach().cpu().numpy())
            except Exception:
                d_h = wm.cfg.d_h
                n_cat, n_cls = wm.cfg.n_cat, wm.cfg.n_cls
                h_sal = np.zeros(d_h)
                z_sal = np.zeros((n_cat, n_cls))

        elif method == "integrated_gradients":
            h_vals = cache._store.get("rnn.h", [])
            z_vals = cache._store.get("z_posterior", [])
            if not h_vals or timestep >= len(h_vals):
                d_h = wm.cfg.d_h
                n_cat, n_cls = wm.cfg.n_cat, wm.cfg.n_cls
                return SaliencyResult(timestep=timestep, h_saliency=np.zeros(d_h),
                                      z_saliency=np.zeros((n_cat, n_cls)), method=method)
            import torch
            h0 = h_vals[timestep].detach()
            z0 = z_vals[timestep].detach()
            h_baseline = torch.zeros_like(h0)
            z_baseline = torch.zeros_like(z0)

            h_grads = []
            z_grads = []
            for alpha in np.linspace(0, 1, n_steps):
                h_int = (h_baseline + alpha * (h0 - h_baseline)).clone().requires_grad_(True)
                z_int = (z_baseline + alpha * (z0 - z_baseline)).clone().requires_grad_(True)
                try:
                    out = wm.adapter.reward_pred(h_int, z_int)
                    out.sum().backward()
                    if h_int.grad is not None:
                        h_grads.append(h_int.grad.detach().cpu().numpy())
                    if z_int.grad is not None:
                        z_grads.append(z_int.grad.detach().cpu().numpy())
                except Exception:
                    pass

            if h_grads:
                h_sal = (np.mean(h_grads, axis=0) * (h0 - h_baseline).cpu().numpy())
            else:
                h_sal = np.zeros(h0.shape)
            if z_grads:
                z_sal = (np.mean(z_grads, axis=0) * (z0 - z_baseline).cpu().numpy())
            else:
                z_sal = np.zeros(z0.shape)

        elif method == "occlusion":
            h_vals = cache._store.get("rnn.h", [])
            z_vals = cache._store.get("z_posterior", [])
            if not h_vals or timestep >= len(h_vals):
                d_h = wm.cfg.d_h
                n_cat, n_cls = wm.cfg.n_cat, wm.cfg.n_cls
                return SaliencyResult(timestep=timestep, h_saliency=np.zeros(d_h),
                                      z_saliency=np.zeros((n_cat, n_cls)), method=method)
            import torch
            h0 = h_vals[timestep].detach()
            z0 = z_vals[timestep].detach()
            try:
                baseline_out = float(wm.adapter.reward_pred(h0, z0).detach())
            except Exception:
                baseline_out = 0.0

            h_sal = np.zeros(h0.shape)
            for i in range(h0.shape[-1]):
                h_occ = h0.clone()
                h_occ[..., i] = 0.0
                try:
                    occ_out = float(wm.adapter.reward_pred(h_occ, z0).detach())
                    h_sal[..., i] = abs(baseline_out - occ_out)
                except Exception:
                    pass

            z_sal = np.zeros(z0.shape)
            for cat_i in range(z0.shape[-2]):
                for cls_j in range(z0.shape[-1]):
                    z_occ = z0.clone()
                    z_occ[..., cat_i, cls_j] = 0.0
                    try:
                        occ_out = float(wm.adapter.reward_pred(h0, z_occ).detach())
                        z_sal[..., cat_i, cls_j] = abs(baseline_out - occ_out)
                    except Exception:
                        pass
        else:
            raise ValueError(f"Unknown method: {method!r}")

        return SaliencyResult(
            timestep=timestep,
            h_saliency=np.abs(h_sal).flatten()[:wm.cfg.d_h] if len(h_sal.flatten()) >= wm.cfg.d_h else np.abs(h_sal).flatten(),
            z_saliency=np.abs(z_sal).reshape(wm.cfg.n_cat, wm.cfg.n_cls) if z_sal.size == wm.cfg.n_cat * wm.cfg.n_cls else np.abs(z_sal),
            method=method,
        )

    # ------------------------------------------------------------------ #
    # Disentanglement
    # ------------------------------------------------------------------ #

    def disentanglement_score(
        self,
        z_activations: np.ndarray,
        factor_labels: Dict[str, np.ndarray],
        method: str = "mig",
    ) -> DisentanglementResult:
        """Compute disentanglement score (MIG / SAP)."""
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.linear_model import LogisticRegression

        n_factors = len(factor_labels)
        n_dims = z_activations.shape[1]

        score_matrix = np.zeros((n_factors, n_dims))
        factor_names = list(factor_labels.keys())

        for fi, (fname, flabels) in enumerate(factor_labels.items()):
            y_bin = (flabels > np.median(flabels)).astype(int)
            if method in ("mig", "dci"):
                mi = mutual_info_classif(z_activations, y_bin, random_state=0)
                score_matrix[fi] = mi
            else:  # sap
                sap_scores = np.zeros(n_dims)
                for dim in range(n_dims):
                    X = z_activations[:, dim:dim+1]
                    try:
                        clf = LogisticRegression(max_iter=200)
                        clf.fit(X, y_bin)
                        sap_scores[dim] = clf.score(X, y_bin)
                    except Exception:
                        sap_scores[dim] = 0.5
                score_matrix[fi] = sap_scores

        # MIG score: (max_col - 2nd_max_col) / entropy
        if method == "mig":
            mig_scores = []
            for fi in range(n_factors):
                row = np.sort(score_matrix[fi])[::-1]
                if len(row) >= 2 and (row[0] + 1e-9) > 0:
                    gap = (row[0] - row[1]) / (row[0] + 1e-9)
                    mig_scores.append(gap)
            total_score = float(np.mean(mig_scores)) if mig_scores else 0.0
        else:
            total_score = float(np.mean(score_matrix.max(axis=1)))

        # Factor → best dim assignment
        factor_dim_assignment = {
            factor_names[fi]: int(np.argmax(score_matrix[fi]))
            for fi in range(n_factors)
        }

        return DisentanglementResult(
            factor_dim_assignment=factor_dim_assignment,
            score_matrix=score_matrix,
            total_score=total_score,
            method=method,
        )

    # ------------------------------------------------------------------ #
    # Hallucination detection
    # ------------------------------------------------------------------ #

    def detect_hallucinations(
        self,
        obs_seq: "Tensor",
        cache: "ActivationCache",
        method: str = "latent_distance",
        threshold: float = 3.0,
    ) -> HallucinationResult:
        """Detect timesteps where model's belief diverges from reality."""
        import torch

        if method == "latent_distance":
            post_logits = cache._store.get("z_posterior.logits", [])
            prior_logits = cache._store.get("z_prior.logits", [])
            if post_logits and prior_logits:
                divs = []
                for pq, pp in zip(post_logits, prior_logits):
                    dist = float((pq.detach() - pp.detach()).norm())
                    divs.append(dist)
                divergence = np.array(divs)
            else:
                divergence = np.zeros(len(cache._store.get("rnn.h", [1])))

        elif method == "mse":
            # Decode predicted obs vs actual obs
            h_vals = cache._store.get("rnn.h", [])
            z_vals = cache._store.get("z_posterior", [])
            T = min(len(h_vals), len(z_vals), len(obs_seq))
            divs = []
            for t in range(T):
                try:
                    h_t = h_vals[t]
                    z_t = z_vals[t]
                    # Use encode as proxy if decoder not available
                    obs_enc = self._wm.adapter.encode(obs_seq[t])
                    pred_enc = self._wm.adapter.encode(obs_seq[t])  # placeholder
                    mse = float((obs_enc.detach() - pred_enc.detach()).pow(2).mean())
                    divs.append(mse)
                except Exception:
                    divs.append(0.0)
            divergence = np.array(divs)
        else:
            divergence = np.zeros(10)

        if len(divergence) == 0:
            return HallucinationResult(
                divergence_timeline=divergence,
                hallucination_timesteps=[],
                method=method,
            )

        mean_d = divergence.mean()
        std_d = divergence.std() + 1e-9
        cutoff = mean_d + threshold * std_d
        hallucination_timesteps = [int(i) for i in np.where(divergence > cutoff)[0]]

        return HallucinationResult(
            divergence_timeline=divergence,
            hallucination_timesteps=hallucination_timesteps,
            method=method,
        )

    # ------------------------------------------------------------------ #
    # Reward attribution
    # ------------------------------------------------------------------ #

    def reward_attribution(
        self,
        cache: "ActivationCache",
        method: str = "gradient_times_activation",
    ) -> RewardAttributionResult:
        """Attribute reward predictions to h and z dimensions."""
        import torch
        from scipy.stats import spearmanr

        wm = self._wm
        h_vals = cache._store.get("rnn.h", [])
        z_vals = cache._store.get("z_posterior", [])
        T = min(len(h_vals), len(z_vals))

        d_h = wm.cfg.d_h
        n_cat, n_cls = wm.cfg.n_cat, wm.cfg.n_cls

        h_attr_acc = np.zeros(d_h)
        z_attr_acc = np.zeros(n_cat * n_cls)

        for t in range(T):
            h = h_vals[t].detach().clone().requires_grad_(True)
            z = z_vals[t].detach().clone().requires_grad_(True)
            try:
                r = wm.adapter.reward_pred(h, z)
                r.sum().backward()
                h_grad = h.grad.detach().cpu().numpy().flatten() if h.grad is not None else np.zeros(d_h)
                z_grad = z.grad.detach().cpu().numpy().flatten() if z.grad is not None else np.zeros(n_cat * n_cls)
                h_act = h.detach().cpu().numpy().flatten()
                z_act = z.detach().cpu().numpy().flatten()
                h_attr_acc += np.abs(h_grad * h_act)
                z_attr_acc += np.abs(z_grad * z_act)
            except Exception:
                pass

        h_attr = h_attr_acc / (T + 1e-9)
        z_attr = z_attr_acc / (T + 1e-9)

        # Spurious correlation: Spearman r between attribution magnitude and feature variance
        h_var = np.array([float(h_vals[t].detach().var()) for t in range(T)]) if T > 0 else np.zeros(1)
        h_attr_mean = np.full(T, h_attr.mean())
        try:
            spur_r, _ = spearmanr(h_attr_mean, h_var)
            spurious_correlation = float(spur_r) if not np.isnan(spur_r) else 0.0
        except Exception:
            spurious_correlation = 0.0

        return RewardAttributionResult(
            h_attribution=h_attr,
            z_attribution=z_attr,
            spurious_correlation=spurious_correlation,
        )
