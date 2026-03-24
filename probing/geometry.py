"""Geometry analysis of activation spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

if TYPE_CHECKING:
    import matplotlib.axes
    from world_model_lens.core import ActivationCache, HookedWorldModel


@dataclass
class GeometryResult:
    """Geometry metrics for an activation space."""

    activation_name: str
    """Name of the activation."""

    cosine_sim_matrix: np.ndarray
    """Pairwise cosine similarities, shape (T, T)."""

    isotropy_score: float
    """Isotropy score: 0=anisotropic, 1=isotropic."""

    spatial_correspondence: Optional[float]
    """Spearman correlation between latent and observation distances."""

    pca_variance_explained: np.ndarray
    """Cumulative variance explained by PCA components."""

    def plot_2d_embedding(
        self,
        method: str = "pca",
        color_by: Optional[np.ndarray] = None,
        ax=None,
    ) -> "matplotlib.axes.Axes":
        """2D PCA or UMAP embedding colored by color_by values.

        Parameters
        ----------
        method:
            "pca" or "umap" for the embedding method.
        color_by:
            Optional array of shape (T,) to color the points.
        ax:
            Matplotlib axes. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # Get embedding (we only have cosine similarities, so we'll use MDS approximation)
        # For PCA, we'd need the original activations, which we don't have
        # So we'll use a simple message instead
        if method == "pca":
            ax.text(
                0.5,
                0.5,
                f"2D embedding not available\n(original activations needed for {method})",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        elif method == "umap":
            try:
                import umap

                # Convert cosine similarities to distances
                distances = 1 - self.cosine_sim_matrix
                embedding = umap.UMAP(
                    n_components=2,
                    metric="precomputed",
                    random_state=42,
                ).fit_transform(distances)

                if color_by is not None:
                    scatter = ax.scatter(
                        embedding[:, 0],
                        embedding[:, 1],
                        c=color_by,
                        cmap="viridis",
                        s=20,
                    )
                    plt.colorbar(scatter, ax=ax)
                else:
                    ax.scatter(embedding[:, 0], embedding[:, 1], s=20)

                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                ax.set_title(f"UMAP embedding: {self.activation_name}")
            except ImportError:
                ax.text(
                    0.5,
                    0.5,
                    "UMAP not installed\nInstall with: pip install umap-learn",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            raise ValueError(f"Unknown method: {method}")

        return ax


class GeometryAnalyzer:
    """Analyze the geometry of activation spaces."""

    def __init__(self, wm: "HookedWorldModel") -> None:
        """Initialize the analyzer.

        Parameters
        ----------
        wm:
            The HookedWorldModel instance.
        """
        self.wm = wm

    def analyze(
        self,
        activations: np.ndarray,
        activation_name: str = "rnn.h",
        obs_distances: Optional[np.ndarray] = None,
    ) -> GeometryResult:
        """Compute geometry metrics.

        Parameters
        ----------
        activations:
            Array of activations, shape (T, d).
        activation_name:
            Name of the activation.
        obs_distances:
            Optional array of pairwise observation distances, shape (T, T).

        Returns
        -------
        GeometryResult
            Geometry metrics.
        """
        T, d = activations.shape

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(activations, axis=1, keepdims=True)
        normalized = activations / (norms + 1e-8)
        cosine_sim = np.dot(normalized, normalized.T)

        # Compute isotropy score as ratio of singular values
        # Center the activations
        activations_centered = activations - activations.mean(axis=0)
        U, S, Vt = np.linalg.svd(activations_centered, full_matrices=False)
        isotropy_score = float(S[-1] / (S[0] + 1e-8))  # min/max singular value

        # Compute PCA variance explained
        variance = (S**2) / (S**2).sum()
        pca_variance_explained = np.cumsum(variance)

        # Compute spatial correspondence if obs_distances provided
        spatial_correspondence = None
        if obs_distances is not None:
            # Convert cosine similarities to distances
            cosine_dist = 1 - cosine_sim
            # Get upper triangle indices
            triu_idx = np.triu_indices(T, k=1)
            cosine_dist_vec = cosine_dist[triu_idx]
            obs_dist_vec = obs_distances[triu_idx]
            # Compute Spearman correlation
            if len(cosine_dist_vec) > 1:
                corr, _ = spearmanr(cosine_dist_vec, obs_dist_vec)
                spatial_correspondence = float(corr)

        return GeometryResult(
            activation_name=activation_name,
            cosine_sim_matrix=cosine_sim,
            isotropy_score=isotropy_score,
            spatial_correspondence=spatial_correspondence,
            pca_variance_explained=pca_variance_explained,
        )

    def from_cache(
        self,
        cache: "ActivationCache",
        activation_name: str = "rnn.h",
        **kwargs,
    ) -> GeometryResult:
        """Extract activations from cache and run analyze().

        Parameters
        ----------
        cache:
            ActivationCache containing the activations.
        activation_name:
            Name of the activation to analyze.
        **kwargs:
            Additional arguments passed to analyze().

        Returns
        -------
        GeometryResult
            Geometry metrics.
        """
        # Extract activations from cache
        activations_tensor = cache[activation_name]
        activations = activations_tensor.detach().cpu().numpy()

        # Flatten spatial dimensions if needed
        if activations.ndim > 2:
            T = activations.shape[0]
            activations = activations.reshape(T, -1)

        return self.analyze(
            activations=activations,
            activation_name=activation_name,
            **kwargs,
        )
