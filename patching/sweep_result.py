"""Results container for patching sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PatchingSweepResult:
    """Results of a full patching sweep over timesteps × activations.

    Attributes
    ----------
    metric_name:
        Name of the metric that was measured.
    recovery_matrix:
        2D array of shape (n_timesteps, n_activations) containing recovery rates.
        Rows are timesteps, columns are activations. NaN indicates the activation
        was not available at that timestep.
    activation_names:
        List of activation names (column headers), length n_activations.
    timesteps:
        List of timestep indices (row headers), length n_timesteps.
    """

    metric_name: str
    recovery_matrix: np.ndarray  # (n_timesteps, n_activations)
    activation_names: List[str]
    timesteps: List[int]

    def heatmap(self, ax=None, cmap: str = "RdYlGn", vmin: float = -1.0, vmax: float = 1.0):
        """Plot recovery_rate as a 2D heatmap.

        Parameters
        ----------
        ax:
            Matplotlib axes object. If None, a new figure is created.
        cmap:
            Colormap name (default "RdYlGn").
        vmin, vmax:
            Colorbar limits.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the heatmap.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for heatmap visualization. "
                "Install it with: pip install matplotlib"
            ) from e

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, max(4, len(self.timesteps) // 2)))

        # Create heatmap
        im = ax.imshow(
            self.recovery_matrix,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
        )

        # Set ticks and labels
        ax.set_xticks(np.arange(len(self.activation_names)))
        ax.set_yticks(np.arange(len(self.timesteps)))
        ax.set_xticklabels(self.activation_names, rotation=45, ha="right")
        ax.set_yticklabels(self.timesteps)

        # Labels and title
        ax.set_xlabel("Activation", fontsize=12)
        ax.set_ylabel("Timestep", fontsize=12)
        ax.set_title(f"Recovery Rates ({self.metric_name})", fontsize=14, fontweight="bold")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Recovery Rate", fontsize=11)

        return ax

    def top_k_patches(self, k: int = 10) -> List[Tuple[int, str, float]]:
        """Return top-k (timestep, activation, recovery_rate) sorted descending.

        Parameters
        ----------
        k:
            Number of top patches to return.

        Returns
        -------
        list of (int, str, float)
            Each tuple is (timestep, activation_name, recovery_rate), sorted
            by recovery_rate in descending order (highest first).
        """
        # Flatten the matrix with indices
        flat_indices = np.ndindex(self.recovery_matrix.shape)
        results = []

        for i, j in flat_indices:
            val = self.recovery_matrix[i, j]
            if not np.isnan(val):
                timestep = self.timesteps[i]
                activation = self.activation_names[j]
                results.append((timestep, activation, float(val)))

        # Sort by recovery rate descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]

    def most_important_activation(self) -> str:
        """Return the activation with highest mean |recovery_rate| across all timesteps.

        Returns
        -------
        str
            Name of the most important activation.
        """
        mean_abs_recovery = np.nanmean(np.abs(self.recovery_matrix), axis=0)
        best_idx = np.nanargmax(mean_abs_recovery)
        return self.activation_names[best_idx]

    def most_important_timestep(self) -> int:
        """Return the timestep with highest mean |recovery_rate| across all activations.

        Returns
        -------
        int
            The timestep index.
        """
        mean_abs_recovery = np.nanmean(np.abs(self.recovery_matrix), axis=1)
        best_idx = np.nanargmax(mean_abs_recovery)
        return self.timesteps[best_idx]

    def summary(self) -> str:
        """Return a text summary of the sweep results.

        Returns
        -------
        str
            Formatted summary including matrix shape, top patches, and key timesteps.
        """
        lines = [
            f"PatchingSweepResult ({self.metric_name})",
            f"  Shape: {self.recovery_matrix.shape} (timesteps × activations)",
            f"  Timesteps: {self.timesteps[0]} to {self.timesteps[-1]} ({len(self.timesteps)} steps)",
            f"  Activations: {len(self.activation_names)} components",
            "",
            "Top 5 patches by recovery rate:",
        ]

        top_5 = self.top_k_patches(k=5)
        for i, (t, act, rec) in enumerate(top_5, 1):
            lines.append(f"  {i}. [t={t:2d}, {act:20s}] → {rec:+.4f}")

        lines.extend([
            "",
            f"Most important activation: {self.most_important_activation()}",
            f"Most important timestep: {self.most_important_timestep()}",
        ])

        return "\n".join(lines)
