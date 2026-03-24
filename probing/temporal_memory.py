"""Temporal memory probing to analyze information retention over time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    import matplotlib.axes
    from world_model_lens.core import ActivationCache
    from world_model_lens.probing.prober import LatentProber


@dataclass
class TemporalMemoryResult:
    """Result of temporal memory analysis."""

    activation_name: str
    """Name of the activation."""

    concept: str
    """Name of the concept."""

    lag_accuracies: Dict[int, float]
    """Mapping from lag to accuracy at that lag."""

    half_life: Optional[float]
    """Steps until accuracy drops to 50% of peak."""

    memory_type: str
    """Type of memory: 'transient', 'episodic', or 'persistent'."""

    def decay_curve(self, ax=None) -> "matplotlib.axes.Axes":
        """Plot accuracy vs lag with half-life marked.

        Parameters
        ----------
        ax:
            Matplotlib axes. If None, creates a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        lags = sorted(self.lag_accuracies.keys())
        accuracies = [self.lag_accuracies[lag] for lag in lags]

        ax.plot(lags, accuracies, "o-", label="Accuracy", linewidth=2, markersize=8)

        if self.half_life is not None:
            ax.axvline(self.half_life, color="red", linestyle="--", label=f"Half-life={self.half_life:.2f}")

        ax.set_xlabel("Lag (timesteps)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Memory Decay: {self.concept} ({self.memory_type})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


class TemporalMemoryProber:
    """Probe how long information is retained in the latent state."""

    def __init__(self, prober: "LatentProber") -> None:
        """Initialize the temporal memory prober.

        Parameters
        ----------
        prober:
            A LatentProber instance to use for training individual probes.
        """
        self.prober = prober

    def probe_temporal_memory(
        self,
        cache: "ActivationCache",
        labels: np.ndarray,
        activation_name: str = "rnn.h",
        concept: str = "unknown",
        lags: Optional[List[int]] = None,
    ) -> TemporalMemoryResult:
        """For each lag L, train a probe to predict labels[t] from activation[t-L].

        Parameters
        ----------
        cache:
            ActivationCache containing the activations.
        labels:
            Array of labels, shape (T,).
        activation_name:
            Name of the activation to probe.
        concept:
            Name of the concept being probed.
        lags:
            List of lags to test. Default [0, 1, 2, 4, 8, 16].

        Returns
        -------
        TemporalMemoryResult
            Result with lag_accuracies, half_life, and memory_type.
        """
        if lags is None:
            lags = [0, 1, 2, 4, 8, 16]

        # Extract activations from cache
        activations_tensor = cache[activation_name]
        activations = activations_tensor.detach().cpu().numpy()

        # Flatten spatial dimensions if needed
        if activations.ndim > 2:
            T = activations.shape[0]
            activations = activations.reshape(T, -1)
        else:
            T = activations.shape[0]

        lag_accuracies = {}

        for lag in lags:
            if lag >= T:
                continue

            # Create lagged activations and labels
            # For lag L: predict labels[L:] from activations[:-L]
            if lag == 0:
                lagged_activations = activations
                lagged_labels = labels
            else:
                lagged_activations = activations[:-lag]
                lagged_labels = labels[lag:]

            # Train probe
            try:
                result = self.prober.train_probe(
                    activation_name=activation_name,
                    labels=lagged_labels,
                    activations=lagged_activations,
                    concept=concept,
                )
                lag_accuracies[lag] = result.accuracy
            except Exception as e:
                print(f"Failed to probe lag {lag}: {e}")
                continue

        # Fit exponential decay if we have enough data points
        half_life = None
        memory_type = "transient"

        if len(lag_accuracies) >= 2:
            lags_sorted = np.array(sorted(lag_accuracies.keys()))
            accuracies = np.array([lag_accuracies[lag] for lag in lags_sorted])

            # Fit: accuracy ~ A * exp(-lag / tau) + baseline
            def decay_func(x, A, tau, baseline):
                return A * np.exp(-x / (tau + 1e-8)) + baseline

            try:
                # Initial guess
                popt, _ = curve_fit(
                    decay_func,
                    lags_sorted,
                    accuracies,
                    p0=[accuracies[0] - accuracies[-1], 5.0, accuracies[-1]],
                    maxfev=10000,
                )
                A, tau, baseline = popt

                # Compute half-life: time until accuracy drops to 50% of peak
                peak_accuracy = accuracies[0]
                if peak_accuracy > baseline:
                    target_accuracy = 0.5 * (peak_accuracy - baseline) + baseline
                    if A > 0 and tau > 0:
                        half_life = tau * np.log(A / (target_accuracy - baseline + 1e-8))
                        if half_life > 0:
                            half_life = float(half_life)

                # Classify memory type based on half-life
                if half_life is not None:
                    if half_life < 3:
                        memory_type = "transient"
                    elif half_life < 20:
                        memory_type = "episodic"
                    else:
                        memory_type = "persistent"
            except Exception:
                # If fitting fails, classify based on accuracy decay
                if len(lag_accuracies) > 1:
                    max_lag = max(lag_accuracies.keys())
                    if lag_accuracies[max_lag] < 0.5 * lag_accuracies[0]:
                        memory_type = "transient"
                    else:
                        memory_type = "episodic"

        return TemporalMemoryResult(
            activation_name=activation_name,
            concept=concept,
            lag_accuracies=lag_accuracies,
            half_life=half_life,
            memory_type=memory_type,
        )
