"""Linear probing of latent activations to decode concepts.

LatentProber trains linear classifiers/regressors on cached activations
to determine what information is linearly decodable from the latent state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd
    from world_model_lens.core import ActivationCache, HookedWorldModel


@dataclass
class ProbeResult:
    """Result of training a linear probe on a single activation."""

    concept: str
    """Name of the concept being probed."""

    activation_name: str
    """Which activation was probed (e.g., 'rnn.h', 'z_posterior')."""

    accuracy: float
    """Classification accuracy (0-1) or R² score for regression."""

    r2: float
    """R² score. Same as accuracy for regression, from validation for classification."""

    direction: Optional[np.ndarray]
    """Unit-norm probe weight vector, shape (d_activation,). None for some models."""

    feature_weights: Optional[np.ndarray]
    """Per-feature weights from the linear model, shape (d_activation,)."""

    confusion_matrix: Optional[np.ndarray]
    """Confusion matrix for classification, shape (n_classes, n_classes)."""

    is_regression: bool
    """Whether this was a regression (True) or classification (False) task."""

    def plot(self, ax=None) -> "matplotlib.axes.Axes":
        """Plot confusion matrix (classification) or scatter plot (regression).

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
            fig, ax = plt.subplots()

        if self.is_regression:
            # For regression, we can't plot much without the actual predictions/labels
            ax.text(
                0.5,
                0.5,
                f"R² = {self.r2:.3f}\n{self.activation_name}\n{self.concept}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"Regression: {self.concept}")
        else:
            # Plot confusion matrix
            if self.confusion_matrix is not None:
                im = ax.imshow(self.confusion_matrix, cmap="Blues")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title(
                    f"Confusion Matrix: {self.concept}\nAccuracy={self.accuracy:.3f}"
                )
                plt.colorbar(im, ax=ax)
                n_classes = self.confusion_matrix.shape[0]
                for i in range(n_classes):
                    for j in range(n_classes):
                        ax.text(
                            j,
                            i,
                            f"{self.confusion_matrix[i, j]:.0f}",
                            ha="center",
                            va="center",
                            color="white"
                            if self.confusion_matrix[i, j]
                            > self.confusion_matrix.max() / 2
                            else "black",
                        )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Accuracy = {self.accuracy:.3f}\n{self.activation_name}\n{self.concept}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Classification: {self.concept}")

        return ax


@dataclass
class SweepResult:
    """Result of probing multiple activations."""

    concept: str
    """Name of the concept."""

    results: Dict[str, ProbeResult]
    """Mapping from activation_name to ProbeResult."""

    @property
    def accuracy_series(self) -> "pd.Series":
        """Return pandas Series of accuracy values indexed by activation_name.

        Returns
        -------
        pandas.Series
            Series with activation names as index and accuracy values.
        """
        import pandas as pd

        return pd.Series(
            {name: result.accuracy for name, result in self.results.items()}
        )

    def best_activation(self) -> str:
        """Return activation_name with highest accuracy.

        Returns
        -------
        str
            Name of the best-performing activation.
        """
        return max(self.results.keys(), key=lambda k: self.results[k].accuracy)

    def heatmap(self, ax=None) -> "matplotlib.axes.Axes":
        """Plot horizontal bar chart of accuracy per activation, sorted descending.

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

        series = self.accuracy_series.sort_values(ascending=True)
        series.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Activation")
        ax.set_title(f"Probe Performance: {self.concept}")
        ax.set_xlim(0, 1)

        return ax


class LatentProber:
    """Train linear probes on cached activations to test concept decodability."""

    def __init__(self, wm: "HookedWorldModel") -> None:
        """Initialize the prober.

        Parameters
        ----------
        wm:
            The HookedWorldModel instance (used for reference, not directly needed).
        """
        self.wm = wm

    def train_probe(
        self,
        activation_name: str,
        labels: np.ndarray,
        activations: np.ndarray,
        concept: str = "unknown",
        test_size: float = 0.2,
        max_iter: int = 1000,
        is_regression: bool = False,
    ) -> ProbeResult:
        """Train a linear probe (LogisticRegression or Ridge) with StandardScaler.

        Auto-detects regression vs classification:
        - If labels are float dtype OR is_regression=True → Ridge regression
        - Otherwise → LogisticRegression with max_iter=max_iter

        Parameters
        ----------
        activation_name:
            Name of the activation being probed (e.g., 'rnn.h').
        labels:
            Array of labels, shape (T,). Can be floats (for regression) or integers/strings.
        activations:
            Array of activations, shape (T, d). Will be scaled with StandardScaler.
        concept:
            Name of the concept being probed.
        test_size:
            Fraction of data to use for testing (default 0.2).
        max_iter:
            Maximum iterations for LogisticRegression.
        is_regression:
            If True, force regression. If False, auto-detect from labels dtype.

        Returns
        -------
        ProbeResult
            Probe result with accuracy/r2, direction, feature_weights, confusion_matrix.
        """
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import (
            confusion_matrix,
            r2_score,
            accuracy_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Determine if regression or classification
        if not is_regression and labels.dtype.kind not in "fc":  # not float/complex
            is_regression = False
        elif is_regression or labels.dtype.kind in "fc":  # float/complex
            is_regression = True

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            activations, labels, test_size=test_size, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        if is_regression:
            model = Ridge(max_iter=max_iter)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = r2_score(y_test, y_pred)
            r2 = accuracy
            confusion_mat = None
            feature_weights = model.coef_
        else:
            model = LogisticRegression(max_iter=max_iter, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            r2 = accuracy  # For classification, use accuracy as r2 proxy
            confusion_mat = confusion_matrix(y_test, y_pred)
            feature_weights = model.coef_[0] if model.coef_.shape[0] == 1 else model.coef_.mean(axis=0)

        # Compute direction (unit-norm weights)
        if feature_weights is not None:
            direction = feature_weights / (np.linalg.norm(feature_weights) + 1e-8)
        else:
            direction = None

        return ProbeResult(
            concept=concept,
            activation_name=activation_name,
            accuracy=float(accuracy),
            r2=float(r2),
            direction=direction,
            feature_weights=feature_weights,
            confusion_matrix=confusion_mat,
            is_regression=is_regression,
        )

    def probe_from_cache(
        self,
        cache: "ActivationCache",
        labels: np.ndarray,
        activation_name: str,
        concept: str = "unknown",
        **kwargs,
    ) -> ProbeResult:
        """Extract activations from cache and train probe.

        Parameters
        ----------
        cache:
            ActivationCache containing the activations.
        labels:
            Array of labels, shape (T,).
        activation_name:
            Name of the activation in the cache (e.g., 'rnn.h').
        concept:
            Name of the concept being probed.
        **kwargs:
            Additional arguments passed to train_probe.

        Returns
        -------
        ProbeResult
            Probe result.
        """
        # Extract activations from cache
        activations_tensor = cache[activation_name]
        activations = activations_tensor.detach().cpu().numpy()

        # Flatten spatial dimensions if needed
        if activations.ndim > 2:
            T = activations.shape[0]
            activations = activations.reshape(T, -1)

        return self.train_probe(
            activation_name=activation_name,
            labels=labels,
            activations=activations,
            concept=concept,
            **kwargs,
        )

    def sweep(
        self,
        cache: "ActivationCache",
        labels: np.ndarray,
        activation_names: List[str],
        concept: str = "unknown",
        **kwargs,
    ) -> SweepResult:
        """Run probe_from_cache for each activation_name.

        Parameters
        ----------
        cache:
            ActivationCache containing the activations.
        labels:
            Array of labels, shape (T,).
        activation_names:
            List of activation names to probe.
        concept:
            Name of the concept being probed.
        **kwargs:
            Additional arguments passed to probe_from_cache.

        Returns
        -------
        SweepResult
            Result containing probes for all activations.
        """
        results = {}
        for activation_name in activation_names:
            try:
                result = self.probe_from_cache(
                    cache=cache,
                    labels=labels,
                    activation_name=activation_name,
                    concept=concept,
                    **kwargs,
                )
                results[activation_name] = result
            except Exception as e:
                print(f"Failed to probe {activation_name}: {e}")
                continue

        return SweepResult(concept=concept, results=results)
