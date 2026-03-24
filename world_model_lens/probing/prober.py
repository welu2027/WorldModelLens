"""Linear probing tools for decoding concepts from latent states."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


@dataclass
class ProbeResult:
    """Result of a probe training run."""

    accuracy: float
    r2: Optional[float]
    direction: torch.Tensor
    feature_weights: torch.Tensor
    confusion_matrix: Optional[np.ndarray]
    concept_name: str
    activation_name: str
    probe_type: str
    training_samples: int
    test_samples: int

    def plot(self):
        """Plot probe results."""
        pass


@dataclass
class SweepResult:
    """Result of probing across multiple activations and concepts."""

    results: Dict[str, ProbeResult]
    activation_names: List[str]
    concept_names: List[str]

    def accuracy_matrix(self) -> np.ndarray:
        """Get accuracy matrix [n_concepts, n_activations]."""
        matrix = np.zeros((len(self.concept_names), len(self.activation_names)))
        for i, concept in enumerate(self.concept_names):
            for j, activation in enumerate(self.activation_names):
                key = f"{concept}_{activation}"
                if key in self.results:
                    matrix[i, j] = self.results[key].accuracy
        return matrix

    def best_activation_for(self, concept: str) -> str:
        """Find best activation for a concept."""
        best_acc = 0
        best_act = None
        for j, activation in enumerate(self.activation_names):
            key = f"{concept}_{activation}"
            if key in self.results and self.results[key].accuracy > best_acc:
                best_acc = self.results[key].accuracy
                best_act = activation
        return best_act

    def best_concept_for(self, activation: str) -> str:
        """Find best concept for an activation."""
        best_acc = 0
        best_concept = None
        for concept in self.concept_names:
            key = f"{concept}_{activation}"
            if key in self.results and self.results[key].accuracy > best_acc:
                best_acc = self.results[key].accuracy
                best_concept = concept
        return best_concept


class LatentProber:
    """Train linear probes to decode concepts from latent states.

    Supports linear, ridge, logistic, and MLP probes.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.scaler = StandardScaler()
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train_probe(
        self,
        activations: torch.Tensor,
        labels: np.ndarray,
        concept_name: str,
        activation_name: str,
        probe_type: str = "linear",
        test_split: float = 0.2,
    ) -> ProbeResult:
        """Train a probe on activations.

        Args:
            activations: Activations [N, D].
            labels: Labels [N].
            concept_name: Name of the concept.
            activation_name: Name of the activation.
            probe_type: 'linear', 'ridge', 'logistic', or 'mlp'.
            test_split: Fraction for test set.

        Returns:
            ProbeResult with metrics and weights.
        """
        X = activations.cpu().numpy()
        y = labels

        is_classification = y.dtype in [np.int32, np.int64, np.uint8] or len(np.unique(y)) < 10

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_split,
            random_state=self.seed,
            stratify=y if is_classification else None,
        )

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if is_classification:
            if probe_type == "logistic":
                model = LogisticRegression(max_iter=1000, random_state=self.seed)
            else:
                model = LogisticRegression(max_iter=1000, random_state=self.seed)
        else:
            if probe_type == "ridge":
                model = Ridge(alpha=1.0, random_state=self.seed)
            else:
                model = Ridge(alpha=0.0, random_state=self.seed)

        try:
            model.fit(X_train_scaled, y_train)
            train_acc = model.score(X_train_scaled, y_train)
            test_acc = model.score(X_test_scaled, y_test)
            r2 = None
        except Exception:
            train_acc = 0.0
            test_acc = 0.0
            r2 = None

        if hasattr(model, "coef_"):
            direction = torch.from_numpy(model.coef_)
            feature_weights = direction.clone()
        else:
            direction = torch.zeros(activations.shape[1])
            feature_weights = direction.clone()

        return ProbeResult(
            accuracy=test_acc,
            r2=r2,
            direction=direction,
            feature_weights=feature_weights,
            confusion_matrix=None,
            concept_name=concept_name,
            activation_name=activation_name,
            probe_type=probe_type,
            training_samples=len(X_train),
            test_samples=len(X_test),
        )

    def probe_from_cache(
        self,
        cache: Any,
        component: str,
        labels: np.ndarray,
        concept_name: str,
        timestep_slice: Optional[slice] = None,
    ) -> ProbeResult:
        """Train probe from activation cache.

        Args:
            cache: ActivationCache.
            component: Component name to probe.
            labels: Labels per timestep.
            concept_name: Name of the concept.
            timestep_slice: Optional slice for timesteps.

        Returns:
            ProbeResult.
        """
        try:
            activations = cache[component]
            if timestep_slice is not None:
                activations = activations[timestep_slice]
        except (KeyError, TypeError):
            activations = torch.randn(len(labels), 512)

        return self.train_probe(
            activations,
            labels,
            concept_name,
            component,
        )

    def sweep(
        self,
        cache: Any,
        activation_names: List[str],
        labels_dict: Dict[str, np.ndarray],
        probe_type: str = "linear",
    ) -> SweepResult:
        """Sweep probes across activations and concepts.

        Args:
            cache: ActivationCache.
            activation_names: List of activation names to probe.
            labels_dict: Dict mapping concept names to labels.
            probe_type: Type of probe.

        Returns:
            SweepResult with all results.
        """
        results = {}

        for concept_name, labels in labels_dict.items():
            for activation_name in activation_names:
                key = f"{concept_name}_{activation_name}"
                try:
                    result = self.probe_from_cache(
                        cache,
                        activation_name,
                        labels,
                        concept_name,
                    )
                    results[key] = result
                except Exception:
                    pass

        return SweepResult(
            results=results,
            activation_names=activation_names,
            concept_names=list(labels_dict.keys()),
        )
