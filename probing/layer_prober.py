"""Layer-wise probing to find concept emergence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import matplotlib.axes
    from world_model_lens.core import ActivationCache
    from world_model_lens.probing.prober import LatentProber, ProbeResult


@dataclass
class LayerProbeResult:
    """Result of probing multiple layers."""

    concept: str
    """Name of the concept."""

    layer_results: Dict[str, "ProbeResult"]
    """Mapping from activation_name to ProbeResult."""

    emergence_layer: Optional[str]
    """First layer where accuracy exceeds emergence_threshold."""

    def plot_by_layer(self, ax=None) -> "matplotlib.axes.Axes":
        """Bar chart of accuracy per layer.

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
            fig, ax = plt.subplots(figsize=(12, 6))

        layer_names = list(self.layer_results.keys())
        accuracies = [self.layer_results[name].accuracy for name in layer_names]

        ax.bar(range(len(layer_names)), accuracies, color="steelblue")
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Layer / Activation")
        ax.set_title(f"Layer-wise Probe Performance: {self.concept}")
        ax.set_ylim(0, 1)

        if self.emergence_layer is not None:
            emergence_idx = list(self.layer_results.keys()).index(self.emergence_layer)
            ax.axvline(emergence_idx - 0.5, color="red", linestyle="--", label=f"Emergence at {self.emergence_layer}")
            ax.legend()

        return ax


class LayerProber:
    """Probe multiple layers/activations to find where a concept emerges."""

    def __init__(self, prober: "LatentProber") -> None:
        """Initialize the layer prober.

        Parameters
        ----------
        prober:
            A LatentProber instance to use for training individual probes.
        """
        self.prober = prober

    def probe_by_layer(
        self,
        cache: "ActivationCache",
        labels,
        concept: str = "unknown",
        activation_names: Optional[List[str]] = None,
        emergence_threshold: float = 0.7,
    ) -> LayerProbeResult:
        """Run probe on each activation in cache. Find emergence_layer.

        Parameters
        ----------
        cache:
            ActivationCache containing the activations.
        labels:
            Array of labels, shape (T,).
        concept:
            Name of the concept being probed.
        activation_names:
            List of activation names to probe. If None, uses all in cache.
        emergence_threshold:
            Accuracy threshold for concept emergence (default 0.7).

        Returns
        -------
        LayerProbeResult
            Result with layer_results and emergence_layer.
        """
        # If no activation names provided, use all from cache
        if activation_names is None:
            activation_names = cache.component_names

        layer_results = {}
        emergence_layer = None

        for activation_name in activation_names:
            try:
                result = self.prober.probe_from_cache(
                    cache=cache,
                    labels=labels,
                    activation_name=activation_name,
                    concept=concept,
                )
                layer_results[activation_name] = result

                # Check for emergence
                if emergence_layer is None and result.accuracy >= emergence_threshold:
                    emergence_layer = activation_name

            except Exception as e:
                print(f"Failed to probe layer {activation_name}: {e}")
                continue

        return LayerProbeResult(
            concept=concept,
            layer_results=layer_results,
            emergence_layer=emergence_layer,
        )
