"""Probing tools for latent representations.

This package provides analysis tools that work with both RL and non-RL world models:

- LatentProber: Train probes on latent representations
- GeometryAnalyzer: Analyze latent space geometry and structure
- TemporalMemoryProber: Analyze temporal dynamics and memory retention
"""

try:
    from world_model_lens.probing.prober import LatentProber
except ImportError:
    LatentProber = None

from world_model_lens.probing.geometry import GeometryAnalyzer
from world_model_lens.probing.temporal_memory import TemporalMemoryProber

__all__ = [
    "LatentProber",
    "GeometryAnalyzer",
    "TemporalMemoryProber",
]
