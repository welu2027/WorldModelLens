"""Probing module for analyzing latent representations.

Provides tools for:
- Training linear probes to decode concepts from activations
- Analyzing activation space geometry
- Studying temporal memory in latent states
- Finding where concepts emerge across layers
"""

from __future__ import annotations

from world_model_lens.probing.geometry import GeometryAnalyzer, GeometryResult
from world_model_lens.probing.layer_prober import LayerProber, LayerProbeResult
from world_model_lens.probing.prober import LatentProber, ProbeResult, SweepResult
from world_model_lens.probing.temporal_memory import (
    TemporalMemoryProber,
    TemporalMemoryResult,
)

__all__ = [
    "LatentProber",
    "ProbeResult",
    "SweepResult",
    "GeometryAnalyzer",
    "GeometryResult",
    "TemporalMemoryProber",
    "TemporalMemoryResult",
    "LayerProber",
    "LayerProbeResult",
]
