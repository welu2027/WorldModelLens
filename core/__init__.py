"""Core abstractions: model wrappers, activation caching, and base classes."""

from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.config import WorldModelConfig
from world_model_lens.core.hooks import HookContext, HookPoint, HookRegistry
from world_model_lens.core.latent_state import LatentState
from world_model_lens.core.latent_trajectory import LatentTrajectory

__all__ = [
    # Data containers
    "LatentState",
    "LatentTrajectory",
    # Activation storage
    "ActivationCache",
    # Hook system
    "HookPoint",
    "HookContext",
    "HookRegistry",
    # Configuration
    "WorldModelConfig",
]
