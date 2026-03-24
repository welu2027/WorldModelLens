"""ModelHub — discover, pull, and push world model checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelCard:
    """Metadata for a world model checkpoint."""
    name: str
    backend: str  # "dreamerv3", "iris", etc.
    env: str      # "atari/pong", "dmc/walker_run", etc.
    description: str
    status: str   # "available" | "coming_soon"
    config: Optional[dict] = None


class ModelHub:
    """Discover, pull, and push world model checkpoints.
    
    Uses huggingface_hub under the hood. All network calls are lazy —
    nothing is downloaded until you call pull().
    """
    
    HUB_REPO = "world-model-lens/checkpoints"  # placeholder HF repo
    
    # 6 planned models, all coming_soon until real checkpoints exist
    _KNOWN_MODELS: List[ModelCard] = [
        ModelCard("dreamerv3-atari-pong", "dreamerv3", "atari/pong", 
                  "DreamerV3 trained on Atari Pong", "coming_soon"),
        ModelCard("dreamerv3-dmc-walker", "dreamerv3", "dmc/walker_run", 
                  "DreamerV3 on DMC Walker", "coming_soon"),
        ModelCard("dreamerv2-atari-breakout", "dreamerv2", "atari/breakout", 
                  "DreamerV2 on Breakout", "coming_soon"),
        ModelCard("iris-atari-pong", "iris", "atari/pong", 
                  "IRIS on Atari Pong", "coming_soon"),
        ModelCard("iris-atari-seaquest", "iris", "atari/seaquest", 
                  "IRIS on Atari Seaquest", "coming_soon"),
        ModelCard("tdmpc2-dmc-cheetah", "tdmpc2", "dmc/cheetah_run", 
                  "TD-MPC2 on DMC Cheetah", "coming_soon"),
    ]
    
    def __init__(self) -> None:
        """Initialize ModelHub."""
        self._models_by_name = {card.name: card for card in self._KNOWN_MODELS}
    
    def list_available(self) -> List[ModelCard]:
        """List all known models."""
        return list(self._KNOWN_MODELS)
    
    def pull(self, name: str, cache_dir: Optional[str] = None) -> str:
        """Download checkpoint to cache_dir. Returns local path.
        
        Args:
            name: Model name (e.g., "dreamerv3-atari-pong")
            cache_dir: Directory to cache the model. If None, uses default.
            
        Returns:
            Path to the downloaded checkpoint.
            
        Raises:
            ValueError: If model name is not found.
            RuntimeError: If model status is "coming_soon".
        """
        if name not in self._models_by_name:
            raise ValueError(f"Unknown model: {name}. Available: {list(self._models_by_name.keys())}")
        
        card = self._models_by_name[name]
        if card.status == "coming_soon":
            raise RuntimeError(f"Model '{name}' is not yet available (status: coming_soon)")
        
        # Try to use huggingface_hub if available
        try:
            from huggingface_hub import hf_hub_download
            
            repo_id = self.HUB_REPO
            filename = f"{name}/model.pt"
            
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
            )
            return local_path
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is not installed. Install it with: "
                "pip install huggingface_hub"
            )
    
    def push(self, name: str, checkpoint_path: str, **kwargs) -> None:
        """Upload checkpoint to HF hub (stub — requires HF credentials).
        
        Args:
            name: Model name.
            checkpoint_path: Path to the checkpoint file.
            **kwargs: Additional arguments (unused).
            
        Raises:
            NotImplementedError: Always, as this is not yet implemented.
        """
        raise NotImplementedError(
            "push not yet implemented; use huggingface_hub directly."
        )
    
    def info(self, name: str) -> ModelCard:
        """Get ModelCard for a named model.
        
        Args:
            name: Model name.
            
        Returns:
            ModelCard for the model.
            
        Raises:
            ValueError: If model name is not found.
        """
        if name not in self._models_by_name:
            raise ValueError(f"Unknown model: {name}. Available: {list(self._models_by_name.keys())}")
        
        return self._models_by_name[name]
