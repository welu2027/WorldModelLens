"""Model hub for downloading and managing pretrained models."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a pretrained model."""

    name: str
    backend: str
    environment: str
    coming_soon: bool = True
    description: str = ""
    url: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class ModelHub:
    """Hub for downloading and managing pretrained world models.

    Example:
        hub = ModelHub()
        models = hub.list_available()
        wm = hub.pull("dreamerv3-atari-breakout")
    """

    _MODELS = {
        "dreamerv3-atari-breakout": ModelInfo(
            name="dreamerv3-atari-breakout",
            backend="dreamerv3",
            environment="Atari/Breakout",
            coming_soon=True,
            description="DreamerV3 trained on Atari Breakout",
        ),
        "dreamerv3-atari-pong": ModelInfo(
            name="dreamerv3-atari-pong",
            backend="dreamerv3",
            environment="Atari/Pong",
            coming_soon=True,
            description="DreamerV3 trained on Atari Pong",
        ),
        "dreamerv3-dmcontrol-cheetah-run": ModelInfo(
            name="dreamerv3-dmcontrol-cheetah-run",
            backend="dreamerv3",
            environment="DeepMind Control/Cheetah-run",
            coming_soon=True,
            description="DreamerV3 trained on DM Control Cheetah Run",
        ),
        "dreamerv3-dmcontrol-walker-walk": ModelInfo(
            name="dreamerv3-dmcontrol-walker-walk",
            backend="dreamerv3",
            environment="DeepMind Control/Walker-walk",
            coming_soon=True,
            description="DreamerV3 trained on DM Control Walker Walk",
        ),
        "iris-atari-breakout": ModelInfo(
            name="iris-atari-breakout",
            backend="iris",
            environment="Atari/Breakout",
            coming_soon=True,
            description="IRIS trained on Atari Breakout",
        ),
        "tdmpc2-dmcontrol-humanoid": ModelInfo(
            name="tdmpc2-dmcontrol-humanoid",
            backend="tdmpc2",
            environment="DeepMind Control/Humanoid",
            coming_soon=True,
            description="TD-MPC2 trained on DM Control Humanoid",
        ),
    }

    @classmethod
    def list_available(cls) -> List[ModelInfo]:
        """List all available pretrained models."""
        return list(cls._MODELS.values())

    @classmethod
    def pull(cls, name: str, download_dir: Optional[str] = None) -> str:
        """Download/pull a pretrained model.

        Args:
            name: Model name.
            download_dir: Optional directory to save the model.

        Returns:
            Path to the downloaded model.

        Raises:
            NotImplementedError: Model downloads not yet implemented.
        """
        if name not in cls._MODELS:
            available = list(cls._MODELS.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")

        model_info = cls._MODELS[name]
        if model_info.coming_soon:
            raise NotImplementedError(
                f"Model '{name}' is not yet available for download. "
                f"This model is marked as coming_soon."
            )

        raise NotImplementedError(
            f"Model download for '{name}' not yet implemented. Check back soon!"
        )

    @classmethod
    def push(cls, model: Any, name: str, environment: str, backend: str) -> None:
        """Upload/push a model to the hub.

        Args:
            model: World model to upload.
            name: Name for the model.
            environment: Environment the model was trained on.
            backend: Backend architecture.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError(
            "Model upload not yet implemented. Please open an issue or PR to contribute models."
        )
