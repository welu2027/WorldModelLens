"""Robotics/embodied world model adapter.

Reference: "Rocketship: Efficient and Safe Real-Time Reinforcement Learning"
          "Latent World Models for Robotic Manipulation"
          "VC-1: Generalist Agents"

Robotics world models for embodied agents:
- Latent dynamics (RSSM, JEPA, etc.)
- Visual-tactile-proprioceptive fusion
- Manipulation and locomotion planning
- Part-centric representations

This adapter supports:
- General robotics manipulation models
- Mobile manipulation models
- Bimanual manipulation models
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_lens.backends.generic_adapter import WorldModelAdapter, WorldModelConfig
from world_model_lens.core.types import LatentType, DynamicsType, WorldModelFamily


class RoboticsEncoder(nn.Module):
    """Multi-modal encoder for robotics (image, proprio, tactile)."""

    def __init__(
        self,
        image_channels: int = 3,
        proprio_dim: int = 14,
        tactile_dim: int = 0,
        d_latent: int = 256,
    ):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        if tactile_dim > 0:
            self.tactile_encoder = nn.Sequential(
                nn.Linear(tactile_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            tactile_out = 32
        else:
            self.tactile_encoder = None
            tactile_out = 0

        total_dim = 128 * 16 + 64 + tactile_out
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, d_latent),
            nn.ReLU(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(
        self,
        image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        tactile: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        img_features = self.image_encoder(image).flatten(1)

        if proprio is not None:
            proprio_features = self.proprio_encoder(proprio)
        else:
            proprio_features = torch.zeros(img_features.shape[0], 64, device=img_features.device)

        if tactile is not None and self.tactile_encoder is not None:
            tactile_features = self.tactile_encoder(tactile)
        else:
            tactile_features = torch.zeros(img_features.shape[0], 32, device=img_features.device)

        fused = torch.cat([img_features, proprio_features, tactile_features], dim=-1)
        return self.fusion(fused)


class RoboticsDynamics(nn.Module):
    """Latent dynamics for robotics."""

    def __init__(self, d_latent: int, d_action: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + d_action, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_latent),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class RoboticsAdapter(WorldModelAdapter):
    """Adapter for Robotics/Embodied world models.

    These models handle:
    - Multi-modal perception (image, proprioception, tactile)
    - Latent dynamics for manipulation/locomotion
    - Object-centric and part-centric representations
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        self.config = config

        self.encoder = RoboticsEncoder(
            image_channels=3,
            proprio_dim=config.d_action + 7,
            tactile_dim=0,
            d_latent=config.d_h,
        )
        self.dynamics = RoboticsDynamics(config.d_h, config.d_action)
        self.decoder = nn.Sequential(
            nn.Linear(config.d_h, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 4 * 4),
        )
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
        )
        self.reward_predictor = nn.Linear(config.d_h, 1)

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return [
            "image_encoder",
            "proprio_encoder",
            "fusion",
            "dynamics",
            "decoder",
            "latent_state",
        ]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.ROBOTICS

    def encode(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode robotics observation to latent state.

        Args:
            observation: Dict with 'image', 'proprio', 'tactile' keys, or image tensor
            context: Optional previous state

        Returns:
            Tuple of (latent state, observation encoding)
        """
        if isinstance(observation, dict):
            image = observation.get("image", observation.get("rgb"))
            proprio = observation.get("proprio")
            tactile = observation.get("tactile")
            obs_encoding = self.encoder(image, proprio, tactile)
        else:
            if observation.dim() == 5:
                B, T, C, H, W = observation.shape
                observation = observation.view(B * T, C, H, W)
                obs_encoding = self.encoder(observation)
                obs_encoding = obs_encoding.view(B, T, -1).mean(dim=1)
            else:
                obs_encoding = self.encoder(observation)

        return obs_encoding, obs_encoding

    def dynamics(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict next latent state."""
        if action is None:
            action = torch.zeros(state.shape[0], self.config.d_action, device=state.device)
        return self.dynamics(state, action)

    def transition(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        input_: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """State transition."""
        return self.dynamics(state, action)

    def decode(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Decode latent to image (optional)."""
        x = self.decoder(state)
        x = x.view(-1, 64, 4, 4)
        return self.image_decoder(x)

    def predict_reward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Predict reward."""
        return self.reward_predictor(state)

    def actor_forward(
        self,
        state: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict action (optional)."""
        return None

    def sample_state(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        """Passthrough for continuous latents."""
        return logits

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Initialize starting state."""
        if device is None:
            device = self._device
        return torch.zeros(batch_size, self.config.d_h, device=device)

    def to(self, device: torch.device) -> "RoboticsAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "RoboticsAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "RoboticsAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import REGISTRY, register
from world_model_lens.core.types import WorldModelFamily

register(
    name="robotics",
    family=WorldModelFamily.ROBOTICS,
    description="Robotics/embodied world models (latent dynamics for manipulation)",
    supports_rl=True,
    supports_video=False,
    supports_planning=True,
)(RoboticsAdapter)
