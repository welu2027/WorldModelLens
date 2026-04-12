"""Autonomous driving world model adapter.

Reference: "WorldDreamer: World Model for Autonomous Driving"
          "DriveWorld: 4D Latent Space for Multi-Modal Memory Autonomous Driving"

Autonomous driving world models fuse camera/LiDAR/radar into latent state
and predict future scenes or ego-motion:

- Multi-modal sensor fusion (camera, LiDAR, radar)
- Temporal prediction of scenes/ego-motion
- Planning in latent space

This adapter supports:
- BEV (Bird's Eye View) based world models
- Latent dynamics for planning
- Multi-modal sensor fusion
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_lens.backends.base_adapter import WorldModelAdapter, AdapterConfig
from world_model_lens.core.types import LatentType, DynamicsType, WorldModelFamily


class MultiModalEncoder(nn.Module):
    """Multi-modal encoder for camera, LiDAR, radar."""

    def __init__(
        self,
        camera_channels: int = 3,
        lidar_channels: int = 1,
        radar_channels: int = 1,
        d_latent: int = 256,
    ):
        super().__init__()

        self.camera_encoder = nn.Sequential(
            nn.Conv2d(camera_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(lidar_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 * 16 + 64 * 16 + 64 * 16, d_latent),
            nn.ReLU(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(
        self,
        camera: torch.Tensor,
        lidar: Optional[torch.Tensor] = None,
        radar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cam_features = self.camera_encoder(camera).flatten(1)

        if lidar is not None:
            lidar_features = self.lidar_encoder(lidar).flatten(1)
        else:
            lidar_features = torch.zeros_like(cam_features)

        if radar is not None:
            radar_features = self.lidar_encoder(radar).flatten(1)
        else:
            radar_features = torch.zeros_like(cam_features)

        fused = torch.cat([cam_features, lidar_features, radar_features], dim=-1)
        return self.fusion(fused)


class DrivingDynamics(nn.Module):
    """Dynamics model for autonomous driving (predicts future states and ego-motion)."""

    def __init__(self, d_latent: int, d_action: int, hidden_dim: int = 256):
        super().__init__()
        self.dynamics = nn.Sequential(
            nn.Linear(d_latent + d_action, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_latent),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.dynamics(x)


class EgoMotionPredictor(nn.Module):
    """Predicts ego-motion (future position/rotation)."""

    def __init__(self, d_latent: int, hidden_dim: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.predictor(state)


class AutonomousDrivingAdapter(WorldModelAdapter):
    """Adapter for Autonomous Driving world models.

    These models:
    - Fuse multi-modal sensors (camera, LiDAR, radar)
    - Predict future scenes and ego-motion
    - Enable planning in latent space

    Supports BEV-based architectures and sensor fusion.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config

        self.encoder = MultiModalEncoder(
            camera_channels=3,
            lidar_channels=config.d_obs // 3 if config.d_obs > 0 else 1,
            radar_channels=1,
            d_latent=config.d_h,
        )
        self.dynamics = DrivingDynamics(config.d_h, config.d_action)
        self.ego_predictor = EgoMotionPredictor(config.d_h)
        self.reward_predictor = nn.Linear(config.d_h, 1)

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return [
            "camera_encoder",
            "lidar_encoder",
            "fusion",
            "dynamics",
            "ego_motion",
            "latent_state",
        ]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.AUTONOMOUS_DRIVING

    def encode(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode multi-modal sensor data to latent state.

        Args:
            observation: Dict with 'camera', 'lidar', 'radar' keys, or single tensor
            context: Optional previous state

        Returns:
            Tuple of (latent state, observation encoding)
        """
        if isinstance(observation, dict):
            camera = observation.get("camera", observation.get("image"))
            lidar = observation.get("lidar")
            radar = observation.get("radar")
            obs_encoding = self.encode_multimodal(camera, lidar, radar)
        else:
            if observation.dim() == 5:
                observation = observation.squeeze(1)
            obs_encoding = self.encoder(observation)

        return obs_encoding, obs_encoding

    def encode_multimodal(
        self,
        camera: torch.Tensor,
        lidar: Optional[torch.Tensor] = None,
        radar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode multi-modal sensor data."""
        if camera.dim() == 5:
            B, T, C, H, W = camera.shape
            camera = camera.view(B * T, C, H, W)
            cam_features = self.encoder.camera_encoder(camera).flatten(1)
            cam_features = cam_features.view(B, T, -1).mean(dim=1)
        else:
            cam_features = self.encoder.camera_encoder(camera).flatten(1)

        if lidar is not None:
            lidar_features = self.encoder.lidar_encoder(lidar).flatten(1)
        else:
            lidar_features = torch.zeros_like(cam_features)

        if radar is not None:
            radar_features = self.encoder.lidar_encoder(radar).flatten(1)
        else:
            radar_features = torch.zeros_like(cam_features)

        fused = torch.cat([cam_features, lidar_features, radar_features], dim=-1)
        return self.encoder.fusion(fused)

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
        """State transition with dynamics."""
        return self.dynamics(state, action)

    def decode(self, state: torch.Tensor) -> None:
        """No decoder (future work: could decode to BEV)."""
        return None

    def predict_reward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Optional driving reward prediction."""
        return self.reward_predictor(state)

    def predict_ego_motion(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future ego-motion (6 DOF: x, y, z, roll, pitch, yaw)."""
        return self.ego_predictor(state)

    def actor_forward(
        self,
        state: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Optional policy for driving."""
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

    def to(self, device: torch.device) -> "AutonomousDrivingAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "AutonomousDrivingAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "AutonomousDrivingAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import REGISTRY, register
from world_model_lens.core.types import WorldModelFamily

register(
    name="autonomous_driving",
    family=WorldModelFamily.AUTONOMOUS_DRIVING,
    description="Autonomous driving world models (camera/LiDAR/radar fusion)",
    supports_rl=True,
    supports_video=True,
    supports_planning=True,
)(AutonomousDrivingAdapter)
