"""Video world model adapter (WorldDreamer-style).

Reference: "WorldDreamer: World Model for Autonomous Driving" (WorldDreamer)
          "Video Diffusion Models" (Video Diffusion)
          "Make-A-Video" (Make-A-Video)

Video world models learn to predict future video frames:
- Latent video diffusion models
- Token-based video generation
- Masked modeling for efficient prediction

This adapter supports:
- Latent video prediction models
- Token-based video world models
- Video diffusion-based world models
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_lens.backends.base_adapter import WorldModelAdapter, AdapterConfig
from world_model_lens.core.types import LatentType, DynamicsType, WorldModelFamily


class VideoEncoder(nn.Module):
    """Video encoder: video frames -> latent tokens."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 256, num_tokens: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(256 * 4 * 4, latent_dim)
        self.num_tokens = num_tokens

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = x / 255.0 if x.max() > 1.0 else x
        features = self.conv(x)
        features = features.flatten(1)
        tokens = self.proj(features)
        tokens = tokens.view(B, T, self.num_tokens, -1)
        return tokens, tokens


class VideoDecoder(nn.Module):
    """Video decoder: latent tokens -> video frames."""

    def __init__(self, latent_dim: int = 256, out_channels: int = 3):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T, N, D = tokens.shape
        x = tokens.flatten(2)
        x = self.proj(x)
        x = x.view(B * T, 256, 4, 4)
        video = self.deconv(x)
        return video.view(B, T, -1, video.shape[2], video.shape[3])


class VideoDynamicsTransformer(nn.Module):
    """Transformer for video dynamics prediction."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(256, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=n_layers,
        )

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, N, D = tokens.shape
        seq_len = T * N
        tokens = tokens.view(B, seq_len, D)

        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_embedding(positions)

        output = self.transformer(tokens, src_key_padding_mask=mask)
        return output.view(B, T, N, D)


class VideoWorldModelAdapter(WorldModelAdapter):
    """Adapter for Video World Models (WorldDreamer-style).

    These models predict future video frames from past observations:
    - Latent video encoding
    - Transformer-based dynamics
    - Video decoder for reconstruction

    This adapter is primarily a non-RL video prediction model,
    but can be extended for RL with reward prediction.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config

        self.encoder = VideoEncoder(
            in_channels=3,
            latent_dim=config.d_embed,
            num_tokens=config.vocab_size,
        )
        self.decoder = VideoDecoder(latent_dim=config.d_embed, out_channels=3)
        self.dynamics = VideoDynamicsTransformer(
            d_model=config.d_embed,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
        )

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return [
            "encoder",
            "decoder",
            "dynamics",
            "latent_tokens",
        ]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.VIDEO_WORLD_MODEL

    def encode(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode video frames to latent tokens.

        Args:
            observation: Video tensor [B, T, C, H, W] or [T, C, H, W]

        Returns:
            Tuple of (latent tokens, observation encoding)
        """
        if observation.dim() == 4:
            observation = observation.unsqueeze(0)
        if observation.dim() == 5:
            observation = observation.unsqueeze(0)

        tokens, obs_encoding = self.encoder(observation)
        return tokens, obs_encoding

    def dynamics(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict future latent tokens (video dynamics).

        Args:
            state: Current latent tokens [B, T, N, D]
            action: Optional action (can be used for conditioned prediction)

        Returns:
            Predicted next latent tokens
        """
        predicted = self.dynamics(state)
        return predicted

    def transition(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        input_: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full transition with dynamics."""
        return self.dynamics(state, action)

    def decode(
        self,
        state: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Decode latent tokens to video frames.

        Args:
            state: Latent tokens [B, T, N, D]

        Returns:
            Reconstructed video [B, T, C, H, W]
        """
        return self.decoder(state)

    def predict_reward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> None:
        """Video models typically don't predict rewards."""
        return None

    def actor_forward(
        self,
        state: torch.Tensor,
    ) -> None:
        """Video models don't have actors."""
        return None

    def sample_state(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        """Passthrough for continuous latents."""
        return logits

    def imagine(
        self,
        start_state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        horizon: int = 50,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """Run imagined rollout for video prediction.

        Args:
            start_state: Starting latent tokens
            actions: Optional actions
            horizon: Number of frames to predict
            temperature: Sampling temperature

        Returns:
            Tuple of (predicted video tokens, reward predictions)
        """
        state = start_state
        state_seq = [state]

        for _ in range(horizon):
            next_state = self.dynamics(state, actions)
            state_seq.append(next_state)
            state = next_state

        video = self.decode(torch.stack(state_seq, dim=1))
        return state_seq, [None] * len(state_seq)

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Initialize starting state."""
        if device is None:
            device = self._device
        return torch.zeros(
            batch_size, 1, self.config.vocab_size, self.config.d_embed, device=device
        )

    def to(self, device: torch.device) -> "VideoWorldModelAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "VideoWorldModelAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "VideoWorldModelAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import REGISTRY, register
from world_model_lens.core.types import WorldModelFamily

register(
    name="video_world_model",
    family=WorldModelFamily.VIDEO_WORLD_MODEL,
    description="Video world models (WorldDreamer-style)",
    supports_rl=False,
    supports_video=True,
    supports_planning=False,
)(VideoWorldModelAdapter)
