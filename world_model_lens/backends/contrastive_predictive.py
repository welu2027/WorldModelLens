"""Contrastive/Predictive world model adapter (CWM, SPR-style).

Reference: "Unsupervised Semantic-Based Planning with Learned Semantic Dynamics" (CWM)
          "Unsupervised Learning of Object Keypoints for World Models" (CWM)
          "Learning Latent Dynamics for Planning from Pixels" (PlaNet)
          "Decoupling Value and Policy for Control in Latent Space" (SPR, etc.)

Contrastive/Predictive world models learn representations by:
- Predicting future latent states via contrastive loss
- Using momentum encoders (like BYOL, MoCo)
- Self-predictive representations (SPR, CWM)

This adapter supports:
- CWM: Semantic-based planning with learned semantic dynamics
- SPR: Self-predictive representations for control
- Generic contrastive latent dynamics models
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_lens.backends.generic_adapter import WorldModelAdapter, WorldModelConfig
from world_model_lens.core.types import LatentType, DynamicsType, WorldModelFamily


class ContrastiveEncoder(nn.Module):
    """Shared encoder for contrastive learning."""

    def __init__(self, d_obs: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_obs, d_latent),
            nn.ReLU(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MomentumEncoder(nn.Module):
    """Momentum encoder for contrastive learning (like MoCo)."""

    def __init__(self, encoder: nn.Module, m: float = 0.99):
        super().__init__()
        self.encoder = encoder
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def update(self, encoder: nn.Module):
        for param_q, param_k in zip(encoder.parameters(), self.parameters()):
            param_k.data.mul_(self.m).add_(param_q.data, alpha=1 - self.m)


class LatentDynamics(nn.Module):
    """Latent dynamics predictor (predictive coding)."""

    def __init__(self, d_latent: int, d_action: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent + d_action, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_latent),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        return self.net(x)


class ContrastiveAdapter(WorldModelAdapter):
    """Adapter for Contrastive/Predictive world models (CWM, SPR-style).

    These models learn latent dynamics through:
    - Contrastive loss between predicted and actual future latents
    - Momentum encoders for stable representation
    - Self-predictive representations

    Architecture:
    - Encoder: observation -> latent
    - Dynamics: (latent, action) -> next latent prediction
    - Projector: latent -> contrastive embedding
    - Predictor: latent -> predicted embedding (for contrastive)
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        self.config = config

        d_latent = config.d_h

        self.encoder = ContrastiveEncoder(config.d_obs, d_latent)
        self.momentum_encoder = MomentumEncoder(self.encoder, m=0.99)
        self.projector = nn.Linear(d_latent, d_latent)
        self.predictor = nn.Linear(d_latent, d_latent)
        self.dynamics = LatentDynamics(d_latent, config.d_action)

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return [
            "encoder",
            "projector",
            "predictor",
            "dynamics",
            "latent",
        ]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.CONTRASTIVE_PREDICTIVE

    def encode(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode observation to latent."""
        if observation.dim() > 1:
            observation = observation.flatten(1)

        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        z = self.encoder(observation)
        return z, z

    def dynamics(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict next latent (prior/dynamics)."""
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
        """No decoder in contrastive models."""
        return None

    def predict_reward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> None:
        """Optional reward prediction (if trained with rewards)."""
        return None

    def actor_forward(
        self,
        state: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Optional policy (if trained with RL)."""
        return None

    def forward_contrastive(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for contrastive learning.

        Args:
            z1: Current latent (online encoder)
            z2: Next latent (momentum encoder)
            action: Action taken

        Returns:
            Tuple of (predicted embedding, target embedding)
        """
        proj1 = self.projector(z1)
        pred1 = self.predictor(proj1)

        with torch.no_grad():
            target = self.momentum_encoder(z2)

        return pred1, target

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

    def to(self, device: torch.device) -> "ContrastiveAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "ContrastiveAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "ContrastiveAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import REGISTRY, register
from world_model_lens.core.types import WorldModelFamily

register(
    name="contrastive_predictive",
    family=WorldModelFamily.CONTRASTIVE_PREDICTIVE,
    description="Contrastive/Predictive world models (CWM, SPR-style)",
    supports_rl=True,
    supports_video=False,
    supports_planning=True,
)(ContrastiveAdapter)
