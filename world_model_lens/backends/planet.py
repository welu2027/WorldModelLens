"""PlaNet adapter implementation.

Reference: "PlaNet: Learning Latent Dynamics for Planning from Pixels" (Hafner et al., 2019)

PlaNet was the predecessor to Dreamer and introduced:
- Latent RSSM: recurrent state-space model with discrete latent variables
- Planning at the latent level using model predictive control (MPC)
- Learning from images without decoder (奖励 prediction only)

Key differences from Dreamer:
- No decoder (learns only from reward)
- Uses image-based latent planning
- Pure model-based RL without policy gradient
- Simpler architecture than DreamerV2+

This adapter implements the latent dynamics model from PlaNet.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_lens.backends.base_adapter import WorldModelAdapter, AdapterConfig
from world_model_lens.core.types import LatentType, DynamicsType, WorldModelFamily


class PlaNetEncoder(nn.Module):
    """PlaNet CNN encoder for images."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 400):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
        )
        self.fc = nn.Linear(256 * 6 * 6, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0 if x.max() > 1.0 else x
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


class PlaNetDynamics(nn.Module):
    """PlaNet RSSM dynamics (prior) model."""

    def __init__(self, d_h: int, d_z: int, d_action: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=d_z + d_action, hidden_size=d_h)
        self.fc = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ELU(),
            nn.Linear(d_h, d_z),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, action], dim=-1)
        h_next = self.gru(x, h)
        prior_logits = self.fc(h_next)
        return h_next, prior_logits


class PlaNetPosterior(nn.Module):
    """PlaNet posterior model (encoder output)."""

    def __init__(self, d_obs: int, d_h: int, d_z: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_obs + d_h, d_h),
            nn.ELU(),
            nn.Linear(d_h, d_h),
            nn.ELU(),
            nn.Linear(d_h, d_z * 2),
        )

    def forward(self, obs_encoding: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_encoding, h], dim=-1)
        return self.fc(x)


class PlaNetRewardPredictor(nn.Module):
    """PlaNet reward predictor (no decoder in original)."""

    def __init__(self, d_h: int, d_z: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_h + d_z, d_h),
            nn.ELU(),
            nn.Linear(d_h, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.fc(x)


class PlaNetAdapter(WorldModelAdapter):
    """Adapter for PlaNet (latent RSSM from pixels).

    PlaNet learns a latent dynamics model from images without
    using a decoder. The model is used for planning/MPC at the
    latent level.

    Architecture:
    - CNN encoder: images -> latent observation encoding
    - RSSM: GRU for dynamics, discrete latent variables
    - Reward predictor: (h, z) -> reward

    This is the foundational architecture that Dreamer built upon.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config

        self.encoder = PlaNetEncoder(in_channels=3, hidden_dim=config.d_obs)
        self.dynamics = PlaNetDynamics(config.d_h, config.d_z, config.d_action)
        self.posterior = PlaNetPosterior(config.d_obs, config.d_h, config.d_z)
        self.reward_predictor = PlaNetRewardPredictor(config.d_h, config.d_z)

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return [
            "encoder",
            "dynamics_gru",
            "posterior",
            "reward",
            "state",
            "prior",
        ]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.PLA_NET

    def encode(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode observation to posterior latent.

        Args:
            observation: Image tensor [..., C, H, W]
            context: Optional hidden state h

        Returns:
            Tuple of (posterior logits, observation encoding)
        """
        if observation.dim() == 3:
            observation = observation.unsqueeze(0)
        if observation.dim() == 4:
            observation = observation.unsqueeze(0)

        obs_encoding = self.encoder(observation)

        h = torch.zeros(observation.shape[0], self.config.d_h, device=observation.device)
        if context is not None and context.dim() == 2:
            h = context

        posterior_logits = self.posterior(obs_encoding, h)

        return posterior_logits, obs_encoding

    def dynamics(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute prior from hidden state (no observation).

        Args:
            state: Current state tuple (h, z) or just h
            action: Optional action

        Returns:
            Prior logits for next z
        """
        if isinstance(state, tuple):
            h, z = state
        else:
            h = state
            z = torch.zeros(h.shape[0], self.config.d_z, device=h.device)

        if action is None:
            action = torch.zeros(h.shape[0], self.config.d_action, device=h.device)

        h_next, prior_logits = self.dynamics(h, z, action)
        return prior_logits

    def transition(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        input_: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full transition with posterior.

        Args:
            state: Current state (h, z)
            action: Action taken
            input_: Optional observation for posterior

        Returns:
            Next hidden state
        """
        if isinstance(state, tuple):
            h, z = state
        else:
            h = state
            z = torch.zeros(h.shape[0], self.config.d_z, device=h.device)

        if action is None:
            action = torch.zeros(h.shape[0], self.config.d_action, device=h.device)

        h_next, _ = self.dynamics(h, z, action)
        return h_next

    def predict_reward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Predict reward from state.

        Args:
            state: State (h, z) tuple or h
            action: Optional action

        Returns:
            Reward prediction
        """
        if isinstance(state, tuple):
            h, z = state
        else:
            h = state
            z = torch.zeros(h.shape[0], self.config.d_z, device=h.device)

        return self.reward_predictor(h, z)

    def decode(self, state: torch.Tensor) -> None:
        """PlaNet has no decoder."""
        return None

    def sample_state(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        """Sample from categorical distribution."""
        if not self.config.is_discrete:
            return logits

        if not sample:
            indices = logits.argmax(dim=-1)
            return F.one_hot(indices, num_classes=logits.shape[-1]).float()

        gumbels = torch.rand_like(logits).log().neg()
        gumbels = (logits + gumbels) / temperature
        soft = F.softmax(gumbels, dim=-1)
        indices = soft.argmax(dim=-1)
        hard = F.one_hot(indices, num_classes=logits.shape[-1]).float()
        return (hard - soft).detach() + soft

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize starting state.

        Args:
            batch_size: Number of initial states
            device: Optional device

        Returns:
            Tuple of (h, z)
        """
        if device is None:
            device = self._device
        h = torch.zeros(batch_size, self.config.d_h, device=device)
        z_logits = torch.zeros(batch_size, self.config.d_z, device=device)
        z = self.sample_state(z_logits, temperature=1.0)
        return h, z

    def to(self, device: torch.device) -> "PlaNetAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "PlaNetAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "PlaNetAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import REGISTRY, register
from world_model_lens.core.types import WorldModelFamily

register(
    name="planet",
    family=WorldModelFamily.PLA_NET,
    description="PlaNet: Learning Latent Dynamics for Planning from Pixels",
    supports_rl=True,
    supports_video=False,
    supports_planning=True,
)(PlaNetAdapter)
