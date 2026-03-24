"""DreamerV1 adapter implementation.

Reference: "Dream to Control: Learning Behaviors by Latent Imagination" (Hafner et al., 2020)

DreamerV1 was the first Dreamer paper. Key characteristics:
- RSSM with discrete latent variables (like later Dreamers)
- Image decoder for reconstruction
- Value function and policy both trained from imagined trajectories
- Uses ELU activation (vs SiLU in V3)
- Simpler architecture than V2/V3

Key differences from DreamerV2/V3:
- ELU activations instead of SiLU
- Simpler encoder/decoder architecture
- Two-hot reward encoding (like V2)
- No layer normalization in some components
- Lower compute requirements
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_lens.backends.generic_adapter import WorldModelAdapter, WorldModelConfig
from world_model_lens.core.types import LatentType, DynamicsType, WorldModelFamily


class DreamerV1Encoder(nn.Module):
    """DreamerV1 CNN encoder."""

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


class DreamerV1Decoder(nn.Module):
    """DreamerV1 CNN decoder."""

    def __init__(self, hidden_dim: int = 400, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 256 * 6 * 6)
        self.deconv = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x).view(-1, 256, 6, 6)
        return self.deconv(h)


class DreamerV1Dynamics(nn.Module):
    """DreamerV1 RSSM dynamics ( GRU + prior)."""

    def __init__(self, d_z: int, d_action: int, d_h: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=d_z + d_action, hidden_size=d_h)
        self.fc = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ELU(),
            nn.Linear(d_h, d_z),
        )

    def forward(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, action], dim=-1)
        h_next = self.gru(x, h)
        prior_logits = self.fc(h_next)
        return h_next, prior_logits


class DreamerV1Posterior(nn.Module):
    """DreamerV1 posterior model."""

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


class DreamerV1RewardHead(nn.Module):
    """DreamerV1 reward predictor (two-hot encoding)."""

    def __init__(self, d_h: int, d_z: int, bins: int = 255, low: float = -20.0, high: float = 20.0):
        super().__init__()
        self.bins = bins
        self.low = low
        self.high = high
        self.fc = nn.Sequential(
            nn.Linear(d_h + d_z, d_h),
            nn.ELU(),
            nn.Linear(d_h, bins),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.fc(x)


class DreamerV1ValueHead(nn.Module):
    """DreamerV1 value predictor."""

    def __init__(self, d_h: int, d_z: int, bins: int = 255):
        super().__init__()
        self.bins = bins
        self.fc = nn.Sequential(
            nn.Linear(d_h + d_z, d_h),
            nn.ELU(),
            nn.Linear(d_h, bins),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.fc(x)


class DreamerV1Actor(nn.Module):
    """DreamerV1 policy (discrete actions)."""

    def __init__(self, d_h: int, d_z: int, d_action: int, n_classes: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_h + d_z, d_h),
            nn.ELU(),
            nn.Linear(d_h, d_action * n_classes),
        )
        self.n_classes = n_classes

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, z], dim=-1)
        return self.fc(x)


class DreamerV1Adapter(WorldModelAdapter):
    """Adapter for DreamerV1.

    The original Dreamer algorithm with:
    - RSSM latent dynamics
    - Image reconstruction decoder
    - Two-hot reward encoding
    - Value and policy from imagined rollouts
    - ELU activations
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)
        self.config = config

        self.encoder = DreamerV1Encoder(in_channels=3, hidden_dim=config.d_obs)
        self.decoder = DreamerV1Decoder(hidden_dim=config.d_h + config.d_z, out_channels=3)
        self.dynamics = DreamerV1Dynamics(config.d_z, config.d_action, config.d_h)
        self.posterior = DreamerV1Posterior(config.d_obs, config.d_h, config.d_z)
        self.reward_head = DreamerV1RewardHead(config.d_h, config.d_z)
        self.value_head = DreamerV1ValueHead(config.d_h, config.d_z)
        self.actor = DreamerV1Actor(config.d_h, config.d_z, config.d_action)

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return [
            "encoder",
            "decoder",
            "dynamics_gru",
            "posterior",
            "reward",
            "value",
            "actor",
            "state",
            "prior",
            "posterior",
        ]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.DREAMER

    def encode(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode observation to posterior."""
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
        """Compute prior from hidden state."""
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
        """Full transition."""
        if isinstance(state, tuple):
            h, z = state
        else:
            h = state
            z = torch.zeros(h.shape[0], self.config.d_z, device=h.device)

        if action is None:
            action = torch.zeros(h.shape[0], self.config.d_action, device=h.device)

        h_next, _ = self.dynamics(h, z, action)
        return h_next

    def decode(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Decode state to observation."""
        if isinstance(state, tuple):
            h, z = state
        else:
            h = state
            z = torch.zeros(h.shape[0], self.config.d_z, device=h.device)

        x = torch.cat([h, z], dim=-1)
        return self.decoder(x)

    def predict_reward(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Predict reward."""
        if isinstance(state, tuple):
            h, z = state
        else:
            h = state
            z = torch.zeros(h.shape[0], self.config.d_z, device=h.device)

        return self.reward_head(h, z)

    def predict_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Predict value."""
        if isinstance(state, tuple):
            h, z = state
        else:
            h = state
            z = torch.zeros(h.shape[0], self.config.d_z, device=h.device)

        return self.value_head(h, z)

    def actor_forward(
        self,
        state: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Predict action."""
        if isinstance(state, tuple):
            h, z = state
        else:
            h = state
            z = torch.zeros(h.shape[0], self.config.d_z, device=h.device)

        return self.actor(h, z)

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
        """Initialize starting state."""
        if device is None:
            device = self._device
        h = torch.zeros(batch_size, self.config.d_h, device=device)
        z_logits = torch.zeros(batch_size, self.config.d_z, device=device)
        z = self.sample_state(z_logits, temperature=1.0)
        return h, z

    def to(self, device: torch.device) -> "DreamerV1Adapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "DreamerV1Adapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "DreamerV1Adapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import REGISTRY, register
from world_model_lens.core.types import WorldModelFamily

register(
    name="dreamerv1",
    family=WorldModelFamily.DREAMER,
    description="DreamerV1: Dream to Control (original Dreamer)",
    supports_rl=True,
    supports_video=False,
    supports_planning=True,
)(DreamerV1Adapter)
