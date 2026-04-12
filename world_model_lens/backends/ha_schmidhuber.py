"""Ha & Schmidhuber "World Models" adapter implementation.

Reference: "World Models" (Ha & Schmidhuber, 2018)

This adapter implements the classic World Models architecture:
- VAE (Variational Autoencoder) for visual compression
- MDN-RNN (Mixture Density Network + RNN) for latent dynamics
- Simple linear controller for action selection

The model was originally used for car racing in VizDoom.
This adapter supports both the original MDN-RNN dynamics and can
be extended to work with modern variants.

Key characteristics:
- Discrete VAE latent (categorical)
- MDN outputs mixture of Gaussians for next latent
- Simple controller: linear(z, h) -> action
- No built-in reward prediction (used with external reward)
- Non-RL by default: can be extended for RL use
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model_lens.backends.base_adapter import WorldModelAdapter, AdapterConfig
from world_model_lens.core.types import LatentType, DynamicsType, WorldModelFamily


class VAEEncoder(nn.Module):
    """VAE Encoder: image -> latent mean + logvar."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 32):
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
        self.fc_mean = nn.Linear(256 * 6 * 6, latent_dim)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x / 255.0 if x.max() > 1.0 else x
        h = self.conv(x)
        h = h.flatten(1)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder: latent -> image reconstruction."""

    def __init__(self, latent_dim: int = 32, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 6 * 6)
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 6, 6)
        return self.deconv(h)


class MDNRNN(nn.Module):
    """Mixture Density Network RNN for latent dynamics.

    Outputs parameters of a mixture of Gaussians for predicting next latent.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int, n_mixtures: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_mixtures = n_mixtures

        input_dim = latent_dim + action_dim
        self.rnn = nn.GRUCell(input_dim, hidden_dim)

        self.mdn_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 3 * n_mixtures),
        )

    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if action is not None:
            x = torch.cat([z, action], dim=-1)
        else:
            x = z

        h_next = self.rnn(x, h)

        mdn_output = self.mdn_fc(h_next)
        mdn_output = mdn_output.view(-1, self.n_mixtures, 3)

        pi = F.softmax(mdn_output[:, :, 0], dim=1)
        mu = mdn_output[:, :, 1]
        log_sigma = mdn_output[:, :, 2]

        return h_next, pi, mu, log_sigma

    def sample_next_z(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        B, M = pi.shape
        pi_idx = torch.multinomial(pi, 1).squeeze(-1)
        mu_selected = mu.gather(1, pi_idx.unsqueeze(-1)).squeeze(-1)
        sigma = log_sigma.exp().gather(1, pi_idx.unsqueeze(-1)).squeeze(-1)
        z = mu_selected + torch.randn_like(mu_selected) * sigma
        return z


class SimpleController(nn.Module):
    """Simple linear controller: (z, h) -> action.

    This is the controller from the original World Models paper.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, h], dim=-1)
        return self.fc(x)


class HaSchmidhuberWorldModelAdapter(WorldModelAdapter):
    """Adapter for Ha & Schmidhuber "World Models" (VAE + MDN-RNN + Controller).

    Architecture:
    - VAE: Compresses images to latent representation
    - MDN-RNN: Predicts next latent (mixture of Gaussians)
    - Controller: Simple linear layer for action selection

    This adapter provides the core world model interface. The controller
    can be used for policy extraction in simple environments.

    For RL use, reward prediction can be added as an extension.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config

        latent_dim = config.d_h
        hidden_dim = config.d_h
        action_dim = config.d_action
        n_mixtures = 5

        self.vae_encoder = VAEEncoder(in_channels=3, latent_dim=latent_dim)
        self.vae_decoder = VAEDecoder(latent_dim=latent_dim, out_channels=3)
        self.mdn_rnn = MDNRNN(latent_dim, hidden_dim, action_dim, n_mixtures)
        self.controller = SimpleController(latent_dim, hidden_dim, action_dim)

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        return [
            "vae_encoder",
            "vae_decoder",
            "mdn_rnn_hidden",
            "controller",
            "observation",
            "latent",
        ]

    @property
    def world_model_family(self) -> WorldModelFamily:
        return WorldModelFamily.HA_SCHMIDHUBER

    def encode(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode observation to VAE latent.

        Args:
            observation: Image tensor [..., C, H, W]
            context: Optional hidden state for VAE

        Returns:
            Tuple of (latent, observation_encoding)
        """
        if observation.dim() == 3:
            observation = observation.unsqueeze(0)
        if observation.dim() == 4:
            observation = observation.unsqueeze(0)

        mean, logvar = self.vae_encoder(observation)
        std = (logvar * 0.5).exp()
        z = mean + std * torch.randn_like(std)

        return z, mean

    def dynamics(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MDN-RNN prior prediction (without observation).

        Args:
            state: Current latent state
            action: Optional action

        Returns:
            Prior latent (mean of mixture)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action is not None and action.dim() == 1:
            action = action.unsqueeze(0)

        B = state.shape[0]
        h = torch.zeros(B, self.config.d_h, device=state.device)

        h_next, pi, mu, log_sigma = self.mdn_rnn(state, h, action)

        z_next = self.mdn_rnn.sample_next_z(pi, mu, log_sigma)
        return z_next

    def transition(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        input_: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full transition with MDN-RNN.

        Args:
            state: Current latent state [B, d_latent]
            action: Action taken [B, d_action]
            input_: Optional additional input

        Returns:
            Next latent state
        """
        B = state.shape[0]
        h = torch.zeros(B, self.config.d_h, device=state.device)

        h_next, pi, mu, log_sigma = self.mdn_rnn(state, h, action)
        z_next = self.mdn_rnn.sample_next_z(pi, mu, log_sigma)

        return z_next

    def decode(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Decode latent to observation (VAE decoder).

        Args:
            state: Latent state

        Returns:
            Reconstructed observation
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.vae_decoder(state)

    def actor_forward(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Controller forward: (z, h) -> action.

        Args:
            state: Current state (z)

        Returns:
            Action logits
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        B = state.shape[0]
        h = torch.zeros(B, self.config.d_h, device=state.device)

        return self.controller(state, h)

    def initial_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Initialize starting state.

        Args:
            batch_size: Number of initial states
            device: Optional device

        Returns:
            Initial latent state
        """
        if device is None:
            device = self._device
        return torch.zeros(batch_size, self.config.d_h, device=device)

    def sample_state(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> torch.Tensor:
        """Sample from latent distribution.

        For continuous latents (VAE), just returns the sample.
        """
        if not sample:
            return logits
        return logits + torch.randn_like(logits) * 0.1

    def to(self, device: torch.device) -> "HaSchmidhuberWorldModelAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "HaSchmidhuberWorldModelAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "HaSchmidhuberWorldModelAdapter":
        super().train(mode)
        return self


from world_model_lens.backends.registry import REGISTRY, register
from world_model_lens.core.types import WorldModelFamily

register(
    name="ha_schmidhuber",
    family=WorldModelFamily.HA_SCHMIDHUBER,
    description="Ha & Schmidhuber World Models (VAE + MDN-RNN + Controller)",
    supports_rl=False,
    supports_video=True,
    supports_planning=True,
)(HaSchmidhuberWorldModelAdapter)
