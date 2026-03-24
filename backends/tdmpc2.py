"""TD-MPC2 backend adapter for world_model_lens.

Implements the TD-MPC2 architecture (Ajay et al., 2024 — "Empowering 12-Year-Old
Algorithm Researchers with Large Language Models") as a :class:`WorldModelAdapter`.

Architecture overview
---------------------
TD-MPC2 is a deterministic-latent world model that differs from RSSM-based models
(DreamerV3, DreamerV2) in several key ways:

* **No recurrent hidden state** — the model operates in latent space only.
  The ``h`` in the interface is still maintained (as zeros) for API compatibility.

* **Encoder** — CNN (ResNet-style) with 4 convolutional blocks (stride 2 each) →
  adaptive average pooling → linear projection to embedding dimension.
  No batch normalization in encoder; uses adaptive avg pool instead.

* **Continuous latent** — The latent ``z`` is a simple Gaussian, not categorical.
  Internally represented with ``n_cat=1, n_cls=d_z`` to match the interface.

* **Dynamics** — No GRU. The dynamics simply concatenates the action encoding with
  the current latent embedding and projects through the prior head.

* **Prior** — MLP that outputs logits for the next latent state (treated as
  Gaussian parameters in the interface).

* **Posterior** — Computed from observation embedding (in contrast to DreamerV3
  which concatenates h with obs_emb).

* **Reward Head** — MLP without two-hot encoding; just a simple scalar prediction.

Named parameter conventions
---------------------------
``encoder.conv_blocks.0.weight``, ``encoder.conv_blocks.1.weight``, ...
``encoder.proj.weight``
``dynamics.prior_head.layers.0.weight``, ``dynamics.prior_head.layers.1.weight``, ...
``reward_head.layers.0.weight``, ``reward_head.layers.1.weight``, ...
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from world_model_lens.backends.base import WorldModelAdapter
from world_model_lens.core.config import WorldModelConfig


# ---------------------------------------------------------------------------
# Component modules
# ---------------------------------------------------------------------------


class TDMPC2Encoder(nn.Module):
    """ResNet-style visual encoder for TD-MPC2.

    Architecture: 4 convolutional blocks with stride-2 downsampling,
    each with Conv2d + BatchNorm + ELU activation. Followed by adaptive
    average pooling and a linear projection to embedding dimension.

    Parameters
    ----------
    obs_channels : int
        Number of input channels (e.g., 3 for RGB).
    image_size : int
        Height/width of input image (assumed square).
    d_emb : int
        Output embedding dimension.
    """

    def __init__(
        self,
        obs_channels: int = 3,
        image_size: int = 64,
        d_emb: int = 512,
    ) -> None:
        super().__init__()
        self.obs_channels = obs_channels
        self.image_size = image_size
        self.d_emb = d_emb

        # Compute channel progression: start small, grow through blocks
        channels = [obs_channels, 32, 64, 128, 256]

        # 4 convolutional blocks with stride-2 downsampling
        self.conv_blocks = nn.ModuleList()
        for i in range(4):
            block = nn.Sequential(
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ELU(),
            )
            self.conv_blocks.append(block)

        # After 4 stride-2 blocks: 64x64 -> 8x8
        # Adaptive avg pool to (1, 1), then flatten and project
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear projection from final conv channels to embedding
        self.proj = nn.Linear(channels[-1], d_emb)

    def forward(self, obs: Tensor) -> Tensor:
        """Encode observation to embedding.

        Parameters
        ----------
        obs : Tensor
            Image tensor of shape (C, H, W) or (B, C, H, W).

        Returns
        -------
        Tensor
            Embedding of shape (d_emb,) or (B, d_emb).
        """
        # Handle both batched and unbatched inputs
        input_shape = obs.shape
        if len(input_shape) == 3:
            obs = obs.unsqueeze(0)  # Add batch dim
            unbatched = True
        else:
            unbatched = False

        # Apply conv blocks
        x = obs
        for block in self.conv_blocks:
            x = block(x)

        # Adaptive pool and flatten
        x = self.adaptive_pool(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)

        # Project to embedding dimension
        emb = self.proj(x)  # (B, d_emb)

        if unbatched:
            emb = emb.squeeze(0)

        return emb


class TDMPC2Dynamics(nn.Module):
    """TD-MPC2 dynamics module (no recurrent state).

    Unlike RSSM-based models, TD-MPC2 dynamics are purely latent-space
    operations: given the current latent embedding and action encoding,
    compute the prior for the next latent state.

    Parameters
    ----------
    d_emb : int
        Latent embedding dimension.
    d_z : int
        Latent state dimension.
    d_action : int
        Action dimension (continuous).
    d_hidden : int
        Hidden dimension for MLP heads.
    n_hidden : int
        Number of hidden layers in prior head.
    """

    def __init__(
        self,
        d_emb: int = 512,
        d_z: int = 512,
        d_action: int = 6,
        d_hidden: int = 512,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.d_emb = d_emb
        self.d_z = d_z
        self.d_action = d_action

        # Action encoder: simple linear projection to embedding dimension
        self.action_encoder = nn.Linear(d_action, d_emb)

        # Prior head: MLP that outputs z logits (Gaussian parameters)
        # Input: concatenated [z_flat, action_emb]
        prior_layers = []
        prior_input_dim = d_emb + d_emb  # z + action_emb
        for i in range(n_hidden):
            if i == 0:
                prior_layers.append(nn.Linear(prior_input_dim, d_hidden))
            else:
                prior_layers.append(nn.Linear(d_hidden, d_hidden))
            prior_layers.append(nn.ELU())

        prior_layers.append(nn.Linear(d_hidden, d_z))
        self.prior_head = nn.Sequential(*prior_layers)

    def forward(
        self,
        z_emb: Tensor,
        action: Tensor,
    ) -> Tensor:
        """Compute prior logits for next latent state.

        Parameters
        ----------
        z_emb : Tensor
            Current latent embedding of shape (d_emb,) or (B, d_emb).
        action : Tensor
            Action taken of shape (d_action,) or (B, d_action).

        Returns
        -------
        Tensor
            Prior logits of shape (d_z,) or (B, d_z).
        """
        # Encode action
        action_emb = self.action_encoder(action)  # (d_emb,) or (B, d_emb)

        # Concatenate z_emb and action_emb
        combined = torch.cat([z_emb, action_emb], dim=-1)

        # Compute prior logits
        prior_logits = self.prior_head(combined)

        return prior_logits


class TDMPC2RewardHead(nn.Module):
    """TD-MPC2 reward prediction head.

    Simple MLP that predicts a scalar reward from latent state.
    No two-hot encoding; just a direct scalar prediction.

    Parameters
    ----------
    d_z : int
        Latent state dimension.
    d_hidden : int
        Hidden dimension for MLP.
    n_hidden : int
        Number of hidden layers.
    """

    def __init__(
        self,
        d_z: int = 512,
        d_hidden: int = 512,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(n_hidden):
            if i == 0:
                layers.append(nn.Linear(d_z, d_hidden))
            else:
                layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.ELU())

        layers.append(nn.Linear(d_hidden, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """Predict reward from latent state.

        Parameters
        ----------
        z : Tensor
            Latent state of shape (d_z,) or (B, d_z).

        Returns
        -------
        Tensor
            Scalar reward of shape () or (B,) or (B, 1).
        """
        reward = self.mlp(z)  # (d_z,) -> (1,) or (B, d_z) -> (B, 1)
        return reward.squeeze(-1) if reward.shape[-1] == 1 else reward


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class TDMPC2Adapter(WorldModelAdapter):
    """TD-MPC2 backend adapter for world_model_lens.

    A deterministic-latent world model with no recurrent hidden state.
    The model operates in embedding space only, computing dynamics as
    simple MLP transformations of the latent state and action.

    Parameters
    ----------
    cfg : WorldModelConfig
        Shared configuration (d_h, d_action, d_obs, n_cat, n_cls).
    obs_type : str
        Type of observation: "visual" or "vector". Default "visual".
    obs_channels : int
        Number of input channels (e.g., 3 for RGB). Default 3.
    image_size : int
        Height/width of square input image. Default 64.
    d_z : int
        Latent embedding dimension. Default 512.
    d_hidden : int
        Hidden dimension for MLP heads. Default 512.
    n_hidden : int
        Number of hidden layers in MLP heads. Default 2.

    Notes
    -----
    - ``h`` (recurrent hidden state) is maintained as zeros for API compatibility.
    - The latent ``z`` is continuous (Gaussian), represented internally as
      ``n_cat=1, n_cls=d_z`` to match the WorldModelAdapter interface.
    - No actor or value heads are implemented; calling those methods raises
      NotImplementedError.
    """

    def __init__(
        self,
        cfg: WorldModelConfig,
        obs_type: str = "visual",
        obs_channels: int = 3,
        image_size: int = 64,
        d_z: int = 512,
        d_hidden: int = 512,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._obs_type = obs_type
        self._obs_channels = obs_channels
        self._image_size = image_size
        self._d_z = d_z
        self._d_hidden = d_hidden
        self._n_hidden = n_hidden
        self._device = torch.device("cpu")

        # Encoder
        if obs_type == "visual":
            self.encoder = TDMPC2Encoder(
                obs_channels=obs_channels,
                image_size=image_size,
                d_emb=d_z,
            )
        else:
            # For vector obs, just a linear projection
            self.encoder = nn.Linear(cfg.d_obs, d_z)

        # Dynamics (prior)
        self.dynamics = TDMPC2Dynamics(
            d_emb=d_z,
            d_z=d_z,
            d_action=cfg.d_action,
            d_hidden=d_hidden,
            n_hidden=n_hidden,
        )

        # Posterior head: MLP that computes posterior logits from obs embedding
        posterior_layers = []
        for i in range(n_hidden):
            if i == 0:
                posterior_layers.append(nn.Linear(d_z, d_hidden))
            else:
                posterior_layers.append(nn.Linear(d_hidden, d_hidden))
            posterior_layers.append(nn.ELU())
        posterior_layers.append(nn.Linear(d_hidden, d_z))
        self.posterior_head = nn.Sequential(*posterior_layers)

        # Reward head
        self.reward_head = TDMPC2RewardHead(
            d_z=d_z,
            d_hidden=d_hidden,
            n_hidden=n_hidden,
        )

    # ──────────────────────────────────────────────────────────────────
    # Required abstract methods
    # ──────────────────────────────────────────────────────────────────

    def encode(self, obs: Tensor) -> Tensor:
        """Encode observation to latent embedding.

        Parameters
        ----------
        obs : Tensor
            Raw observation of shape (*obs_shape) or (B, *obs_shape).

        Returns
        -------
        Tensor
            Latent embedding of shape (d_z,) or (B, d_z).
        """
        if self._obs_type == "visual":
            # Reshape if needed: (C, H, W) or (B, C, H, W)
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)
                unbatched = True
            else:
                unbatched = False

            emb = self.encoder(obs)

            if unbatched:
                emb = emb.squeeze(0)
            return emb
        else:
            # Vector observation: flatten and project
            if obs.ndim > 1:
                original_shape = obs.shape[:-1]
                obs_flat = obs.view(-1, obs.shape[-1])
                emb = self.encoder(obs_flat)
                emb = emb.view(*original_shape, -1)
            else:
                emb = self.encoder(obs)
            return emb

    def initial_state(self, batch_size: int = 1) -> Tensor:
        """Return initial (zero) recurrent hidden state.

        TD-MPC2 has no true recurrent state, so this returns zeros
        for API compatibility.

        Parameters
        ----------
        batch_size : int
            Number of parallel episodes. Default 1.

        Returns
        -------
        Tensor
            Zeros of shape (d_h,) or (B, d_h).
        """
        h = torch.zeros(batch_size, self._cfg.d_h, device=self._device)
        if batch_size == 1:
            h = h.squeeze(0)
        return h

    def dynamics_step(
        self,
        h: Tensor,
        z: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """One dynamics step (no recurrent update in TD-MPC2).

        Computes the prior logits for the next latent state given
        the current latent and action. The hidden state ``h`` is
        ignored (it is always zeros in TD-MPC2).

        Parameters
        ----------
        h : Tensor
            Current (zero) hidden state, shape (d_h,) or (B, d_h).
        z : Tensor
            Current stochastic latent, shape (d_z,) or (B, d_z).
        action : Tensor
            Action taken, shape () or (d_a,) or (B, d_a).

        Returns
        -------
        h_next : Tensor
            Updated (still zero) hidden state, shape (d_h,) or (B, d_h).
        z_prior_logits : Tensor
            Prior logits, shape (d_z,) or (B, d_z).
        """
        # Handle both batched and unbatched inputs
        if h.ndim == 1:
            batch_size = 1
            z = z.unsqueeze(0) if z.ndim == 1 else z
            action = action.unsqueeze(0) if action.ndim == 1 else action
            unbatched = True
        else:
            batch_size = h.shape[0]
            unbatched = False

        # Compute prior logits from current latent and action
        prior_logits = self.dynamics(z, action)

        # Next hidden state is still zeros (no recurrence)
        h_next = torch.zeros_like(h)

        if unbatched:
            h_next = h_next.squeeze(0)
            prior_logits = prior_logits.squeeze(0)

        return h_next, prior_logits

    def posterior(self, h: Tensor, obs_emb: Tensor) -> Tensor:
        """Compute posterior logits from observation embedding.

        Unlike RSSM-based models, TD-MPC2 posterior depends only on the
        observation embedding, not the hidden state.

        Parameters
        ----------
        h : Tensor
            Hidden state (ignored in TD-MPC2), shape (d_h,) or (B, d_h).
        obs_emb : Tensor
            Observation embedding from :meth:`encode`, shape (d_z,) or (B, d_z).

        Returns
        -------
        Tensor
            Posterior logits, shape (d_z,) or (B, d_z).
        """
        posterior_logits = self.posterior_head(obs_emb)
        return posterior_logits

    # ──────────────────────────────────────────────────────────────────
    # Optional heads
    # ──────────────────────────────────────────────────────────────────

    def reward_pred(self, h: Tensor, z: Tensor) -> Tensor:
        """Predict reward from latent state.

        Parameters
        ----------
        h : Tensor
            Hidden state (ignored), shape (d_h,) or (B, d_h).
        z : Tensor
            Stochastic latent, shape (d_z,) or (B, d_z).

        Returns
        -------
        Tensor
            Scalar reward, shape () or (1,) or (B,) or (B, 1).
        """
        return self.reward_head(z)

    # ──────────────────────────────────────────────────────────────────
    # Parameter access & device management
    # ──────────────────────────────────────────────────────────────────

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        """Iterate over (name, parameter) pairs of all model weights.

        Delegates to all nn.Module sub-networks.

        Yields
        ------
        (str, Tensor)
        """
        for name, param in self.encoder.named_parameters():
            yield f"encoder.{name}", param
        for name, param in self.dynamics.named_parameters():
            yield f"dynamics.{name}", param
        for name, param in self.posterior_head.named_parameters():
            yield f"posterior_head.{name}", param
        for name, param in self.reward_head.named_parameters():
            yield f"reward_head.{name}", param

    def to(self, device: Union[str, torch.device]) -> "TDMPC2Adapter":
        """Move all sub-networks to device.

        Parameters
        ----------
        device : str | torch.device
            Target device.

        Returns
        -------
        TDMPC2Adapter
            Self, for chaining.
        """
        self._device = torch.device(device) if isinstance(device, str) else device
        self.encoder = self.encoder.to(self._device)
        self.dynamics = self.dynamics.to(self._device)
        self.posterior_head = self.posterior_head.to(self._device)
        self.reward_head = self.reward_head.to(self._device)
        return self

    def eval(self) -> "TDMPC2Adapter":
        """Switch to eval mode.

        Returns
        -------
        TDMPC2Adapter
            Self, for chaining.
        """
        self.encoder.eval()
        self.dynamics.eval()
        self.posterior_head.eval()
        self.reward_head.eval()
        return self

    def train(self, mode: bool = True) -> "TDMPC2Adapter":
        """Switch to train mode.

        Parameters
        ----------
        mode : bool
            Whether to enable training. Default True.

        Returns
        -------
        TDMPC2Adapter
            Self, for chaining.
        """
        self.encoder.train(mode)
        self.dynamics.train(mode)
        self.posterior_head.train(mode)
        self.reward_head.train(mode)
        return self

    # ──────────────────────────────────────────────────────────────────
    # Metadata
    # ──────────────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "tdmpc2"

    @property
    def cfg(self) -> WorldModelConfig:
        """Return configuration."""
        return self._cfg

    def __repr__(self) -> str:
        return (
            f"TDMPC2Adapter("
            f"obs={self._obs_type}, "
            f"d_z={self._d_z}, "
            f"d_hidden={self._d_hidden}, "
            f"d_action={self._cfg.d_action})"
        )


__all__ = [
    "TDMPC2Encoder",
    "TDMPC2Dynamics",
    "TDMPC2RewardHead",
    "TDMPC2Adapter",
]
