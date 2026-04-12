"""TD-MPC2 adapter implementation.

TD-MPC2 has no explicit decoder and no recurrent state.
Uses ResNet encoder and continuous latent.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import WorldModelAdapter, AdapterConfig


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetEncoder(nn.Module):
    def __init__(self, d_obs: int, d_h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_obs, d_h),
            nn.ReLU(),
            *[ResBlock(d_h) for _ in range(3)],
            nn.LayerNorm(d_h),
        )
        self.out_dim = d_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPDynamics(nn.Module):
    def __init__(self, d_h: int, d_action: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h + d_action, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
        )

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, action], dim=-1)
        return self.net(x)


class RewardPredictor(nn.Module):
    def __init__(self, d_h: int, d_action: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h + d_action, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, action], dim=-1)
        return self.net(x)


class PolicyHead(nn.Module):
    def __init__(self, d_h: int, d_action: int):
        super().__init__()
        self.mean = nn.Linear(d_h, d_action)
        self.log_std = nn.Parameter(torch.zeros(d_action))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mean = self.mean(h)
        std = torch.exp(self.log_std)
        return torch.cat([mean, std], dim=-1)


class ValueHead(nn.Module):
    def __init__(self, d_h: int, d_action: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h + d_action, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, action], dim=-1)
        return self.net(x)


class TDMPC2Adapter(WorldModelAdapter):
    """TD-MPC2: Temporal Difference MPC.

    Continuous latent with no recurrent state. No decoder.
    """

    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.config = config

        self.encoder = ResNetEncoder(config.d_obs, config.d_h)
        self.dynamics = MLPDynamics(config.d_h, config.d_action)
        self.reward_predictor = RewardPredictor(config.d_h, config.d_action)
        self.policy = PolicyHead(config.d_h, config.d_action)
        self.value = ValueHead(config.d_h, config.d_action)

        self._device = torch.device("cpu")

    def encode(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() > 1:
            obs = (
                obs.flatten(start_dim=1)
                if obs.dim() > 2
                else obs.squeeze(0)
                if obs.dim() == 2
                else obs
            )
        z = self.encoder(obs)
        return z, z

    def dynamics_fn(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(h, action)

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        return h

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "TD-MPC2 has no decoder. The model uses continuous latent "
            "and learns implicitly through MPC planning."
        )

    def predict_reward(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.reward_predictor(h, action)

    def predict_continue(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, device=h.device)

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.policy(h)

    def critic_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        action = torch.zeros(h.shape[0], self.config.d_action, device=h.device)
        return self.value(h, action)

    def transition(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.dynamics(z, action)

    def initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.config.d_h, device=self._device)
        z = torch.zeros(batch_size, self.config.d_h, device=self._device)
        return h, z

    def named_parameters(self) -> Dict[str, torch.Tensor]:
        params = {}
        for name, param in self.named_parameters(full=True):
            params[name] = param
        return params

    def to(self, device: torch.device) -> "TDMPC2Adapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "TDMPC2Adapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "TDMPC2Adapter":
        super().train(mode)
        return self
