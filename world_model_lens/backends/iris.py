"""IRIS adapter implementation.

IRIS uses VQVAE to discretize observations, then a GPT-style
transformer predicts next tokens.
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import WorldModelAdapter
from world_model_lens.core.config import WorldModelConfig


class VectorQuantizer(nn.Module):
    def __init__(self, n_e: int, e_dim: int):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_flat = z.flatten(start_dim=1)
        d = (
            torch.sum(z_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        z_q = z + (z_q - z).detach()
        return z_q, min_encoding_indices, torch.zeros(z.shape[0], device=z.device)


class VQVAEEncoder(nn.Module):
    def __init__(self, d_obs: int, d_h: int, n_e: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_obs, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
        )
        self.vq = VectorQuantizer(n_e=n_e, e_dim=d_h)
        self.out_dim = d_h

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        z_q, indices, commitment = self.vq(z)
        return z_q, indices


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), key_padding_mask=mask)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class IRISTransformer(nn.Module):
    def __init__(
        self, d_model: int, n_layers: int, n_head: int, vocab_size: int, dropout: float = 0.1
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(x) + self.pos_embedding(pos)

        hidden_states = []
        for block in self.blocks:
            x = block(x, mask)
            hidden_states.append(self.ln(x))

        logits = self.head(x)
        return logits, hidden_states[-1]


class IRISRewardHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class IRISContinueHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class IRISAdapter(WorldModelAdapter):
    """IRIS: Image Reinforcement with Implicit Skills.

    Transformer-based world model with VQVAE discretization.
    """

    def __init__(
        self,
        config: WorldModelConfig,
        d_model: int = 256,
        n_layers: int = 4,
        n_head: int = 4,
        vocab_size: int = 512,
    ):
        super().__init__(config)
        self.config = config
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.encoder = VQVAEEncoder(config.d_obs, d_model, n_e=vocab_size)
        self.transformer = IRISTransformer(d_model, n_layers, n_head, vocab_size)
        self.reward_head = IRISRewardHead(d_model)
        self.continue_head = IRISContinueHead(d_model)

        self._device = torch.device("cpu")

    @property
    def hook_point_names(self) -> List[str]:
        names = super().hook_point_names
        for i in range(self.config.n_gru_layers or 4):
            names.append(f"transformer_hidden_{i}")
        return names

    def encode(self, obs: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() > 1:
            obs = (
                obs.flatten(start_dim=1)
                if obs.dim() > 2
                else obs.squeeze(0)
                if obs.dim() == 2
                else obs
            )

        z_q, indices = self.encoder(obs)
        logits = torch.zeros(self.d_model, device=obs.device)
        return logits, z_q

    def dynamics(self, h: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.vocab_size, device=h.device)

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("IRIS uses VQVAE decoder, decode not standard")

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.reward_head(h)

    def predict_continue(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.continue_head(h)

    def actor_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        logits = self.transformer.head(h)
        return logits

    def critic_forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.reward_head(h)

    def transition(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return z

    def initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.d_model, device=self._device)
        z = torch.zeros(batch_size, self.d_model, device=self._device)
        return h, z

    def named_parameters(self) -> Dict[str, torch.Tensor]:
        params = {}
        for name, param in self.named_parameters(full=True):
            params[name] = param
        return params

    def to(self, device: torch.device) -> "IRISAdapter":
        super().to(device)
        self._device = device
        return self

    def eval(self) -> "IRISAdapter":
        super().eval()
        return self

    def train(self, mode: bool = True) -> "IRISAdapter":
        super().train(mode)
        return self
