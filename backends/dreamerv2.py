"""DreamerV2 backend adapter for world_model_lens.

Implements the DreamerV2 architecture (Hafner et al., 2020 — "Mastering Atari
with Discrete World Models") as a :class:`WorldModelAdapter`.

Key differences from DreamerV3
-------------------------------
1. **ELU activations** throughout (not SiLU).
2. **Gaussian reward/value heads** — predict a Normal distribution mean; no
   two-hot categorical regression, no symlog transform.
3. **Discrete categorical** ``z_t`` (same RSSM structure) but gradient flows
   via straight-through estimator, not through soft reparameterisation.
4. No LayerNorm in some original configurations; we keep it as an option
   (``use_norm=True`` by default) to match modern reproductions.
5. No symlog/symexp anywhere — targets are used at face value.

Shared with DreamerV3
---------------------
* Same RSSM skeleton: GRU transition, categorical latent ``z_t``, separate
  prior (dynamics predictor, ``h_t`` only) and posterior (encoder +
  ``(h_t, obs_emb)``).
* Same ``_build_mlp`` / ``_cnn_output_size`` utilities (imported from
  ``dreamerv3`` with ``act=nn.ELU``).
* Identical GRU cell: ``DreamerV2Transition ≡ DreamerV3Transition``.
* Same parameter-naming conventions.

Named parameter conventions
---------------------------
``encoder.conv1.weight``, ``encoder.post_head.weight``
``transition.gru.weight_ih``, ``transition.gru.weight_hh``
``dynamics.layers.0.weight``, ``dynamics.layers.3.weight``
``reward.layers.0.weight``, ``cont.layers.0.weight``
``actor.layers.0.weight``, ``critic.layers.0.weight``
``decoder.in_proj.weight``, ``decoder.layers.0.weight``
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from world_model_lens.backends.base import WorldModelAdapter
from world_model_lens.backends.dreamerv3 import (
    DreamerV3Transition,   # GRUCell — identical in V2 and V3
    _build_mlp,            # shared MLP factory; we pass act=nn.ELU below
    _cnn_output_size,      # shared CNN spatial-size calculator
)
from world_model_lens.core.config import WorldModelConfig


# ---------------------------------------------------------------------------
# Utility: straight-through estimator for categorical z
# ---------------------------------------------------------------------------

def straight_through_z(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """Produce a hard one-hot ``z`` with gradients through the soft distribution.

    Forward pass: argmax → one-hot (discrete, non-differentiable).
    Backward pass: gradients treated as if taken through ``softmax(logits)``.

    This implements Bengio et al.'s straight-through estimator so that
    discrete ``z_t`` can be trained end-to-end without the KL variance of
    REINFORCE or the bias of Gumbel-softmax.

    Parameters
    ----------
    logits : Tensor
        Unnormalised log-probabilities, shape ``(*batch, n_cat, n_cls)``.
    temperature : float
        Softmax temperature (applied before argmax and softmax; default 1.0).

    Returns
    -------
    Tensor
        Straight-through one-hot tensor, same shape as *logits*.

    Examples
    --------
    >>> logits = torch.randn(4, 4)            # (n_cat=4, n_cls=4)
    >>> z_st = straight_through_z(logits)
    >>> z_st.sum(-1)                          # each row sums to 1
    tensor([1., 1., 1., 1.])
    """
    scaled = logits / max(temperature, 1e-6)
    soft = scaled.softmax(dim=-1)                            # (*batch, n_cat, n_cls)
    indices = soft.argmax(dim=-1, keepdim=True)              # (*batch, n_cat, 1)
    hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0) # one-hot
    # Straight-through: hard in forward, soft gradient in backward
    return hard - soft.detach() + soft


def _build_mlp_elu(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_hidden: int = 2,
    use_norm: bool = True,
) -> nn.Sequential:
    """Convenience wrapper around :func:`_build_mlp` with ELU activation.

    All DreamerV2 sub-networks use ELU rather than SiLU (one of the
    key differences from V3).
    """
    return _build_mlp(in_dim, hidden_dim, out_dim, n_hidden,
                      act=nn.ELU, use_norm=use_norm)


# ---------------------------------------------------------------------------
# Component nn.Modules
# ---------------------------------------------------------------------------

class DreamerV2Encoder(nn.Module):
    """Observation encoder + posterior head (ELU variant of the V3 encoder).

    Two modes:

    * **Visual** — four stride-2 Conv2d(k=4, s=2) layers with ELU activations.
    * **Vector** — MLP with ELU activations.

    A ``post_head`` linear maps
    ``concat(h_t, obs_emb) → z_posterior_logits``.

    ⚠ The posterior takes BOTH ``h_t`` and the observation (same as V3).

    Parameters
    ----------
    obs_type : ``"visual"`` | ``"vector"``
    obs_channels : int
    image_size : int
    cnn_channels : list of int
    d_obs : int
    d_hidden : int
    n_hidden : int
    d_h : int
    n_cat, n_cls : int
    """

    def __init__(
        self,
        obs_type: str,
        obs_channels: int,
        image_size: int,
        cnn_channels: List[int],
        d_obs: int,
        d_hidden: int,
        n_hidden: int,
        d_h: int,
        n_cat: int,
        n_cls: int,
    ) -> None:
        super().__init__()
        self._obs_type = obs_type
        self._n_cat = n_cat
        self._n_cls = n_cls

        if obs_type == "visual":
            ch = cnn_channels
            self.conv1 = nn.Conv2d(obs_channels, ch[0], kernel_size=4, stride=2)
            self.conv2 = nn.Conv2d(ch[0], ch[1], kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(ch[1], ch[2], kernel_size=4, stride=2)
            self.conv4 = nn.Conv2d(ch[2], ch[3], kernel_size=4, stride=2)
            d_emb = _cnn_output_size(image_size, cnn_channels)
            self.out_norm = nn.LayerNorm(d_emb)
        else:
            layers: List[nn.Module] = []
            d = d_obs
            for _ in range(max(n_hidden, 1)):
                # ELU instead of SiLU — key V2 difference
                layers.extend([nn.Linear(d, d_hidden), nn.LayerNorm(d_hidden), nn.ELU()])
                d = d_hidden
            self.mlp = nn.Sequential(*layers)
            d_emb = d_hidden

        self._d_emb = d_emb
        self.post_head = nn.Linear(d_h + d_emb, n_cat * n_cls)

    def forward(self, obs: Tensor) -> Tensor:
        """Encode observation → embedding (ELU, no ``h_t`` involved here)."""
        if self._obs_type == "visual":
            x = F.elu(self.conv1(obs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))
            x = x.flatten(start_dim=-3)
            return self.out_norm(x)
        else:
            return self.mlp(obs)

    def posterior_logits(self, h: Tensor, obs_emb: Tensor) -> Tensor:
        """Compute z_posterior_logits from ``(h_t, obs_emb)`` — takes BOTH."""
        combined = torch.cat([h, obs_emb], dim=-1)
        logits = self.post_head(combined)
        return logits.view(*h.shape[:-1], self._n_cat, self._n_cls)


class DreamerV2DynamicsPredictor(nn.Module):
    """Prior network: ELU MLP over ``h_t`` → ``z_prior_logits`` (h only, no obs).

    ⚠ Identical to V3's DynamicsPredictor except uses ELU instead of SiLU.

    Parameters
    ----------
    d_h : int
    n_cat, n_cls : int
    d_hidden : int
    n_hidden : int
    """

    def __init__(
        self,
        d_h: int,
        n_cat: int,
        n_cls: int,
        d_hidden: int = 1024,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self._n_cat = n_cat
        self._n_cls = n_cls
        self.layers = _build_mlp_elu(d_h, d_hidden, n_cat * n_cls, n_hidden)

    def forward(self, h: Tensor) -> Tensor:
        """Prior logits from recurrent state only (ELU MLP)."""
        logits = self.layers(h)
        return logits.view(*h.shape[:-1], self._n_cat, self._n_cls)


# DreamerV2 transition is identical to V3: GRUCell(cat(z_flat, action) → h_next)
DreamerV2Transition = DreamerV3Transition


class DreamerV2Decoder(nn.Module):
    """Transpose-CNN decoder: ``(h_t, z_t)`` → reconstructed obs (ELU variant).

    Parameters
    ----------
    d_h, d_z, obs_channels, cnn_channels : see DreamerV3Decoder.
    """

    def __init__(
        self,
        d_h: int,
        d_z: int,
        obs_channels: int,
        cnn_channels: List[int],
    ) -> None:
        super().__init__()
        d_latent = d_h + d_z
        rev_ch = list(reversed(cnn_channels))
        start_size = 4

        self.in_proj = nn.Linear(d_latent, rev_ch[0] * start_size * start_size)
        self._start_channels = rev_ch[0]
        self._start_size = start_size

        layers: List[nn.Module] = []
        c_in = rev_ch[0]
        for c_out in rev_ch[1:]:
            layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ELU())  # ELU instead of SiLU
            c_in = c_out
        layers.append(nn.ConvTranspose2d(c_in, obs_channels, kernel_size=4, stride=2, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        z_flat = z.flatten(start_dim=-2) if z.dim() >= 2 else z
        x = torch.cat([h, z_flat], dim=-1)
        x = self.in_proj(x)
        if x.dim() == 1:
            x = x.view(self._start_channels, self._start_size, self._start_size)
        else:
            B = x.shape[0]
            x = x.view(B, self._start_channels, self._start_size, self._start_size)
        return self.layers(x)


class DreamerV2RewardHead(nn.Module):
    """Reward head predicting the **mean** of a Gaussian distribution (ELU MLP).

    Unlike DreamerV3's two-hot categorical regression, DreamerV2 trains the
    reward head with a simple Gaussian log-likelihood (MSE in practice).
    The head outputs a scalar mean; the variance is a fixed hyper-parameter
    during training.

    Parameters
    ----------
    d_h, d_z : int
    d_hidden, n_hidden : int
    """

    def __init__(
        self,
        d_h: int,
        d_z: int,
        d_hidden: int = 1024,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        # Output: 1 (mean of Gaussian) — no two-hot, no symlog
        self.layers = _build_mlp_elu(d_h + d_z, d_hidden, 1, n_hidden)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Return raw Gaussian mean (scalar per batch element)."""
        z_flat = z.flatten(start_dim=-2) if z.dim() >= 2 else z
        return self.layers(torch.cat([h, z_flat], dim=-1))


class DreamerV2ContinueHead(nn.Module):
    """Episode-continuation head: ELU MLP → sigmoid probability.

    Parameters
    ----------
    d_h, d_z : int
    d_hidden, n_hidden : int
    """

    def __init__(
        self,
        d_h: int,
        d_z: int,
        d_hidden: int = 1024,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.layers = _build_mlp_elu(d_h + d_z, d_hidden, 1, n_hidden)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Return continuation probability in ``[0, 1]``."""
        z_flat = z.flatten(start_dim=-2) if z.dim() >= 2 else z
        return torch.sigmoid(self.layers(torch.cat([h, z_flat], dim=-1)))


class DreamerV2Actor(nn.Module):
    """Policy head: ELU MLP → categorical logits (discrete) or ``(μ, σ)`` (continuous).

    Parameters
    ----------
    d_h, d_z : int
    d_action : int
    action_discrete : bool
    d_hidden, n_hidden : int
    """

    def __init__(
        self,
        d_h: int,
        d_z: int,
        d_action: int,
        action_discrete: bool = True,
        d_hidden: int = 1024,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self._action_discrete = action_discrete
        self._d_action = d_action
        out_dim = d_action if action_discrete else 2 * d_action
        self.layers = _build_mlp_elu(d_h + d_z, d_hidden, out_dim, n_hidden)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Return discrete logits or ``cat(mean, log_std)`` for continuous."""
        z_flat = z.flatten(start_dim=-2) if z.dim() >= 2 else z
        return self.layers(torch.cat([h, z_flat], dim=-1))

    def sample(self, h: Tensor, z: Tensor, temperature: float = 1.0) -> Tensor:
        """Sample action from actor distribution."""
        out = self.forward(h, z)
        if self._action_discrete:
            return torch.distributions.Categorical(
                logits=out / max(temperature, 1e-6)
            ).sample()
        else:
            mean, log_std = out.chunk(2, dim=-1)
            std = (log_std + math.log(temperature)).exp().clamp(min=1e-6)
            return torch.distributions.Normal(mean, std).rsample()


class DreamerV2Critic(nn.Module):
    """Value head predicting the **mean** of a Gaussian distribution (ELU MLP).

    Mirrors :class:`DreamerV2RewardHead` in structure but estimates state
    value rather than immediate reward.

    Parameters
    ----------
    d_h, d_z : int
    d_hidden, n_hidden : int
    """

    def __init__(
        self,
        d_h: int,
        d_z: int,
        d_hidden: int = 1024,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.layers = _build_mlp_elu(d_h + d_z, d_hidden, 1, n_hidden)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Return raw Gaussian mean value estimate (scalar per batch element)."""
        z_flat = z.flatten(start_dim=-2) if z.dim() >= 2 else z
        return self.layers(torch.cat([h, z_flat], dim=-1))


# ---------------------------------------------------------------------------
# DreamerV2Adapter
# ---------------------------------------------------------------------------

class DreamerV2Adapter(WorldModelAdapter):
    """Full DreamerV2 backend implementing :class:`WorldModelAdapter`.

    Key differences from :class:`~world_model_lens.backends.DreamerV3Adapter`:

    * **ELU activations** throughout all MLPs and CNN layers.
    * **Gaussian reward and value heads** — no two-hot encoding, no symlog.
      :meth:`reward_pred` and :meth:`value` return the scalar *mean* of the
      predicted Normal distribution.
    * **Straight-through gradient** for the discrete ``z_t``.  During
      inference the categorical is the same (softmax probabilities), but
      :meth:`straight_through_z` can be used to obtain a hard one-hot
      ``z`` with low-variance gradients for policy learning.

    Parameters
    ----------
    cfg : WorldModelConfig
    obs_type : ``"visual"`` | ``"vector"``
    obs_channels : int
    image_size : int
    cnn_channels : list of int
    d_hidden : int
    n_hidden : int
    action_discrete : bool
    """

    DEFAULT_CNN_CHANNELS: List[int] = [48, 96, 192, 384]

    def __init__(
        self,
        cfg: WorldModelConfig,
        obs_type: str = "visual",
        obs_channels: int = 3,
        image_size: int = 64,
        cnn_channels: Optional[List[int]] = None,
        d_hidden: int = 1024,
        n_hidden: int = 2,
        action_discrete: Optional[bool] = None,
    ) -> None:
        self._cfg = cfg
        self._obs_type = obs_type
        self._obs_channels = obs_channels
        self._image_size = image_size
        self._cnn_channels = cnn_channels or self.DEFAULT_CNN_CHANNELS
        self._d_hidden = d_hidden
        self._n_hidden = n_hidden
        self._action_discrete = (
            action_discrete if action_discrete is not None else cfg.action_discrete
        )
        self._d_action = cfg.d_action

        d_h   = cfg.d_h
        n_cat = cfg.n_cat
        n_cls = cfg.n_cls
        d_z   = n_cat * n_cls
        d_obs = cfg.d_obs

        # ── Sub-networks (ELU throughout) ─────────────────────────────
        self._encoder = DreamerV2Encoder(
            obs_type=obs_type,
            obs_channels=obs_channels,
            image_size=image_size,
            cnn_channels=self._cnn_channels,
            d_obs=d_obs,
            d_hidden=d_hidden,
            n_hidden=n_hidden,
            d_h=d_h,
            n_cat=n_cat,
            n_cls=n_cls,
        )
        self._transition = DreamerV2Transition(
            d_z=d_z,
            d_action_in=self._d_action,
            d_h=d_h,
        )
        self._dynamics = DreamerV2DynamicsPredictor(
            d_h=d_h,
            n_cat=n_cat,
            n_cls=n_cls,
            d_hidden=d_hidden,
            n_hidden=n_hidden,
        )
        self._decoder = DreamerV2Decoder(
            d_h=d_h,
            d_z=d_z,
            obs_channels=obs_channels,
            cnn_channels=self._cnn_channels,
        )
        self._reward_head = DreamerV2RewardHead(
            d_h=d_h, d_z=d_z,
            d_hidden=d_hidden, n_hidden=n_hidden,
        )
        self._cont_head = DreamerV2ContinueHead(
            d_h=d_h, d_z=d_z,
            d_hidden=d_hidden, n_hidden=n_hidden,
        )
        self._actor = DreamerV2Actor(
            d_h=d_h, d_z=d_z,
            d_action=self._d_action,
            action_discrete=self._action_discrete,
            d_hidden=d_hidden,
            n_hidden=n_hidden,
        )
        self._critic = DreamerV2Critic(
            d_h=d_h, d_z=d_z,
            d_hidden=d_hidden, n_hidden=n_hidden,
        )

    # ── Sub-network list (for device/mode management) ─────────────────────
    @property
    def _all_modules(self) -> List[Tuple[str, nn.Module]]:
        return [
            ("encoder",    self._encoder),
            ("transition", self._transition),
            ("dynamics",   self._dynamics),
            ("decoder",    self._decoder),
            ("reward",     self._reward_head),
            ("cont",       self._cont_head),
            ("actor",      self._actor),
            ("critic",     self._critic),
        ]

    # ── WorldModelAdapter abstract methods ───────────────────────────────

    def encode(self, obs: Tensor) -> Tensor:
        """Encode observation → embedding via ELU CNN or MLP."""
        return self._encoder(obs)

    def initial_state(self, batch_size: int = 1) -> Tensor:
        """Return zeros as the initial recurrent state ``(d_h,)``."""
        dev = next(
            (p.device for _, m in self._all_modules for p in m.parameters()),
            torch.device("cpu"),
        )
        return torch.zeros(self._cfg.d_h, device=dev)

    def dynamics_step(
        self, h: Tensor, z: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """One RSSM step: ``(h, z, a) → (h_next, z_prior_logits)``.

        Uses straight-through z internally for the GRU input so that
        gradients can flow during training.  In inference mode this is
        equivalent to using soft ``z_t``.

        ⚠ The prior is computed from ``h_next`` **only** — no observation.
        """
        # Use soft z for transition (straight-through not needed here;
        # gradient flows through z_flat → GRU input during training)
        z_flat = z.flatten()
        if self._action_discrete:
            a_enc = F.one_hot(action.long().squeeze(), self._d_action).float()
        else:
            a_enc = action.float()
        h_next = self._transition(h, z_flat, a_enc)
        # ⚠ Prior: h_next → z_prior_logits (no observation)
        z_prior_logits = self._dynamics(h_next)
        return h_next, z_prior_logits

    def posterior(self, h: Tensor, obs_emb: Tensor) -> Tensor:
        """Compute posterior logits from ``(h_t, obs_emb)`` — takes BOTH."""
        return self._encoder.posterior_logits(h, obs_emb)

    # ── Optional heads ───────────────────────────────────────────────────

    def reward_pred(self, h: Tensor, z: Tensor) -> Tensor:
        """Predict reward as the *mean* of a Gaussian (scalar, no symlog)."""
        return self._reward_head(h, z).squeeze(-1)

    def cont_pred(self, h: Tensor, z: Tensor) -> Tensor:
        """Predict episode-continuation probability in ``[0, 1]``."""
        return self._cont_head(h, z)

    def actor(self, h: Tensor, z: Tensor) -> Tensor:
        """Return action logits (discrete) or ``(mean, log_std)`` (continuous)."""
        return self._actor(h, z)

    def value(self, h: Tensor, z: Tensor) -> Tensor:
        """Estimate state value as Gaussian mean (scalar, no symlog)."""
        return self._critic(h, z).squeeze(-1)

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        """Reconstruct observation from latent state via ELU decoder."""
        return self._decoder(h, z)

    # ── Straight-through helper ───────────────────────────────────────────

    def straight_through_z(self, logits: Tensor, temperature: float = 1.0) -> Tensor:
        """Produce hard one-hot ``z`` with straight-through gradients.

        Convenience wrapper around the module-level
        :func:`straight_through_z` function.

        Parameters
        ----------
        logits : Tensor
            Shape ``(n_cat, n_cls)`` — raw logits from :meth:`posterior`.
        temperature : float
            Softmax temperature before argmax.

        Returns
        -------
        Tensor
            Hard one-hot ``z``, shape ``(n_cat, n_cls)``.
        """
        return straight_through_z(logits, temperature)

    # ── Parameter access ─────────────────────────────────────────────────

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        """Yield ``(name, param)`` with DreamerV2-style naming."""
        for prefix, module in self._all_modules:
            for name, param in module.named_parameters():
                yield f"{prefix}.{name}", param

    # ── Device / mode ────────────────────────────────────────────────────

    def to(self, device: Union[str, torch.device]) -> "DreamerV2Adapter":
        dev = torch.device(device)
        for _, m in self._all_modules:
            m.to(dev)
        return self

    def eval(self) -> "DreamerV2Adapter":
        for _, m in self._all_modules:
            m.eval()
        return self

    def train(self, mode: bool = True) -> "DreamerV2Adapter":
        for _, m in self._all_modules:
            m.train(mode)
        return self

    # ── Checkpoint ───────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Tensor]:
        """Flat state dict with DreamerV2-style parameter names."""
        return {n: p.clone() for n, p in self.named_parameters()}

    def load_state_dict(
        self, state_dict: Dict[str, Tensor], strict: bool = True
    ) -> None:
        """Load parameters from a flat state dict."""
        own_keys = {n for n, _ in self.named_parameters()}
        ckpt_keys = set(state_dict.keys())
        if strict:
            missing    = own_keys - ckpt_keys
            unexpected = ckpt_keys - own_keys
            if missing or unexpected:
                raise RuntimeError(
                    f"load_state_dict (strict=True): "
                    f"missing={sorted(missing)[:5]}, "
                    f"unexpected={sorted(unexpected)[:5]}"
                )
        prefix_to_module = {p: m for p, m in self._all_modules}
        mod_dicts: Dict[str, Dict[str, Tensor]] = {p: {} for p in prefix_to_module}
        for name, tensor in state_dict.items():
            parts = name.split(".", 1)
            if len(parts) == 2 and parts[0] in mod_dicts:
                mod_dicts[parts[0]][parts[1]] = tensor
        for prefix, module in self._all_modules:
            if mod_dicts[prefix]:
                module.load_state_dict(mod_dicts[prefix], strict=strict)

    @classmethod
    def infer_config(cls, state_dict: Dict[str, Tensor]) -> WorldModelConfig:
        """Infer :class:`WorldModelConfig` from a state dict.

        Uses the same heuristics as DreamerV3's ``infer_config``; the
        reward head now has output dim 1 (Gaussian mean) rather than 255.
        """
        gru_wh = "transition.gru.weight_hh"
        if gru_wh not in state_dict:
            raise KeyError(f"Cannot infer config: '{gru_wh}' not in state_dict.")
        d_h = state_dict[gru_wh].shape[1]

        dyn_keys = sorted(
            [k for k in state_dict if k.startswith("dynamics.layers") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[2]),
        )
        d_z = state_dict[dyn_keys[-1]].shape[0]
        sqrt_dz = int(math.isqrt(d_z))
        if sqrt_dz * sqrt_dz == d_z:
            n_cat = n_cls = sqrt_dz
        elif d_z % 32 == 0:
            n_cat, n_cls = 32, d_z // 32
        else:
            n_cat, n_cls = 1, d_z

        gru_wi = "transition.gru.weight_ih"
        d_action = (state_dict[gru_wi].shape[1] - d_z) if gru_wi in state_dict else 1

        if "encoder.conv1.weight" in state_dict:
            obs_type, d_obs = "visual", 1
        elif "encoder.mlp.0.weight" in state_dict:
            obs_type, d_obs = "vector", state_dict["encoder.mlp.0.weight"].shape[1]
        else:
            obs_type, d_obs = "unknown", 1

        return WorldModelConfig(
            d_h=d_h, d_action=max(d_action, 1), d_obs=max(d_obs, 1),
            n_cat=n_cat, n_cls=n_cls,
            backend="dreamerv2",
            encoder_type="cnn" if obs_type == "visual" else "mlp",
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        cfg: Optional[WorldModelConfig] = None,
        strict: bool = True,
        **kwargs,
    ) -> "DreamerV2Adapter":
        """Load from a checkpoint file.  Config is inferred if not provided."""
        path = Path(path)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in raw and isinstance(raw[key], dict):
                    raw = raw[key]
                    break
        if cfg is None:
            cfg = cls.infer_config(raw)
        adapter = cls(cfg=cfg, **kwargs)
        adapter.load_state_dict(raw, strict=strict)
        return adapter

    # ── Metadata ─────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return "dreamer_v2"

    @property
    def cfg(self) -> WorldModelConfig:
        return self._cfg

    def __repr__(self) -> str:
        return (
            f"DreamerV2Adapter("
            f"obs={self._obs_type}, "
            f"d_h={self._cfg.d_h}, "
            f"n_cat={self._cfg.n_cat}, n_cls={self._cfg.n_cls}, "
            f"d_action={self._d_action}, "
            f"discrete={self._action_discrete})"
        )
