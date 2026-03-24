"""DreamerV3 backend adapter for world_model_lens.

Implements the full DreamerV3 architecture (Hafner et al., 2023 — "Mastering
Diverse Domains with World Models") as a :class:`WorldModelAdapter`.

Architecture overview
---------------------
The model is a Recurrent State-Space Model (RSSM) with:

* **Encoder** — CNN (visual) or MLP (vector) maps raw obs to ``obs_emb``.
  The *posterior* head then concatenates ``(h_t, obs_emb)`` → ``z_post_logits``.
  ⚠ The posterior takes BOTH ``h_t`` and the observation embedding.

* **DynamicsPredictor** — MLP maps ONLY ``h_t`` → ``z_prior_logits``
  (no observation access). ⚠ The prior takes ONLY ``h_t``.

* **Transition** — ``GRUCell(cat(z_flat, a_onehot))`` → ``h_{t+1}``.

* **Decoder** — Transpose-CNN maps ``(h_t, z_t)`` → reconstructed obs.

* **RewardHead / Critic** — MLP + two-hot distribution over 255 symlog-spaced
  bins, enabling robust regression on heavy-tailed targets.

* **ContinueHead** — MLP + sigmoid → episode-continuation probability.

* **Actor** — MLP → categorical logits (discrete) or ``(μ, log σ)`` (continuous).

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
from world_model_lens.core.config import WorldModelConfig


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def symlog(x: Tensor) -> Tensor:
    """Symmetric log transform: ``sign(x) * log(|x| + 1)``.

    Compresses large values while preserving the sign and zero.
    Used to normalise reward and value targets in DreamerV3.

    Parameters
    ----------
    x : Tensor
        Input tensor of any shape.

    Returns
    -------
    Tensor
        Same shape as *x*.

    Examples
    --------
    >>> symlog(torch.tensor([0., 1., 10., -5.]))
    tensor([ 0.0000,  0.6931,  2.3979, -1.7918])
    """
    return x.sign() * (x.abs() + 1.0).log()


def symexp(x: Tensor) -> Tensor:
    """Inverse of :func:`symlog`: ``sign(x) * (exp(|x|) - 1)``.

    Parameters
    ----------
    x : Tensor
        Symlog-transformed tensor.

    Returns
    -------
    Tensor
        Original-scale tensor.

    Examples
    --------
    >>> symexp(symlog(torch.tensor([0., 1., 10., -5.])))
    tensor([ 0.,  1., 10., -5.])   # (up to float precision)
    """
    return x.sign() * (x.abs().exp() - 1.0)


def twohot_encode(x: Tensor, bins: Tensor) -> Tensor:
    """Encode scalar(s) as a two-hot vector over *bins*.

    For each scalar value *x*, finds the two adjacent bins that bracket it
    and assigns fractional weights (summing to 1) to those two positions.
    All other bins receive weight 0.

    The *bins* tensor represents the expected value of each class in symlog
    space (evenly spaced between -20 and 20 by default).  Pass the raw
    (untransformed) target value — this function applies :func:`symlog`
    internally before finding the bracket.

    Parameters
    ----------
    x : Tensor
        Scalar targets, shape ``(*batch,)``.
    bins : Tensor
        1-D bin centres, shape ``(B,)`` in symlog space.

    Returns
    -------
    Tensor
        Two-hot tensor of shape ``(*batch, B)``.

    Examples
    --------
    >>> bins = torch.linspace(-20, 20, 255)
    >>> enc = twohot_encode(torch.tensor([1.0, -3.0]), bins)
    >>> enc.shape
    torch.Size([2, 255])
    >>> enc.sum(-1)               # each row sums to 1
    tensor([1., 1.])
    """
    B = bins.shape[0]
    # Work in symlog space
    x_sym = symlog(x)                             # (*batch,)
    x_sym = x_sym.unsqueeze(-1)                   # (*batch, 1)
    bins_ = bins.to(x_sym.device)                 # (B,)

    # Clamp to valid range
    x_sym_clamped = x_sym.clamp(bins_[0], bins_[-1])

    # Find lower bin index via searchsorted (returns insertion point)
    lo = torch.searchsorted(bins_.contiguous(), x_sym_clamped.contiguous()) - 1
    lo = lo.clamp(0, B - 2)                       # (*batch, 1)
    hi = lo + 1                                   # (*batch, 1)

    # Fractional weights
    lo_val = bins_[lo]                            # (*batch, 1)
    hi_val = bins_[hi]                            # (*batch, 1)
    span = (hi_val - lo_val).clamp(min=1e-8)
    weight_hi = (x_sym_clamped - lo_val) / span  # (*batch, 1)
    weight_lo = 1.0 - weight_hi                  # (*batch, 1)

    # Scatter into one-hot tensors
    batch_shape = x.shape
    out = torch.zeros(*batch_shape, B, device=x.device, dtype=x.dtype)
    lo_squeezed = lo.squeeze(-1)                  # (*batch,)
    hi_squeezed = hi.squeeze(-1)                  # (*batch,)
    out.scatter_(-1, lo_squeezed.unsqueeze(-1), weight_lo)
    out.scatter_(-1, hi_squeezed.unsqueeze(-1), weight_hi)
    return out


def twohot_decode(logits: Tensor, bins: Tensor) -> Tensor:
    """Decode two-hot logits to scalar predictions via symexp of expected bin.

    Computes ``symexp(Σ softmax(logits) * bins)`` — i.e. the expected value
    in symlog space, then inverts the symlog transform.

    Parameters
    ----------
    logits : Tensor
        Raw (un-normalised) two-hot logits, shape ``(*batch, B)``.
    bins : Tensor
        Bin centres in symlog space, shape ``(B,)``.

    Returns
    -------
    Tensor
        Scalar predictions in original space, shape ``(*batch,)``.

    Examples
    --------
    >>> bins = torch.linspace(-20, 20, 255)
    >>> x = torch.tensor([2.5, -1.0])
    >>> twohot_decode(twohot_encode(x, bins), bins)
    tensor([ 2.5000, -1.0000])    # near-exact round-trip
    """
    probs = logits.softmax(dim=-1)                    # (*batch, B)
    bins_ = bins.to(logits.device)
    expected_symlog = (probs * bins_).sum(dim=-1)     # (*batch,)
    return symexp(expected_symlog)


def _cnn_output_size(image_size: int, cnn_channels: List[int]) -> int:
    """Spatial size after stacking ``len(cnn_channels)`` Conv2d(k=4, s=2, p=0) layers."""
    h = image_size
    for _ in cnn_channels:
        h = (h - 4) // 2 + 1
    return h * h * cnn_channels[-1]


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_hidden: int = 2,
    act: type = nn.SiLU,
    use_norm: bool = True,
) -> nn.Sequential:
    """Build a fully-connected MLP with optional LayerNorm.

    Produces an ``nn.Sequential`` whose integer-keyed layers expose named
    parameters as ``layers.<idx>.weight`` — matching the convention required
    by ``DreamerV3DynamicsPredictor``, ``DreamerV3RewardHead``, etc.

    Structure (n_hidden=2)::

        Linear(in, h) → [LayerNorm(h)] → Act()
        Linear(h, h)  → [LayerNorm(h)] → Act()
        Linear(h, out)

    Parameters
    ----------
    in_dim, hidden_dim, out_dim : int
        Input, hidden, and output widths.
    n_hidden : int
        Number of hidden layers (each followed by norm + activation).
    act : type
        Activation class (default ``nn.SiLU``).
    use_norm : bool
        Whether to insert ``nn.LayerNorm`` after each hidden linear.

    Returns
    -------
    nn.Sequential
    """
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        layers.append(nn.Linear(d, hidden_dim))
        if use_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act())
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Component nn.Modules
# ---------------------------------------------------------------------------

class DreamerV3Encoder(nn.Module):
    """Observation encoder + posterior head.

    Two modes:

    * **Visual** — four stride-2 Conv2d layers with channel progression
      ``[obs_channels → 48 → 96 → 192 → 384]``, followed by LayerNorm.
    * **Vector** — MLP over the raw observation vector.

    In both modes a ``post_head`` linear maps
    ``concat(h_t, obs_emb) → z_posterior_logits``.

    ⚠ The posterior takes BOTH ``h_t`` and the observation.  The ``forward``
    method returns only the obs embedding; the posterior logits are computed
    via :meth:`posterior_logits`.

    Parameters
    ----------
    obs_type : ``"visual"`` | ``"vector"``
    obs_channels : int
        Number of input channels (1=grayscale, 3=RGB) for visual obs.
    image_size : int
        Spatial extent H=W of the input image (for visual obs only).
    cnn_channels : list of int
        Output channel counts for each conv layer (default 4 layers).
    d_obs : int
        Raw observation dimension for vector obs.
    d_hidden : int
        Hidden size for vector-obs MLP layers.
    n_hidden : int
        Number of hidden layers in the vector MLP.
    d_h : int
        Recurrent hidden state size (needed for ``post_head`` input size).
    n_cat, n_cls : int
        Categorical latent dimensions (output of ``post_head``).
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
            # Named conv1..conv4 so parameter paths match `encoder.conv1.weight`
            ch = cnn_channels
            self.conv1 = nn.Conv2d(obs_channels, ch[0], kernel_size=4, stride=2)
            self.conv2 = nn.Conv2d(ch[0], ch[1], kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(ch[1], ch[2], kernel_size=4, stride=2)
            self.conv4 = nn.Conv2d(ch[2], ch[3], kernel_size=4, stride=2)
            d_emb = _cnn_output_size(image_size, cnn_channels)
            self.out_norm = nn.LayerNorm(d_emb)
        else:  # vector
            mlp_layers: List[nn.Module] = []
            d = d_obs
            for _ in range(max(n_hidden, 1)):
                mlp_layers.extend([nn.Linear(d, d_hidden), nn.LayerNorm(d_hidden), nn.SiLU()])
                d = d_hidden
            self.mlp = nn.Sequential(*mlp_layers)
            d_emb = d_hidden

        self._d_emb = d_emb
        # Posterior head: Linear([h_t ‖ obs_emb] → z_post_logits)
        # ⚠ Takes BOTH h_t and observation embedding
        self.post_head = nn.Linear(d_h + d_emb, n_cat * n_cls)

    def forward(self, obs: Tensor) -> Tensor:
        """Encode observation to embedding vector (no ``h_t`` involved here).

        Parameters
        ----------
        obs : Tensor
            ``(*batch, C, H, W)`` for visual or ``(*batch, d_obs)`` for vector.

        Returns
        -------
        Tensor
            Observation embedding ``(*batch, d_emb)``.
        """
        if self._obs_type == "visual":
            x = F.silu(self.conv1(obs))
            x = F.silu(self.conv2(x))
            x = F.silu(self.conv3(x))
            x = F.silu(self.conv4(x))
            # Flatten spatial + channel dims; works for batched and unbatched
            x = x.flatten(start_dim=-3)
            return self.out_norm(x)
        else:
            return self.mlp(obs)

    def posterior_logits(self, h: Tensor, obs_emb: Tensor) -> Tensor:
        """Compute z_posterior_logits from ``(h_t, obs_emb)``.

        ⚠ Takes BOTH ``h_t`` and the observation embedding (computed by
        :meth:`forward`).  Concatenates them before the linear projection.

        Parameters
        ----------
        h : Tensor
            Recurrent state ``(*batch, d_h)``.
        obs_emb : Tensor
            Observation embedding ``(*batch, d_emb)`` from :meth:`forward`.

        Returns
        -------
        Tensor
            Logits shape ``(*batch, n_cat, n_cls)``.
        """
        combined = torch.cat([h, obs_emb], dim=-1)
        logits = self.post_head(combined)
        return logits.view(*h.shape[:-1], self._n_cat, self._n_cls)


class DreamerV3DynamicsPredictor(nn.Module):
    """Prior network: MLP over ``h_t`` → ``z_prior_logits``.

    ⚠ Takes ONLY ``h_t`` (no observation).  This is the key distinction from
    the posterior: the prior predicts *without* seeing the current observation.

    Parameters
    ----------
    d_h : int
        Recurrent state size.
    n_cat, n_cls : int
        Categorical latent dimensions.
    d_hidden : int
        MLP hidden width.
    n_hidden : int
        Number of hidden layers.
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
        # layers.0.weight, layers.3.weight, ... match the required naming
        self.layers = _build_mlp(d_h, d_hidden, n_cat * n_cls, n_hidden)

    def forward(self, h: Tensor) -> Tensor:
        """Predict prior logits from recurrent state only.

        Parameters
        ----------
        h : Tensor
            Recurrent state ``(*batch, d_h)``.

        Returns
        -------
        Tensor
            Prior logits ``(*batch, n_cat, n_cls)``.
        """
        logits = self.layers(h)
        return logits.view(*h.shape[:-1], self._n_cat, self._n_cls)


class DreamerV3Transition(nn.Module):
    """Recurrent transition: ``GRUCell(cat(z_flat, a)) → h_next``.

    Parameters
    ----------
    d_z : int
        Flattened stochastic latent size (``n_cat * n_cls``).
    d_action_in : int
        Action input size (``d_action`` for one-hot discrete or raw continuous).
    d_h : int
        GRU hidden state size.
    """

    def __init__(self, d_z: int, d_action_in: int, d_h: int) -> None:
        super().__init__()
        # Named `gru` → parameter paths become `transition.gru.weight_ih` etc.
        self.gru = nn.GRUCell(input_size=d_z + d_action_in, hidden_size=d_h)

    def forward(self, h: Tensor, z_flat: Tensor, action: Tensor) -> Tensor:
        """Single GRU step.

        Parameters
        ----------
        h : Tensor
            Previous hidden state ``(d_h,)`` or ``(B, d_h)``.
        z_flat : Tensor
            Flattened stochastic latent ``(d_z,)`` or ``(B, d_z)``.
        action : Tensor
            Action vector ``(d_action_in,)`` or ``(B, d_action_in)``.

        Returns
        -------
        Tensor
            Updated hidden state, same shape as ``h``.
        """
        inp = torch.cat([z_flat, action], dim=-1)
        # GRUCell requires batch dimension; add/remove as needed
        unbatched = inp.dim() == 1
        if unbatched:
            inp = inp.unsqueeze(0)
            h   = h.unsqueeze(0)
        h_next = self.gru(inp, h)
        return h_next.squeeze(0) if unbatched else h_next


class DreamerV3Decoder(nn.Module):
    """Transpose-CNN decoder: ``(h_t, z_t)`` → reconstructed obs.

    Projects the latent to a 4×4 spatial feature map, then applies four
    ConvTranspose2d(k=4, s=2, p=1) layers for exact 2× upsampling per step,
    reaching 64×64 for a 64-wide image (4 → 8 → 16 → 32 → 64).

    Parameters
    ----------
    d_h : int
        Recurrent state size.
    d_z : int
        Flattened stochastic latent size.
    obs_channels : int
        Output image channels (1 or 3).
    cnn_channels : list of int
        CNN channel progression used in the encoder (reversed for decoder).
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
        rev_ch = list(reversed(cnn_channels))   # [384, 192, 96, 48]
        start_size = 4                           # start from 4×4 spatial

        # Named `in_proj` for the linear projection
        self.in_proj = nn.Linear(d_latent, rev_ch[0] * start_size * start_size)
        self._start_channels = rev_ch[0]
        self._start_size = start_size

        # ConvTranspose layers: 4→8→16→32→64 (for 64×64 target)
        # ConvTranspose2d(k=4, s=2, p=1): output = 2 × input (exact)
        layers: List[nn.Module] = []
        c_in = rev_ch[0]
        for c_out in rev_ch[1:]:
            layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.SiLU())
            c_in = c_out
        # Final layer: no activation, outputs obs_channels
        layers.append(nn.ConvTranspose2d(c_in, obs_channels, kernel_size=4, stride=2, padding=1))
        # Named `layers` → `decoder.layers.0.weight`, `decoder.layers.2.weight`, ...
        self.layers = nn.Sequential(*layers)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Reconstruct observation from latent state.

        Parameters
        ----------
        h : Tensor
            Recurrent state ``(*batch, d_h)``.
        z : Tensor
            Stochastic latent ``(*batch, n_cat, n_cls)`` or ``(*batch, d_z)``.

        Returns
        -------
        Tensor
            Reconstructed observation ``(*batch, C, H, W)``.
        """
        z_flat = z.flatten(start_dim=-2) if z.dim() > 1 and z.shape[-2:] != z.shape[-1:] else z.flatten(start_dim=-2 if z.dim() >= 2 else 0)
        x = torch.cat([h, z_flat], dim=-1)
        x = self.in_proj(x)
        # Reshape to spatial feature map
        if x.dim() == 1:  # unbatched
            x = x.view(self._start_channels, self._start_size, self._start_size)
        else:
            B = x.shape[0]
            x = x.view(B, self._start_channels, self._start_size, self._start_size)
        return self.layers(x)


class DreamerV3RewardHead(nn.Module):
    """Reward prediction head using two-hot categorical regression.

    Predicts a distribution over ``reward_bins`` symlog-spaced bins.
    The expected value (decoded via :func:`twohot_decode`) gives the
    predicted reward in original scale.

    Parameters
    ----------
    d_h, d_z : int
        Latent state dimensions.
    d_hidden, n_hidden : int
        MLP hidden width and depth.
    reward_bins : int
        Number of two-hot bins (DreamerV3 default: 255).
    """

    def __init__(
        self,
        d_h: int,
        d_z: int,
        d_hidden: int = 1024,
        n_hidden: int = 2,
        reward_bins: int = 255,
    ) -> None:
        super().__init__()
        self._bins = reward_bins
        self.layers = _build_mlp(d_h + d_z, d_hidden, reward_bins, n_hidden)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Return raw two-hot logits (``reward_bins``-dim vector)."""
        z_flat = z.flatten(start_dim=-2) if z.dim() >= 2 else z
        return self.layers(torch.cat([h, z_flat], dim=-1))


class DreamerV3ContinueHead(nn.Module):
    """Episode-continuation head: MLP → sigmoid probability.

    Parameters
    ----------
    d_h, d_z : int
        Latent state dimensions.
    d_hidden, n_hidden : int
        MLP hidden width and depth.
    """

    def __init__(
        self,
        d_h: int,
        d_z: int,
        d_hidden: int = 1024,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.layers = _build_mlp(d_h + d_z, d_hidden, 1, n_hidden)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Return continuation probability in ``[0, 1]``.

        Returns
        -------
        Tensor
            Shape ``(*batch, 1)``.
        """
        z_flat = z.flatten(start_dim=-2) if z.dim() >= 2 else z
        return torch.sigmoid(self.layers(torch.cat([h, z_flat], dim=-1)))


class DreamerV3Actor(nn.Module):
    """Policy head supporting discrete and continuous action spaces.

    * **Discrete** — outputs ``d_action`` logits; sample with
      ``Categorical(logits=out).sample()``.
    * **Continuous** — outputs ``(μ, log σ)`` concatenated; split at
      ``d_action`` for the Normal distribution parameters.

    Parameters
    ----------
    d_h, d_z : int
        Latent state dimensions.
    d_action : int
        Action space size.
    action_discrete : bool
        Whether the action space is discrete.
    d_hidden, n_hidden : int
        MLP hidden width and depth.
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
        self.layers = _build_mlp(d_h + d_z, d_hidden, out_dim, n_hidden)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Return action logits (discrete) or ``cat(mean, log_std)`` (continuous).

        Returns
        -------
        Tensor
            Shape ``(*batch, d_action)`` or ``(*batch, 2 * d_action)``.
        """
        z_flat = z.flatten(start_dim=-2) if z.dim() >= 2 else z
        return self.layers(torch.cat([h, z_flat], dim=-1))

    def sample(self, h: Tensor, z: Tensor, temperature: float = 1.0) -> Tensor:
        """Sample an action from the actor's distribution.

        Parameters
        ----------
        h, z : Tensor
            Latent state.
        temperature : float
            Softmax temperature (discrete) or log_std offset (continuous).

        Returns
        -------
        Tensor
            Sampled action, shape ``(*batch,)`` for discrete or
            ``(*batch, d_action)`` for continuous.
        """
        out = self.forward(h, z)
        if self._action_discrete:
            return torch.distributions.Categorical(
                logits=out / max(temperature, 1e-6)
            ).sample()
        else:
            mean, log_std = out.chunk(2, dim=-1)
            std = (log_std + math.log(temperature)).exp().clamp(min=1e-6)
            return torch.distributions.Normal(mean, std).rsample()


class DreamerV3Critic(nn.Module):
    """Value (critic) head using two-hot categorical regression.

    Mirrors :class:`DreamerV3RewardHead` but estimates the *state value*
    rather than the immediate reward.

    Parameters
    ----------
    d_h, d_z : int
        Latent state dimensions.
    d_hidden, n_hidden : int
        MLP hidden width and depth.
    reward_bins : int
        Number of two-hot bins.
    """

    def __init__(
        self,
        d_h: int,
        d_z: int,
        d_hidden: int = 1024,
        n_hidden: int = 2,
        reward_bins: int = 255,
    ) -> None:
        super().__init__()
        self._bins = reward_bins
        self.layers = _build_mlp(d_h + d_z, d_hidden, reward_bins, n_hidden)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Return raw two-hot logits for the value estimate."""
        z_flat = z.flatten(start_dim=-2) if z.dim() >= 2 else z
        return self.layers(torch.cat([h, z_flat], dim=-1))


# ---------------------------------------------------------------------------
# DreamerV3Adapter
# ---------------------------------------------------------------------------

class DreamerV3Adapter(WorldModelAdapter):
    """Full DreamerV3 backend implementing :class:`WorldModelAdapter`.

    Parameters
    ----------
    cfg : WorldModelConfig
        Architectural config with at minimum ``d_h``, ``d_action``, ``d_obs``,
        ``n_cat``, ``n_cls``.
    obs_type : ``"visual"`` | ``"vector"``
        Whether observations are images or flat vectors.
    obs_channels : int
        Number of image channels (1–3 for visual; ignored for vector).
    image_size : int
        H=W of the input image (ignored for vector obs).
    cnn_channels : list of int
        Four output channel counts for the encoder CNN.
    d_hidden : int
        Hidden width for all MLP sub-networks.
    n_hidden : int
        Number of hidden layers in all MLP sub-networks.
    reward_bins : int
        Two-hot bin count for reward and value heads.
    action_discrete : bool
        Whether the action space is discrete.

    Notes
    -----
    All ``nn.Module`` sub-networks are stored as plain instance attributes
    (not registered via ``nn.Module.register_module``) because
    :class:`WorldModelAdapter` does not inherit from ``nn.Module``.  Use
    :meth:`named_parameters` and :meth:`to` / :meth:`eval` / :meth:`train`
    for device and mode management.
    """

    # Default CNN channel progression (user-specified: [1,2,3,48,96,192,384]
    # is interpreted as obs_channels ∈ {1,2,3} + cnn_channels [48,96,192,384])
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
        reward_bins: int = 255,
        action_discrete: Optional[bool] = None,
    ) -> None:
        self._cfg = cfg
        self._obs_type = obs_type
        self._obs_channels = obs_channels
        self._image_size = image_size
        self._cnn_channels = cnn_channels or self.DEFAULT_CNN_CHANNELS
        self._d_hidden = d_hidden
        self._n_hidden = n_hidden
        self._reward_bins = reward_bins
        self._action_discrete = (
            action_discrete if action_discrete is not None else cfg.action_discrete
        )
        self._d_action = cfg.d_action
        d_h   = cfg.d_h
        n_cat = cfg.n_cat
        n_cls = cfg.n_cls
        d_z   = n_cat * n_cls
        d_obs = cfg.d_obs
        d_action_in = d_z  # dummy for cont actions; overridden below
        # For discrete actions, action is one-hot (d_action dims)
        # For continuous, action is raw vector (d_action dims)
        d_action_in = self._d_action

        # ── Sub-networks ──────────────────────────────────────────────
        self._encoder = DreamerV3Encoder(
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
        self._transition = DreamerV3Transition(
            d_z=d_z,
            d_action_in=d_action_in,
            d_h=d_h,
        )
        self._dynamics = DreamerV3DynamicsPredictor(
            d_h=d_h,
            n_cat=n_cat,
            n_cls=n_cls,
            d_hidden=d_hidden,
            n_hidden=n_hidden,
        )
        self._decoder = DreamerV3Decoder(
            d_h=d_h,
            d_z=d_z,
            obs_channels=obs_channels,
            cnn_channels=self._cnn_channels,
        )
        self._reward_head = DreamerV3RewardHead(
            d_h=d_h, d_z=d_z,
            d_hidden=d_hidden, n_hidden=n_hidden,
            reward_bins=reward_bins,
        )
        self._cont_head = DreamerV3ContinueHead(
            d_h=d_h, d_z=d_z,
            d_hidden=d_hidden, n_hidden=n_hidden,
        )
        self._actor = DreamerV3Actor(
            d_h=d_h, d_z=d_z,
            d_action=self._d_action,
            action_discrete=self._action_discrete,
            d_hidden=d_hidden,
            n_hidden=n_hidden,
        )
        self._critic = DreamerV3Critic(
            d_h=d_h, d_z=d_z,
            d_hidden=d_hidden, n_hidden=n_hidden,
            reward_bins=reward_bins,
        )

        # Two-hot bin centres (in symlog space, evenly spaced −20 to 20)
        self.register_buffer(
            torch.linspace(-20.0, 20.0, reward_bins), name="_bins"
        )

    # ── Helper: register_buffer (adapter is not nn.Module) ───────────────────
    def register_buffer(self, tensor: Tensor, name: str = "_bins") -> None:
        """Store a non-parameter tensor attribute by *name*."""
        setattr(self, name, tensor)

    # ── All sub-networks as an ordered list (used by device/mode helpers) ─────
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

    # ── WorldModelAdapter abstract methods ───────────────────────────────────

    def encode(self, obs: Tensor) -> Tensor:
        """Encode observation → embedding (no ``h_t`` involvement).

        Parameters
        ----------
        obs : Tensor
            Raw observation ``(*batch, C, H, W)`` or ``(*batch, d_obs)``.

        Returns
        -------
        Tensor
            Observation embedding ``(*batch, d_emb)``.
        """
        return self._encoder(obs)

    def initial_state(self, batch_size: int = 1) -> Tensor:
        """Return zeros as the initial recurrent state.

        Returns
        -------
        Tensor
            Shape ``(d_h,)`` (unbatched, matching the single-step convention).
        """
        dev = next(iter(p for _, m in self._all_modules for p in m.parameters()),
                   None)
        device = dev.device if dev is not None else torch.device("cpu")
        return torch.zeros(self._cfg.d_h, device=device)

    def dynamics_step(self, h: Tensor, z: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """One RSSM recurrent step: ``(h, z, a) → (h_next, z_prior_logits)``.

        Sequence of operations:

        1. Encode action as one-hot (discrete) or use directly (continuous).
        2. Transition (GRUCell): ``cat(z_flat, action) → h_next``.
        3. Dynamics predictor (prior): ``h_next → z_prior_logits``.
           ⚠ ONLY ``h_next`` feeds the prior — no observation.

        Parameters
        ----------
        h : Tensor  ``(d_h,)``
        z : Tensor  ``(n_cat, n_cls)`` — current posterior latent (soft probs).
        action : Tensor — discrete index (scalar) or continuous vector.

        Returns
        -------
        h_next : Tensor ``(d_h,)``
        z_prior_logits : Tensor ``(n_cat, n_cls)``
        """
        z_flat = z.flatten()
        if self._action_discrete:
            a_enc = F.one_hot(action.long().squeeze(), self._d_action).float()
        else:
            a_enc = action.float()
        h_next = self._transition(h, z_flat, a_enc)
        # ⚠ Prior takes ONLY h_next — no observation here
        z_prior_logits = self._dynamics(h_next)
        return h_next, z_prior_logits

    def posterior(self, h: Tensor, obs_emb: Tensor) -> Tensor:
        """Compute z_posterior_logits from ``(h_t, obs_emb)``.

        ⚠ Takes BOTH ``h_t`` and the observation embedding.

        Parameters
        ----------
        h : Tensor
            Recurrent state ``(d_h,)``.
        obs_emb : Tensor
            Observation embedding from :meth:`encode`, shape ``(d_emb,)``.

        Returns
        -------
        Tensor
            Posterior logits ``(n_cat, n_cls)``.
        """
        return self._encoder.posterior_logits(h, obs_emb)

    def reward_pred(self, h: Tensor, z: Tensor) -> Tensor:
        """Predict reward via two-hot decode of the reward head's logits.

        Returns a scalar tensor (decoded via :func:`twohot_decode` + symexp).
        """
        logits = self._reward_head(h, z)
        return twohot_decode(logits, self._bins.to(h.device))

    def reward_logits(self, h: Tensor, z: Tensor) -> Tensor:
        """Return raw two-hot logits (255-dim) from the reward head."""
        return self._reward_head(h, z)

    def cont_pred(self, h: Tensor, z: Tensor) -> Tensor:
        """Predict episode-continuation probability in ``[0, 1]``."""
        return self._cont_head(h, z)

    def actor(self, h: Tensor, z: Tensor) -> Tensor:
        """Return action logits (discrete) or ``(mean, log_std)`` (continuous)."""
        return self._actor(h, z)

    def value(self, h: Tensor, z: Tensor) -> Tensor:
        """Estimate state value via two-hot decode of the critic's logits."""
        logits = self._critic(h, z)
        return twohot_decode(logits, self._bins.to(h.device))

    def value_logits(self, h: Tensor, z: Tensor) -> Tensor:
        """Return raw two-hot logits (255-dim) from the critic."""
        return self._critic(h, z)

    def decode(self, h: Tensor, z: Tensor) -> Tensor:
        """Reconstruct observation from latent state via the decoder."""
        return self._decoder(h, z)

    # ── Parameter access ─────────────────────────────────────────────────────

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        """Yield ``(name, param)`` with structured DreamerV3 naming.

        Parameter path format:
        ``<module>.<submodule>.<leaf_name>``

        Examples:
        ``encoder.conv1.weight``, ``transition.gru.weight_ih``,
        ``dynamics.layers.0.weight``, ``reward.layers.0.weight``
        """
        for prefix, module in self._all_modules:
            for name, param in module.named_parameters():
                yield f"{prefix}.{name}", param

    # ── Device / mode management ──────────────────────────────────────────────

    def to(self, device: Union[str, torch.device]) -> "DreamerV3Adapter":
        """Move all sub-networks and buffers to *device*."""
        dev = torch.device(device)
        for _, module in self._all_modules:
            module.to(dev)
        self._bins = self._bins.to(dev)
        return self

    def eval(self) -> "DreamerV3Adapter":
        """Switch all sub-networks to eval mode."""
        for _, module in self._all_modules:
            module.eval()
        return self

    def train(self, mode: bool = True) -> "DreamerV3Adapter":
        """Switch all sub-networks to train/eval mode."""
        for _, module in self._all_modules:
            module.train(mode)
        return self

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Tensor]:
        """Return a flat state dict with DreamerV3-style parameter names."""
        return {name: param.clone() for name, param in self.named_parameters()}

    def load_state_dict(self, state_dict: Dict[str, Tensor], strict: bool = True) -> None:
        """Load parameters from a flat state dict.

        Parameters
        ----------
        state_dict : dict
            ``{parameter_name: tensor}`` mapping produced by :meth:`state_dict`
            or :meth:`infer_config` + checkpoint loading.
        strict : bool
            If ``True`` (default), raise on missing or unexpected keys.
        """
        own_keys = set(name for name, _ in self.named_parameters())
        ckpt_keys = set(state_dict.keys())
        if strict:
            missing  = own_keys - ckpt_keys
            unexpected = ckpt_keys - own_keys
            if missing or unexpected:
                raise RuntimeError(
                    f"load_state_dict (strict=True): "
                    f"missing keys: {sorted(missing)[:5]}{'...' if len(missing)>5 else ''}, "
                    f"unexpected keys: {sorted(unexpected)[:5]}{'...' if len(unexpected)>5 else ''}"
                )

        prefix_to_module = {prefix: module for prefix, module in self._all_modules}
        # Rebuild per-module state dicts
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
        """Infer :class:`WorldModelConfig` from a checkpoint's state dict.

        Reads parameter shapes to reconstruct the architectural hyperparameters
        without needing a separately saved config file.

        Heuristics
        ----------
        * ``d_h``     ← ``transition.gru.weight_hh`` shape ``(3*d_h, d_h)``
        * ``d_z``     ← last linear layer of ``dynamics.layers`` output dim
        * ``n_cat``   ← 32 if ``d_z % 32 == 0``, else ``sqrt(d_z)``
        * ``n_cls``   ← ``d_z // n_cat``
        * ``d_action``← GRU input size minus d_z (``transition.gru.weight_ih``)
        * ``d_obs``   ← ``encoder.mlp.0.weight`` input dim (vector) or
                        from ``encoder.conv1.weight`` (visual)

        Parameters
        ----------
        state_dict : dict

        Returns
        -------
        WorldModelConfig
        """
        # d_h from GRU hidden-hidden weight: shape (3*d_h, d_h)
        gru_wh_key = "transition.gru.weight_hh"
        if gru_wh_key not in state_dict:
            raise KeyError(f"Cannot infer config: '{gru_wh_key}' not in state_dict.")
        d_h = state_dict[gru_wh_key].shape[1]

        # d_z from last linear layer in dynamics.layers
        dyn_weight_keys = sorted(
            [k for k in state_dict if k.startswith("dynamics.layers") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[2]),
        )
        if not dyn_weight_keys:
            raise KeyError("Cannot find 'dynamics.layers.*.weight' in state_dict.")
        d_z = state_dict[dyn_weight_keys[-1]].shape[0]

        # n_cat, n_cls
        sqrt_dz = int(math.isqrt(d_z))
        if sqrt_dz * sqrt_dz == d_z:
            n_cat = n_cls = sqrt_dz
        elif d_z % 32 == 0:
            n_cat = 32
            n_cls = d_z // 32
        else:
            n_cat = 1
            n_cls = d_z

        # d_action from GRU input size
        gru_wi_key = "transition.gru.weight_ih"
        if gru_wi_key in state_dict:
            d_action = state_dict[gru_wi_key].shape[1] - d_z
        else:
            d_action = 0

        # obs_type and d_obs
        if "encoder.conv1.weight" in state_dict:
            obs_type = "visual"
            obs_channels = state_dict["encoder.conv1.weight"].shape[1]
            d_obs = -1  # visual — placeholder
        elif "encoder.mlp.0.weight" in state_dict:
            obs_type = "vector"
            obs_channels = 1
            d_obs = state_dict["encoder.mlp.0.weight"].shape[1]
        else:
            obs_type = "unknown"
            obs_channels = 3
            d_obs = 1

        # Infer reward_bins from reward head
        rew_keys = sorted(
            [k for k in state_dict if k.startswith("reward.layers") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[2]),
        )
        reward_bins = state_dict[rew_keys[-1]].shape[0] if rew_keys else 255

        return WorldModelConfig(
            d_h=d_h,
            d_action=max(d_action, 1),
            d_obs=max(d_obs, 1),
            n_cat=n_cat,
            n_cls=n_cls,
            backend="dreamer",
            encoder_type="cnn" if obs_type == "visual" else "mlp",
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        cfg: Optional[WorldModelConfig] = None,
        strict: bool = True,
        **kwargs,
    ) -> "DreamerV3Adapter":
        """Load a :class:`DreamerV3Adapter` from a saved checkpoint.

        The checkpoint may be:

        * A plain ``torch.save`` of the adapter's :meth:`state_dict`.
        * A dict with a ``"state_dict"`` or ``"model_state_dict"`` key.

        Parameters
        ----------
        path : str | Path
            Path to the ``.pt`` / ``.pth`` checkpoint file.
        cfg : WorldModelConfig, optional
            If ``None``, the config is inferred via :meth:`infer_config`.
        strict : bool
            Passed to :meth:`load_state_dict`.
        **kwargs
            Extra keyword arguments forwarded to the constructor.

        Returns
        -------
        DreamerV3Adapter
        """
        path = Path(path)
        raw = torch.load(path, map_location="cpu", weights_only=False)

        # Unwrap common checkpoint wrappers
        if isinstance(raw, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in raw and isinstance(raw[key], dict):
                    raw = raw[key]
                    break

        state_dict: Dict[str, Tensor] = raw  # type: ignore[assignment]

        if cfg is None:
            cfg = cls.infer_config(state_dict)

        adapter = cls(cfg=cfg, **kwargs)
        adapter.load_state_dict(state_dict, strict=strict)
        return adapter

    # ── Metadata ─────────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return "dreamer_v3"

    @property
    def cfg(self) -> WorldModelConfig:
        return self._cfg

    def __repr__(self) -> str:
        return (
            f"DreamerV3Adapter("
            f"obs={self._obs_type}, "
            f"d_h={self._cfg.d_h}, "
            f"n_cat={self._cfg.n_cat}, n_cls={self._cfg.n_cls}, "
            f"d_action={self._d_action}, "
            f"discrete={self._action_discrete})"
        )
