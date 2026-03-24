"""IRIS (Imagination with auto-Regressive transformerS) backend adapter.

Implements the IRIS architecture (Micheli et al., 2023 — "Transformers are
Sample Efficient World Models") as a :class:`WorldModelAdapter`.

Architecture overview
---------------------
IRIS combines a **VQVAE** for discrete observation tokenisation with a
**GPT-style causal transformer** that predicts the next observation token
sequence given the history.

VQVAE:
    * Encoder CNN: ``(C, H, W)`` → feature map → ``(n_tokens, d_emb)``
      (e.g. 64×64 → 4 strided convs → 4×4 = 16 spatial locations per frame).
    * Codebook: ``(n_vocab, d_emb)`` learned embeddings; nearest-neighbour
      quantisation with straight-through gradient.
    * Decoder: transpose-CNN mirroring the encoder (for training the VQVAE).

Transformer:
    * GPT-style causal multi-head self-attention; ``n_layers`` blocks.
    * Input token sequence: ``[obs_tokens₀, a₀, obs_tokens₁, a₁, …]``
      where each step contributes ``n_tokens + 1`` tokens.
    * Output: per-token logits over the vocabulary for next-token prediction.

WorldModelAdapter mapping
-------------------------
* ``h_t``         — last-layer **mean-pooled hidden state** ``(d_model,)``.
  The full sequence context is maintained internally in a circular buffer.
* ``z_t``         — current step's token probabilities ``(n_cat, n_cls)``
  = ``(n_tokens_per_step, n_vocab)``.
* ``encode(obs)`` — VQVAE CNN encoder → raw embeddings ``(n_tokens * d_emb,)``.
* ``posterior(h, obs_emb)`` — negative L2 distances to codebook entries as
  z-logits ``(n_cat, n_cls)`` (closer = higher logit).
* ``dynamics_step(h, z, a)`` — append ``(z, a)`` to context buffer; run
  transformer; return mean-pooled hidden state + predicted token logits.

Statefulness warning
--------------------
The context buffer is **mutated in-place** on every :meth:`dynamics_step`
call.  Call :meth:`reset_context` (or :meth:`initial_state`) before
starting a new episode.  This is clearly at odds with a purely functional
interface but is necessary given the transformer's need for full context;
the adapter documents this contract explicitly.

Named parameter conventions
---------------------------
``encoder.conv1.weight``, ``encoder.conv4.weight``, ``encoder.out_proj.weight``
``codebook.embeddings``
``decoder.in_proj.weight``, ``decoder.conv1.weight``
``transformer.layer.{i}.attn.q_proj.weight``
``transformer.layer.{i}.attn.k_proj.weight``
``transformer.layer.{i}.attn.v_proj.weight``
``transformer.layer.{i}.attn.out_proj.weight``
``transformer.layer.{i}.ffn.0.weight``
``transformer.embed.weight``
``transformer.action_embed.weight``

Hook-point extensions
---------------------
In addition to the 11 standard RSSM hook points, IRIS exposes:
``transformer_hidden_{i}`` for ``i`` in ``0 … n_layers-1`` (the full
sequence tensor ``(seq_len, d_model)`` at each transformer layer output).
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from world_model_lens.backends.base import WorldModelAdapter
from world_model_lens.backends.dreamerv3 import _build_mlp, _cnn_output_size
from world_model_lens.core.config import WorldModelConfig


# ---------------------------------------------------------------------------
# VQVAE: Codebook
# ---------------------------------------------------------------------------

class VQCodebook(nn.Module):
    """Learnable VQ codebook with straight-through gradient estimator.

    Performs nearest-neighbour quantisation in embedding space.  During the
    forward pass the straight-through trick is used so gradients bypass the
    non-differentiable argmin operation.

    Parameters
    ----------
    n_vocab : int
        Number of codebook entries (vocabulary size).
    d_emb : int
        Embedding dimension.

    Examples
    --------
    >>> cb = VQCodebook(512, 64)
    >>> z_e = torch.randn(16, 64)   # 16 tokens, each d_emb-dim
    >>> z_q, indices, loss = cb(z_e)
    >>> z_q.shape
    torch.Size([16, 64])
    >>> indices.shape
    torch.Size([16])
    """

    def __init__(self, n_vocab: int, d_emb: int) -> None:
        super().__init__()
        # Named `embeddings` → accessible as `codebook.embeddings`
        self.embeddings = nn.Embedding(n_vocab, d_emb)
        nn.init.uniform_(self.embeddings.weight, -1.0 / n_vocab, 1.0 / n_vocab)
        self._n_vocab = n_vocab
        self._d_emb = d_emb

    def forward(
        self, z_e: Tensor, commitment_cost: float = 0.25
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Quantise encoder outputs to the nearest codebook entry.

        Parameters
        ----------
        z_e : Tensor
            Encoder output, shape ``(*batch, n_tokens, d_emb)``.
        commitment_cost : float
            Weight for the commitment loss (encoder tries to stay close to the
            chosen codebook vector).

        Returns
        -------
        z_q : Tensor
            Quantised (codebook-aligned) embeddings, same shape as ``z_e``.
            Gradients flow through via the straight-through estimator.
        indices : Tensor
            Nearest codebook indices, shape ``(*batch, n_tokens)``.
        vq_loss : Tensor
            VQ commitment + embedding loss scalar.
        """
        cb = self.embeddings.weight  # (n_vocab, d_emb)
        # Distances: ||z_e||² − 2 z_e @ cb.T + ||cb||²
        dist = (
            z_e.pow(2).sum(-1, keepdim=True)
            - 2.0 * z_e @ cb.T
            + cb.pow(2).sum(-1)
        )  # (*batch, n_tokens, n_vocab)
        indices = dist.argmin(-1)                      # (*batch, n_tokens)
        z_q = self.embeddings(indices)                 # (*batch, n_tokens, d_emb)

        # VQ loss: embedding loss + commitment loss
        embed_loss  = (z_q.detach() - z_e).pow(2).mean()
        commit_loss = (z_q - z_e.detach()).pow(2).mean()
        vq_loss = embed_loss + commitment_cost * commit_loss

        # Straight-through: use z_q values but z_e gradients
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices, vq_loss

    def lookup(self, indices: Tensor) -> Tensor:
        """Look up codebook vectors by index (no gradient tracking)."""
        return self.embeddings(indices)

    def soft_assignments(self, z_e: Tensor, temperature: float = 1.0) -> Tensor:
        """Compute soft assignment probabilities (for the posterior z-logits).

        Returns negative L2 distances divided by temperature as logits:
        closer codebook entries get higher (less negative) logits.

        Parameters
        ----------
        z_e : Tensor
            Encoder output, shape ``(*batch, n_tokens, d_emb)``.
        temperature : float
            Controls softmax sharpness.

        Returns
        -------
        Tensor
            Logits ``(*batch, n_tokens, n_vocab)``.
        """
        cb = self.embeddings.weight
        dist = (
            z_e.pow(2).sum(-1, keepdim=True)
            - 2.0 * z_e @ cb.T
            + cb.pow(2).sum(-1)
        )  # (*batch, n_tokens, n_vocab)
        return -dist / max(temperature, 1e-6)


# ---------------------------------------------------------------------------
# VQVAE: Encoder and Decoder
# ---------------------------------------------------------------------------

class IRISEncoder(nn.Module):
    """VQVAE encoder: ``(C, H, W)`` → ``(n_tokens, d_emb)`` raw embeddings.

    Uses four stride-2 Conv2d(k=4) layers (same as DreamerV3) followed by
    a linear projection to ``d_emb``.

    The output is **not** quantised — quantisation is done by
    :class:`VQCodebook`.  This separation allows the encoder to be used for
    both posterior computation (soft distances) and for passing embeddings
    into the transformer.

    Parameters
    ----------
    obs_channels : int
    cnn_channels : list of int
        Four-element list, e.g. ``[48, 96, 192, 384]``.
    d_emb : int
        Codebook / token embedding dimension.
    image_size : int
        H=W of the input image.
    """

    def __init__(
        self,
        obs_channels: int,
        cnn_channels: List[int],
        d_emb: int,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        ch = cnn_channels
        self.conv1 = nn.Conv2d(obs_channels, ch[0], kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(ch[0], ch[1], kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(ch[1], ch[2], kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(ch[2], ch[3], kernel_size=4, stride=2)
        self.out_norm = nn.LayerNorm(ch[3])
        # Spatial size after 4 stride-2 convolutions
        spatial = _cnn_output_size(image_size, cnn_channels)
        n_spatial = spatial // ch[3]  # number of spatial locations
        self._n_spatial = n_spatial
        self._ch_last = ch[3]
        # Project each spatial feature to d_emb
        self.out_proj = nn.Linear(ch[3], d_emb)

    def forward(self, obs: Tensor) -> Tensor:
        """Encode image to a sequence of token embeddings.

        Parameters
        ----------
        obs : Tensor
            ``(*batch, C, H, W)``

        Returns
        -------
        Tensor
            ``(*batch, n_tokens, d_emb)``
        """
        x = F.silu(self.conv1(obs))
        x = F.silu(self.conv2(x))
        x = F.silu(self.conv3(x))
        x = F.silu(self.conv4(x))
        # x: (*batch, C_last, h_spatial, w_spatial)
        # Flatten spatial → (*batch, n_tokens, C_last)
        batch_shape = x.shape[:-3]
        C, Hs, Ws = x.shape[-3], x.shape[-2], x.shape[-1]
        x = x.flatten(start_dim=-2).transpose(-1, -2)  # (*batch, n_tokens, C)
        x = self.out_norm(x)
        return self.out_proj(x)                         # (*batch, n_tokens, d_emb)


class IRISDecoder(nn.Module):
    """VQVAE decoder: ``(n_tokens, d_emb)`` → reconstructed ``(C, H, W)``.

    Parameters
    ----------
    obs_channels, cnn_channels, d_emb, image_size : mirror of
        :class:`IRISEncoder` parameters.
    """

    def __init__(
        self,
        obs_channels: int,
        cnn_channels: List[int],
        d_emb: int,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        rev_ch = list(reversed(cnn_channels))
        spatial_h = image_size
        for _ in cnn_channels:
            spatial_h = (spatial_h - 4) // 2 + 1
        # n_tokens spatial locations; project d_emb → rev_ch[0] each
        self._start_h = spatial_h
        self.in_proj = nn.Linear(d_emb, rev_ch[0])

        layers: List[nn.Module] = []
        c_in = rev_ch[0]
        for c_out in rev_ch[1:]:
            layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            layers.append(nn.SiLU())
            c_in = c_out
        layers.append(nn.ConvTranspose2d(c_in, obs_channels, kernel_size=4, stride=2, padding=1))
        self.layers = nn.Sequential(*layers)
        self._rev_ch0 = rev_ch[0]

    def forward(self, z_q: Tensor) -> Tensor:
        """Reconstruct image from quantised token embeddings.

        Parameters
        ----------
        z_q : Tensor
            ``(*batch, n_tokens, d_emb)``

        Returns
        -------
        Tensor
            ``(*batch, C, H, W)``
        """
        x = self.in_proj(z_q)      # (*batch, n_tokens, rev_ch[0])
        # Reshape spatial: (*batch, n_tokens, C) → (*batch, C, h, w)
        batch_shape = x.shape[:-2]
        n_tok = x.shape[-2]
        h = w = int(math.isqrt(n_tok))
        # (*batch, C, h, w)
        x = x.transpose(-1, -2).contiguous().view(*batch_shape, self._rev_ch0, h, w)
        return self.layers(x)


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    Parameters
    ----------
    d_model : int
        Model / embedding dimension.
    n_heads : int
        Number of attention heads.
    max_seq_len : int
        Maximum sequence length for the causal mask.
    attn_dropout : float
        Dropout on attention weights.

    Named parameters (accessible via the transformer prefix):
    ``attn.q_proj.weight``, ``attn.k_proj.weight``,
    ``attn.v_proj.weight``, ``attn.out_proj.weight``
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self._n_heads = n_heads
        self._d_head = d_model // n_heads
        # Named q/k/v/out projections for interpretable parameter paths
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        # Causal mask (lower triangular)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("_causal_mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        """Causal multi-head self-attention.

        Parameters
        ----------
        x : Tensor
            ``(batch, seq_len, d_model)`` or ``(seq_len, d_model)`` (unbatched).

        Returns
        -------
        Tensor
            Same shape as *x*.
        """
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)

        B, T, _ = x.shape
        H, Dh = self._n_heads, self._d_head

        # Project and reshape to (B, H, T, Dh)
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        scale = 1.0 / math.sqrt(Dh)
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Apply causal mask (attend only to past + present)
        mask = self._causal_mask[:T, :T].bool()
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)

        return out.squeeze(0) if unbatched else out


class TransformerBlock(nn.Module):
    """Single GPT-style transformer block: LN → Attn → residual → LN → FFN → residual.

    Parameters
    ----------
    d_model : int
    n_heads : int
    ffn_ratio : int
        FFN hidden dimension = ``ffn_ratio * d_model``.
    max_seq_len : int
    attn_dropout, ffn_dropout : float
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_ratio: int = 4,
        max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        # Named `attn` so parameter paths become `transformer.layer.{i}.attn.q_proj.weight`
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, attn_dropout)
        self.ln2  = nn.LayerNorm(d_model)
        # Named `ffn` so parameter paths become `transformer.layer.{i}.ffn.0.weight`
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_ratio * d_model, d_model),
            nn.Dropout(ffn_dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class IRISTransformer(nn.Module):
    """GPT-style causal transformer with per-layer hidden state access.

    Input: sequence of token embeddings + optional action embeddings.
    The token embedding table (``embed``) projects codebook indices to the
    model dimension.  Actions are embedded separately (``action_embed``).

    Per-layer hidden states are stored in :attr:`layer_hiddens` after every
    :meth:`forward` call, enabling the :class:`IRISAdapter` to expose them
    as ``transformer_hidden_{i}`` hook points.

    Parameters
    ----------
    n_vocab : int
        Codebook vocabulary size.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Attention heads per block.
    n_layers : int
        Number of transformer blocks.
    d_action : int
        One-hot action dimension (for action embeddings).
    max_seq_len : int
        Maximum context length in tokens.
    ffn_ratio : int
    attn_dropout, ffn_dropout : float
    """

    def __init__(
        self,
        n_vocab: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_action: int,
        max_seq_len: int = 512,
        ffn_ratio: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._n_vocab    = n_vocab
        self._d_model    = d_model
        self._n_layers   = n_layers
        self._max_seq    = max_seq_len

        # Token embedding + positional embedding
        self.embed     = nn.Embedding(n_vocab, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        # Action embedding (one-hot → d_model)
        self.action_embed = nn.Linear(d_action, d_model, bias=False)
        self.drop_in   = nn.Dropout(0.0)
        # Named `layer` module list for clean parameter paths
        self.layer     = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_ratio, max_seq_len,
                             attn_dropout, ffn_dropout)
            for _ in range(n_layers)
        ])
        self.ln_f      = nn.LayerNorm(d_model)
        # Output projection: predicts logits over n_vocab for each token
        self.head      = nn.Linear(d_model, n_vocab, bias=False)

        # Per-layer hidden states (populated during forward, for hook support)
        self.layer_hiddens: List[Tensor] = []

        # Weight tying between embed and head (standard GPT practice)
        self.head.weight = self.embed.weight

    def forward_from_embeddings(self, embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        """Run transformer on pre-computed embeddings (no index lookup).

        Parameters
        ----------
        embeddings : Tensor
            ``(seq_len, d_model)`` or ``(B, seq_len, d_model)``.

        Returns
        -------
        logits : Tensor
            ``(*batch, seq_len, n_vocab)``
        last_hidden : Tensor
            Mean-pooled last-layer hidden state ``(*batch, d_model)``.
        """
        unbatched = embeddings.dim() == 2
        x = embeddings.unsqueeze(0) if unbatched else embeddings
        B, T, _ = x.shape

        # Positional embeddings
        positions = torch.arange(T, device=x.device)
        x = self.drop_in(x + self.pos_embed(positions))

        # Transformer blocks — store per-layer outputs
        self.layer_hiddens = []
        for block in self.layer:
            x = block(x)
            self.layer_hiddens.append(x)  # (B, T, d_model) per layer

        x = self.ln_f(x)
        logits = self.head(x)     # (B, T, n_vocab)
        last_hidden = x.mean(dim=-2)   # mean pool over sequence: (B, d_model)

        if unbatched:
            return logits.squeeze(0), last_hidden.squeeze(0)
        return logits, last_hidden

    def token_embedding(self, indices: Tensor) -> Tensor:
        """Look up token embeddings from codebook indices."""
        return self.embed(indices)

    def action_embedding(self, action_onehot: Tensor) -> Tensor:
        """Project one-hot action to model dimension."""
        return self.action_embed(action_onehot)


# ---------------------------------------------------------------------------
# IRISAdapter
# ---------------------------------------------------------------------------

class IRISAdapter(WorldModelAdapter):
    """IRIS world model backend implementing :class:`WorldModelAdapter`.

    Maps the IRIS architecture onto the standard RSSM-style interface:

    * ``h_t``  — last-layer mean-pooled transformer hidden state ``(d_model,)``.
    * ``z_t``  — token soft-assignment logits ``(n_tokens, n_vocab)``
      = ``(n_cat, n_cls)``.
    * ``encode`` — VQVAE CNN encoder → raw embeddings (not quantised).
    * ``posterior`` — negative L2 distances to codebook → z-logits.
    * ``dynamics_step`` — append tokens+action to context, run transformer.

    ⚠ **Statefulness**: the adapter maintains an internal context buffer
    ``_context_embeds`` (a :class:`collections.deque`) that is mutated by
    every :meth:`dynamics_step` call.  Call :meth:`reset_context` before
    each new episode.  :meth:`initial_state` also resets the context.

    Parameters
    ----------
    cfg : WorldModelConfig
        ``n_cat`` = tokens per step, ``n_cls`` = vocab size, ``d_h`` = d_model.
    obs_channels : int
    image_size : int
    cnn_channels : list of int
    d_emb : int
        VQVAE embedding dimension (may differ from d_model).
    n_heads : int
    n_layers : int
    max_seq_len : int
    ffn_ratio : int
    attn_dropout, ffn_dropout : float
    """

    DEFAULT_CNN_CHANNELS: List[int] = [48, 96, 192, 384]

    def __init__(
        self,
        cfg: WorldModelConfig,
        obs_channels: int = 3,
        image_size: int = 64,
        cnn_channels: Optional[List[int]] = None,
        d_emb: int = 64,
        n_heads: int = 8,
        n_layers: int = 10,
        max_seq_len: int = 512,
        ffn_ratio: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ) -> None:
        self._cfg          = cfg
        self._obs_channels = obs_channels
        self._image_size   = image_size
        self._cnn_channels = cnn_channels or self.DEFAULT_CNN_CHANNELS
        self._d_emb        = d_emb
        self._n_heads      = n_heads
        self._n_layers     = n_layers
        self._max_seq_len  = max_seq_len

        n_vocab   = cfg.n_cls          # codebook size
        n_tokens  = cfg.n_cat          # tokens per step
        d_model   = cfg.d_h            # transformer hidden dim
        d_action  = cfg.d_action

        self._n_vocab  = n_vocab
        self._n_tokens = n_tokens
        self._d_model  = d_model
        self._d_action = d_action

        # ── Sub-networks ────────────────────────────────────────────────
        self._encoder = IRISEncoder(
            obs_channels=obs_channels,
            cnn_channels=self._cnn_channels,
            d_emb=d_emb,
            image_size=image_size,
        )
        self._codebook = VQCodebook(n_vocab=n_vocab, d_emb=d_emb)
        self._decoder  = IRISDecoder(
            obs_channels=obs_channels,
            cnn_channels=self._cnn_channels,
            d_emb=d_emb,
            image_size=image_size,
        )
        self._transformer = IRISTransformer(
            n_vocab=n_vocab,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_action=d_action,
            max_seq_len=max_seq_len,
            ffn_ratio=ffn_ratio,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )
        # Optional reward head on top of transformer last hidden state
        self._reward_head = _build_mlp(d_model, d_model, 1, n_hidden=1)

        # ── Stateful context buffer ──────────────────────────────────────
        # Each entry is a tensor of shape (n_tokens + 1, d_model):
        # [z_token_embeds (n_tokens rows), action_embed (1 row)]
        self._context_embeds: Deque[Tensor] = deque(maxlen=max_seq_len)

    # ── Context management ────────────────────────────────────────────────

    def reset_context(self) -> None:
        """Clear the internal context buffer.

        Must be called at the start of every new episode.
        :meth:`initial_state` also calls this automatically.
        """
        self._context_embeds.clear()

    def _get_context_tensor(self) -> Optional[Tensor]:
        """Stack the context buffer into a single sequence tensor.

        Returns
        -------
        Tensor or None
            ``(seq_len, d_model)`` stacked context, or ``None`` if empty.
        """
        if not self._context_embeds:
            return None
        return torch.cat(list(self._context_embeds), dim=0)  # (seq_len, d_model)

    def _context_length(self) -> int:
        """Current context length in tokens."""
        return sum(t.shape[0] for t in self._context_embeds)

    # ── WorldModelAdapter abstract methods ───────────────────────────────

    def encode(self, obs: Tensor) -> Tensor:
        """VQVAE-encode observation to raw (unquantised) token embeddings.

        Parameters
        ----------
        obs : Tensor
            ``(*batch, C, H, W)``

        Returns
        -------
        Tensor
            Raw embeddings ``(*batch, n_tokens * d_emb)``.
            (Flattened so it fits the ``obs_emb`` convention.)
        """
        emb = self._encoder(obs)   # (*batch, n_tokens, d_emb)
        return emb.flatten(start_dim=-2)   # (*batch, n_tokens * d_emb)

    def initial_state(self, batch_size: int = 1) -> Tensor:
        """Reset context buffer and return zero hidden state ``(d_model,)``.

        ⚠ Resets the internal context buffer — must be called before each
        new episode.
        """
        self.reset_context()
        dev = next(
            (p.device for p in self._transformer.parameters()),
            torch.device("cpu"),
        )
        return torch.zeros(self._d_model, device=dev)

    def dynamics_step(
        self, h: Tensor, z: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """One IRIS step: append (z_tokens, action) to context, run transformer.

        Appends the current step's token embeddings and the action embedding
        to the context buffer, then runs the transformer to predict the next
        step's token logits.

        Parameters
        ----------
        h : Tensor
            Current mean-pooled hidden state ``(d_model,)`` (informational;
            the adapter uses its internal context buffer, not this value).
        z : Tensor
            Current step's z logits ``(n_cat, n_cls)`` = ``(n_tokens, n_vocab)``.
        action : Tensor
            Discrete action index ``()`` or ``(1,)``.

        Returns
        -------
        h_next : Tensor
            Mean-pooled last-layer hidden state ``(d_model,)``.
        z_prior_logits : Tensor
            Transformer-predicted next token logits ``(n_cat, n_cls)``.
        """
        dev = h.device

        # Convert z (soft probs or logits) to token embeddings via codebook
        # We use the argmax (most likely token) to get discrete token indices
        z_probs = z.softmax(dim=-1) if z.min() < 0 else z  # ensure probs
        token_indices = z_probs.argmax(dim=-1)              # (n_tokens,)
        z_embeds = self._transformer.token_embedding(token_indices)  # (n_tokens, d_model)

        # Action embedding: one-hot encode discrete action
        a_onehot = F.one_hot(action.long().squeeze(), self._d_action).float()
        a_embed  = self._transformer.action_embedding(a_onehot).unsqueeze(0)  # (1, d_model)

        # Append (z_embeds, a_embed) as a single step to the context
        step_embed = torch.cat([z_embeds, a_embed], dim=0)  # (n_tokens+1, d_model)
        self._context_embeds.append(step_embed)

        # Run transformer on full context
        ctx = self._get_context_tensor()  # (seq_len, d_model)
        logits, h_next = self._transformer.forward_from_embeddings(ctx)

        # Prior logits = the n_tokens predictions BEFORE the last action token
        # The transformer predicts next tokens; we want the prediction
        # at the positions of the last n_tokens steps
        # Last seq positions: -(n_tokens+1) … -1 (before the action embed we just added)
        # We predict from position -n_tokens-1 to position -2 (inclusive)
        n_tok = self._n_tokens
        if logits.shape[0] >= n_tok + 1:
            # Shape: (seq_len, n_vocab); take last n_tokens predictions
            prior_logits_flat = logits[-(n_tok + 1):-1]   # (n_tokens, n_vocab)
        else:
            # Fallback: use last available logits
            prior_logits_flat = logits[-n_tok:] if logits.shape[0] >= n_tok else logits
            if prior_logits_flat.shape[0] < n_tok:
                pad = torch.zeros(n_tok - prior_logits_flat.shape[0], self._n_vocab, device=dev)
                prior_logits_flat = torch.cat([pad, prior_logits_flat], dim=0)

        z_prior_logits = prior_logits_flat  # (n_tokens, n_vocab) = (n_cat, n_cls)
        return h_next, z_prior_logits

    def posterior(self, h: Tensor, obs_emb: Tensor) -> Tensor:
        """Compute z-posterior logits from raw encoder embeddings.

        For IRIS the "posterior" is determined by the VQVAE quantisation:
        the soft assignment scores (negative L2 distances to codebook entries)
        serve as posterior logits — closer codebook entries get higher
        (less negative) logit values.

        Parameters
        ----------
        h : Tensor
            ``(d_model,)`` — not used for IRIS (codebook assignment is obs-only).
        obs_emb : Tensor
            Flattened raw encoder output ``(n_tokens * d_emb,)``.

        Returns
        -------
        Tensor
            Posterior logits ``(n_cat, n_cls)`` = ``(n_tokens, n_vocab)``.
        """
        z_e = obs_emb.view(self._n_tokens, self._d_emb)     # (n_tokens, d_emb)
        return self._codebook.soft_assignments(z_e)          # (n_tokens, n_vocab)

    # ── Optional heads ────────────────────────────────────────────────────

    def reward_pred(self, h: Tensor, z: Tensor) -> Tensor:
        """Predict reward from mean-pooled transformer hidden state."""
        return self._reward_head(h).squeeze(-1)

    # ── Transformer hidden states (for hook points) ───────────────────────

    @property
    def last_layer_hiddens(self) -> List[Tensor]:
        """Per-layer hidden states from the most recent transformer forward pass.

        Each entry is ``(seq_len, d_model)`` (or batched).  Available only
        after :meth:`dynamics_step` has been called at least once.

        These are exposed as ``transformer_hidden_{i}`` hook points.
        """
        return self._transformer.layer_hiddens

    # ── hook_point_names ─────────────────────────────────────────────────

    @property
    def hook_point_names(self) -> List[str]:
        """Standard RSSM names + per-layer transformer hidden states."""
        from world_model_lens.backends.base import WorldModelAdapter
        base = super().hook_point_names
        extra = [f"transformer_hidden_{i}" for i in range(self._n_layers)]
        return base + extra

    # ── Parameter access ─────────────────────────────────────────────────

    @property
    def _all_modules(self) -> List[Tuple[str, nn.Module]]:
        return [
            ("encoder",     self._encoder),
            ("codebook",    self._codebook),
            ("decoder",     self._decoder),
            ("transformer", self._transformer),
            ("reward",      self._reward_head),
        ]

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        """Yield ``(name, param)`` with structured IRIS naming.

        Transformer attention projections are accessible as:
        ``transformer.layer.{i}.attn.q_proj.weight``
        ``transformer.layer.{i}.attn.k_proj.weight``
        ``transformer.layer.{i}.attn.v_proj.weight``
        ``transformer.layer.{i}.attn.out_proj.weight``
        ``transformer.layer.{i}.ffn.0.weight``
        ``transformer.embed.weight``, ``transformer.action_embed.weight``
        """
        for prefix, module in self._all_modules:
            for name, param in module.named_parameters():
                yield f"{prefix}.{name}", param

    # ── Device / mode ────────────────────────────────────────────────────

    def to(self, device: Union[str, torch.device]) -> "IRISAdapter":
        dev = torch.device(device)
        for _, m in self._all_modules:
            m.to(dev)
        # Move buffered context tensors
        self._context_embeds = deque(
            (t.to(dev) for t in self._context_embeds),
            maxlen=self._max_seq_len,
        )
        return self

    def eval(self) -> "IRISAdapter":
        for _, m in self._all_modules:
            m.eval()
        return self

    def train(self, mode: bool = True) -> "IRISAdapter":
        for _, m in self._all_modules:
            m.train(mode)
        return self

    # ── Checkpoint ───────────────────────────────────────────────────────

    def state_dict(self) -> Dict[str, Tensor]:
        return {n: p.clone() for n, p in self.named_parameters()}

    def load_state_dict(
        self, state_dict: Dict[str, Tensor], strict: bool = True
    ) -> None:
        own_keys  = {n for n, _ in self.named_parameters()}
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
        """Infer :class:`WorldModelConfig` from an IRIS state dict.

        Heuristics:
        * ``d_model``  ← ``transformer.embed.weight`` shape ``(n_vocab, d_model)``
        * ``n_vocab``  ← same weight's first dim
        * ``n_tokens`` ← cannot be directly inferred; defaults to 16 (4×4 patches)
        * ``d_action`` ← ``transformer.action_embed.weight`` shape ``(d_model, d_action)``
        """
        emb_key = "transformer.embed.weight"
        if emb_key not in state_dict:
            raise KeyError(f"Cannot infer config: '{emb_key}' not in state_dict.")
        n_vocab, d_model = state_dict[emb_key].shape

        act_key = "transformer.action_embed.weight"
        d_action = state_dict[act_key].shape[1] if act_key in state_dict else 1

        return WorldModelConfig(
            d_h=d_model,
            d_action=max(d_action, 1),
            d_obs=1,         # not directly used for visual IRIS
            n_cat=16,        # default 4×4 tokens; override if known
            n_cls=n_vocab,
            backend="iris",
            encoder_type="cnn",
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        cfg: Optional[WorldModelConfig] = None,
        strict: bool = True,
        **kwargs,
    ) -> "IRISAdapter":
        """Load from checkpoint.  Config is inferred if not provided."""
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
        return "iris"

    @property
    def cfg(self) -> WorldModelConfig:
        return self._cfg

    def __repr__(self) -> str:
        return (
            f"IRISAdapter("
            f"n_tokens={self._n_tokens}, "
            f"n_vocab={self._n_vocab}, "
            f"d_model={self._d_model}, "
            f"n_layers={self._n_layers}, "
            f"n_heads={self._n_heads})"
        )
