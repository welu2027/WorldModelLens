"""LatentState — a single time-step snapshot of a world model's internal state.

A ``LatentState`` records every tensor and scalar that characterises one step
of a world model's latent dynamics:

* the recurrent hidden state ``h_t``
* the categorical (RSSM-style) latent ``z``, represented as both the
  posterior (inferred from observations) and the prior (predicted from
  dynamics only)
* optional per-step quantities: action, rewards, continuation flag,
  value & actor logits

The class is a standard Python :class:`dataclasses.dataclass` with an
``eq=False`` to avoid element-wise tensor comparison (use :meth:`torch.equal`
for that).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(eq=False)
class LatentState:
    """Single time-step snapshot of a world model's latent state.

    Parameters
    ----------
    h_t:
        Recurrent hidden state.  Shape ``(d_h,)``.
    z_posterior:
        Categorical latent posterior (inferred from the real observation).
        Shape ``(n_cat, n_cls)``.  Logits or soft probabilities are both
        accepted; properties that require probabilities apply ``softmax``
        internally.
    z_prior:
        Categorical latent prior (predicted purely from dynamics, without
        observing ``o_t``).  Same shape as ``z_posterior``.
    timestep:
        Integer step index within the episode (0-indexed).
    action:
        Action taken at this step.  Shape ``()`` (discrete) or ``(d_a,)``
        (continuous).  ``None`` if unavailable (e.g. terminal step).
    reward_pred:
        Scalar reward *predicted* by the world model's reward head.
    reward_real:
        Ground-truth reward observed from the environment.
    cont_pred:
        Predicted episode-continuation probability in ``[0, 1]``.
    value_pred:
        Scalar value estimate from the critic.
    actor_logits:
        Raw (un-normalised) action logits from the actor.  Shape ``(d_a,)``.
    metadata:
        Arbitrary key-value annotations (e.g. ``{"env_obs": obs_array}``).

    Examples
    --------
    >>> import torch
    >>> s = LatentState(
    ...     h_t=torch.randn(512),
    ...     z_posterior=torch.randn(32, 32),
    ...     z_prior=torch.randn(32, 32),
    ...     timestep=0,
    ... )
    >>> s.flat.shape
    torch.Size([1536])       # 512 + 32*32
    >>> s.kl.item() > 0
    True
    """

    # ------------------------------------------------------------------
    # Required fields
    # ------------------------------------------------------------------
    h_t: torch.Tensor
    """Recurrent hidden state, shape ``(d_h,)``."""

    z_posterior: torch.Tensor
    """Categorical posterior latent, shape ``(n_cat, n_cls)``."""

    z_prior: torch.Tensor
    """Categorical prior latent, shape ``(n_cat, n_cls)``."""

    timestep: int
    """Step index within the episode."""

    # ------------------------------------------------------------------
    # Optional per-step quantities
    # ------------------------------------------------------------------
    action: torch.Tensor | None = None
    """Action tensor, shape ``()`` or ``(d_a,)``."""

    reward_pred: float | None = None
    """World-model-predicted reward."""

    reward_real: float | None = None
    """Ground-truth environment reward."""

    cont_pred: float | None = None
    """Predicted continuation probability."""

    value_pred: float | None = None
    """Critic value estimate."""

    actor_logits: torch.Tensor | None = None
    """Actor output logits, shape ``(d_a,)``."""

    metadata: dict[str, Any] | None = field(default=None)
    """Arbitrary extra annotations."""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def z_flat(self) -> torch.Tensor:
        """Flattened categorical latent (posterior), shape ``(n_cat * n_cls,)``.

        Returns
        -------
        torch.Tensor
            1-D view of ``z_posterior``.

        Examples
        --------
        >>> s.z_flat.shape
        torch.Size([1024])    # for n_cat=32, n_cls=32
        """
        return self.z_posterior.flatten()

    @property
    def flat(self) -> torch.Tensor:
        """Full flattened latent: ``[h_t ‖ z_flat]``, shape ``(d_h + n_cat*n_cls,)``.

        Concatenates the deterministic recurrent state with the flattened
        stochastic categorical posterior into a single 1-D feature vector,
        suitable as input to probes or analysis pipelines.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(d_h + n_cat * n_cls,)``.

        Examples
        --------
        >>> s = LatentState(h_t=torch.randn(512), z_posterior=torch.randn(32, 32), ...)
        >>> s.flat.shape
        torch.Size([1536])
        """
        return torch.cat([self.h_t.flatten(), self.z_flat])

    @property
    def kl(self) -> torch.Tensor:
        """KL divergence KL(posterior ‖ prior), summed over categorical dimensions.

        Computes the analytically exact KL between two batches of categorical
        distributions (one per categorical variable), using numerically stable
        clamped log-probabilities.  The result is summed over all ``n_cat``
        categorical variables to give a single scalar.

        Returns
        -------
        torch.Tensor
            Scalar (0-D) non-negative tensor.

        Notes
        -----
        Both ``z_posterior`` and ``z_prior`` are treated as *logits* and
        passed through ``softmax`` before computing the KL, matching the
        convention used in DreamerV3 and similar models.

        Derivation::

            KL = Σ_c Σ_k  p_c(k) * [log p_c(k) − log q_c(k)]
        """
        p = self.z_posterior.softmax(dim=-1).clamp(min=1e-8)  # [n_cat, n_cls]
        q = self.z_prior.softmax(dim=-1).clamp(min=1e-8)      # [n_cat, n_cls]
        # Per-categorical KL: [n_cat]
        kl_per_cat = (p * (p.log() - q.log())).sum(dim=-1)
        return kl_per_cat.sum()

    @property
    def surprise(self) -> torch.Tensor:
        """Alias for :attr:`kl`.

        Treating the KL divergence between posterior and prior as a measure
        of *surprise* (how much the observation updated the model's beliefs)
        is a common interpretation in the active-inference / free-energy
        literature.

        Returns
        -------
        torch.Tensor
            Same scalar as :attr:`kl`.
        """
        return self.kl

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def to_device(self, device: str | torch.device) -> LatentState:
        """Return a copy of this state with all tensors moved to *device*.

        Parameters
        ----------
        device:
            Target device, e.g. ``"cuda"``, ``"cpu"``, ``torch.device("cuda:1")``.

        Returns
        -------
        LatentState
            New :class:`LatentState` instance on the requested device.  The
            original state is *not* modified (tensors are moved, not copied,
            but Python creates new references).

        Examples
        --------
        >>> s_cuda = s.to_device("cuda")
        >>> s_cuda.h_t.device
        device(type='cuda', index=0)
        """
        dev = torch.device(device)

        def _move(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(dev) if t is not None else None

        return dataclasses.replace(
            self,
            h_t=self.h_t.to(dev),
            z_posterior=self.z_posterior.to(dev),
            z_prior=self.z_prior.to(dev),
            action=_move(self.action),
            actor_logits=_move(self.actor_logits),
        )

    def detach(self) -> LatentState:
        """Return a copy with all tensor gradients detached.

        Useful when checkpointing states during training to prevent
        accidental backprop through stored latents.

        Returns
        -------
        LatentState
            New :class:`LatentState` with ``.detach()`` applied to every
            tensor field.  Non-tensor fields are shallow-copied.

        Examples
        --------
        >>> s_no_grad = s.detach()
        >>> s_no_grad.h_t.requires_grad
        False
        """
        def _detach(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.detach() if t is not None else None

        return dataclasses.replace(
            self,
            h_t=self.h_t.detach(),
            z_posterior=self.z_posterior.detach(),
            z_prior=self.z_prior.detach(),
            action=_detach(self.action),
            actor_logits=_detach(self.actor_logits),
        )

    def __repr__(self) -> str:
        d_h = self.h_t.shape[-1]
        n_cat, n_cls = self.z_posterior.shape
        device = self.h_t.device
        return (
            f"LatentState("
            f"t={self.timestep}, d_h={d_h}, "
            f"n_cat={n_cat}, n_cls={n_cls}, "
            f"device={device})"
        )
