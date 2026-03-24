"""WorldModelAdapter — abstract interface every backend must implement.

A *backend* is a thin adapter that wraps a concrete model implementation
(DreamerV3, TD-MPC, custom RSSM, …) and exposes the standardised interface
that :class:`~world_model_lens.HookedWorldModel` relies on.

Implementors override only the methods relevant to their architecture; all
optional heads (actor, value, continuation) raise :exc:`NotImplementedError`
by default so callers can test for their presence gracefully via ``hasattr``
or ``try/except``.

Contract
--------
Every method that returns a tensor must:

* Accept *device*-agnostic inputs (move internally if needed, or expect the
  caller to have already moved the tensor).
* Return a *detached* tensor when used in inference / caching mode.
* Support single-step (non-batched) calls, i.e. all leading "batch" dims may
  be absent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor


class WorldModelAdapter(ABC):
    """Abstract base class for world-model backend adapters.

    Sub-class this and implement the abstract methods to plug any world-model
    architecture into :class:`~world_model_lens.HookedWorldModel`.

    Shape conventions
    -----------------
    Most methods operate on *single-step, single-episode* tensors unless
    otherwise noted.  Shapes are described in terms of:

    * ``d_h``    — recurrent hidden state dimension
    * ``d_obs``  — raw observation dimension (post-flattening)
    * ``d_emb``  — encoder output dimension
    * ``n_cat``  — number of categorical latent variables
    * ``n_cls``  — number of classes per categorical
    * ``d_z``    — ``n_cat * n_cls`` (stochastic latent size)
    * ``d_a``    — action dimension

    Examples
    --------
    >>> class MyAdapter(WorldModelAdapter):
    ...     def encode(self, obs): ...
    ...     def initial_state(self): ...
    ...     def dynamics_step(self, h, z, action): ...
    ...     def posterior(self, h, obs_emb): ...
    """

    # ------------------------------------------------------------------
    # Required methods
    # ------------------------------------------------------------------

    @abstractmethod
    def encode(self, obs: Tensor) -> Tensor:
        """Encode a raw observation into a latent embedding.

        Parameters
        ----------
        obs:
            Raw observation tensor of shape ``(*obs_shape,)`` or
            ``(B, *obs_shape)``.

        Returns
        -------
        Tensor
            Embedding of shape ``(d_emb,)`` or ``(B, d_emb)``.
        """
        ...

    @abstractmethod
    def initial_state(self, batch_size: int = 1) -> Tensor:
        """Return the initial (zero or learned) recurrent hidden state.

        Parameters
        ----------
        batch_size:
            Number of parallel episodes.  Default 1.

        Returns
        -------
        Tensor
            Shape ``(d_h,)`` or ``(B, d_h)``.
        """
        ...

    @abstractmethod
    def dynamics_step(
        self,
        h: Tensor,
        z: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """One recurrent dynamics step (the deterministic + prior update).

        Computes:
        ``h_next = GRU(h, concat(z_flat, action))``
        ``z_prior_logits = prior_head(h_next)``

        Parameters
        ----------
        h:
            Current recurrent state, shape ``(d_h,)``.
        z:
            Current stochastic latent (soft or hard), shape ``(n_cat, n_cls)``
            or ``(d_z,)`` — the adapter handles flattening internally.
        action:
            Action taken at this step, shape ``()`` or ``(d_a,)``.

        Returns
        -------
        h_next : Tensor
            Updated recurrent state, shape ``(d_h,)``.
        z_prior_logits : Tensor
            Prior logits, shape ``(n_cat, n_cls)``.
        """
        ...

    @abstractmethod
    def posterior(self, h: Tensor, obs_emb: Tensor) -> Tensor:
        """Compute posterior logits from recurrent state + observation embedding.

        Parameters
        ----------
        h:
            Recurrent state ``h_t``, shape ``(d_h,)``.
        obs_emb:
            Observation embedding from :meth:`encode`, shape ``(d_emb,)``.

        Returns
        -------
        Tensor
            Posterior logits, shape ``(n_cat, n_cls)``.
        """
        ...

    # ------------------------------------------------------------------
    # Optional heads (raise NotImplementedError by default)
    # ------------------------------------------------------------------

    def reward_pred(self, h: Tensor, z: Tensor) -> Tensor:
        """Predict reward from latent state ``(h, z)``.

        Parameters
        ----------
        h:
            Recurrent state, shape ``(d_h,)``.
        z:
            Stochastic latent (soft), shape ``(n_cat, n_cls)`` or ``(d_z,)``.

        Returns
        -------
        Tensor
            Scalar reward prediction, shape ``()`` or ``(1,)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a reward head."
        )

    def cont_pred(self, h: Tensor, z: Tensor) -> Tensor:
        """Predict episode-continuation probability from latent state.

        Parameters
        ----------
        h:
            Recurrent state, shape ``(d_h,)``.
        z:
            Stochastic latent, shape ``(n_cat, n_cls)`` or ``(d_z,)``.

        Returns
        -------
        Tensor
            Continuation probability in ``[0, 1]``, shape ``()`` or ``(1,)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a continuation head."
        )

    def actor(self, h: Tensor, z: Tensor) -> Tensor:
        """Compute action logits from latent state.

        Parameters
        ----------
        h:
            Recurrent state, shape ``(d_h,)``.
        z:
            Stochastic latent, shape ``(n_cat, n_cls)`` or ``(d_z,)``.

        Returns
        -------
        Tensor
            Action logits, shape ``(d_a,)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement an actor head."
        )

    def value(self, h: Tensor, z: Tensor) -> Tensor:
        """Estimate value from latent state.

        Parameters
        ----------
        h:
            Recurrent state, shape ``(d_h,)``.
        z:
            Stochastic latent, shape ``(n_cat, n_cls)`` or ``(d_z,)``.

        Returns
        -------
        Tensor
            Scalar value estimate, shape ``()`` or ``(1,)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a value head."
        )

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        """Iterate over ``(name, parameter)`` pairs of all model weights.

        Defaults to an empty iterator.  Backends that wrap a :mod:`torch.nn`
        module should delegate to ``model.named_parameters()``.

        Yields
        ------
        (str, Tensor)
        """
        return iter([])

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device: Union[str, torch.device]) -> "WorldModelAdapter":
        """Move the adapter's internal model to *device*.

        Returns ``self`` for chaining.  Override in sub-classes that hold
        :mod:`torch.nn` modules.
        """
        return self

    def eval(self) -> "WorldModelAdapter":
        """Switch to eval mode (disables dropout, BN running stats, etc.).

        Returns ``self`` for chaining.
        """
        return self

    def train(self, mode: bool = True) -> "WorldModelAdapter":
        """Switch to train mode.

        Returns ``self`` for chaining.
        """
        return self

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Human-readable backend identifier.  Override in sub-classes."""
        return type(self).__name__

    @property
    def hook_point_names(self) -> List[str]:
        """All named activation hook-points exposed by this adapter.

        :class:`~world_model_lens.HookedWorldModel` uses this to enumerate
        the activation points it will intercept.  The default set covers the
        11 standard RSSM activation names used in every RSSM-based backend.

        Adapters with extra internals (e.g. per-layer transformer hidden
        states in IRIS) should *extend* this list by overriding the property.

        Returns
        -------
        list of str
        """
        return [
            "encoder.out",
            "rnn.h",
            "z_prior.logits",
            "z_prior",
            "z_posterior.logits",
            "z_posterior",
            "kl",
            "reward_pred",
            "cont_pred",
            "actor.logits",
            "value_pred",
        ]
