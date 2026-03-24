"""Template for implementing a custom world model backend adapter.

This module provides a heavily documented skeleton (:class:`CustomWorldModelAdapter`)
that shows all required and optional methods, their signatures, expected return
shapes, and implementation checklist items. It serves as a reference guide for
adding any new world-model architecture to world_model_lens.

Quick Start Checklist
---------------------
When implementing a custom backend, follow these steps:

1. **Subclass WorldModelAdapter**
   Create a new class: ``class MyAdapter(WorldModelAdapter): ...``

2. **Implement required abstract methods**
   - :meth:`encode(obs)` — encode raw observation to latent embedding
   - :meth:`initial_state(batch_size)` — return initial recurrent hidden state
   - :meth:`dynamics_step(h, z, action)` — compute one step of dynamics
   - :meth:`posterior(h, obs_emb)` — compute posterior latent logits

3. **Optionally implement optional heads**
   - :meth:`reward_pred(h, z)` — scalar reward prediction
   - :meth:`cont_pred(h, z)` — episode-continuation probability
   - :meth:`actor(h, z)` — action logits (for reinforcement learning)
   - :meth:`value(h, z)` — scalar value estimate (for reinforcement learning)

4. **Implement parameter access & device management**
   - :meth:`named_parameters()` — yield (name, param) tuples
   - :meth:`to(device)` — move model to device
   - :meth:`eval()` / :meth:`train()` — switch modes

5. **Set metadata properties**
   - :prop:`backend_name` — return string identifier (e.g., "my_model")
   - :prop:`hook_point_names` — list of activation points to hook (optional)

6. **Register your adapter** in :mod:`world_model_lens.backends`
   Add to ``BACKEND_REGISTRY`` dict: ``"my_model": MyAdapter``

Shape Conventions
-----------------
Most methods operate on single-step, single-episode tensors:

- ``d_h``    — recurrent hidden state dimension
- ``d_obs``  — raw observation dimension (post-flattening)
- ``d_emb``  — encoder output dimension
- ``n_cat``  — number of categorical latent variables
- ``n_cls``  — number of classes per categorical
- ``d_z``    — ``n_cat * n_cls`` (total stochastic latent size)
- ``d_a``    — action dimension

Example shapes:
  - obs: ``(64, 64, 3)`` for RGB image or ``(18,)`` for vector
  - obs_emb: ``(512,)`` for embedding
  - h: ``(256,)`` for hidden state
  - z: ``(1024,)`` or ``(32, 32)`` for latent
  - action: ``(6,)`` for 6-D continuous action

Full Example
------------
Here's a minimal working adapter::

    from world_model_lens.backends.base import WorldModelAdapter
    from world_model_lens.core.config import WorldModelConfig
    import torch
    import torch.nn as nn

    class MinimalAdapter(WorldModelAdapter):
        def __init__(self, cfg: WorldModelConfig):
            super().__init__()
            self._cfg = cfg
            self.enc = nn.Linear(cfg.d_obs, 512)
            self.dyn = nn.Linear(512 + cfg.d_action, 512)

        def encode(self, obs):
            return self.enc(obs)

        def initial_state(self, batch_size=1):
            return torch.zeros(batch_size, self._cfg.d_h)

        def dynamics_step(self, h, z, action):
            z_next = self.dyn(torch.cat([z, action], dim=-1))
            return h, z_next

        def posterior(self, h, obs_emb):
            return obs_emb  # or compute from obs_emb

        @property
        def backend_name(self):
            return "minimal"

        def named_parameters(self):
            for name, p in self.enc.named_parameters():
                yield f"enc.{name}", p
            for name, p in self.dyn.named_parameters():
                yield f"dyn.{name}", p

        def to(self, device):
            self.enc.to(device)
            self.dyn.to(device)
            return self

        def eval(self):
            self.enc.eval()
            self.dyn.eval()
            return self

        def train(self, mode=True):
            self.enc.train(mode)
            self.dyn.train(mode)
            return self
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from world_model_lens.backends.base import WorldModelAdapter
from world_model_lens.core.config import WorldModelConfig


class CustomWorldModelAdapter(WorldModelAdapter):
    """Template for custom world-model backend adapters.

    This class contains all required and optional methods with detailed docstrings,
    shape information, and implementation checklists. Copy this class and override
    the methods relevant to your architecture.

    Parameters
    ----------
    cfg : WorldModelConfig
        Shared configuration object containing d_h, d_action, d_obs, n_cat, n_cls.
    **kwargs :
        Additional hyperparameters specific to your architecture.

    Examples
    --------
    To implement a custom backend:

    1. Copy this class to your module (e.g., ``my_backends.py``):
       >>> class MyAdapter(CustomWorldModelAdapter):
       ...     pass

    2. Override only the methods you need; delete the rest.

    3. Implement the required abstract methods:
       >>> def encode(self, obs): ...
       >>> def initial_state(self, batch_size): ...
       >>> def dynamics_step(self, h, z, action): ...
       >>> def posterior(self, h, obs_emb): ...

    4. Set the ``backend_name`` property.

    5. Register in the backend registry.
    """

    def __init__(self, cfg: WorldModelConfig, **kwargs) -> None:
        """Initialize the custom adapter.

        Parameters
        ----------
        cfg : WorldModelConfig
            Configuration with d_h, d_action, d_obs, n_cat, n_cls, etc.
        **kwargs :
            Hyperparameters like learning_rate, hidden_dim, etc.
        """
        super().__init__()
        self._cfg = cfg
        # TODO: Store configuration and initialize nn.Module sub-networks here.
        # Examples:
        #   self.encoder = MyEncoder(cfg.d_obs, d_hidden=512)
        #   self.dynamics = MyDynamics(d_z=cfg.d_z, d_action=cfg.d_action)
        raise NotImplementedError(
            "CustomWorldModelAdapter is a template. Override __init__ in your subclass."
        )

    # ──────────────────────────────────────────────────────────────────
    # REQUIRED ABSTRACT METHODS
    # ──────────────────────────────────────────────────────────────────

    def encode(self, obs: Tensor) -> Tensor:
        """Encode raw observation to latent embedding.

        This is the first step of inference: convert pixel data or vector
        observations to a fixed-size latent representation that the rest
        of the model operates on.

        ✓ Checklist:
          - Accept batched and unbatched inputs (handle both gracefully)
          - Return embedding of shape (d_emb,) or (B, d_emb)
          - Detach output if using in inference mode
          - Handle device movement (move tensors internally if needed)

        Parameters
        ----------
        obs : Tensor
            Raw observation of shape ``(*obs_shape)`` or ``(B, *obs_shape)``.
            For visual: ``(C, H, W)`` or ``(B, C, H, W)``.
            For vector: ``(d_obs,)`` or ``(B, d_obs,)``.

        Returns
        -------
        Tensor
            Latent embedding of shape ``(d_emb,)`` or ``(B, d_emb)``.
            Matches batch dimension of input.

        Examples
        --------
        >>> cfg = WorldModelConfig(d_h=256, d_action=4, d_obs=12288)
        >>> adapter = CustomAdapter(cfg)
        >>> obs = torch.randn(3, 64, 64)  # Single RGB image
        >>> emb = adapter.encode(obs)
        >>> emb.shape
        torch.Size([512])  # d_emb=512

        >>> obs_batch = torch.randn(8, 3, 64, 64)  # Batch of 8 images
        >>> emb_batch = adapter.encode(obs_batch)
        >>> emb_batch.shape
        torch.Size([8, 512])
        """
        raise NotImplementedError(
            f"{type(self).__name__}.encode() must be implemented. "
            "Return a tensor of shape (d_emb,) or (B, d_emb)."
        )

    def initial_state(self, batch_size: int = 1) -> Tensor:
        """Return initial (zero or learned) recurrent hidden state.

        This is called at the start of a rollout to initialize the recurrent
        core. For stateless models (e.g., TD-MPC2), return zeros. For models
        with a learned initial state, return a parameter or computed value.

        ✓ Checklist:
          - Return shape (d_h,) if batch_size=1
          - Return shape (B, d_h) if batch_size > 1
          - All zeros is a safe default
          - Place tensor on correct device (check self._device if tracked)

        Parameters
        ----------
        batch_size : int
            Number of parallel episodes. Default 1.

        Returns
        -------
        Tensor
            Initial hidden state of shape ``(d_h,)`` or ``(B, d_h)``.
            For batch_size=1, typically return unbatched shape (d_h,).

        Examples
        --------
        >>> h0 = adapter.initial_state(batch_size=1)
        >>> h0.shape
        torch.Size([256])

        >>> h0_batch = adapter.initial_state(batch_size=16)
        >>> h0_batch.shape
        torch.Size([16, 256])

        Notes
        -----
        If you have a learned initial state, register it as a parameter::

            self.h0 = nn.Parameter(torch.randn(d_h))

        Then in ``initial_state``, return::

            return self.h0.expand(batch_size, -1).squeeze(0) if batch_size == 1 else ...
        """
        raise NotImplementedError(
            f"{type(self).__name__}.initial_state() must be implemented. "
            "Return zeros of shape (d_h,) or (B, d_h)."
        )

    def dynamics_step(
        self,
        h: Tensor,
        z: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute one step of latent dynamics.

        This implements the deterministic transition (typically a GRU or MLP)
        and the prior distribution over the next latent state. This is the
        core of the world model.

        Common implementations:
        - **RSSM**: ``h_next = GRU(h, cat(z_flat, action))``; compute prior from h_next
        - **TD-MPC**: ``z_prior = MLP(cat(z, action))``; h_next stays constant

        ✓ Checklist:
          - Accept h, z, action in batched or unbatched form
          - Return (h_next, z_prior_logits) with matching batch dims
          - z_prior_logits should be raw logits, not probabilities
          - For continuous latents, logits are Gaussian parameters (mean, log_var)
          - For categorical latents, logits are class scores

        Parameters
        ----------
        h : Tensor
            Current recurrent hidden state, shape ``(d_h,)`` or ``(B, d_h)``.
        z : Tensor
            Current stochastic latent, shape ``(d_z,)`` or ``(B, d_z)``.
            May also be ``(n_cat, n_cls)`` or ``(B, n_cat, n_cls)`` depending
            on representation. Adapter handles flattening/reshaping internally.
        action : Tensor
            Action taken at this step, shape ``(d_a,)`` or ``(B, d_a)``.

        Returns
        -------
        h_next : Tensor
            Updated hidden state, shape ``(d_h,)`` or ``(B, d_h)``.
            Matches batch dimension of input.
        z_prior_logits : Tensor
            Prior logits for the next latent, shape ``(d_z,)`` or ``(B, d_z)``.
            These are *logits*, not probabilities. The caller will sample or
            take argmax depending on the latent type.

        Examples
        --------
        >>> h = torch.randn(256)
        >>> z = torch.randn(1024)
        >>> action = torch.randn(6)
        >>> h_next, z_prior = adapter.dynamics_step(h, z, action)
        >>> h_next.shape, z_prior.shape
        (torch.Size([256]), torch.Size([1024]))

        >>> # Batched
        >>> h_batch = torch.randn(8, 256)
        >>> z_batch = torch.randn(8, 1024)
        >>> action_batch = torch.randn(8, 6)
        >>> h_next, z_prior = adapter.dynamics_step(h_batch, z_batch, action_batch)
        >>> h_next.shape, z_prior.shape
        (torch.Size([8, 256]), torch.Size([8, 1024]))

        Notes
        -----
        For RSSM-like models::

            def dynamics_step(self, h, z, action):
                x = torch.cat([z.view(z.shape[0], -1), action], dim=-1)
                h_next = self.gru_cell(x, h)
                z_prior_logits = self.prior_head(h_next)
                return h_next, z_prior_logits

        For stateless models (TD-MPC2)::

            def dynamics_step(self, h, z, action):
                z_prior_logits = self.prior_mlp(torch.cat([z, action], dim=-1))
                return h, z_prior_logits  # h unchanged
        """
        raise NotImplementedError(
            f"{type(self).__name__}.dynamics_step() must be implemented. "
            "Return (h_next, z_prior_logits) tuple."
        )

    def posterior(self, h: Tensor, obs_emb: Tensor) -> Tensor:
        """Compute posterior distribution over latent state.

        Given the current hidden state and observation embedding, compute
        the posterior distribution (logits) over the stochastic latent.
        This is used to sample the latent during training (teacher forcing)
        and inference.

        ✓ Checklist:
          - Accept h and obs_emb in batched or unbatched form
          - Return logits (not probabilities) of shape (d_z,) or (B, d_z)
          - For categorical latents, return (n_cat, n_cls) or (B, n_cat, n_cls)
          - For continuous latents, return raw parameters or samples
          - Posterior typically depends on both h and obs_emb (unlike prior)

        Parameters
        ----------
        h : Tensor
            Current recurrent hidden state, shape ``(d_h,)`` or ``(B, d_h)``.
            Some architectures (e.g., RSSM) condition the posterior on h.
            Others (e.g., TD-MPC2) may ignore h and use only obs_emb.
        obs_emb : Tensor
            Observation embedding from :meth:`encode`, shape ``(d_emb,)`` or ``(B, d_emb)``.

        Returns
        -------
        Tensor
            Posterior logits, shape ``(d_z,)`` or ``(B, d_z)``.
            May also be ``(n_cat, n_cls)`` or ``(B, n_cat, n_cls)``.

        Examples
        --------
        >>> h = torch.randn(256)
        >>> obs_emb = torch.randn(512)
        >>> z_post = adapter.posterior(h, obs_emb)
        >>> z_post.shape
        torch.Size([1024])

        >>> # Categorical representation
        >>> z_post = adapter.posterior(h, obs_emb)
        >>> z_post.shape
        torch.Size([32, 32])  # (n_cat, n_cls)

        Notes
        -----
        A typical RSSM posterior head::

            def posterior(self, h, obs_emb):
                x = torch.cat([h, obs_emb], dim=-1)
                return self.posterior_mlp(x)

        Some models (TD-MPC2) only condition on obs_emb::

            def posterior(self, h, obs_emb):
                return self.posterior_mlp(obs_emb)
        """
        raise NotImplementedError(
            f"{type(self).__name__}.posterior() must be implemented. "
            "Return logits of shape (d_z,) or (B, d_z)."
        )

    # ──────────────────────────────────────────────────────────────────
    # OPTIONAL HEADS (raise NotImplementedError by default)
    # ──────────────────────────────────────────────────────────────────

    def reward_pred(self, h: Tensor, z: Tensor) -> Tensor:
        """Predict reward from latent state.

        Optional: override if your model has a reward prediction head.
        Used for model-based RL and planning.

        ✓ Checklist:
          - Accept h, z in batched or unbatched form
          - Return scalar reward(s) of shape () or (1,) or (B,) or (B, 1)
          - For distributional rewards (two-hot), return logits of shape (B,) or (B, n_bins)

        Parameters
        ----------
        h : Tensor
            Hidden state, shape ``(d_h,)`` or ``(B, d_h)``.
        z : Tensor
            Stochastic latent (soft), shape ``(d_z,)`` or ``(B, d_z)``.

        Returns
        -------
        Tensor
            Scalar reward prediction(s), shape ``()`` or ``(1,)`` or ``(B,)`` or ``(B, 1)``.

        Examples
        --------
        >>> h = torch.randn(256)
        >>> z = torch.randn(1024)
        >>> reward = adapter.reward_pred(h, z)
        >>> reward.shape
        torch.Size([])  # scalar

        >>> reward = reward.item()  # Convert to Python float
        >>> print(f"Predicted reward: {reward:.3f}")
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a reward head."
        )

    def cont_pred(self, h: Tensor, z: Tensor) -> Tensor:
        """Predict episode-continuation probability.

        Optional: override if your model predicts whether episodes end.
        Used to compute episode masks and expected return predictions.

        ✓ Checklist:
          - Accept h, z in batched or unbatched form
          - Return probability in [0, 1] of shape () or (1,) or (B,) or (B, 1)
          - Typically computed as sigmoid(logits)

        Parameters
        ----------
        h : Tensor
            Hidden state, shape ``(d_h,)`` or ``(B, d_h)``.
        z : Tensor
            Stochastic latent, shape ``(d_z,)`` or ``(B, d_z)``.

        Returns
        -------
        Tensor
            Continuation probability in [0, 1], shape ``()`` or ``(1,)`` or ``(B,)`` or ``(B, 1)``.

        Examples
        --------
        >>> h = torch.randn(256)
        >>> z = torch.randn(1024)
        >>> cont = adapter.cont_pred(h, z)
        >>> cont.shape
        torch.Size([])

        >>> print(f"Episode continues with probability {cont.item():.2%}")
        Episode continues with probability 95.32%
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a continuation head."
        )

    def actor(self, h: Tensor, z: Tensor) -> Tensor:
        """Compute action logits or means from latent state.

        Optional: override if your model has an actor (policy) head.
        Used for reinforcement learning and planning.

        ✓ Checklist:
          - Accept h, z in batched or unbatched form
          - For discrete actions: return logits of shape (d_a,) or (B, d_a)
          - For continuous actions: return means of shape (d_a,) or (B, d_a)
                                     or return (mean, log_std) tuple

        Parameters
        ----------
        h : Tensor
            Hidden state, shape ``(d_h,)`` or ``(B, d_h)``.
        z : Tensor
            Stochastic latent, shape ``(d_z,)`` or ``(B, d_z)``.

        Returns
        -------
        Tensor
            Action logits (discrete) or means (continuous), shape ``(d_a,)`` or ``(B, d_a)``.

        Examples
        --------
        >>> h = torch.randn(256)
        >>> z = torch.randn(1024)
        >>> action_logits = adapter.actor(h, z)
        >>> action_logits.shape
        torch.Size([6])  # 6-D action space

        >>> # Sample action
        >>> action = torch.softmax(action_logits, dim=-1).multinomial(1)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement an actor head."
        )

    def value(self, h: Tensor, z: Tensor) -> Tensor:
        """Estimate value (expected return) from latent state.

        Optional: override if your model has a value (critic) head.
        Used for reinforcement learning and planning.

        ✓ Checklist:
          - Accept h, z in batched or unbatched form
          - Return scalar value(s) of shape () or (1,) or (B,) or (B, 1)
          - For distributional values (two-hot), return logits of shape (B, n_bins)

        Parameters
        ----------
        h : Tensor
            Hidden state, shape ``(d_h,)`` or ``(B, d_h)``.
        z : Tensor
            Stochastic latent, shape ``(d_z,)`` or ``(B, d_z)``.

        Returns
        -------
        Tensor
            Scalar value estimate(s), shape ``()`` or ``(1,)`` or ``(B,)`` or ``(B, 1)``.

        Examples
        --------
        >>> h = torch.randn(256)
        >>> z = torch.randn(1024)
        >>> value = adapter.value(h, z)
        >>> value.shape
        torch.Size([])  # scalar

        >>> print(f"Estimated value: ${value.item():.2f}")
        Estimated value: $42.15
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement a value head."
        )

    # ──────────────────────────────────────────────────────────────────
    # PARAMETER ACCESS & DEVICE MANAGEMENT
    # ──────────────────────────────────────────────────────────────────

    def named_parameters(self) -> Iterator[Tuple[str, Tensor]]:
        """Iterate over (name, parameter) pairs of all model weights.

        Override this to expose parameters of your nn.Module sub-networks
        to the caller. The :class:`~world_model_lens.HookedWorldModel` uses
        this to enumerate all model parameters.

        ✓ Checklist:
          - Yield all trainable weights (parameters)
          - Use descriptive hierarchical names: "encoder.conv1.weight"
          - If not overridden, defaults to empty iterator

        Yields
        ------
        (str, Tensor)
            Name and parameter tensor pairs.

        Examples
        --------
        >>> for name, param in adapter.named_parameters():
        ...     print(f"{name}: {param.shape}")
        encoder.conv1.weight: torch.Size([32, 3, 4, 4])
        encoder.conv1.bias: torch.Size([32])
        ...

        Typical override pattern::

            def named_parameters(self):
                for name, param in self.encoder.named_parameters():
                    yield f"encoder.{name}", param
                for name, param in self.dynamics.named_parameters():
                    yield f"dynamics.{name}", param
                for name, param in self.reward.named_parameters():
                    yield f"reward.{name}", param
        """
        return iter([])

    def to(self, device: Union[str, torch.device]) -> "CustomWorldModelAdapter":
        """Move all internal modules to the specified device.

        Override this to move all nn.Module sub-networks to a target device
        (e.g., "cuda:0", "cpu"). Called by :class:`~world_model_lens.HookedWorldModel`
        to synchronize device placement.

        ✓ Checklist:
          - Move all torch.nn.Module attributes to device
          - Store device for later reference (useful for initial_state)
          - Return self for method chaining
          - Handle both str and torch.device inputs

        Parameters
        ----------
        device : str | torch.device
            Target device, e.g., "cuda:0", "cpu", torch.device("cuda:0").

        Returns
        -------
        CustomWorldModelAdapter
            self, for chaining.

        Examples
        --------
        >>> adapter.to("cuda:0")
        >>> adapter.to(torch.device("mps"))

        Typical override pattern::

            def to(self, device):
                self._device = torch.device(device) if isinstance(device, str) else device
                self.encoder = self.encoder.to(self._device)
                self.dynamics = self.dynamics.to(self._device)
                return self
        """
        return self

    def eval(self) -> "CustomWorldModelAdapter":
        """Switch to evaluation mode.

        Override this to set all nn.Module sub-networks to evaluation mode
        (disables dropout, batch norm running stats, etc.). Called before
        running inference or validation.

        ✓ Checklist:
          - Call ``.eval()`` on all nn.Module attributes
          - Return self for method chaining

        Returns
        -------
        CustomWorldModelAdapter
            self, for chaining.

        Examples
        --------
        >>> adapter.eval()
        >>> with torch.no_grad():
        ...     output = adapter.encode(obs)

        Typical override pattern::

            def eval(self):
                self.encoder.eval()
                self.dynamics.eval()
                return self
        """
        return self

    def train(self, mode: bool = True) -> "CustomWorldModelAdapter":
        """Switch to training mode.

        Override this to set all nn.Module sub-networks to training mode
        (enables dropout, batch norm updates, etc.). Called before running
        training loops.

        ✓ Checklist:
          - Call ``.train(mode)`` on all nn.Module attributes
          - Return self for method chaining

        Parameters
        ----------
        mode : bool
            Whether to enable training. Default True.
            Pass False to disable training (equivalent to :meth:`eval`).

        Returns
        -------
        CustomWorldModelAdapter
            self, for chaining.

        Examples
        --------
        >>> adapter.train()
        >>> optimizer.zero_grad()
        >>> loss = compute_loss(adapter, batch)
        >>> loss.backward()

        >>> adapter.train(False)  # Equivalent to adapter.eval()

        Typical override pattern::

            def train(self, mode=True):
                self.encoder.train(mode)
                self.dynamics.train(mode)
                return self
        """
        return self

    # ──────────────────────────────────────────────────────────────────
    # METADATA
    # ──────────────────────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        """Return a human-readable backend identifier.

        Override this to return a unique string name for your backend.
        Used in logging, checkpointing, and registry lookups.

        Returns
        -------
        str
            Backend name, e.g., "my_model", "custom_rssm", "tdmpc2".

        Examples
        --------
        >>> adapter.backend_name
        'my_model'

        Typical override::

            @property
            def backend_name(self):
                return "my_model"
        """
        return type(self).__name__

    @property
    def hook_point_names(self) -> list[str]:
        """Return all named activation hook-points exposed by this adapter.

        Override this to expose custom activation points (internal hidden states,
        intermediate features, etc.) that can be hooked and cached by
        :class:`~world_model_lens.HookedWorldModel`.

        The default list covers the 11 standard RSSM hook-points. If your model
        has extra internals (e.g., transformer layers, attention heads), extend
        this list.

        Returns
        -------
        list of str
            Activation point names. The base list includes:

            - ``"encoder.out"`` — encoder output / obs embedding
            - ``"rnn.h"`` — recurrent hidden state after dynamics
            - ``"z_prior.logits"`` — prior logits before sampling
            - ``"z_prior"`` — prior sample / mode
            - ``"z_posterior.logits"`` — posterior logits before sampling
            - ``"z_posterior"`` — posterior sample / mode
            - ``"kl"`` — KL divergence (if computed)
            - ``"reward_pred"`` — reward prediction
            - ``"cont_pred"`` — continuation prediction
            - ``"actor.logits"`` — actor output logits
            - ``"value_pred"`` — value prediction

        Examples
        --------
        >>> adapter.hook_point_names
        ['encoder.out', 'rnn.h', 'z_prior.logits', ..., 'transformer.layer_0', ...]

        Typical override for a transformer model::

            @property
            def hook_point_names(self):
                base = super().hook_point_names
                transformer_hooks = [
                    f"transformer.layer_{i}" for i in range(self.n_layers)
                ]
                return base + transformer_hooks
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

    def __repr__(self) -> str:
        """Return a concise string representation of this adapter.

        Override to provide useful debugging info.

        Returns
        -------
        str

        Examples
        --------
        >>> repr(adapter)
        'CustomWorldModelAdapter(d_h=256, d_z=1024, d_action=6)'
        """
        return f"{type(self).__name__}()"


__all__ = [
    "CustomWorldModelAdapter",
]
