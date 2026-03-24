"""WorldModelConfig — declarative configuration for world model architecture.

:class:`WorldModelConfig` is the single source of truth for all architectural
hyperparameters.  It is passed to model builders, analysis tools, and the CLI
so that every component shares the same view of the model's shape.

Usage
-----
>>> cfg = WorldModelConfig(d_h=512, d_action=18, d_obs=64*64*3)
>>> cfg.d_latent
1536    # 512 + 32*32
>>> cfg.d_z
1024    # 32*32

Serialisation
-------------
>>> cfg.save("config.yaml")
>>> cfg2 = WorldModelConfig.load("config.yaml")
>>> cfg == cfg2
True
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import yaml


# Sentinel to detect missing required fields in from_dict
_MISSING = object()


@dataclass
class WorldModelConfig:
    """Full architectural configuration for a world model.

    Required Parameters
    -------------------
    d_h:
        Dimensionality of the recurrent deterministic hidden state ``h_t``.
    d_action:
        Dimensionality of the action space.  For discrete actions this is the
        number of possible actions; for continuous actions it is the vector
        length.
    d_obs:
        Dimensionality of the raw observation (e.g. ``64*64*3 = 12288`` for
        a 64×64 RGB image that has been flattened by the encoder input layer).

    Optional Parameters (with defaults)
    ------------------------------------
    n_cat:
        Number of categorical latent variables (DreamerV3 default: 32).
    n_cls:
        Number of classes per categorical (DreamerV3 default: 32).
    encoder_type:
        Architecture of the observation encoder.  One of
        ``"cnn"`` | ``"mlp"`` | ``"transformer"``.
    backend:
        Model family identifier.  One of
        ``"dreamer"`` | ``"tdmpc"`` | ``"rssm"`` | ``"custom"``.
    n_gru_layers:
        Number of stacked GRU layers in the recurrent core.
    reward_head:
        Architecture of the reward prediction head.
        One of ``"mlp"`` | ``"linear"``.
    action_discrete:
        Whether the action space is discrete.  Inferred from context when
        not explicitly set.
    name:
        Human-readable name for this configuration (for logging/checkpoints).

    Derived Properties
    ------------------
    d_z : int
        Stochastic latent dimension = ``n_cat * n_cls``.
    d_latent : int
        Full latent dimension = ``d_h + d_z``.
    latent_shape : tuple[int, int]
        ``(n_cat, n_cls)`` — the 2-D shape of the categorical latent.

    Examples
    --------
    >>> cfg = WorldModelConfig(d_h=512, d_action=4, d_obs=4)
    >>> cfg.d_z
    1024
    >>> cfg.d_latent
    1536
    >>> cfg.latent_shape
    (32, 32)
    """

    # ------------------------------------------------------------------
    # Required fields (no defaults — must be supplied)
    # ------------------------------------------------------------------
    d_h: int
    """Recurrent hidden state dimension."""

    d_action: int
    """Action space dimensionality."""

    d_obs: int
    """Observation dimensionality (post-encoder input)."""

    # ------------------------------------------------------------------
    # Optional fields with defaults
    # ------------------------------------------------------------------
    n_cat: int = 32
    """Number of categorical latent variables."""

    n_cls: int = 32
    """Number of classes per categorical."""

    encoder_type: str = "cnn"
    """Observation encoder architecture: 'cnn' | 'mlp' | 'transformer'."""

    backend: str = "dreamer"
    """Model family: 'dreamer' | 'tdmpc' | 'rssm' | 'custom'."""

    n_gru_layers: int = 1
    """Number of stacked GRU layers in the recurrent core."""

    reward_head: str = "mlp"
    """Reward-prediction head architecture: 'mlp' | 'linear'."""

    action_discrete: bool = True
    """Whether the action space is discrete."""

    name: str = "world_model"
    """Human-readable identifier for logging and checkpointing."""

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        errors: list[str] = []

        if self.d_h <= 0:
            errors.append(f"d_h must be positive, got {self.d_h}.")
        if self.d_action <= 0:
            errors.append(f"d_action must be positive, got {self.d_action}.")
        if self.d_obs <= 0:
            errors.append(f"d_obs must be positive, got {self.d_obs}.")
        if self.n_cat <= 0:
            errors.append(f"n_cat must be positive, got {self.n_cat}.")
        if self.n_cls <= 0:
            errors.append(f"n_cls must be positive, got {self.n_cls}.")
        if self.n_gru_layers <= 0:
            errors.append(f"n_gru_layers must be positive, got {self.n_gru_layers}.")

        _valid_encoders = {"cnn", "mlp", "transformer"}
        if self.encoder_type not in _valid_encoders:
            errors.append(
                f"encoder_type must be one of {_valid_encoders}, "
                f"got {self.encoder_type!r}."
            )

        _valid_backends = {"dreamer", "tdmpc", "rssm", "custom"}
        if self.backend not in _valid_backends:
            errors.append(
                f"backend must be one of {_valid_backends}, "
                f"got {self.backend!r}."
            )

        _valid_reward_heads = {"mlp", "linear"}
        if self.reward_head not in _valid_reward_heads:
            errors.append(
                f"reward_head must be one of {_valid_reward_heads}, "
                f"got {self.reward_head!r}."
            )

        if errors:
            raise ValueError(
                "WorldModelConfig validation failed:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def d_z(self) -> int:
        """Stochastic latent dimension: ``n_cat × n_cls``.

        Returns
        -------
        int
            Total number of categorical class logits.

        Examples
        --------
        >>> WorldModelConfig(512, 4, 64, n_cat=32, n_cls=32).d_z
        1024
        """
        return self.n_cat * self.n_cls

    @property
    def d_latent(self) -> int:
        """Full latent dimension: ``d_h + d_z``.

        This is the size of the concatenated ``[h_t ‖ z_flat]`` vector
        used by probes and downstream analysis (cf. :attr:`LatentState.flat`).

        Returns
        -------
        int

        Examples
        --------
        >>> WorldModelConfig(512, 4, 64).d_latent
        1536    # 512 + 1024
        """
        return self.d_h + self.d_z

    @property
    def latent_shape(self) -> Tuple[int, int]:
        """2-D shape of the categorical latent: ``(n_cat, n_cls)``.

        Returns
        -------
        tuple[int, int]

        Examples
        --------
        >>> cfg.latent_shape
        (32, 32)
        """
        return (self.n_cat, self.n_cls)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain Python dictionary.

        All fields (including defaults) are included.  The dict is suitable
        for JSON / YAML serialisation, logging, or passing to experiment
        trackers.

        Returns
        -------
        dict

        Examples
        --------
        >>> cfg.to_dict()['d_h']
        512
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldModelConfig":
        """Construct a :class:`WorldModelConfig` from a plain dict.

        Unknown keys are silently ignored so that configs from future
        versions of the library can be loaded without error.

        Parameters
        ----------
        data:
            Dictionary of field values (typically loaded from YAML/JSON).

        Returns
        -------
        WorldModelConfig

        Raises
        ------
        TypeError
            If any *required* field (``d_h``, ``d_action``, ``d_obs``) is
            missing from *data*.
        """
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        for required in ("d_h", "d_action", "d_obs"):
            if required not in filtered:
                raise TypeError(
                    f"WorldModelConfig.from_dict(): required field "
                    f"'{required}' is missing from the provided dict."
                )
        return cls(**filtered)

    def save(self, path: Union[str, Path]) -> None:
        """Serialise to a YAML file.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created if they
            do not exist.

        Examples
        --------
        >>> cfg.save("experiments/run_001/config.yaml")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=True, default_flow_style=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WorldModelConfig":
        """Load a :class:`WorldModelConfig` from a YAML file.

        Parameters
        ----------
        path:
            Source YAML file path (produced by :meth:`save`).

        Returns
        -------
        WorldModelConfig

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.

        Examples
        --------
        >>> cfg = WorldModelConfig.load("experiments/run_001/config.yaml")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def replace(self, **changes: Any) -> "WorldModelConfig":
        """Return a copy of this config with *changes* applied.

        Equivalent to :func:`dataclasses.replace` but more discoverable.

        Parameters
        ----------
        **changes:
            Field names and their new values.

        Returns
        -------
        WorldModelConfig

        Examples
        --------
        >>> big_cfg = cfg.replace(d_h=1024, n_cat=64)
        """
        return dataclasses.replace(self, **changes)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"WorldModelConfig("
            f"name={self.name!r}, "
            f"d_h={self.d_h}, d_z={self.d_z}, d_latent={self.d_latent}, "
            f"d_action={self.d_action}, d_obs={self.d_obs}, "
            f"backend={self.backend!r}, encoder={self.encoder_type!r})"
        )
