"""ActivationCache — keyed storage for per-component, per-timestep activations.

Activations are stored under ``(component_name: str, timestep: int)`` keys.
The cache supports *lazy evaluation*: any value stored as a zero-argument
callable is only evaluated the first time it is accessed, then memoised.

Indexing interface
------------------
``cache[name, timestep]``       → single :class:`torch.Tensor`
``cache[name, 2:5]``            → steps 2,3,4 stacked along dim 0
``cache[name, ':']``            → all timesteps for *name*, stacked
``cache[name]``                 → same as ``cache[name, ':']``
``cache['z_posterior', 3]``     → posterior latent at t=3
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence

import torch

# Lazy value: either a tensor or a zero-arg callable that returns one.
_StoredValue = torch.Tensor | Callable[[], torch.Tensor]


class ActivationCache:
    """Activation storage keyed by ``(component_name, timestep)``.

    Parameters
    ----------
    data:
        Optional initial data as a dict mapping ``(name, timestep)``
        tuples to tensors (or callables that return tensors).

    Examples
    --------
    >>> cache = ActivationCache()
    >>> cache["encoder.out", 0] = torch.randn(512)
    >>> cache["encoder.out", 1] = torch.randn(512)
    >>> cache["encoder.out"].shape          # all timesteps stacked
    torch.Size([2, 512])
    >>> cache["encoder.out", 0].shape       # single step
    torch.Size([512])
    >>> cache["encoder.out", 0:2].shape     # slice
    torch.Size([2, 512])
    >>> cache["encoder.out", ':'].shape     # explicit 'all'
    torch.Size([2, 512])

    Lazy evaluation
    ~~~~~~~~~~~~~~~
    >>> cache["slow_feat", 0] = lambda: expensive_computation()
    >>> cache["slow_feat", 0]   # callable is called once; result is cached
    tensor(...)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, data: dict[tuple[str, int], _StoredValue] | None = None) -> None:
        # _store: (name, t) → tensor-or-callable
        self._store: dict[tuple[str, int], _StoredValue] = {}
        if data:
            for k, v in data.items():
                self._store[k] = v

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, key: tuple[str, int]) -> torch.Tensor:
        """Evaluate and memoise a lazy callable if necessary."""
        val = self._store[key]
        if callable(val):
            val = val()
            if not isinstance(val, torch.Tensor):
                raise TypeError(
                    f"Lazy callable for key {key!r} did not return a torch.Tensor "
                    f"(got {type(val).__name__})."
                )
            self._store[key] = val  # memoise
        return val  # type: ignore[return-value]

    def _timesteps_for(self, name: str) -> list[int]:
        """Sorted list of timesteps stored for *name*."""
        return sorted(t for (n, t) in self._store if n == name)

    def _stack_for(self, name: str, timesteps: Sequence[int]) -> torch.Tensor:
        """Stack tensors for *name* at *timesteps* along dim 0."""
        tensors = [self._resolve((name, t)) for t in timesteps]
        if not tensors:
            raise KeyError(f"No entries found for component {name!r} at the requested timesteps.")
        return torch.stack(tensors, dim=0)

    # ------------------------------------------------------------------
    # Core dunder methods
    # ------------------------------------------------------------------

    def __setitem__(self, key: str | tuple[str, int], value: _StoredValue) -> None:
        """Store a tensor (or lazy callable) in the cache.

        Parameters
        ----------
        key:
            Either ``(name, timestep)`` or just ``name``.  If only ``name``
            is supplied a ``timestep`` of ``-1`` is assumed (useful for
            non-temporal activations).
        value:
            A :class:`torch.Tensor` or a zero-argument callable returning
            one (lazy value).

        Examples
        --------
        >>> cache["rnn.h", 0] = torch.randn(512)
        >>> cache["rnn.h", 1] = lambda: compute_hidden(input_t1)
        """
        if isinstance(key, str):
            key = (key, -1)
        if not (
            isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[0], str)
            and isinstance(key[1], int)
        ):
            raise KeyError(f"Cache key must be (str, int) or str, got {key!r}.")
        self._store[key] = value

    def __getitem__(self, key: str | tuple[str, int | slice | str]) -> torch.Tensor:
        """Retrieve one or more cached activations.

        Supported indexing patterns
        ---------------------------
        ``cache[name]``             all timesteps stacked (dim 0)
        ``cache[name, ':']``        same as above
        ``cache[name, t]``          single tensor at integer timestep *t*
        ``cache[name, start:stop]`` slice of timesteps, stacked

        Raises
        ------
        KeyError
            If *name* has no entries, or the requested timestep is absent.
        """
        # ---- normalise key ----------------------------------------
        if isinstance(key, str):
            # cache[name]  →  all timesteps
            name: str = key
            return self._all_timesteps(name)

        if not (isinstance(key, tuple) and len(key) == 2):
            raise KeyError(f"Invalid cache key {key!r}.")

        name, idx = key  # type: ignore[misc]

        # ---- cache[name, ':']  →  all timesteps -------------------
        if idx == ":":
            return self._all_timesteps(name)

        # ---- cache[name, t]  →  single tensor ---------------------
        if isinstance(idx, int):
            full_key = (name, idx)
            if full_key not in self._store:
                raise KeyError(
                    f"No activation cached for component={name!r}, timestep={idx}. "
                    f"Available timesteps: {self._timesteps_for(name)}"
                )
            return self._resolve(full_key)

        # ---- cache[name, start:stop]  →  sliced stack -------------
        if isinstance(idx, slice):
            all_ts = self._timesteps_for(name)
            selected = all_ts[idx]
            if not selected:
                raise KeyError(
                    f"Slice {idx} produced no timesteps for component {name!r}. Available: {all_ts}"
                )
            return self._stack_for(name, selected)

        raise KeyError(
            f"Unsupported index type {type(idx).__name__!r} for key {key!r}. "
            "Use int, slice, or ':' (str)."
        )

    def _all_timesteps(self, name: str) -> torch.Tensor:
        ts = self._timesteps_for(name)
        if not ts:
            raise KeyError(f"Component {name!r} has no entries in the cache.")
        return self._stack_for(name, ts)

    def __contains__(self, key: object) -> bool:
        """Test membership.

        ``(name, t) in cache``  → exact key lookup
        ``name in cache``       → True if any timestep is stored for *name*
        """
        if isinstance(key, tuple) and len(key) == 2:
            return key in self._store
        if isinstance(key, str):
            return any(n == key for (n, _) in self._store)
        return False

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[tuple[str, int]]:
        return iter(self._store)

    def __repr__(self) -> str:
        n_components = len(self.component_names)
        n_entries = len(self._store)
        return f"ActivationCache(components={n_components}, entries={n_entries})"

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def keys(self) -> list[tuple[str, int]]:
        """All ``(component_name, timestep)`` keys, sorted.

        Returns
        -------
        list of (str, int)
        """
        return sorted(self._store.keys())

    @property
    def component_names(self) -> list[str]:
        """Unique component names, sorted alphabetically.

        Returns
        -------
        list of str
        """
        return sorted({name for (name, _) in self._store})

    @property
    def timesteps(self) -> list[int]:
        """Unique timestep indices across all components, sorted.

        Returns
        -------
        list of int
        """
        return sorted({t for (_, t) in self._store})

    # ------------------------------------------------------------------
    # Safe access
    # ------------------------------------------------------------------

    def get(
        self, name: str, timestep: int, default: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Return the activation at ``(name, timestep)`` or *default*.

        Parameters
        ----------
        name:
            Component name.
        timestep:
            Step index.
        default:
            Value to return when the key is absent (default ``None``).

        Returns
        -------
        torch.Tensor | None
        """
        key = (name, timestep)
        if key not in self._store:
            return default
        return self._resolve(key)

    # ------------------------------------------------------------------
    # Device / gradient management
    # ------------------------------------------------------------------

    def to_device(self, device: str | torch.device) -> ActivationCache:
        """Return a new cache with all *evaluated* tensors moved to *device*.

        Lazy (callable) entries are forced before being moved.

        Parameters
        ----------
        device:
            Target device.

        Returns
        -------
        ActivationCache
        """
        dev = torch.device(device)
        new: dict[tuple[str, int], _StoredValue] = {}
        for k in self._store:
            new[k] = self._resolve(k).to(dev)
        return ActivationCache(new)

    def detach(self) -> ActivationCache:
        """Return a new cache with all tensors detached from the autograd graph.

        Lazy entries are forced before detaching.

        Returns
        -------
        ActivationCache
        """
        new: dict[tuple[str, int], _StoredValue] = {}
        for k in self._store:
            new[k] = self._resolve(k).detach()
        return ActivationCache(new)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(self, names: Iterable[str]) -> ActivationCache:
        """Return a new cache containing only the specified component names.

        Parameters
        ----------
        names:
            Component names to keep.  Names not present in the cache are
            silently ignored.

        Returns
        -------
        ActivationCache

        Examples
        --------
        >>> sub = cache.filter(["z_posterior", "z_prior"])
        >>> sub.component_names
        ['z_posterior', 'z_prior']
        """
        keep = set(names)
        new = {k: v for k, v in self._store.items() if k[0] in keep}
        return ActivationCache(new)

    # ------------------------------------------------------------------
    # Domain-specific helpers
    # ------------------------------------------------------------------

    def surprise(self) -> dict[int, torch.Tensor]:
        """Compute per-timestep KL divergence KL(z_posterior ‖ z_prior).

        Expects the cache to contain entries keyed ``"z_posterior"`` and
        ``"z_prior"`` at matching timesteps.  The KL is computed for every
        timestep at which *both* keys are present.

        The computation mirrors :attr:`LatentState.kl`: tensors are treated
        as logits, passed through ``softmax``, then the categorical KL is
        summed over the ``n_cat`` dimension.

        Returns
        -------
        dict mapping int → scalar torch.Tensor
            ``{timestep: kl_scalar}`` for every shared timestep.  Empty
            dict if ``"z_posterior"`` or ``"z_prior"`` are absent.

        Examples
        --------
        >>> surprises = cache.surprise()
        >>> print(surprises[0])
        tensor(27.8)
        """
        post_ts = set(self._timesteps_for("z_posterior"))
        prior_ts = set(self._timesteps_for("z_prior"))
        shared = sorted(post_ts & prior_ts)

        result: dict[int, torch.Tensor] = {}
        for t in shared:
            p = self._resolve(("z_posterior", t)).softmax(dim=-1).clamp(min=1e-8)
            q = self._resolve(("z_prior", t)).softmax(dim=-1).clamp(min=1e-8)
            # KL summed over categorical variables
            kl = (p * (p.log() - q.log())).sum(dim=-1).sum()
            result[t] = kl
        return result

    def to_dataframe(self, flatten: bool = True):  # → pd.DataFrame
        """Export cache contents to a :class:`pandas.DataFrame`.

        Each row corresponds to one ``(component_name, timestep)`` entry.
        Tensor values are flattened (or kept as nested lists when
        ``flatten=False``) and stored in a single ``"value"`` column, with
        the flat feature dimension reflected in ``"feature_dim"``.

        Parameters
        ----------
        flatten:
            If ``True`` (default), 1-D numpy arrays are stored per row.
            If ``False``, raw tensors are kept (useful for further
            tensor-level processing without round-tripping through numpy).

        Returns
        -------
        pandas.DataFrame
            Columns: ``component``, ``timestep``, ``shape``, ``value``
            (plus one column per feature index when ``flatten=True`` and
            tensors are 1-D, i.e. the "wide" format).

        Raises
        ------
        ImportError
            If :mod:`pandas` is not installed.

        Examples
        --------
        >>> df = cache.to_dataframe()
        >>> df.columns.tolist()
        ['component', 'timestep', 'shape', 'value']
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for ActivationCache.to_dataframe(). "
                "Install it with: pip install pandas"
            ) from e

        rows = []
        for name, t in sorted(self._store.keys()):
            tensor = self._resolve((name, t))
            flat = tensor.detach().cpu().float()
            shape_str = str(tuple(flat.shape))
            if flatten:
                rows.append(
                    {
                        "component": name,
                        "timestep": t,
                        "shape": shape_str,
                        "value": flat.flatten().numpy(),
                    }
                )
            else:
                rows.append(
                    {
                        "component": name,
                        "timestep": t,
                        "shape": shape_str,
                        "value": flat,
                    }
                )

        return pd.DataFrame(rows)
