"""Activation cache for storing and retrieving intermediate activations."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import torch
import pandas as pd


class ActivationCache:
    """Storage for activations keyed by (component_name, timestep).

    Supports single-item access, slicing, lazy evaluation, and export.

    Example:
        cache = ActivationCache()
        cache["z_posterior", 0] = tensor
        single = cache["z_posterior", 0]           # Single tensor
        sequence = cache["z_posterior", :]          # Stacked tensors
        sequence = cache["z_posterior", slice(0,5)] # Slice
    """

    def __init__(self):
        self._store: Dict[Tuple[str, int], Union[torch.Tensor, Callable[[], torch.Tensor]]] = {}
        self._evaluated: Dict[Tuple[str, int], torch.Tensor] = {}

    def __getitem__(
        self,
        key: Union[Tuple[str, int], Tuple[str, slice], Tuple[str, str], str],
    ) -> torch.Tensor:
        """Access cached activations.

        Args:
            key: Either:
                - (name, timestep): Single tensor
                - (name, slice): Tensor sequence via slice
                - (name, ':'): All timesteps stacked
                - name: All timesteps stacked

        Returns:
            torch.Tensor: Requested activation(s).
        """
        if isinstance(key, str):
            return self._get_all(key)

        name, second = key

        if isinstance(second, int):
            return self._get_single(name, second)
        elif isinstance(second, slice):
            return self._get_slice(name, second)
        elif second == ":":
            return self._get_all(name)
        else:
            raise KeyError(f"Invalid cache key: {key}")

    def _get_single(self, name: str, timestep: int) -> torch.Tensor:
        """Get a single cached tensor."""
        key = (name, timestep)
        if key not in self._store and key not in self._evaluated:
            raise KeyError(f"No cached activation for {name} at t={timestep}")

        if key in self._evaluated:
            return self._evaluated[key]

        value = self._store[key]
        if callable(value):
            value = value()
            self._evaluated[key] = value
        return value

    def _get_all(self, name: str) -> torch.Tensor:
        """Get all timesteps stacked."""
        timesteps = sorted(set(t for n, t in self._store.keys() if n == name))
        if not timesteps:
            raise KeyError(f"No cached activations for '{name}'")
        return torch.stack([self._get_single(name, t) for t in timesteps], dim=0)

    def _get_slice(self, name: str, slc: slice) -> torch.Tensor:
        """Get a slice of timesteps."""
        timesteps = sorted(set(t for n, t in self._store.keys() if n == name))
        sliced = timesteps[slc]
        return torch.stack([self._get_single(name, t) for t in sliced], dim=0)

    def __setitem__(
        self,
        key: Tuple[str, int],
        value: Union[torch.Tensor, Callable[[], torch.Tensor]],
    ) -> None:
        """Store an activation.

        Args:
            key: Tuple of (component_name, timestep).
            value: Tensor or lazy callable.
        """
        name, timestep = key
        self._store[key] = value
        self._evaluated.pop(key, None)

    def __contains__(self, key: Tuple[str, int]) -> bool:
        """Check if a key exists in the cache."""
        return key in self._store or key in self._evaluated

    def keys(self) -> Iterable[Tuple[str, int]]:
        """Iterate over all (component, timestep) pairs."""
        return self._store.keys()

    @property
    def component_names(self) -> List[str]:
        """List of unique component names in cache."""
        return sorted(set(n for n, _ in self._store.keys()))

    @property
    def timesteps(self) -> List[int]:
        """List of timesteps with cached activations."""
        return sorted(set(t for _, t in self._store.keys()))

    def get(
        self,
        name: str,
        timestep: int,
        default: Any = None,
    ) -> Optional[torch.Tensor]:
        """Get with default if not found."""
        try:
            return self._get_single(name, timestep)
        except KeyError:
            return default

    def to_device(self, device: torch.device) -> "ActivationCache":
        """Move all tensors to a device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        for key in list(self._store.keys()):
            val = self._store[key]
            if isinstance(val, torch.Tensor):
                self._store[key] = val.to(device)
                self._evaluated.pop(key, None)
        for key in list(self._evaluated.keys()):
            self._evaluated[key] = self._evaluated[key].to(device)
        return self

    def detach(self) -> "ActivationCache":
        """Detach all tensors from the computation graph."""
        for key in list(self._store.keys()):
            val = self._store[key]
            if isinstance(val, torch.Tensor):
                self._store[key] = val.detach()
                self._evaluated.pop(key, None)
        for key in list(self._evaluated.keys()):
            self._evaluated[key] = self._evaluated[key].detach()
        return self

    def filter(self, names: List[str]) -> "ActivationCache":
        """Return a new cache with only specified components.

        Args:
            names: Components to keep.

        Returns:
            New ActivationCache with filtered contents.
        """
        new_cache = ActivationCache()
        for (name, t), val in self._store.items():
            if name in names:
                new_cache._store[(name, t)] = val
        return new_cache

    def surprise(self) -> torch.Tensor:
        """Compute per-timestep KL divergence (surprise) between z_posterior and z_prior.

        Returns:
            Tensor of shape [T] with KL values per timestep.
        """
        try:
            posterior = self["z_posterior"]
            prior = self["z_prior"]
            T = posterior.shape[0]
            kl_vals = []
            for t in range(T):
                p = posterior[t]
                q = prior[t]
                p = p.clamp(min=1e-8)
                q = q.clamp(min=1e-8)
                p = p / p.sum(dim=-1, keepdim=True)
                q = q / q.sum(dim=-1, keepdim=True)
                kl = (p * (p.log() - q.log())).sum(dim=-1)
                kl_vals.append(kl.sum().item())
            return torch.tensor(kl_vals)
        except KeyError:
            raise KeyError("Cache must contain z_posterior and z_prior for surprise()")

    def to_dataframe(self) -> pd.DataFrame:
        """Export cache to a pandas DataFrame.

        Returns:
            DataFrame with columns: component, timestep, shape.
        """
        records = []
        for (name, t), val in self._store.items():
            if isinstance(val, torch.Tensor):
                records.append(
                    {
                        "component": name,
                        "timestep": t,
                        "shape": str(list(val.shape)),
                        "dtype": str(val.dtype),
                    }
                )
        return pd.DataFrame(records)

    def materialize(
        self,
        names: Optional[List[str]] = None,
        timesteps: Optional[List[int]] = None,
    ) -> "ActivationCache":
        """Pre-compute a subset of lazy callables into tensors.

        Args:
            names: Component names to materialize (None = all).
            timesteps: Timesteps to materialize (None = all).

        Returns:
            Self for chaining.
        """
        if names is None:
            names = self.component_names
        if timesteps is None:
            timesteps = self.timesteps

        for name in names:
            for t in timesteps:
                key = (name, t)
                if key in self._store and callable(self._store[key]):
                    self._get_single(name, t)

        return self

    def estimate_memory_gb(self) -> float:
        """Predict RAM usage before materializing all callables.

        Returns:
            Estimated memory in gigabytes.
        """
        total_bytes = 0
        for key, val in self._store.items():
            if key in self._evaluated:
                total_bytes += self._evaluated[key].element_size() * self._evaluated[key].nelement()
            elif isinstance(val, torch.Tensor):
                total_bytes += val.element_size() * val.nelement()
            elif callable(val):
                estimated_size = 4 * 1024 * 1024
                total_bytes += estimated_size
        return total_bytes / (1024**3)
