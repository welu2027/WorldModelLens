"""BackendRegistry — a typed registry for adapter classes.

This module provides a simple typed registry (dict-like) that maps string keys
to :class:`~world_model_lens.backends.base.WorldModelAdapter` classes. It enables
dynamic lookup and registration of backends without hardcoding imports.

Usage
-----
Register a new adapter class::

    from world_model_lens.backends.registry import REGISTRY

    # Manual registration
    REGISTRY.register("my_model", MyAdapter)

    # Or use the decorator
    @REGISTRY.register_decorator("my_model")
    class MyAdapter(WorldModelAdapter):
        ...

    # Lookup
    adapter_cls = REGISTRY.get("my_model")
    adapter = adapter_cls(cfg)

    # Check if registered
    if "my_model" in REGISTRY:
        print(REGISTRY.list_backends())
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Type

from world_model_lens.backends.base import WorldModelAdapter


class BackendRegistry:
    """A simple dict-like registry for backend adapter classes.

    Provides typed access to :class:`~world_model_lens.backends.base.WorldModelAdapter`
    implementations. Useful for plugin architectures, CLI tools, and dynamic
    adapter instantiation.

    Examples
    --------
    >>> from world_model_lens.backends.registry import REGISTRY
    >>> "dreamerv3" in REGISTRY
    True
    >>> adapter_cls = REGISTRY.get("dreamerv3")
    >>> adapter_cls.__name__
    'DreamerV3Adapter'

    >>> REGISTRY.register("custom", MyAdapter)
    >>> REGISTRY["custom"] is MyAdapter
    True

    >>> REGISTRY.list_backends()
    ['custom', 'dreamerv2', 'dreamerv3', 'iris', 'tdmpc2']
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._registry: Dict[str, Type[WorldModelAdapter]] = {}

    def register(self, name: str, cls: Type[WorldModelAdapter]) -> None:
        """Register an adapter class under a name.

        Parameters
        ----------
        name : str
            Backend identifier (e.g., "dreamerv3", "tdmpc2", "custom_rssm").
        cls : type[WorldModelAdapter]
            Adapter class to register. Must be a subclass of
            :class:`~world_model_lens.backends.base.WorldModelAdapter`.

        Raises
        ------
        TypeError
            If *cls* is not a subclass of WorldModelAdapter.

        Examples
        --------
        >>> registry = BackendRegistry()
        >>> registry.register("my_model", MyAdapter)
        >>> registry.get("my_model") is MyAdapter
        True
        """
        if not issubclass(cls, WorldModelAdapter):
            raise TypeError(
                f"Cannot register {cls.__name__}: must be a subclass of "
                f"WorldModelAdapter, got {type(cls)}."
            )
        self._registry[name] = cls

    def get(self, name: str) -> Type[WorldModelAdapter]:
        """Get an adapter class by name.

        Parameters
        ----------
        name : str
            Backend identifier.

        Returns
        -------
        type[WorldModelAdapter]
            The registered adapter class.

        Raises
        ------
        KeyError
            If *name* is not registered. Includes a helpful message listing
            all available backends.

        Examples
        --------
        >>> adapter_cls = REGISTRY.get("dreamerv3")
        >>> adapter = adapter_cls(cfg)
        """
        try:
            return self._registry[name]
        except KeyError:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"Backend '{name}' not found in registry. "
                f"Available backends: {available}."
            ) from None

    def __contains__(self, name: str) -> bool:
        """Check if a backend is registered.

        Parameters
        ----------
        name : str
            Backend identifier.

        Returns
        -------
        bool

        Examples
        --------
        >>> "dreamerv3" in REGISTRY
        True
        >>> "nonexistent" in REGISTRY
        False
        """
        return name in self._registry

    def __getitem__(self, name: str) -> Type[WorldModelAdapter]:
        """Get adapter class via indexing (equivalent to :meth:`get`).

        Parameters
        ----------
        name : str
            Backend identifier.

        Returns
        -------
        type[WorldModelAdapter]

        Raises
        ------
        KeyError
            If *name* is not registered.

        Examples
        --------
        >>> adapter_cls = REGISTRY["dreamerv3"]
        """
        return self.get(name)

    def __setitem__(self, name: str, cls: Type[WorldModelAdapter]) -> None:
        """Register adapter class via assignment (equivalent to :meth:`register`).

        Parameters
        ----------
        name : str
            Backend identifier.
        cls : type[WorldModelAdapter]
            Adapter class.

        Examples
        --------
        >>> REGISTRY["my_model"] = MyAdapter
        """
        self.register(name, cls)

    def keys(self) -> List[str]:
        """Return sorted list of all registered backend names.

        Returns
        -------
        list of str

        Examples
        --------
        >>> REGISTRY.keys()
        ['dreamerv2', 'dreamerv3', 'iris', 'tdmpc2']
        """
        return sorted(self._registry.keys())

    def items(self) -> Iterator[Tuple[str, Type[WorldModelAdapter]]]:
        """Iterate over (name, class) pairs of all registered backends.

        Yields
        ------
        (str, type[WorldModelAdapter])

        Examples
        --------
        >>> for name, cls in REGISTRY.items():
        ...     print(f"{name}: {cls.__name__}")
        dreamerv2: DreamerV2Adapter
        dreamerv3: DreamerV3Adapter
        iris: IRISAdapter
        tdmpc2: TDMPC2Adapter
        """
        for name in self.keys():
            yield name, self._registry[name]

    def register_decorator(self, name: str):
        """Decorator to register a class under a name.

        Parameters
        ----------
        name : str
            Backend identifier.

        Returns
        -------
        callable
            Decorator function.

        Examples
        --------
        >>> @REGISTRY.register_decorator("my_model")
        ... class MyAdapter(WorldModelAdapter):
        ...     pass
        >>> "my_model" in REGISTRY
        True
        """

        def decorator(cls: Type[WorldModelAdapter]) -> Type[WorldModelAdapter]:
            self.register(name, cls)
            return cls

        return decorator

    def list_backends(self) -> List[str]:
        """Return a sorted list of all registered backend names.

        Alias for :meth:`keys` for convenience.

        Returns
        -------
        list of str
            Sorted backend identifiers.

        Examples
        --------
        >>> REGISTRY.list_backends()
        ['dreamerv2', 'dreamerv3', 'iris', 'tdmpc2']
        """
        return self.keys()

    def __repr__(self) -> str:
        """Return a string representation of the registry.

        Returns
        -------
        str

        Examples
        --------
        >>> repr(REGISTRY)
        'BackendRegistry(4 backends: dreamerv2, dreamerv3, iris, tdmpc2)'
        """
        count = len(self._registry)
        backends = ", ".join(self.keys())
        return f"BackendRegistry({count} backends: {backends})"


# Module-level singleton registry
REGISTRY: BackendRegistry = BackendRegistry()
"""
Global registry instance used throughout world_model_lens.

All built-in adapters are pre-registered. Users can add custom adapters via
:meth:`register` or the :meth:`register_decorator`.

Examples
--------
>>> from world_model_lens.backends.registry import REGISTRY
>>> REGISTRY.list_backends()
['dreamerv2', 'dreamerv3', 'iris', 'tdmpc2']
>>> adapter_cls = REGISTRY["dreamerv3"]
>>> REGISTRY.register("my_model", MyAdapter)
"""

__all__ = [
    "BackendRegistry",
    "REGISTRY",
]
