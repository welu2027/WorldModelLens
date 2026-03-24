"""Backend adapters for world_model_lens.

Exports all built-in :class:`~world_model_lens.backends.base.WorldModelAdapter`
implementations, the :class:`~world_model_lens.backends.registry.BackendRegistry`
for dynamic adapter lookup, and a module-level ``BACKEND_REGISTRY`` dict for
backward compatibility.

Built-in Adapters
------------------
- ``"dreamerv3"`` → :class:`DreamerV3Adapter`
- ``"dreamerv2"`` → :class:`DreamerV2Adapter`
- ``"iris"`` → :class:`IRISAdapter`
- ``"tdmpc2"`` → :class:`TDMPC2Adapter`

Global Registry
---------------
The module-level :data:`REGISTRY` instance provides typed access to all registered
backends::

    from world_model_lens.backends import REGISTRY
    adapter_cls = REGISTRY["dreamerv3"]

For backward compatibility, :data:`BACKEND_REGISTRY` is a dict mapping strings
to adapter classes (same content as REGISTRY._registry).

Custom Backends
---------------
Register custom adapters via the decorator or manually::

    from world_model_lens.backends import REGISTRY

    @REGISTRY.register_decorator("my_model")
    class MyAdapter(WorldModelAdapter):
        ...

    # Or manually
    REGISTRY.register("my_model", MyAdapter)

Examples
--------
List all available backends::

    from world_model_lens.backends import REGISTRY
    print(REGISTRY.list_backends())
    # Output: ['dreamerv2', 'dreamerv3', 'iris', 'tdmpc2']

Instantiate an adapter::

    from world_model_lens.backends import REGISTRY
    from world_model_lens.core import WorldModelConfig

    cfg = WorldModelConfig(d_h=256, d_action=6, d_obs=12288)
    adapter_cls = REGISTRY["dreamerv3"]
    adapter = adapter_cls(cfg=cfg)
"""

from world_model_lens.backends.base import WorldModelAdapter
from world_model_lens.backends.custom_adapter_template import CustomWorldModelAdapter
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.backends.dreamerv2 import DreamerV2Adapter
from world_model_lens.backends.iris import IRISAdapter
from world_model_lens.backends.registry import REGISTRY, BackendRegistry
from world_model_lens.backends.tdmpc2 import TDMPC2Adapter

# Pre-populate the global registry with built-in adapters
REGISTRY.register("dreamerv3", DreamerV3Adapter)
REGISTRY.register("dreamerv2", DreamerV2Adapter)
REGISTRY.register("iris", IRISAdapter)
REGISTRY.register("tdmpc2", TDMPC2Adapter)

# Backward-compatibility dict: maps backend name → adapter class
# This is the same as REGISTRY._registry but exposed as a plain dict
BACKEND_REGISTRY: dict = {
    "dreamerv3": DreamerV3Adapter,
    "dreamerv2": DreamerV2Adapter,
    "iris": IRISAdapter,
    "tdmpc2": TDMPC2Adapter,
}

__all__ = [
    # Base class
    "WorldModelAdapter",
    # Built-in adapters
    "DreamerV3Adapter",
    "DreamerV2Adapter",
    "IRISAdapter",
    "TDMPC2Adapter",
    # Template
    "CustomWorldModelAdapter",
    # Registry
    "REGISTRY",
    "BackendRegistry",
    # Backward compatibility
    "BACKEND_REGISTRY",
]
