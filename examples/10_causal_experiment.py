import torch
import torch.nn as nn

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.causal import CounterfactualEngine, Intervention


def main():
    print("=" * 60)
    print("World Model Lens - Activation Patching Example")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)

    obs_seq = torch.randn(15, 3, 64, 64)
    action_seq = torch.randn(15, cfg.d_action)

    engine = CounterfactualEngine(wm)
    engine.intervene(
        observations=obs_seq,
        intervention=Intervention(
            target_timestep=5,
            target_type="action",
            intervention_fn=None,
        ),
    )
    print("intervened something here!")


if __name__ == "__main__":
    main()
