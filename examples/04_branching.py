"""Example 04: Imagination Branching - Fork and compare imagined futures.

This example demonstrates:
1. Forking imagination from a real trajectory
2. Running multiple imagined branches
3. Comparing branch states
4. Measuring divergence
"""

import numpy as np
import torch

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.branching import ImaginationBrancher


def main():
    print("=" * 60)
    print("World Model Lens - Imagination Branching Example")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)

    print("\n[1] Collecting real trajectory...")

    obs_seq = torch.randn(30, 3, 64, 64)
    action_seq = torch.randn(30, cfg.d_action)

    real_traj, cache = wm.run_with_cache(obs_seq, action_seq)
    print(f"    Real trajectory: {real_traj.length} steps")

    print("\n[2] Finding surprise peak for fork point...")

    # Get KL divergence from cache for fork point selection
    kl_vals = []
    for t in range(real_traj.length):
        kl = cache.get("kl", t)
        if kl is not None:
            kl_vals.append(kl.sum().item())
        else:
            kl_vals.append(0.0)
    kl_vals = np.array(kl_vals)
    fork_at = int(np.argmax(kl_vals[5:])) + 5
    print(f"    Surprise peak at t={fork_at} (KL={kl_vals[fork_at]:.3f})")

    print("\n[3] Forking 5 imagined branches using ImaginationBrancher")

    brancher = ImaginationBrancher(wm)
    action_sequences = [torch.randn(20, cfg.d_action) for _ in range(5)]
    branch_collection = brancher.fork(
        real_traj=real_traj,
        fork_at=fork_at,
        action_sequences=action_sequences,
        horizon=20,
    )

    print(f"    Created {len(branch_collection.branches)} branches")

    print("\n[4] Comparing branch trajectories")

    for i, branch in enumerate(branch_collection.branches):
        states_tensor = torch.stack([s.state for s in branch.states])
        print(f"    Branch {i}: {branch.length} steps, state norm={states_tensor.norm():.3f}")

    print("\n[5] Computing divergence between branches")

    ref_states = torch.stack([s.state for s in branch_collection.branches[0].states])
    for i, branch in enumerate(branch_collection.branches[1:], 1):
        branch_states = torch.stack([s.state for s in branch.states])
        min_len = min(len(ref_states), len(branch_states))
        divergence = (ref_states[:min_len] - branch_states[:min_len]).norm(dim=-1)
        print(f"    Branch 0 vs {i}: mean L2={divergence.mean():.4f}, max L2={divergence.max():.4f}")

    print("\n" + "=" * 60)
    print("Branching complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
