"""Example 04: Imagination Branching - Fork and compare imagined futures.

This example demonstrates:
1. Forking imagination from a real trajectory
2. Running multiple imagined branches
3. Comparing reward predictions
4. Measuring divergence
"""

import torch
import numpy as np

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.branching.brancher import ImaginationBrancher


def main():
    print("=" * 60)
    print("World Model Lens - Imagination Branching Example")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, cfg=cfg)
    brancher = ImaginationBrancher(wm)

    print("\n[1] Collecting real trajectory...")

    obs_seq = torch.randn(30, 3, 64, 64)
    action_seq = torch.randn(30, cfg.d_action)

    real_traj, cache = wm.run_with_cache(obs_seq, action_seq)
    print(f"    Real trajectory: {real_traj.length} steps")

    print("\n[2] Finding surprise peak for fork point...")

    kl_vals = [s.kl.item() for s in real_traj.states]
    fork_at = np.argmax(kl_vals[5:]) + 5
    print(f"    Surprise peak at t={fork_at} (KL={kl_vals[fork_at]:.3f})")

    print("\n[3] Creating 5 imagined branches...")

    action_sequences = [torch.randn(20, cfg.d_action) for _ in range(5)]

    branches = brancher.fork(
        real_traj=real_traj,
        fork_at=fork_at,
        action_sequences=action_sequences,
        horizon=20,
    )

    print(f"    Created {len(branches.branches)} branches")

    print("\n[4] Comparing reward predictions across branches:")

    reward_df = branches.compare_reward_predictions()
    print(reward_df.to_string(index=False))

    best_branch_idx = branches.best_branch(metric="total_reward")
    print(f"\n    Best branch: #{best_branch_idx}")

    print("\n[5] Computing divergence over time...")

    divergence = branches.latent_divergence_over_time(metric="cosine")
    print(f"    Mean divergence: {divergence.mean():.4f}")
    print(f"    Max divergence: {divergence.max():.4f}")

    print("\n" + "=" * 60)
    print("Branching complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
