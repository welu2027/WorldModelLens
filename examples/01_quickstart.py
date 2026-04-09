"""Example 01: Quickstart - Load model, run cache, train probe.

This example demonstrates the basic workflow:
1. Create a world model adapter
2. Wrap it in HookedWorldModel
3. Run forward pass with caching
4. Train a linear probe on cached activations
"""

import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OUTPUT_DIR = pathlib.Path("assets/examples")

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.visualization import (
    CacheSignalPlotter,
    InterventionVisualizer,
    PredictionVisualizer,
)


def main():
    print("=" * 60)
    print("World Model Lens - Quickstart Example")
    print("=" * 60)

    cfg = WorldModelConfig(
        d_h=256,
        n_cat=32,
        n_cls=32,
        d_action=4,
        d_obs=12288,
    )
    print(f"\n[1] Config created: d_h={cfg.d_h}, n_cat={cfg.n_cat}, n_cls={cfg.n_cls}")

    adapter = DreamerV3Adapter(cfg)
    print("\n[2] DreamerV3Adapter initialized")

    wm = HookedWorldModel(adapter=adapter, config=cfg, name="quickstart")
    print("\n[3] HookedWorldModel wrapper created")

    T, C, H, W = 10, 3, 64, 64
    obs_seq = torch.randn(T, C, H, W)
    action_seq = torch.randn(T, cfg.d_action)
    print(f"\n[4] Created fake data: obs={obs_seq.shape}, actions={action_seq.shape}")

    traj, cache = wm.run_with_cache(obs_seq, action_seq)
    print(f"\n[5] Forward pass complete!")
    print(f"    Trajectory length: {traj.length}")
    print(f"    Cache keys: {cache.component_names}")

    h_t = cache["h", 0]
    z_posterior = cache["z_posterior", 0]
    print(f"\n[6] Sample activations:")
    print(f"    h_t shape: {h_t.shape}")
    print(f"    z_posterior shape: {z_posterior.shape}")

    imagined = wm.imagine(start_state=traj.states[5], horizon=20)
    print(f"\n[7] Imagination complete: {imagined.length} steps")

    print("\n[8] Building visualization dashboard...")

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("WorldModelLens Quickstart Dashboard", fontsize=14)

    # 1. Hidden-state PCA
    n_cat, n_cls = cfg.n_cat, cfg.n_cls

    h_tensors = torch.stack([cache["h", t] for t in range(T)])  # [T, d_h]
    h_centered = h_tensors - h_tensors.mean(dim=0)
    _, _, Vt = torch.pca_lowrank(h_centered, q=2)
    pca_proj = (h_centered @ Vt).detach().numpy()  # [T, 2]

    ax1 = fig.add_subplot(2, 3, 1)
    sc = ax1.scatter(pca_proj[:, 0], pca_proj[:, 1], c=np.arange(T), cmap="viridis")
    for i in range(T):
        ax1.annotate(str(i), (pca_proj[i, 0], pca_proj[i, 1]), fontsize=7)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("Hidden-State PCA")

    # 2. Surprise Timeline (computed manually: DreamerV3 stores z as 1D so
    # the library's KL cache key is never written; derive KL from z_posterior/z_prior)
    kl_vals_timeline = []
    for t in range(T):
        post = cache["z_posterior", t].reshape(n_cat, n_cls)
        prior = cache["z_prior", t].reshape(n_cat, n_cls)
        p = post.softmax(dim=-1).clamp(min=1e-8)
        q = prior.softmax(dim=-1).clamp(min=1e-8)
        kl_vals_timeline.append((p * (p.log() - q.log())).sum().item())
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(range(T), kl_vals_timeline, marker="o")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("KL")
    ax2.set_title("Surprise Timeline")

    # 3. Per-Category Surprise Heatmap
    kl_per_cat = []
    for t in range(T):
        post = cache["z_posterior", t].reshape(n_cat, n_cls)
        prior = cache["z_prior", t].reshape(n_cat, n_cls)
        p = post.softmax(dim=-1).clamp(min=1e-8)
        q = prior.softmax(dim=-1).clamp(min=1e-8)
        kl_cat = (p * (p.log() - q.log())).sum(dim=-1).detach().numpy()
        kl_per_cat.append(kl_cat)
    heatmap_matrix = np.stack(kl_per_cat, axis=0)  # [T, n_cat]

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(heatmap_matrix.T, aspect="auto", origin="lower", cmap="hot")
    ax3.set_xlabel("Timestep")
    ax3.set_ylabel("Latent Category")
    ax3.set_title("Per-Category Surprise Heatmap")

    # 4. Activation / Head Timelines
    h_signal = CacheSignalPlotter.plot_cache_signal(cache, "h")
    reward_data = CacheSignalPlotter.plot_reward_timeline(traj)
    value_data = CacheSignalPlotter.plot_value_timeline(traj)

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(h_signal["timesteps"], h_signal["norms"], label="||h_t||")
    if len(reward_data["predicted"]) > 0:
        ax4.plot(reward_data["timesteps"], reward_data["predicted"], label="Reward Pred")
    if len(value_data["value_pred"]) > 0:
        ax4.plot(value_data["timesteps"], value_data["value_pred"], label="Value Pred")
    ax4.set_xlabel("Timestep")
    ax4.legend(fontsize=7)
    ax4.set_title("Activation / Head Timelines")

    # 5. State Trajectory in 3D
    pred_viz = PredictionVisualizer(wm)
    traj_3d = pred_viz.state_trajectory_3d(traj, components=[0, 1, 2])
    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    if len(traj_3d["x"]) > 1:
        ax5.plot(traj_3d["x"], traj_3d["y"], traj_3d["z"], marker="o", markersize=3)
    ax5.set_xlabel("Dim 0")
    ax5.set_ylabel("Dim 1")
    ax5.set_zlabel("Dim 2")
    ax5.set_title("State Trajectory (3D)")

    # 6. Counterfactual Divergence
    iv = InterventionVisualizer(wm)
    heatmap = iv.intervention_heatmap(traj, imagined)
    ax6 = fig.add_subplot(2, 3, 6)
    divergence_curve = iv.divergence_curve(traj, imagined)
    ts = sorted(divergence_curve.keys())
    divs = [divergence_curve[t] for t in ts]
    cf_matrix = np.zeros((heatmap.shape[1], len(ts)))
    for i, t in enumerate(ts):
        if t < heatmap.shape[0]:
            cf_matrix[:, i] = heatmap[t, : cf_matrix.shape[0]]
    ax6.imshow(cf_matrix, aspect="auto", cmap="RdBu_r", origin="lower")
    ax6.plot(
        range(len(ts)),
        [d * cf_matrix.shape[0] for d in divs],
        color="navy",
        label="Divergence",
    )
    ax6.legend(fontsize=7)
    ax6.set_xlabel("Timestep")
    ax6.set_ylabel("Dims / Divergence")
    ax6.set_title("Counterfactual Divergence")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "quickstart_dashboard.png", dpi=120, bbox_inches="tight")
    print("    Saved quickstart_dashboard.png")
    plt.show()

    print("\n" + "=" * 60)
    print("Quickstart complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
