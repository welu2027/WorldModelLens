"""Counterfactual engine: interventions, divergence, and branch comparison.

Demonstrates `CounterfactualEngine` from `world_model_lens.causal.counterfactual`:
baseline rollout, latent ablation, trajectory comparison, divergence tracing,
branch tree, and side-by-side intervention metrics.
"""

import pathlib

import torch
import numpy as np
import matplotlib.pyplot as plt

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter
from world_model_lens.causal import (
    CounterfactualEngine,
    Intervention,
    rollout_comparison,
)
from world_model_lens.visualization import CacheSignalPlotter, InterventionVisualizer

OUTPUT_DIR = pathlib.Path("assets/examples")


def main():
    print("=" * 60)
    print("World Model Lens — CounterfactualEngine walkthrough")
    print("=" * 60)

    cfg = WorldModelConfig(d_h=128, n_cat=16, n_cls=16, d_action=4, d_obs=12288)
    adapter = DreamerV3Adapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)

    T = 15
    obs_seq = torch.randn(T, 3, 64, 64)
    action_seq = torch.randn(T, cfg.d_action)

    engine = CounterfactualEngine(wm)

    # --- Baseline trajectory (no hooks)
    baseline_traj, cache = wm.run_with_cache(obs_seq, action_seq)
    print(f"\nBaseline trajectory: {len(baseline_traj.states)} states")
    print(f"  State dim: {baseline_traj.states[0].state.shape}")

    # --- Single intervention: zero selected posterior dimensions at t=5
    ablate_z = Intervention(
        target_timestep=5,
        target_type="dimension",
        target_indices=[0, 1, 2],
        description="Ablate z dims 0–2 at t=5",
    )
    cf_traj = engine.intervene(
        observations=obs_seq,
        intervention=ablate_z,
        actions=action_seq,
    )
    print("\nCounterfactual (dimension ablation on z_posterior): done.")

    # --- Compare trajectories
    metrics = rollout_comparison(baseline_traj, cf_traj)
    print("\nrollout_comparison (baseline vs counterfactual):")
    for k, v in metrics.to_dict().items():
        print(f"  {k}: {v:.6f}")

    # --- Cumulative divergence over time
    div_curve = engine.trace_divergence(baseline_traj, cf_traj)
    sample_ts = [0, T // 2, T - 1]  # 0, 7, 14
    print("\ntrace_divergence (cumulative MSE-style drift) at sample timesteps:")
    for t in sample_ts:
        if t in div_curve:
            print(f"  t={t}: cumulative = {div_curve[t]:.6f}")

    # --- Branch tree: several interventions at the same fork
    interventions = [
        Intervention(
            target_timestep=5,
            target_type="dimension",
            target_indices=[0],
            description="Ablate dim 0",
        ),
        Intervention(
            target_timestep=5,
            target_type="dimension",
            target_indices=[1],
            description="Ablate dim 1",
        ),
        Intervention(
            target_timestep=5,
            target_type="action",
            target_indices=None,
            description="Zero action at t=5",
        ),
    ]
    tree = engine.build_branch_tree(
        observations=obs_seq,
        interventions=interventions,
        base_actions=action_seq,
    )
    print(f"\nBranchTree: {len(tree.branches)} branches from baseline")
    for i, br in enumerate(tree.branches):
        print(
            f"  [{i}] fork={br.fork_point} "
            f"divergence(trajectory_distance)={br.divergence:.6f} "
            f"— {br.intervention.description}"
        )

    # --- Tabular comparison of multiple interventions
    compared = engine.compare_interventions(
        observations=obs_seq,
        interventions=interventions,
        base_actions=action_seq,
        target_metric="reward_pred",
    )
    print("\ncompare_interventions (reward_pred outcome + divergence metrics):")
    for idx, row in compared.items():
        print(f"  intervention {idx}:")
        for key in (
            "intervention_description",
            "target_timestep",
            "baseline_outcome",
            "counterfactual_outcome",
            "outcome_delta",
            "l2_distance",
            "trajectory_distance",
        ):
            if key in row:
                print(f"    {key}: {row[key]}")

    # --- Build visualization dashboard ---
    print("\nBuilding visualization dashboard...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("WorldModelLens - Causal Engine Dashboard", fontsize=14)

    # 1. Cumulative divergence curve 
    ax = axes[0]
    ts = sorted(div_curve.keys())
    divs = [div_curve[t] for t in ts]
    ax.plot(ts, divs, marker="o", color="tomato")
    ax.axvline(5, color="gray", linestyle="--", linewidth=1.5, label="Intervention at t=5")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Divergence")
    ax.set_title("Baseline vs Counterfactual\nCumulative Divergence")
    ax.legend(fontsize=8)

    # 2. Branch divergence bar chart 
    ax = axes[1]
    branch_labels = [br.intervention.description for br in tree.branches]
    branch_divs = [br.divergence for br in tree.branches]
    colors = ["steelblue", "darkorange", "mediumseagreen"]
    bars = ax.bar(range(len(branch_labels)), branch_divs, color=colors[: len(branch_labels)])
    ax.set_xticks(range(len(branch_labels)))
    ax.set_xticklabels(branch_labels, fontsize=8, rotation=10)
    ax.set_ylabel("Trajectory Distance")
    ax.set_title("Divergence per Intervention Branch")
    for bar, val in zip(bars, branch_divs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 3. Outcome delta per intervention 
    ax = axes[2]
    iv = InterventionVisualizer(wm)
    h_signal = CacheSignalPlotter.plot_cache_signal(cache, "h")
    ax.plot(h_signal["timesteps"], h_signal["norms"], marker="o", color="steelblue")
    ax.axvline(5, color="red", linestyle="--", linewidth=1.5, label="Intervention at t=5")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("||h_t||")
    ax.set_title("Baseline Hidden State Norms")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "causal_engine_dashboard.png", dpi=80, bbox_inches="tight")
    print("    Saved causal_engine_dashboard.png")
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
