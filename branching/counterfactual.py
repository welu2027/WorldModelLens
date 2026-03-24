"""CounterfactualAnalyzer — test what-if interventions on latent belief states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from world_model_lens.branching.brancher import ImaginationBrancher
    from world_model_lens.core.latent_trajectory import LatentTrajectory


# ---------------------------------------------------------------------------
# CounterfactualReport
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualReport:
    """Results of a multi-intervention counterfactual analysis."""

    baseline_trajectory: "LatentTrajectory"
    counterfactual_trajectories: Dict[str, "LatentTrajectory"]
    reward_deltas: Dict[str, float]
    kl_divergences: Dict[str, np.ndarray]   # name → (T,)

    # ------------------------------------------------------------------ #

    def most_impactful_intervention(self) -> str:
        """Return intervention name with highest |reward_delta|."""
        return max(self.reward_deltas, key=lambda k: abs(self.reward_deltas[k]))

    def render_html(self) -> str:
        """Return an HTML table summarising each intervention."""
        rows = []
        for name, delta in sorted(
            self.reward_deltas.items(), key=lambda x: -abs(x[1])
        ):
            mean_kl = float(np.mean(self.kl_divergences.get(name, [0])))
            rows.append(
                f"<tr><td>{name}</td>"
                f"<td>{delta:+.4f}</td>"
                f"<td>{mean_kl:.4f}</td></tr>"
            )

        rows_str = "\n".join(rows)
        return f"""<!DOCTYPE html>
<html>
<head><title>Counterfactual Report</title>
<style>
  table {{ border-collapse: collapse; font-family: monospace; font-size: 13px; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; }}
  th {{ background: #f0f0f0; }}
  tr:hover {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h2>Counterfactual Analysis</h2>
<table>
  <thead>
    <tr><th>Intervention</th><th>ΔReward</th><th>Mean KL / L2</th></tr>
  </thead>
  <tbody>
{rows_str}
  </tbody>
</table>
</body>
</html>"""

    def plot_reward_deltas(self, ax=None):
        """Horizontal bar chart of reward deltas."""
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots(figsize=(6, 3))

        names = list(self.reward_deltas.keys())
        deltas = [self.reward_deltas[n] for n in names]
        colors = ["tab:green" if d > 0 else "tab:red" for d in deltas]

        ax.barh(names, deltas, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("ΔReward (vs baseline)")
        ax.set_title("Counterfactual Reward Deltas")
        ax.grid(True, axis="x", alpha=0.3)
        return ax


# ---------------------------------------------------------------------------
# CounterfactualAnalyzer
# ---------------------------------------------------------------------------

class CounterfactualAnalyzer:
    """Test counterfactual z-space interventions on latent belief states.

    Usage
    -----
    >>> brancher = ImaginationBrancher(wm)
    >>> analyzer = CounterfactualAnalyzer(brancher)
    >>> report = analyzer.analyze(trajectory, fork_timestep=10, action_seq)
    >>> print(report.most_impactful_intervention())
    """

    # Default intervention functions (applied to z at the fork timestep)
    _DEFAULT_INTERVENTIONS: Dict[str, Callable[[Tensor], Tensor]] = {
        "zero_z":   lambda z: torch.zeros_like(z),
        "negate_z": lambda z: -z,
        "random_z": lambda z: torch.randn_like(z),
        "scale_2x": lambda z: z * 2.0,
        "scale_0x": lambda z: z * 0.0,
    }

    def __init__(self, brancher: "ImaginationBrancher") -> None:
        self._brancher = brancher

    # ------------------------------------------------------------------ #

    def analyze(
        self,
        trajectory: "LatentTrajectory",
        fork_timestep: int,
        action_sequence: Tensor,
        horizon: int = 20,
        interventions: Optional[Dict[str, Callable[[Tensor], Tensor]]] = None,
    ) -> CounterfactualReport:
        """Run multiple z-space interventions and compare to baseline.

        Parameters
        ----------
        interventions:
            Dict mapping intervention name → z_patch_fn (Tensor → Tensor).
            If None, uses the default interventions: zero_z, negate_z, random_z,
            scale_2x, scale_0x.
        """
        if interventions is None:
            interventions = self._DEFAULT_INTERVENTIONS

        # Baseline (unmodified)
        brancher = self._brancher
        baseline_cmp = brancher.belief_manipulation_fork(
            trajectory, fork_timestep, action_sequence, horizon=horizon,
        )
        baseline = baseline_cmp.baseline

        # Compute baseline total reward
        r_base = sum(
            s.reward_pred for s in baseline.states if s.reward_pred is not None
        )

        counterfactuals: Dict[str, "LatentTrajectory"] = {}
        reward_deltas: Dict[str, float] = {}
        kl_divergences: Dict[str, np.ndarray] = {}

        for name, z_fn in interventions.items():
            comparison = brancher.belief_manipulation_fork(
                trajectory, fork_timestep, action_sequence,
                horizon=horizon, z_patch_fn=z_fn,
            )
            cf_traj = comparison.modified
            r_cf = sum(
                s.reward_pred for s in cf_traj.states if s.reward_pred is not None
            )
            counterfactuals[name] = cf_traj
            reward_deltas[name] = float(r_cf - r_base)  # type: ignore[operator]
            kl_divergences[name] = comparison.kl_divergence

        return CounterfactualReport(
            baseline_trajectory=baseline,
            counterfactual_trajectories=counterfactuals,
            reward_deltas=reward_deltas,
            kl_divergences=kl_divergences,
        )
