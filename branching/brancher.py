"""ImaginationBrancher — fork imagined rollouts at any point in a trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from world_model_lens.core.activation_cache import ActivationCache
    from world_model_lens.core.latent_trajectory import LatentTrajectory
    from world_model_lens.hooked_world_model import HookedWorldModel
    import pandas as pd


# ---------------------------------------------------------------------------
# BranchCollection
# ---------------------------------------------------------------------------


@dataclass
class BranchCollection:
    """A collection of imagined trajectory branches forked at a common timestep."""

    branches: List["LatentTrajectory"]
    branch_labels: List[str]
    fork_timestep: int

    # ------------------------------------------------------------------ #

    def compare_reward_predictions(self) -> "pd.DataFrame":
        """Return DataFrame: columns = branch names, rows = timestep, values = reward."""
        import pandas as pd

        data: Dict[str, List[float]] = {}
        for label, traj in zip(self.branch_labels, self.branches):
            rewards = [
                float(s.reward_pred) if s.reward_pred is not None else float("nan")
                for s in traj.states
            ]
            data[label] = rewards
        return pd.DataFrame(data)

    def latent_divergence_over_time(self) -> np.ndarray:
        """Return (n_branches, T) L2 distance of each branch's h from branch-0."""
        if not self.branches:
            return np.zeros((0, 0))

        ref = self.branches[0]
        T = len(ref.states)
        n = len(self.branches)
        out = np.zeros((n, T))

        for bi, traj in enumerate(self.branches):
            for t in range(min(T, len(traj.states))):
                h_ref = ref.states[t].h
                h_cmp = traj.states[t].h
                if h_ref is not None and h_cmp is not None:
                    diff = h_cmp.detach() - h_ref.detach()
                    out[bi, t] = float(diff.norm())
        return out

    def best_branch(self, criterion: str = "total_reward") -> "LatentTrajectory":
        """Return branch with highest total reward (or first branch if tie)."""
        if criterion == "total_reward":
            scores = []
            for traj in self.branches:
                r = sum(
                    float(s.reward_pred) if s.reward_pred is not None else 0.0 for s in traj.states
                )
                scores.append(r)
            return self.branches[int(np.argmax(scores))]
        raise ValueError(f"Unknown criterion: {criterion!r}")

    def plot_decoded_futures(
        self,
        decoder_fn: Optional[Callable] = None,
        n_frames: int = 5,
        ax=None,
    ):
        """Grid of reward-prediction curves (decoder_fn optional for images)."""
        import matplotlib.pyplot as plt

        n_branches = len(self.branches)
        fig, axes = plt.subplots(1, n_branches, figsize=(3 * n_branches, 3))
        if n_branches == 1:
            axes = [axes]

        for ax_i, (label, traj) in enumerate(zip(self.branch_labels, self.branches)):
            rewards = [
                float(s.reward_pred) if s.reward_pred is not None else 0.0 for s in traj.states
            ]
            axes[ax_i].plot(rewards, marker="o", markersize=3)
            axes[ax_i].set_title(label, fontsize=9)
            axes[ax_i].set_xlabel("Step")
            if ax_i == 0:
                axes[ax_i].set_ylabel("Reward pred")

        fig.suptitle(f"Imagined Futures (fork t={self.fork_timestep})", fontsize=10)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# UncertaintyResult
# ---------------------------------------------------------------------------


@dataclass
class UncertaintyResult:
    """Uncertainty estimate from an ensemble of imagined rollouts."""

    branches: List["LatentTrajectory"]
    mean_rewards: np.ndarray  # (T,)
    epistemic_uncertainty: np.ndarray  # (T,) std of h across branches
    reward_uncertainty: np.ndarray  # (T,) std of reward across branches

    def plot_uncertainty_bands(self, ax=None):
        """Plot mean ± 2*std reward bands."""
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots(figsize=(8, 4))

        T = len(self.mean_rewards)
        ts = np.arange(T)
        ax.fill_between(
            ts,
            self.mean_rewards - 2 * self.reward_uncertainty,
            self.mean_rewards + 2 * self.reward_uncertainty,
            alpha=0.3,
            label="±2σ reward",
        )
        ax.plot(ts, self.mean_rewards, lw=2, label="mean reward")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Predicted reward")
        ax.set_title("Imagination Uncertainty")
        ax.legend()
        return ax


# ---------------------------------------------------------------------------
# BehaviorComparison
# ---------------------------------------------------------------------------


@dataclass
class BehaviorComparison:
    """Comparison between a baseline and a modified imagined trajectory."""

    baseline: "LatentTrajectory"
    modified: "LatentTrajectory"
    reward_delta: float
    kl_divergence: np.ndarray  # (T,) per-timestep KL or L2 distance
    description: str


# ---------------------------------------------------------------------------
# ImaginationBrancher
# ---------------------------------------------------------------------------


class ImaginationBrancher:
    """Fork imagined rollouts at any point in a trajectory.

    All imagined rollouts use the world model's *prior* dynamics
    (``dynamics_step``) without further observations — i.e. open-loop
    imagination starting from the forked latent state.
    """

    def __init__(self, wm: "HookedWorldModel") -> None:
        self._wm = wm

    # ------------------------------------------------------------------ #

    def _imagine_from_state(
        self,
        h: Tensor,
        z: Tensor,
        action_seq: Tensor,
        horizon: int,
    ) -> "LatentTrajectory":
        """Imagine forward from (h, z) using action_seq for *horizon* steps."""
        from world_model_lens.core.latent_state import LatentState
        from world_model_lens.core.latent_trajectory import LatentTrajectory

        wm = self._wm
        adapter = wm.adapter

        states: List[LatentState] = []
        h_t = h.detach().clone()
        z_t = z.detach().clone()

        T_act = action_seq.shape[0] if action_seq.ndim > 1 else 1
        for t in range(min(horizon, T_act if action_seq.ndim > 1 else horizon)):
            a_t = action_seq[t] if action_seq.ndim > 1 else action_seq

            # Optional heads
            try:
                r = float(adapter.reward_pred(h_t, z_t).detach())
            except NotImplementedError:
                r = None
            try:
                c = float(adapter.cont_pred(h_t, z_t).detach())
            except NotImplementedError:
                c = None
            try:
                v = float(adapter.value(h_t, z_t).detach())
            except NotImplementedError:
                v = None

            state = LatentState(
                timestep=t,
                h=h_t.detach().clone(),
                z=z_t.detach().clone(),
                reward_pred=r,
                value_pred=v,
                cont_pred=c,
            )
            states.append(state)

            h_t, z_prior_logits = adapter.dynamics_step(h_t, z_t, a_t)
            z_t = torch.softmax(z_prior_logits.float(), dim=-1)

        return LatentTrajectory(states=states)

    # ------------------------------------------------------------------ #

    def fork(
        self,
        trajectory: "LatentTrajectory",
        fork_timestep: int,
        action_sequences: List[Tensor],
        horizon: int = 20,
        branch_labels: Optional[List[str]] = None,
    ) -> BranchCollection:
        """Fork at fork_timestep and imagine forward with each action sequence."""
        if fork_timestep >= len(trajectory.states):
            raise ValueError(
                f"fork_timestep={fork_timestep} >= trajectory length {len(trajectory.states)}"
            )

        state = trajectory.states[fork_timestep]
        h0 = state.h
        z0 = state.z

        if branch_labels is None:
            branch_labels = [f"branch_{i}" for i in range(len(action_sequences))]

        branches = [
            self._imagine_from_state(h0, z0, act_seq, horizon) for act_seq in action_sequences
        ]
        return BranchCollection(
            branches=branches,
            branch_labels=branch_labels,
            fork_timestep=fork_timestep,
        )

    # ------------------------------------------------------------------ #

    def belief_manipulation_fork(
        self,
        trajectory: "LatentTrajectory",
        fork_timestep: int,
        action_sequence: Tensor,
        horizon: int = 20,
        z_patch_fn: Optional[Callable[[Tensor], Tensor]] = None,
        h_patch_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> BehaviorComparison:
        """Fork with optionally modified belief state at fork_timestep."""
        state = trajectory.states[fork_timestep]
        h0 = state.h.detach().clone()
        z0 = state.z.detach().clone()

        # Baseline
        baseline = self._imagine_from_state(h0, z0, action_sequence, horizon)

        # Modified
        h_mod = h_patch_fn(h0.clone()) if h_patch_fn else h0.clone()
        z_mod = z_patch_fn(z0.clone()) if z_patch_fn else z0.clone()
        modified = self._imagine_from_state(h_mod, z_mod, action_sequence, horizon)

        # Compute metrics
        r_base = sum(s.reward_pred for s in baseline.states if s.reward_pred is not None)
        r_mod = sum(s.reward_pred for s in modified.states if s.reward_pred is not None)
        reward_delta = float(r_mod - r_base)  # type: ignore[operator]

        T = min(len(baseline.states), len(modified.states))
        kl_div = np.zeros(T)
        for t in range(T):
            z_b = baseline.states[t].z
            z_m = modified.states[t].z
            if z_b is not None and z_m is not None:
                kl_div[t] = float((z_b.detach() - z_m.detach()).norm())

        return BehaviorComparison(
            baseline=baseline,
            modified=modified,
            reward_delta=reward_delta,
            kl_divergence=kl_div,
            description=(
                f"Belief manipulation at t={fork_timestep}: reward delta = {reward_delta:+.4f}"
            ),
        )

    # ------------------------------------------------------------------ #

    def compare_behavior(
        self,
        trajectory: "LatentTrajectory",
        fork_timestep: int,
        action_sequence: Tensor,
        horizon: int = 20,
        n_samples: int = 1,
    ) -> BehaviorComparison:
        """Compare imagined behavior with/without a small random noise perturbation."""
        state = trajectory.states[fork_timestep]
        h0 = state.h.detach().clone()
        z0 = state.z.detach().clone()

        noise_std = 0.01

        def _add_noise(z: Tensor) -> Tensor:
            return z + torch.randn_like(z) * noise_std

        return self.belief_manipulation_fork(
            trajectory,
            fork_timestep,
            action_sequence,
            horizon=horizon,
            z_patch_fn=_add_noise,
        )

    # ------------------------------------------------------------------ #

    def ensemble_branch(
        self,
        trajectory: "LatentTrajectory",
        fork_timestep: int,
        action_sequence: Tensor,
        horizon: int = 20,
        n_samples: int = 50,
    ) -> UncertaintyResult:
        """Sample n_samples rollouts by adding noise to z at fork."""
        state = trajectory.states[fork_timestep]
        h0 = state.h.detach().clone()
        z0 = state.z.detach().clone()

        noise_std = 0.05
        branches: List["LatentTrajectory"] = []
        for _ in range(n_samples):
            z_noisy = z0 + torch.randn_like(z0) * noise_std
            branches.append(self._imagine_from_state(h0, z_noisy, action_sequence, horizon))

        T = min(len(b.states) for b in branches)

        # Rewards
        all_rewards = np.array(
            [
                [s.reward_pred if s.reward_pred is not None else 0.0 for s in b.states[:T]]
                for b in branches
            ]
        )  # (n_samples, T)
        mean_rewards = all_rewards.mean(axis=0)
        reward_uncertainty = all_rewards.std(axis=0)

        # Epistemic uncertainty (std of h vectors)
        h_dim = h0.shape[-1]
        all_h = np.zeros((n_samples, T, h_dim))
        for si, b in enumerate(branches):
            for t in range(min(T, len(b.states))):
                if b.states[t].h is not None:
                    all_h[si, t] = b.states[t].h.detach().cpu().numpy()
        epistemic_uncertainty = all_h.std(axis=0).mean(axis=-1)  # (T,)

        return UncertaintyResult(
            branches=branches,
            mean_rewards=mean_rewards,
            epistemic_uncertainty=epistemic_uncertainty,
            reward_uncertainty=reward_uncertainty,
        )

    # ------------------------------------------------------------------ #

    def context_fork(
        self,
        trajectory: "LatentTrajectory",
        fork_timestep: int,
        context_window: int = 8,
        action_sequence: Optional[Tensor] = None,
        horizon: int = 20,
    ) -> BranchCollection:
        """Fork but vary the context window by zeroing history."""
        state = trajectory.states[fork_timestep]
        h0 = state.h.detach().clone()
        z0 = state.z.detach().clone()

        if action_sequence is None:
            # Use zeros as action sequence
            d_a = self._wm.cfg.d_action
            action_sequence = torch.zeros(horizon, d_a)

        # Branch 1: full context (original h)
        # Branch 2: zeroed h (no context)
        branches = [
            self._imagine_from_state(h0, z0, action_sequence, horizon),
            self._imagine_from_state(torch.zeros_like(h0), z0, action_sequence, horizon),
        ]
        return BranchCollection(
            branches=branches,
            branch_labels=["full_context", "no_context"],
            fork_timestep=fork_timestep,
        )

    # ------------------------------------------------------------------ #

    def resample_from_point(
        self,
        trajectory: "LatentTrajectory",
        timestep: int,
        n_samples: int = 10,
        temperature: float = 1.0,
        horizon: int = 20,
    ) -> BranchCollection:
        """Resample z_t from the prior at the given timestep and imagine forward."""
        state = trajectory.states[timestep]
        h0 = state.h.detach().clone()

        d_a = self._wm.cfg.d_action
        action_seq = torch.zeros(horizon, d_a)

        branches = []
        for i in range(n_samples):
            # Sample z from Gaussian noise (approximate prior resampling)
            z_sample = torch.randn_like(state.z) * temperature
            branches.append(self._imagine_from_state(h0, z_sample, action_seq, horizon))

        labels = [f"sample_{i}" for i in range(n_samples)]
        return BranchCollection(
            branches=branches,
            branch_labels=labels,
            fork_timestep=timestep,
        )
