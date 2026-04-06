"""Branching and imagination tools."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import pandas as pd

from world_model_lens import HookedWorldModel, WorldTrajectory, WorldState


@dataclass
class BranchCollection:
    """Collection of imagined branches from a fork point."""

    branches: List[WorldTrajectory]
    reference: WorldTrajectory
    fork_point: int

    def compare_reward_predictions(self) -> "pd.DataFrame":
        """Compare reward predictions across branches."""
        import pandas as pd

        data = []
        for i, branch in enumerate(self.branches):
            rewards = branch.reward_sequence
            if rewards is not None:
                data.append(
                    {
                        "branch": i,
                        "mean_reward": rewards.mean().item(),
                        "std_reward": rewards.std().item(),
                    }
                )
        return pd.DataFrame(data)

    def latent_divergence_over_time(self, metric: str = "cosine") -> torch.Tensor:
        """Compute divergence between branches over time."""
        if not self.branches:
            return torch.tensor([])

        divergences = []
        for t in range(min(len(b) for b in self.branches)):
            h_values = torch.stack([b.states[t].state for b in self.branches])
            if metric == "cosine":
                mean_h = h_values.mean(dim=0)
                cos_sim = torch.nn.functional.cosine_similarity(
                    h_values - mean_h.unsqueeze(0),
                    mean_h.unsqueeze(0) - mean_h.unsqueeze(0),
                    dim=1,
                )
                divergences.append(1 - cos_sim.mean().abs())
            else:
                std = h_values.std(dim=0).mean().item()
                divergences.append(std)

        return torch.tensor(divergences)

    def best_branch(self, metric: str = "total_reward") -> int:
        """Find best branch by metric."""
        best_idx = 0
        best_score = float("-inf")

        for i, branch in enumerate(self.branches):
            rewards = branch.reward_sequence
            if metric == "total_reward" and rewards is not None:
                score = rewards.sum().item()
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx


@dataclass
class BehaviorComparison:
    """Comparison between two trajectories."""

    action_divergence: torch.Tensor
    reward_divergence: torch.Tensor
    value_divergence: torch.Tensor

    def plot(self, figsize=(12, 4)):
        """Plot divergence over time."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        if len(self.action_divergence) > 0:
            axes[0].plot(self.action_divergence.cpu().numpy())
            axes[0].set_title("Action Divergence")
        if len(self.reward_divergence) > 0:
            axes[1].plot(self.reward_divergence.cpu().numpy())
            axes[1].set_title("Reward Divergence")
        if len(self.value_divergence) > 0:
            axes[2].plot(self.value_divergence.cpu().numpy())
            axes[2].set_title("Value Divergence")

        for ax in axes:
            ax.set_xlabel("Timestep")
        return fig


@dataclass
class UncertaintyResult:
    """Result of ensemble imagination."""

    trajectories: List[WorldTrajectory]
    mean_latent: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    reward_uncertainty: torch.Tensor


class ImaginationBrancher:
    """Fork and manipulate imagined trajectories.

    Tools for belief manipulation, counterfactual analysis,
    and ensemble imagination.
    """

    def __init__(self, wm: HookedWorldModel):
        self.wm = wm

    def fork(
        self,
        real_traj: WorldTrajectory,
        fork_at: int,
        action_sequences: List[torch.Tensor],
        horizon: int = 20,
    ) -> BranchCollection:
        """Fork imagination from a real trajectory.

        Args:
            real_traj: Real trajectory.
            fork_at: Fork point timestep.
            action_sequences: List of action sequences to execute.
            horizon: Imagination horizon.

        Returns:
            BranchCollection with imagined branches.
        """
        start_state = real_traj.states[fork_at]
        branches = []

        for actions in action_sequences:
            imagined = self.wm.imagine(
                start_state=start_state,
                actions=actions,
                horizon=horizon,
            )
            imagined.fork_point = fork_at
            branches.append(imagined)

        return BranchCollection(
            branches=branches,
            reference=real_traj,
            fork_point=fork_at,
        )

    def belief_manipulation_fork(
        self,
        real_traj: WorldTrajectory,
        fork_at: int,
        z_patch_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        h_patch_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        z_replacement: Optional[torch.Tensor] = None,
        policy: Optional[Callable] = None,
        actions: Optional[torch.Tensor] = None,
        horizon: int = 30,
    ) -> Tuple[WorldTrajectory, WorldTrajectory]:
        """Fork with belief manipulation.

        Args:
            real_traj: Real trajectory.
            fork_at: Fork point.
            z_patch_fn: Function to transform z.
            h_patch_fn: Function to transform h.
            z_replacement: Direct replacement for z.
            policy: Policy for imagination.
            actions: Fixed actions.
            horizon: Imagination horizon.

        Returns:
            Tuple of (original_branch, manipulated_branch).
        """
        start_state = real_traj.states[fork_at]

        original = self.wm.imagine(
            start_state=start_state,
            actions=actions,
            horizon=horizon,
        )
        original.fork_point = fork_at

        manipulated_state = start_state
        if z_replacement is not None:
            manipulated_state.obs_encoding = z_replacement.clone()

        manipulated = self.wm.imagine(
            start_state=manipulated_state,
            actions=actions,
            horizon=horizon,
        )
        manipulated.fork_point = fork_at

        return original, manipulated

    def compare_behavior(
        self,
        traj_a: WorldTrajectory,
        traj_b: WorldTrajectory,
        timestep_range: Optional[Tuple[int, int]] = None,
    ) -> BehaviorComparison:
        """Compare behavior between two trajectories.

        Args:
            traj_a: First trajectory.
            traj_b: Second trajectory.
            timestep_range: Optional (start, end) for comparison.

        Returns:
            BehaviorComparison with divergence metrics.
        """
        start, end = timestep_range or (0, min(len(traj_a), len(traj_b)))

        action_div = []
        reward_div = []
        value_div = []

        for t in range(start, end):
            if t < len(traj_a) and t < len(traj_b):
                a_state = traj_a.states[t]
                b_state = traj_b.states[t]

                if a_state.action is not None and b_state.action is not None:
                    action_div.append(
                        torch.nn.functional.mse_loss(a_state.action, b_state.action).item()
                    )

                if a_state.reward_pred is not None and b_state.reward_pred is not None:
                    reward_div.append(
                        torch.nn.functional.mse_loss(
                            a_state.reward_pred, b_state.reward_pred
                        ).item()
                    )

                if a_state.value_pred is not None and b_state.value_pred is not None:
                    value_div.append(
                        torch.nn.functional.mse_loss(a_state.value_pred, b_state.value_pred).item()
                    )

        return BehaviorComparison(
            action_divergence=torch.tensor(action_div),
            reward_divergence=torch.tensor(reward_div),
            value_divergence=torch.tensor(value_div),
        )

    def ensemble_branch(
        self,
        start_state: WorldState,
        policy: Optional[Callable] = None,
        horizon: int = 20,
        n_samples: int = 50,
        temperature: float = 1.0,
    ) -> UncertaintyResult:
        """Run ensemble imagination for uncertainty estimation.

        Args:
            start_state: Starting state.
            policy: Policy function.
            horizon: Imagination horizon.
            n_samples: Number of samples.
            temperature: Sampling temperature.

        Returns:
            UncertaintyResult with uncertainty quantification.
        """
        trajectories = []

        for _ in range(n_samples):
            traj = self.wm.imagine(
                start_state=start_state,
                horizon=horizon,
                temperature=temperature,
            )
            trajectories.append(traj)

        all_h = torch.stack([t.state_sequence for t in trajectories], dim=0)
        mean_h = all_h.mean(dim=0)
        epistemic = all_h.std(dim=0)

        reward_unc = torch.zeros(horizon)
        for traj in trajectories:
            rewards = traj.reward_sequence
            if rewards is not None:
                reward_unc += rewards.std(dim=0)
        reward_unc = reward_unc / n_samples

        return UncertaintyResult(
            trajectories=trajectories,
            mean_latent=mean_h,
            epistemic_uncertainty=epistemic,
            reward_uncertainty=reward_unc,
        )
