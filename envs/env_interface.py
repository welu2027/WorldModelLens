"""EpisodeCollector — collects episodes using Gymnasium API."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from world_model_lens.hooked_world_model import HookedWorldModel

from world_model_lens.core import LatentState, LatentTrajectory
from world_model_lens.hub.trajectory_hub import TrajectoryDataset

# optional imports handled at module import time so __init__ can raise helpful
# errors without triggering import-time side effects inside methods
try:
    import gymnasium as gym
except ImportError:
    gym = None

try:
    import torch
except ImportError:
    # torch is an optional runtime dependency for some helpers; tests install it
    import types

    torch = types.SimpleNamespace(zeros=lambda *a, **k: None)


@dataclass
class CollectedEpisode:
    """A single episode collected from an environment."""

    observations: list[Any]  # raw observations from env
    actions: list[Any]
    rewards: list[float]
    dones: list[bool]
    infos: list[dict]
    total_reward: float
    length: int


class EpisodeCollector:
    """Collect episodes from a Gymnasium environment using a world model.

    Works with any env that follows the Gymnasium API
    (obs, reward, terminated, truncated, info).
    """

    def __init__(
        self,
        wm: HookedWorldModel,
        env_id: str,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        """Initialize EpisodeCollector.

        Args:
            wm: HookedWorldModel instance for running trajectory.
            env_id: Gymnasium environment ID (e.g., "Pong-v5").
            device: Device to run model on (default "cpu").
            seed: Random seed for environment.

        Raises:
            ImportError: If gymnasium is not installed.
        """
        if gym is None:
            raise ImportError("gymnasium is not installed. Install it with: pip install gymnasium")

        self.wm = wm
        self.env_id = env_id
        self.device = device
        self.seed = seed

        self.env = gym.make(env_id)
        if seed is not None:
            self.env.reset(seed=seed)

    def collect_episode(
        self,
        max_steps: int = 1000,
        *,
        render: bool = False,
        policy: Callable | None = None,
    ) -> CollectedEpisode:
        """Collect a single episode.

        Args:
            max_steps: Maximum steps per episode.
            render: Whether to render the environment.
            policy: Optional policy function (h, z, env) -> action.
                   If None, uses random actions.

        Returns:
            CollectedEpisode with observations, actions, rewards, etc.
        """
        observations: list[Any] = []
        actions: list[Any] = []
        rewards: list[float] = []
        dones: list[bool] = []
        infos: list[dict] = []
        total_reward = 0.0

        obs, info = self.env.reset()
        observations.append(obs)
        infos.append(info)

        if render:
            self.env.render()

        # Initialize hidden state and latent state
        h = None
        z = None

        for _step in range(max_steps):
            # Decide action
            if policy is not None:
                action = policy(h, z, self.env)
            else:
                # Random action
                action = self.env.action_space.sample()

            actions.append(action)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            observations.append(obs)
            rewards.append(float(reward))
            dones.append(done)
            infos.append(info)
            total_reward += reward

            if render:
                self.env.render()

            if done:
                break

        return CollectedEpisode(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            infos=infos,
            total_reward=total_reward,
            length=len(actions),
        )

    def collect_dataset(
        self,
        n_episodes: int,
        max_steps_per_episode: int = 1000,
        policy: Callable | None = None,
        *,
        show_progress: bool = True,
    ) -> TrajectoryDataset:
        """Collect n_episodes and return as TrajectoryDataset.

        Args:
            n_episodes: Number of episodes to collect.
            max_steps_per_episode: Max steps per episode.
            policy: Optional policy function.
            show_progress: Whether to show progress bar.

        Returns:
            TrajectoryDataset with collected episodes.
        """
        trajectories: list[LatentTrajectory] = []

        if show_progress:
            if "tqdm" not in globals():
                try:
                    from tqdm import tqdm  # type: ignore
                except ImportError:
                    warnings.warn("tqdm not installed; progress bar disabled", stacklevel=2)
                    episode_range = range(n_episodes)
                else:
                    episode_range = tqdm(range(n_episodes), desc="Collecting episodes")
            else:
                episode_range = globals()["tqdm"](range(n_episodes), desc="Collecting episodes")
        else:
            episode_range = range(n_episodes)

        for ep_idx in episode_range:
            episode = self.collect_episode(
                max_steps=max_steps_per_episode,
                render=False,
                policy=policy,
            )

            # Convert to LatentTrajectory with LatentState objects
            states: list[LatentState] = []
            n_steps = len(episode.observations) - 1
            for t in range(n_steps):
                # Create minimal LatentState with observations and actions
                # h_t: dummy hidden state
                # z_posterior, z_prior: dummy categorical latents (minimal shape)
                h_t = torch.zeros(1, dtype=torch.float32)
                z_posterior = torch.zeros(1, 1, dtype=torch.float32)
                z_prior = torch.zeros(1, 1, dtype=torch.float32)

                state = LatentState(
                    h_t=h_t,
                    z_posterior=z_posterior,
                    z_prior=z_prior,
                    timestep=t,
                    action=episode.actions[t],
                    reward_pred=float(episode.rewards[t]),
                    reward_real=float(episode.rewards[t]),
                )
                states.append(state)

            # Add final state without action
            h_t = torch.zeros(1, dtype=torch.float32)
            z_posterior = torch.zeros(1, 1, dtype=torch.float32)
            z_prior = torch.zeros(1, 1, dtype=torch.float32)
            final_state = LatentState(
                h_t=h_t,
                z_posterior=z_posterior,
                z_prior=z_prior,
                timestep=len(episode.observations) - 1,
                action=None,
            )
            states.append(final_state)

            traj = LatentTrajectory(
                states=states,
                env_name=self.env_id,
                episode_id=f"ep_{ep_idx:06d}",
                imagined=False,
                metadata={
                    "observations": episode.observations,
                    "raw_actions": episode.actions,
                    "raw_rewards": episode.rewards,
                    "raw_dones": episode.dones,
                    "raw_infos": episode.infos,
                    "total_reward": episode.total_reward,
                },
            )
            trajectories.append(traj)

        return TrajectoryDataset(trajectories)

    def close(self) -> None:
        """Close the Gymnasium environment."""
        if self.env is not None:
            self.env.close()
