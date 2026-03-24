"""TrajectoryHub — local storage for trajectory datasets."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from world_model_lens.core import LatentTrajectory


@dataclass
class TrajectoryDataset:
    """A collection of LatentTrajectory objects with dataset-like API."""
    trajectories: List[LatentTrajectory]
    
    def __len__(self) -> int:
        """Return number of trajectories."""
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> LatentTrajectory:
        """Get trajectory by index."""
        return self.trajectories[idx]
    
    def filter(self, fn: Callable[[LatentTrajectory], bool]) -> TrajectoryDataset:
        """Return new dataset with only trajectories where fn(traj) is True.
        
        Args:
            fn: Function that takes a LatentTrajectory and returns bool.
            
        Returns:
            New TrajectoryDataset with filtered trajectories.
        """
        filtered = [traj for traj in self.trajectories if fn(traj)]
        return TrajectoryDataset(filtered)
    
    def map(
        self, 
        fn: Callable[[LatentTrajectory], LatentTrajectory]
    ) -> TrajectoryDataset:
        """Apply fn to each trajectory, return new dataset.
        
        Args:
            fn: Function that transforms a LatentTrajectory.
            
        Returns:
            New TrajectoryDataset with transformed trajectories.
        """
        mapped = [fn(traj) for traj in self.trajectories]
        return TrajectoryDataset(mapped)
    
    def split(self, ratio: float = 0.8) -> Tuple[TrajectoryDataset, TrajectoryDataset]:
        """Split into train/val by ratio.
        
        Args:
            ratio: Fraction of trajectories for train set (default 0.8).
            
        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        if not (0 < ratio < 1):
            raise ValueError(f"ratio must be between 0 and 1, got {ratio}")
        
        n_train = int(len(self.trajectories) * ratio)
        train_trajs = self.trajectories[:n_train]
        val_trajs = self.trajectories[n_train:]
        
        return (TrajectoryDataset(train_trajs), TrajectoryDataset(val_trajs))
    
    def statistics(self) -> dict:
        """Return dict with trajectory statistics.
        
        Returns dict with keys:
            - n_trajectories: Number of trajectories
            - total_steps: Total number of steps across all trajectories
            - mean_episode_length: Average trajectory length
            - mean_reward: Mean of trajectory total rewards
            - std_reward: Standard deviation of trajectory total rewards
            
        Returns zeros/None if trajectories empty.
        """
        if not self.trajectories:
            return {
                "n_trajectories": 0,
                "total_steps": 0,
                "mean_episode_length": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
            }
        
        n_trajs = len(self.trajectories)
        total_steps = sum(traj.length for traj in self.trajectories)
        mean_length = total_steps / n_trajs if n_trajs > 0 else 0
        
        # Extract total rewards from trajectories
        rewards = []
        for traj in self.trajectories:
            # Try to get from metadata first
            if traj.metadata and "total_reward" in traj.metadata:
                rewards.append(traj.metadata["total_reward"])
            # Fall back to summing reward_real from states
            elif traj.rewards_real is not None:
                rewards.append(traj.rewards_real.sum().item())
            else:
                rewards.append(0.0)
        
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        # Calculate std deviation
        if len(rewards) > 1:
            variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
            std_reward = variance ** 0.5
        else:
            std_reward = 0.0
        
        return {
            "n_trajectories": n_trajs,
            "total_steps": total_steps,
            "mean_episode_length": mean_length,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }
    
    def save(self, path: str) -> None:
        """Save to .pkl file.
        
        Args:
            path: File path to save to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> TrajectoryDataset:
        """Load from .pkl file.
        
        Args:
            path: File path to load from.
            
        Returns:
            Loaded TrajectoryDataset.
        """
        with open(path, "rb") as f:
            return pickle.load(f)


class TrajectoryHub:
    """Simple local storage for trajectory datasets."""
    
    def __init__(self, root_dir: str = "~/.cache/world_model_lens/trajectories") -> None:
        """Initialize TrajectoryHub.
        
        Args:
            root_dir: Root directory for storing trajectory datasets.
        """
        self.root_dir = Path(root_dir).expanduser()
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, name: str, dataset: TrajectoryDataset) -> str:
        """Save dataset under name. Returns path.
        
        Args:
            name: Name of the dataset.
            dataset: TrajectoryDataset to save.
            
        Returns:
            Path where dataset was saved.
        """
        path = self.root_dir / f"{name}.pkl"
        dataset.save(str(path))
        return str(path)
    
    def load(self, name: str) -> TrajectoryDataset:
        """Load dataset by name.
        
        Args:
            name: Name of the dataset.
            
        Returns:
            Loaded TrajectoryDataset.
            
        Raises:
            FileNotFoundError: If dataset not found.
        """
        path = self.root_dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found at {path}")
        
        return TrajectoryDataset.load(str(path))
    
    def list_datasets(self) -> List[str]:
        """List all saved dataset names.
        
        Returns:
            List of dataset names (without .pkl extension).
        """
        pkl_files = self.root_dir.glob("*.pkl")
        return [f.stem for f in pkl_files]
    
    def delete(self, name: str) -> None:
        """Delete dataset by name.
        
        Args:
            name: Name of the dataset to delete.
            
        Raises:
            FileNotFoundError: If dataset not found.
        """
        path = self.root_dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found at {path}")
        
        path.unlink()
