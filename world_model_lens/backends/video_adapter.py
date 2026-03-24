"""Video Prediction Model Adapter.

Example of a non-RL world model that works with World Model Lens.

Video prediction models predict future frames from current frames.
They fit the "encode + dynamics + decode" pattern but have no
reward/value/action predictions.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.generic_adapter import WorldModelAdapter, WorldModelConfig


class VideoEncoder(nn.Module):
    """Encode video frames into latent representation."""

    def __init__(self, d_obs: int, d_state: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_obs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, d_state),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(1))


class VideoDecoder(nn.Module):
    """Decode latent state back to video frames."""

    def __init__(self, d_state: int, d_obs: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_state, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, d_obs),
        )

    def forward(self, x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        recon = self.decoder(x)
        return recon.view(*target_shape)


class VideoDynamics(nn.Module):
    """Predict next frame from current latent state."""

    def __init__(self, d_state: int):
        super().__init__()
        self.dynamics = nn.Sequential(
            nn.Linear(d_state, 256),
            nn.ReLU(),
            nn.Linear(256, d_state),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.dynamics(state)


class VideoWorldModelAdapter(WorldModelAdapter):
    """World model adapter for video prediction.

    This is a non-RL world model - it has no reward, value, or action predictions.
    It only predicts future frames using latent dynamics.

    This demonstrates that World Model Lens works with ANY world model,
    not just RL agents.
    """

    def __init__(self, config: WorldModelConfig, d_obs: int, frame_shape: Tuple[int, ...]):
        super().__init__(config)
        self.d_obs = d_obs
        self.frame_shape = frame_shape
        self.frame_size = 1
        for dim in frame_shape:
            self.frame_size *= dim

        self.encoder = VideoEncoder(self.frame_size, config.d_state)
        self.dynamics = VideoDynamics(config.d_state)
        self.decoder = VideoDecoder(config.d_state, self.frame_size)

    @property
    def hook_point_names(self) -> List[str]:
        """Video model hook points."""
        return [
            "state",
            "observation",
            "reconstruction",
            "dynamics",
        ]

    def encode(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode video frame into latent state."""
        obs_encoding = self.encoder(observation)
        if context is not None:
            obs_encoding = torch.cat([obs_encoding, context], dim=-1)
        return obs_encoding, obs_encoding

    def dynamics(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict next latent state."""
        return self.dynamics(state)

    def transition(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        input_: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Transition state (same as dynamics for video model)."""
        return self.dynamics(state)

    def decode(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Decode latent state to video frame."""
        return self.decoder(state, self.frame_shape)

    def forward(
        self,
        observation: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full video prediction forward pass."""
        posterior, obs_encoding = self.encode(observation, state)
        prior = self.dynamics(state if state is not None else self.initial_state())
        next_state = self.transition(posterior, action)
        reconstruction = self.decode(posterior)

        return {
            "posterior": posterior,
            "prior": prior,
            "next_state": next_state,
            "reconstruction": reconstruction,
            "obs_encoding": obs_encoding,
        }

    def predict_next_frame(
        self,
        current_frame: torch.Tensor,
        n_frames: int = 1,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Predict multiple future frames.

        Args:
            current_frame: Current video frame
            n_frames: Number of frames to predict

        Returns:
            Tuple of (predicted_frames, latent_states)
        """
        state, _ = self.encode(current_frame)
        predictions = []
        states = [state]

        for _ in range(n_frames):
            state = self.dynamics(state)
            pred_frame = self.decode(state)
            predictions.append(pred_frame)
            states.append(state)

        return torch.stack(predictions, dim=0), torch.stack(states, dim=0)
