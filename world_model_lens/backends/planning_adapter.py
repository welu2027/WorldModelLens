"""Planning Model Adapter.

Example of a planning-oriented world model that works with World Model Lens.

Planning models predict action sequences to achieve goals.
They fit the "encode + dynamics + plan" pattern.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from world_model_lens.backends.base_adapter import WorldModelAdapter, AdapterConfig


class PlanningAdapter(WorldModelAdapter):
    """World model adapter for planning-oriented models.

    This model:
    - Encodes observations into state
    - Uses dynamics for imagination
    - Predicts actions for planning
    - Optionally predicts goal achievement

    No reward or value predictions - planning is goal-based.
    """

    def __init__(self, config: AdapterConfig, goal_dim: int = 32):
        super().__init__(config)
        self.goal_dim = goal_dim

        self.encoder = nn.Sequential(
            nn.Linear(config.d_obs + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.d_state),
        )

        self.dynamics = nn.Sequential(
            nn.Linear(config.d_state + config.d_action, 256),
            nn.ReLU(),
            nn.Linear(256, config.d_state),
        )

        self.planner = nn.Sequential(
            nn.Linear(config.d_state + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.d_action * 10),
        )

    def encode(
        self,
        observation: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode observation + goal into state."""
        if observation.dim() == 4:
            obs_flat = observation.flatten(1) if observation.shape[0] == 1 else observation
        else:
            obs_flat = observation

        goal = (
            context
            if context is not None
            else torch.zeros(1, self.goal_dim, device=obs_flat.device)
        )
        x = torch.cat([obs_flat, goal], dim=-1)
        state = self.encoder(x)
        return state, state

    def dynamics(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict next state given current state and action."""
        if action is None:
            action = torch.zeros(state.shape[0], self.config.d_action, device=state.device)
        x = torch.cat([state, action], dim=-1)
        return self.dynamics(x)

    def plan(
        self,
        current_state: torch.Tensor,
        goal: torch.Tensor,
        horizon: int = 10,
    ) -> torch.Tensor:
        """Plan action sequence to reach goal.

        Args:
            current_state: Current state
            goal: Goal state
            horizon: Planning horizon

        Returns:
            Planned action sequence [horizon, d_action]
        """
        x = torch.cat([current_state, goal], dim=-1)
        action_sequence = self.planner(x)
        return action_sequence.view(-1, horizon, self.config.d_action)

    def predict_goal_achievement(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """Predict whether goal is achieved."""
        x = torch.cat([state, goal], dim=-1)
        return torch.sigmoid(self.planner[2](self.planner[1](self.planner[0](x)[:, :1])))
