"""Tests for HookedWorldModel."""

import pytest
import torch
import torch.nn as nn

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.backends.base_adapter import WorldModelAdapter, WorldModelCapabilities


def test_run_with_cache_returns_correct_length(hooked_wm, fake_obs_seq, fake_action_seq):
    """Test run_with_cache returns trajectory of correct length."""
    traj, cache = hooked_wm.run_with_cache(fake_obs_seq, fake_action_seq)
    assert traj.length == 10
    assert len(cache.timesteps) == 10


def test_run_with_cache_caches_all_components(hooked_wm, fake_obs_seq, fake_action_seq):
    """Test cache contains all expected components."""
    traj, cache = hooked_wm.run_with_cache(fake_obs_seq, fake_action_seq)
    assert "state" in cache.component_names
    assert "z_posterior" in cache.component_names
    assert "z_prior" in cache.component_names


def test_imagine_returns_correct_horizon(hooked_wm, fake_trajectory):
    """Test imagine returns correct horizon length."""
    start_state = fake_trajectory.states[0]
    horizon = 20
    imagined = hooked_wm.imagine(start_state=start_state, horizon=horizon)
    assert imagined.length == horizon


def test_named_weights_accessible(hooked_wm):
    """Test all named weights are accessible."""
    weights = hooked_wm.named_weights
    assert isinstance(weights, dict)


def test_add_and_clear_hooks(hooked_wm):
    """Test hook management."""
    from world_model_lens import HookPoint

    def dummy_hook(tensor, ctx):
        return tensor

    hook = HookPoint(name="state", fn=dummy_hook)
    hooked_wm.add_hook(hook)
    assert len(hooked_wm._hooks) > 0

    hooked_wm.clear_hooks()
    assert len(hooked_wm._hooks) == 0


class LegacyAdapter(nn.Module):
    """Legacy-style adapter without a capabilities property."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transition_calls = []
        self.decode_calls = []
        self.value_calls = []
        self.encoder_net = nn.Linear(config.d_obs, config.d_h)
        self.dynamics_net = nn.Linear(config.d_h, config.d_h)
        self.value_net = nn.Linear(config.d_h, 1)

    def encode(self, obs, context=None):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        encoded = self.encoder_net(obs)
        return encoded, encoded

    def dynamics(self, state, action=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.dynamics_net(state)

    def transition(self, state, action=None, input_=None):
        self.transition_calls.append((state, action, input_))
        return self.dynamics(state, action)

    def decode(self, state):
        self.decode_calls.append((state,))
        return state

    def predict_value(self, state, action=None):
        self.value_calls.append((state, action))
        return self.value_net(state)

    def initial_state(self, batch_size=1, device=None):
        return torch.zeros(batch_size, self.config.d_h, device=device or torch.device("cpu"))

    def sample_z(self, logits_or_repr, temperature=1.0, sample=True):
        return logits_or_repr


class CapabilitiesAdapter(WorldModelAdapter):
    """New-style adapter with capabilities and (h, z, action) transition."""

    def __init__(self, config):
        super().__init__(config)
        self.transition_calls = []
        self.decode_calls = []
        self.encoder_net = nn.Linear(config.d_obs, config.d_h)
        self.dynamics_net = nn.Linear(config.d_h, config.d_h)
        self._capabilities = WorldModelCapabilities(has_decoder=True)

    def encode(self, obs, h_prev):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        encoded = self.encoder_net(obs)
        return encoded, encoded

    def dynamics(self, state, action=None):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.dynamics_net(state)

    def transition(self, h, z, action=None):
        self.transition_calls.append((h, z, action))
        return h + z

    def decode(self, h, z):
        self.decode_calls.append((h, z))
        return h + z


def test_capabilities_fallback_for_legacy_adapter():
    """Legacy adapters without .capabilities should still work."""
    cfg = WorldModelConfig(
        d_h=16,
        d_obs=8,
        d_action=3,
        has_decoder=False,
        has_reward_head=False,
        has_value_head=True,
        has_policy_head=False,
        has_done_head=True,
    )
    wm = HookedWorldModel(adapter=LegacyAdapter(cfg), config=cfg)

    caps = wm.capabilities

    assert caps.has_critic is True
    assert caps.uses_actions is True
    assert caps.has_continue_head is True


def test_call_transition_uses_legacy_signature_order():
    """Legacy adapters should receive transition(state, action, input_)."""
    cfg = WorldModelConfig(d_h=8, d_obs=4, has_decoder=False, has_value_head=False)
    adapter = LegacyAdapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)

    state = torch.randn(1, cfg.d_h)
    posterior = torch.randn(1, cfg.d_h)
    action = torch.randn(1, 2)

    wm._call_transition(state, posterior, action)

    called_state, called_action, called_input = adapter.transition_calls[-1]
    assert torch.equal(called_state, state)
    assert torch.equal(called_action, action)
    assert torch.equal(called_input, posterior)


def test_call_transition_uses_capabilities_signature_order():
    """Capabilities adapters should receive transition(h, z, action)."""
    cfg = WorldModelConfig(d_h=8, d_obs=4, has_decoder=True, has_value_head=False)
    adapter = CapabilitiesAdapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)

    state = torch.randn(1, cfg.d_h)
    posterior = torch.randn(1, cfg.d_h)
    action = torch.randn(1, 2)

    wm._call_transition(state, posterior, action)

    called_state, called_posterior, called_action = adapter.transition_calls[-1]
    assert torch.equal(called_state, state)
    assert torch.equal(called_posterior, posterior)
    assert torch.equal(called_action, action)


def test_call_decode_uses_both_adapter_styles():
    """Decode dispatch should support both old and new adapter APIs."""
    legacy_cfg = WorldModelConfig(d_h=8, d_obs=4, has_decoder=True, has_value_head=False)
    legacy_adapter = LegacyAdapter(legacy_cfg)
    legacy_wm = HookedWorldModel(adapter=legacy_adapter, config=legacy_cfg)

    state = torch.randn(1, legacy_cfg.d_h)
    posterior = torch.randn(1, legacy_cfg.d_h)
    legacy_wm._call_decode(state, posterior)
    assert len(legacy_adapter.decode_calls[-1]) == 1

    new_cfg = WorldModelConfig(d_h=8, d_obs=4, has_decoder=True, has_value_head=False)
    new_adapter = CapabilitiesAdapter(new_cfg)
    new_wm = HookedWorldModel(adapter=new_adapter, config=new_cfg)
    new_wm._call_decode(state, posterior)
    assert len(new_adapter.decode_calls[-1]) == 2


def test_run_with_cache_falls_back_to_predict_value_for_legacy_adapter():
    """Legacy adapters should use predict_value when critic_forward is absent."""
    cfg = WorldModelConfig(
        d_h=8,
        d_obs=4,
        d_action=2,
        has_decoder=False,
        has_reward_head=False,
        has_value_head=True,
        has_policy_head=False,
        has_done_head=False,
    )
    adapter = LegacyAdapter(cfg)
    wm = HookedWorldModel(adapter=adapter, config=cfg)

    observations = torch.randn(3, cfg.d_obs)
    actions = torch.randn(3, cfg.d_action)
    _, cache = wm.run_with_cache(observations, actions)

    assert len(adapter.value_calls) == 3
    assert "value" in cache.component_names
