"""Tests for HookedWorldModel."""

import pytest
import torch


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
