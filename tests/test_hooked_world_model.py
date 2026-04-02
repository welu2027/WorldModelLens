"""Tests for HookedWorldModel."""


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


def test_hook_time_slice_matching():
    """Test that time_slice correctly filters hooks by timestep range."""
    from world_model_lens import HookPoint, HookRegistry

    def dummy_hook(tensor, ctx):
        return tensor

    registry = HookRegistry()
    hook = HookPoint(name="encoder.out", stage="post", fn=dummy_hook, time_slice=[5, 10])
    registry.register(hook)

    assert len(registry.get_hooks_for("encoder.out", 0)) == 0
    assert len(registry.get_hooks_for("encoder.out", 4)) == 0
    assert len(registry.get_hooks_for("encoder.out", 5)) == 1
    assert len(registry.get_hooks_for("encoder.out", 7)) == 1
    assert len(registry.get_hooks_for("encoder.out", 9)) == 1
    assert len(registry.get_hooks_for("encoder.out", 10)) == 0


def test_hook_timestep_overrides_time_slice():
    """Test that timestep takes precedence over time_slice when both set."""
    from world_model_lens import HookPoint, HookRegistry

    def dummy_hook(tensor, ctx):
        return tensor

    registry = HookRegistry()
    hook = HookPoint(name="state", stage="post", fn=dummy_hook, timestep=3, time_slice=[5, 10])
    registry.register(hook)

    assert len(registry.get_hooks_for("state", 3)) == 1
    assert len(registry.get_hooks_for("state", 5)) == 0
    assert len(registry.get_hooks_for("state", 7)) == 0
