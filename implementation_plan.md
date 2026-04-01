# Visualization Audit & Contribution Plan

Fix bugs and add new visualizer classes across `world_model_lens/visualization/`.

---

## Bugs Found (Cross-Referencing with Core Classes)

### 1. `latent_plots.py` — Typo: `TSN` should be `TSNE`
`from sklearn.manifold import TSN` → `from sklearn.manifold import TSNE`.  
Will crash on import at runtime (line 151).

### 2. `latent_plots.py` — `color_by_reward` uses non-existent `state.predictions` dict
`WorldState` has no `.predictions` dict. Reward lives in `state.reward_pred` or `state.reward`.

### 3. `prediction_plots.py` — `reward_timeline` uses `state.predictions.get(...)`
Same issue — `state.predictions` doesn't exist. Should use `state.reward_pred` and `state.reward`.

### 4. `prediction_plots.py` — `latent_distribution` uses `state.predictions` (would crash)
Same pattern; `z` should be extracted from `state.obs_encoding or state.state`.

### 5. `temporal_maps.py` — Calls `self.wm.run_with_advanced_hooks()`
`HookedWorldModel` **has no `run_with_advanced_hooks` method**. It only has `run_with_hooks()`.  
The temporal attribution computation will always crash. Needs to be rewritten using `run_with_hooks()` + `HookPoint`.

### 6. `temporal_maps.py` — `ablate_hook` closure captures wrong `source_t` in loop
The `ablate_hook` closures inside the `for source_t in range(T)` loop capture the loop variable by reference — classic Python closure-in-loop bug. All hooks end up ablating the same (last) `source_t`.

### 7. `intervention_plots.py` — `dimension_importance` accesses `trajectory.states[0].state.shape[-1]`
This works but silently gives wrong `d_z` if batch dim is present (shape `[1, d]` gives `d` not `d_z`). Should call `.flatten()` first.

### 8. `visualization/__init__.py` — Missing new classes from plan (tracked below)
Once new classes are added, they need to be exported from `__init__.py`.

---

## Missing Capabilities (New Classes)

### A. `SurpriseHeatmap` in `latent_plots.py`
The `ActivationCache` has a `.surprise()` method that returns `[T]` KL values, but there is NO visualizer to produce a 2D KL heatmap over `[T, d_latent]`. This is the most important MI visualization for a world model (shows per-dimension contribution to surprise). Will add as a new class in `latent_plots.py`.

### B. `CacheSignalPlotter` in a new file `world_model_lens/visualization/cache_plots.py`
No way to plot arbitrary cache signals (e.g., KL over time, reward over time, value over time as line charts). This is a core MI workflow: after `run_with_cache()`, plot key scalars vs timestep. Will add `CacheSignalPlotter` with `plot_surprise_timeline`, `plot_reward_timeline`, `plot_value_timeline`, and `plot_cache_signal`.

---

## Proposed Changes

### Core Fixes

#### [MODIFY] [latent_plots.py](file:///d:/WorldModelLens/world_model_lens/visualization/latent_plots.py)
- Fix `TSN` → `TSNE` typo
- Fix `color_by_reward()` to use `state.reward_pred` / `state.reward` instead of `state.predictions`
- Add `SurpriseHeatmap` class using `ActivationCache.get("kl", t)`

#### [MODIFY] [prediction_plots.py](file:///d:/WorldModelLens/world_model_lens/visualization/prediction_plots.py)
- Fix `reward_timeline()` to use `state.reward_pred` / `state.reward`
- Fix `latent_distribution()` to read from `state.obs_encoding or state.state`

#### [MODIFY] [temporal_maps.py](file:///d:/WorldModelLens/world_model_lens/visualization/temporal_maps.py)
- Replace all `run_with_advanced_hooks(hook_specs=...)` calls with correct `run_with_hooks(fwd_hooks=[HookPoint(...)])` API
- Fix closure-in-loop bug in `compute_influence_matrix` and `influence_from_timestep` by using a factory function to capture `source_t` correctly

#### [MODIFY] [intervention_plots.py](file:///d:/WorldModelLens/world_model_lens/visualization/intervention_plots.py)
- Fix `dimension_importance()` to flatten state before computing shape

### New Classes

#### [NEW] [cache_plots.py](file:///d:/WorldModelLens/world_model_lens/visualization/cache_plots.py)
New `CacheSignalPlotter` class with methods:
- `plot_surprise_timeline(cache)` → `dict` with `timesteps`, `kl_values`
- `plot_reward_timeline(trajectory)` → `dict` with `timesteps`, `predicted`, `actual`
- `plot_value_timeline(trajectory)` → `dict` with `timesteps`, `value_pred`
- `plot_cache_signal(cache, component)` → `dict` with `timesteps`, `norms` (L2 norm of the activation over time)

#### [MODIFY] [\_\_init\_\_.py](file:///d:/WorldModelLens/world_model_lens/visualization/__init__.py)
- Export `SurpriseHeatmap` and `CacheSignalPlotter`

### Tests

#### [NEW] [test_visualization.py](file:///d:/WorldModelLens/tests/test_visualization.py)
New test file covering:
- `LatentTrajectoryPlotter.project_pca()` — smoke test, 10 fake states
- `LatentTrajectoryPlotter.project_tsne()` — smoke test
- `LatentTrajectoryPlotter.color_by_reward()` — confirm correct dtype, no crash
- `InterventionVisualizer.divergence_curve()` — confirm shape matches T
- `InterventionVisualizer.intervention_heatmap()` — confirm 2D `[T, d]`
- `PredictionVisualizer.reward_timeline()` — confirm correct field access
- `CacheSignalPlotter.plot_surprise_timeline()` — confirm output keys

---

## Verification Plan

### Automated Tests
```bash
# Run only visualization tests (fast)
cd d:\WorldModelLens
pytest tests/test_visualization.py -v

# Run all tests to check for regressions
pytest tests/ -v --tb=short
```

### Manual Sanity Check (optional)
```bash
cd d:\WorldModelLens
python -c "
from world_model_lens.visualization import (
    LatentTrajectoryPlotter, PredictionVisualizer,
    InterventionVisualizer, TemporalAttributionMap,
    SurpriseHeatmap, CacheSignalPlotter,
)
print('All imports OK')
"
```
