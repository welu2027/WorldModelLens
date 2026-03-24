"""Causal tracing for world model activations.

CausalTracer implements a greedy path-finding algorithm that identifies which
(timestep, activation_name) nodes form the minimal causal chain connecting an
initial cause to a downstream effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from world_model_lens.core.activation_cache import ActivationCache
    from world_model_lens.hooked_world_model import HookedWorldModel
    from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CausalLink:
    """One directed edge in a causal chain."""
    src_timestep: int
    src_activation: str
    dst_timestep: int
    dst_activation: str
    recovery_contribution: float   # how much recovery this link adds


@dataclass
class CausalChain:
    """Ordered sequence of causal links from cause to effect."""
    links: List[CausalLink]
    total_recovery: float          # cumulative recovery rate of the whole chain
    metric_clean: float
    metric_corrupted: float

    # ------------------------------------------------------------------ #
    def nodes(self) -> List[Tuple[int, str]]:
        """All unique (timestep, activation) nodes in the chain."""
        seen: set = set()
        out: List[Tuple[int, str]] = []
        for lk in self.links:
            for node in [(lk.src_timestep, lk.src_activation),
                         (lk.dst_timestep, lk.dst_activation)]:
                if node not in seen:
                    seen.add(node)
                    out.append(node)
        return out

    def summary(self) -> str:
        lines = [
            f"CausalChain  |  recovery={self.total_recovery:.3f}  |  "
            f"clean={self.metric_clean:.4f}  corrupted={self.metric_corrupted:.4f}",
            f"  {len(self.links)} links:",
        ]
        for i, lk in enumerate(self.links):
            lines.append(
                f"    [{i}] t={lk.src_timestep} {lk.src_activation!r}"
                f" → t={lk.dst_timestep} {lk.dst_activation!r}"
                f"  (+{lk.recovery_contribution:.3f})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def plot_causal_flow(self, ax=None):
        """Draw a directed graph of the causal chain.

        Nodes are positioned by (timestep, activation_rank).
        Edges are coloured by recovery contribution (green = high, red = low).

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if ax is None:
            _fig, ax = plt.subplots(figsize=(max(6, len(self.links) * 2), 4))

        act_names = list(dict.fromkeys(
            name
            for lk in self.links
            for name in [lk.src_activation, lk.dst_activation]
        ))
        act_rank = {n: i for i, n in enumerate(act_names)}

        # Determine node positions
        node_pos: Dict[Tuple[int, str], Tuple[float, float]] = {}
        for lk in self.links:
            for t, name in [(lk.src_timestep, lk.src_activation),
                            (lk.dst_timestep, lk.dst_activation)]:
                node_pos[(t, name)] = (float(t), float(act_rank[name]))

        # Draw edges
        contributions = [lk.recovery_contribution for lk in self.links]
        max_c = max(abs(c) for c in contributions) if contributions else 1.0

        for lk in self.links:
            x0, y0 = node_pos[(lk.src_timestep, lk.src_activation)]
            x1, y1 = node_pos[(lk.dst_timestep, lk.dst_activation)]
            alpha = 0.4 + 0.6 * abs(lk.recovery_contribution) / (max_c + 1e-9)
            color = "tab:green" if lk.recovery_contribution > 0 else "tab:red"
            ax.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, alpha=alpha, lw=1.5),
            )

        # Draw nodes
        for (t, name), (x, y) in node_pos.items():
            ax.scatter(x, y, s=100, zorder=5, color="steelblue")
            ax.text(x, y + 0.15, f"t={t}\n{name}", ha="center", va="bottom",
                    fontsize=7, clip_on=True)

        ax.set_xlabel("Timestep")
        ax.set_yticks([act_rank[n] for n in act_names])
        ax.set_yticklabels(act_names, fontsize=8)
        ax.set_title(
            f"Causal Flow  (recovery={self.total_recovery:.3f})", fontsize=10
        )
        ax.grid(True, alpha=0.3)
        return ax


# ---------------------------------------------------------------------------
# CausalTracer
# ---------------------------------------------------------------------------

class CausalTracer:
    """Greedy causal-tracing algorithm for world-model activations.

    Algorithm
    ---------
    Given a *clean* cache and a *corrupted* rollout:

    1. Compute ``metric_clean`` and ``metric_corrupted`` from the respective
       full rollouts.
    2. For each candidate ``(timestep, activation_name)`` node:
       a. Patch that single node with its clean value into the corrupted run.
       b. Measure ``metric_patched``.
       c. ``recovery = (patched - corrupted) / (clean - corrupted)``.
    3. Greedily select the node with the highest recovery at each hop.
    4. Continue for ``n_hops`` hops, accumulating a chain.

    The result is a :class:`CausalChain` showing which nodes are most
    causally responsible for the metric gap.
    """

    def __init__(self, wm: "HookedWorldModel") -> None:
        self._wm = wm

    # ------------------------------------------------------------------ #

    def causal_trace(
        self,
        clean_obs_seq: "Tensor",
        clean_action_seq: "Tensor",
        corrupted_obs_seq: "Tensor",
        corrupted_action_seq: "Tensor",
        metric_fn: Callable[["ActivationCache"], float],
        activation_names: Optional[List[str]] = None,
        n_hops: int = 5,
        patch_value: str = "clean",   # "clean" — patch corrupted with clean values
    ) -> CausalChain:
        """Run greedy causal tracing.

        Parameters
        ----------
        clean_obs_seq, clean_action_seq:
            The "clean" (reference) trajectory.
        corrupted_obs_seq, corrupted_action_seq:
            The "corrupted" (interventional) trajectory.
        metric_fn:
            A callable that maps an :class:`ActivationCache` to a scalar
            performance metric.  Higher is better.
        activation_names:
            Names of activations to consider as candidate nodes.  Defaults to
            all standard hook-point names from the adapter.
        n_hops:
            Maximum number of nodes to include in the chain.
        patch_value:
            Currently only ``"clean"`` is supported.
        """
        import torch

        wm = self._wm

        # -- Step 1: baseline caches ----------------------------------------
        _, clean_cache = wm.run_with_cache(clean_obs_seq, clean_action_seq)
        _, corrupted_cache = wm.run_with_cache(
            corrupted_obs_seq, corrupted_action_seq
        )
        metric_clean = metric_fn(clean_cache)
        metric_corrupted = metric_fn(corrupted_cache)
        denom = metric_clean - metric_corrupted

        # -- activation names to probe ----------------------------------------
        if activation_names is None:
            activation_names = wm.adapter.hook_point_names
            # keep only names present in the clean cache
            activation_names = [
                n for n in activation_names
                if n in clean_cache._store and clean_cache._store[n]
            ]

        T = len(clean_cache._store.get(activation_names[0], []))  # timesteps

        # -- Step 2: score all candidate nodes --------------------------------
        candidate_scores: Dict[Tuple[int, str], float] = {}

        for act_name in activation_names:
            clean_vals = clean_cache._store.get(act_name, [])
            if not clean_vals:
                continue
            for t in range(min(T, len(clean_vals))):
                recovery = self._patch_and_score(
                    wm,
                    corrupted_obs_seq,
                    corrupted_action_seq,
                    metric_fn,
                    metric_corrupted,
                    denom,
                    patch_spec=[(t, act_name, clean_vals[t])],
                )
                candidate_scores[(t, act_name)] = recovery

        # -- Step 3: greedy chain building ------------------------------------
        links: List[CausalLink] = []
        committed_patches: List[Tuple[int, str, "Tensor"]] = []
        prev_recovery = 0.0

        for _hop in range(n_hops):
            if not candidate_scores:
                break

            # Pick best remaining candidate
            best_node = max(candidate_scores, key=lambda k: candidate_scores[k])
            best_recovery = candidate_scores.pop(best_node)

            t_best, name_best = best_node
            clean_val = clean_cache._store[name_best][t_best]
            committed_patches.append((t_best, name_best, clean_val))

            # Add link from previous node (or "corrupted input" at hop 0)
            if links:
                src_t, src_n = links[-1].dst_timestep, links[-1].dst_activation
            else:
                src_t, src_n = -1, "corrupted_input"

            contrib = best_recovery - prev_recovery
            links.append(CausalLink(
                src_timestep=src_t, src_activation=src_n,
                dst_timestep=t_best, dst_activation=name_best,
                recovery_contribution=contrib,
            ))
            prev_recovery = best_recovery

            # Re-score remaining candidates with current committed patches
            # (only re-score if we have remaining budget)
            if _hop < n_hops - 1:
                candidate_scores = {}
                for act_name in activation_names:
                    clean_vals = clean_cache._store.get(act_name, [])
                    if not clean_vals:
                        continue
                    for t in range(min(T, len(clean_vals))):
                        if (t, act_name) in [(lk.dst_timestep, lk.dst_activation)
                                              for lk in links]:
                            continue
                        patches = committed_patches + [(t, act_name, clean_vals[t])]
                        recovery = self._patch_and_score(
                            wm, corrupted_obs_seq, corrupted_action_seq,
                            metric_fn, metric_corrupted, denom,
                            patch_spec=patches,
                        )
                        candidate_scores[(t, act_name)] = recovery

        return CausalChain(
            links=links,
            total_recovery=prev_recovery,
            metric_clean=metric_clean,
            metric_corrupted=metric_corrupted,
        )

    # ------------------------------------------------------------------ #

    def _patch_and_score(
        self,
        wm: "HookedWorldModel",
        corrupted_obs_seq: "Tensor",
        corrupted_action_seq: "Tensor",
        metric_fn: Callable,
        metric_corrupted: float,
        denom: float,
        patch_spec: List[Tuple[int, str, "Tensor"]],
    ) -> float:
        """Apply patches and compute recovery rate."""
        import torch

        if abs(denom) < 1e-10:
            return 0.0

        # Build patch hooks
        hook_fns = {}
        for t_patch, act_name, patch_val in patch_spec:
            existing = hook_fns.get(act_name, [])
            existing.append((t_patch, patch_val.detach().clone()))
            hook_fns[act_name] = existing

        # Convert to per-activation callable
        patch_hooks: Dict[str, Callable] = {}
        for act_name, patches_for_act in hook_fns.items():
            # Sort by timestep for efficient lookup
            patch_map = {t: v for t, v in patches_for_act}

            def make_hook(pm):
                counter = [0]
                def hook(value):
                    t = counter[0]
                    counter[0] += 1
                    if t in pm:
                        return pm[t]
                    return value
                return hook

            patch_hooks[act_name] = make_hook(patch_map)

        # Run patched forward
        try:
            _, patched_cache = wm.run_with_cache(
                corrupted_obs_seq,
                corrupted_action_seq,
                hooks=patch_hooks,
            )
            metric_patched = metric_fn(patched_cache)
        except Exception:
            return 0.0

        recovery = (metric_patched - metric_corrupted) / denom
        return float(np.clip(recovery, -1.0, 2.0))
