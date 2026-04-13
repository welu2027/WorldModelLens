from __future__ import annotations

import contextlib
import uuid
from typing import Any, Optional, Set

import torch
from torch import Tensor

from world_model_lens.core.hooks import HookContext
from world_model_lens.core.activation_cache import ActivationCache
from world_model_lens.core.latent_trajectory import LatentTrajectory


class ForwardRunner:
    """Isolated runner that performs the forward pass for a HookedWorldModel.

    This class delegates computations to the provided *hooked* object which is
    expected to implement the small helpers used by the forward loop (for
    example: _encode, _dynamics_and_prior, _posterior, _compute_kl_and_cache,
    _compute_heads, _build_state, _apply_and_cache, and _cache_prior_equivalent_at_t0).

    Keeping the loop here makes it easier to test the sequencing and to reuse
    the runner from different entry points (run_with_cache, run_with_hooks,
    etc.) without duplicating control-flow.
    """

    def __init__(self, hooked: Any):
        self.hooked = hooked

    def run_forward(
        self,
        obs_seq: Tensor,
        action_seq: Tensor,
        cache: Optional[ActivationCache],
        names_filter: Optional[Set[str]],
        no_grad: bool = True,
    ) -> LatentTrajectory:
        T = obs_seq.shape[0]
        ctx_mgr = torch.no_grad() if no_grad else contextlib.nullcontext()

        states = []
        h: Tensor = self.hooked._adapter.initial_state()
        z: Tensor | None = None

        manager = getattr(self.hooked, "_hook_cache_manager", None)

        with ctx_mgr:
            for t in range(T):
                traj_so_far = (
                    LatentTrajectory(states=list(states), env_name="", episode_id=None)
                    if states
                    else None
                )
                ctx = (
                    HookContext(timestep=t, component="forward", trajectory_so_far=[])
                    if traj_so_far is None
                    else HookContext(
                        timestep=t, component="forward", trajectory_so_far=list(states)
                    )
                )

                if manager is not None:
                    obs_emb = self.hooked._adapter.encode(obs_seq[t])
                    obs_emb = manager.apply_and_cache(
                        "encoder.out", t, obs_emb, ctx, cache, names_filter
                    )
                else:
                    obs_emb = self.hooked._encode(obs_seq[t], t, ctx, cache, names_filter)

                if t == 0:
                    h = self.hooked._apply_and_cache("rnn.h", 0, h, ctx, cache, names_filter)
                    z_prior_logits = None
                else:
                    a_prev = action_seq[t - 1]
                    h, z_prior_raw, z_prior_prob = self.hooked._dynamics_and_prior(
                        h, z, a_prev, t, ctx, cache, names_filter
                    )
                    z_prior_logits = z_prior_raw

                z_post_raw, z_post_prob = self.hooked._posterior(
                    h, obs_emb, t, ctx, cache, names_filter
                )
                z = z_post_prob

                if t == 0:
                    z_prior_logits = z_post_raw
                    z_prior_prob = z_post_prob
                    # prefer HookCacheManager helper when available
                    manager = getattr(self.hooked, "_hook_cache_manager", None)
                    if manager is not None:
                        manager.set_prior_equivalent(
                            cache,
                            0,
                            z_post_raw.detach(),
                            z_post_prob.detach(),
                            names_filter,
                        )
                    else:
                        # fall back to cache helper
                        try:
                            if cache is not None:
                                cache.set_prior_equivalent(
                                    0,
                                    z_post_raw.detach(),
                                    z_post_prob.detach(),
                                    list(names_filter) if names_filter is not None else None,
                                )
                        except Exception:
                            # last-resort fallback to hooked shim
                            self.hooked._cache_prior_equivalent_at_t0(
                                z_post_raw, z_post_prob, 0, cache, names_filter
                            )

                kl = self.hooked._compute_kl_and_cache(
                    z_post_prob, z_prior_prob, t, ctx, cache, names_filter
                )

                reward_val, cont_val, actor_logits_out, value_val = self.hooked._compute_heads(
                    h, z_post_prob, t, ctx, cache, names_filter
                )

                state = self.hooked._build_state(
                    h,
                    z_post_prob,
                    z_prior_prob,
                    t,
                    action_seq,
                    reward_val,
                    cont_val,
                    actor_logits_out,
                    value_val,
                )
                states.append(state)

        return LatentTrajectory(
            states=states,
            env_name=self.hooked.name,
            episode_id=f"run_{uuid.uuid4().hex[:8]}",
            imagined=False,
        )
