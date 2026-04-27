"""Tests for additional adapter hookable-point introspection."""

from __future__ import annotations

import pytest

from world_model_lens.backends.base_adapter import AdapterConfig
from world_model_lens.backends.iris import IRISAdapter
from world_model_lens.backends.planet import PlaNetAdapter
from world_model_lens.backends.tdmpc2 import TDMPC2Adapter
from world_model_lens.core.config import WorldModelConfig


def _cfg() -> AdapterConfig:
    return AdapterConfig(
        d_h=8,
        d_obs=16,
        d_action=4,
        n_categories=2,
        n_classes=4,
        encoder_type="vector",
    )


@pytest.mark.parametrize(
    "adapter_cls, expected_semantic, expected_modules, kwargs, config_factory",
    [
        (
            PlaNetAdapter,
            {"encoder", "dynamics_gru", "posterior", "reward", "state", "prior"},
            {"encoder.conv.0", "encoder.fc", "dynamics_model.gru", "reward_predictor.fc.0"},
            {},
            _cfg,
        ),
        (
            TDMPC2Adapter,
            {"h", "z_posterior", "z_prior", "reward_pred", "cont_pred", "actor_logits", "value_pred"},
            {"encoder.net.0", "dynamics_model.net.0", "policy.mean", "value.net.0"},
            {},
            _cfg,
        ),
        (
            IRISAdapter,
            {"h", "z_posterior", "z_prior", "reward_pred", "cont_pred", "transformer_hidden_0"},
            {"encoder.encoder.0", "transformer.blocks.0.attn", "reward_head.fc.0", "continue_head.fc"},
            {"d_model": 16, "n_layers": 2, "n_head": 2, "vocab_size": 32},
            lambda: WorldModelConfig(
                d_h=8,
                d_obs=16,
                d_action=4,
                n_cat=2,
                n_cls=4,
                backend="iris",
                d_embed=16,
                n_layers=2,
                n_heads=2,
                vocab_size=32,
            ),
        ),
    ],
)
def test_additional_adapters_expose_semantic_and_module_names(
    adapter_cls,
    expected_semantic,
    expected_modules,
    kwargs,
    config_factory,
):
    adapter = adapter_cls(config_factory(), **kwargs)

    points = adapter.list_hookable_points()

    assert "" not in points
    assert len(points) == len(set(points))
    assert expected_semantic.issubset(points)
    assert expected_modules.issubset(points)
