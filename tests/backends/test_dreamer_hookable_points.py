"""Tests for Dreamer adapter hookable-point introspection."""

from __future__ import annotations

import pytest

from world_model_lens.backends.base_adapter import AdapterConfig
from world_model_lens.backends.dreamerv1 import DreamerV1Adapter
from world_model_lens.backends.dreamerv2 import DreamerV2Adapter
from world_model_lens.backends.dreamerv3 import DreamerV3Adapter


def _tiny_config(*, encoder_type: str = "vector") -> AdapterConfig:
    return AdapterConfig(
        d_h=8,
        d_obs=16,
        d_action=4,
        n_categories=2,
        n_classes=4,
        encoder_type=encoder_type,
    )


@pytest.mark.parametrize(
    "adapter_cls, expected_semantic, expected_modules",
    [
        (
            DreamerV1Adapter,
            {"encoder", "decoder", "dynamics_gru", "posterior", "reward", "value", "actor"},
            {"encoder.conv.0", "decoder.deconv.1", "dynamics_model.gru", "posterior.fc.0"},
        ),
        (
            DreamerV2Adapter,
            {
                "h",
                "z_posterior",
                "z_prior",
                "reward_pred",
                "cont_pred",
                "actor_logits",
                "value_pred",
                "obs_reconstruction",
            },
            {"encoder.vector.mlp.layers.0", "dynamics.mlp.layers.0", "transition_layer.gru"},
        ),
        (
            DreamerV3Adapter,
            {
                "h",
                "z_posterior",
                "z_prior",
                "reward_pred",
                "cont_pred",
                "actor_logits",
                "value_pred",
                "obs_reconstruction",
            },
            {"encoder.vector.mlp.layers.0", "dynamics_model.mlp.layers.0", "transition_layer.gru"},
        ),
    ],
)
def test_dreamer_list_hookable_points_exposes_semantic_and_module_names(
    adapter_cls,
    expected_semantic,
    expected_modules,
):
    adapter = adapter_cls(_tiny_config())

    points = adapter.list_hookable_points()

    assert "" not in points
    assert len(points) == len(set(points))
    assert expected_semantic.issubset(points)
    assert expected_modules.issubset(points)
