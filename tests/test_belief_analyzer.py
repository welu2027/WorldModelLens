"""Tests for the analysis module."""

import torch
import pytest

from world_model_lens import HookedWorldModel, WorldModelConfig
from world_model_lens.analysis.belief_analyzer import BeliefAnalyzer
from tests.conftest import SimpleTestAdapter


class TestBeliefAnalyzer:
    """Tests for BeliefAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a belief analyzer."""
        config = WorldModelConfig(d_h=32, d_action=4, d_obs=64)
        adapter = SimpleTestAdapter(config)
        wm = HookedWorldModel(adapter=adapter, config=config)
        return BeliefAnalyzer(wm)

    def test_analyzer_creation(self, analyzer):
        """Test analyzer creates correctly."""
        assert analyzer is not None

    def test_capabilities_property(self, analyzer):
        """Test capabilities property."""
        caps = analyzer.capabilities
        assert caps is not None

    def test_has_surprise_timeline_method(self, analyzer):
        """Test surprise_timeline method exists."""
        assert hasattr(analyzer, "surprise_timeline")
        assert callable(analyzer.surprise_timeline)

    def test_has_concept_search_method(self, analyzer):
        """Test concept_search method exists."""
        assert hasattr(analyzer, "concept_search")
        assert callable(analyzer.concept_search)

    def test_has_latent_saliency_method(self, analyzer):
        """Test latent_saliency method exists."""
        assert hasattr(analyzer, "latent_saliency")
        assert callable(analyzer.latent_saliency)

    def test_has_disentanglement_score_method(self, analyzer):
        """Test disentanglement_score method exists."""
        assert hasattr(analyzer, "disentanglement_score")
        assert callable(analyzer.disentanglement_score)
