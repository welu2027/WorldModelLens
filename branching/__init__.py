"""Imagination branching and counterfactual analysis."""

from world_model_lens.branching.brancher import (
    BehaviorComparison,
    BranchCollection,
    ImaginationBrancher,
    UncertaintyResult,
)
from world_model_lens.branching.counterfactual import (
    CounterfactualAnalyzer,
    CounterfactualReport,
)

__all__ = [
    "ImaginationBrancher",
    "BranchCollection",
    "BehaviorComparison",
    "UncertaintyResult",
    "CounterfactualAnalyzer",
    "CounterfactualReport",
]
