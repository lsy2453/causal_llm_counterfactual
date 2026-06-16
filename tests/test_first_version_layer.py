from __future__ import annotations

import numpy as np

from llm_contrastive_cbo import (
    ContrastPoolBuilder,
    ContrastValidator,
    ContrastiveDecisionLayer,
    HeuristicContrastSelector,
    InterventionCandidate,
    InterventionSpace,
    build_near_tie_pool,
)
from llm_contrastive_cbo.posterior import CallablePosteriorProvider


def test_near_tie_pool_delta() -> None:
    candidates = (
        InterventionCandidate(np.array([0.0]), 1.0, candidate_id="a"),
        InterventionCandidate(np.array([1.0]), 0.95, candidate_id="b"),
        InterventionCandidate(np.array([2.0]), 0.50, candidate_id="c"),
    )
    pool = build_near_tie_pool(candidates, delta=0.06, mode="max")
    assert [c.candidate_id for c in pool] == ["a", "b"]


def test_decision_layer_returns_near_tie_candidate() -> None:
    def mean_fn(x: np.ndarray) -> np.ndarray:
        return np.array([row[0] for row in x])

    def cov_fn(x: np.ndarray) -> np.ndarray:
        return np.eye(len(x)) * 0.1

    space = InterventionSpace(("x",), np.array([0.0]), np.array([1.0]))
    layer = ContrastiveDecisionLayer(
        posterior=CallablePosteriorProvider(mean_fn, cov_fn),
        pool_builder=ContrastPoolBuilder(space, local_step_fraction=0.2),
        validator=ContrastValidator(space),
        selector=HeuristicContrastSelector(max_selected=2),
        delta=0.02,
    )
    candidates = (
        InterventionCandidate(np.array([0.9]), 1.0, candidate_id="a"),
        InterventionCandidate(np.array([0.7]), 0.99, candidate_id="b"),
        InterventionCandidate(np.array([0.1]), 0.5, candidate_id="c"),
    )
    result = layer.choose(candidates)
    assert result.selected.candidate_id in {"a", "b"}
    assert result.contrast_results
    assert result.explanation_payload["method"] == "posterior_interventional_contrastive_reranking"
