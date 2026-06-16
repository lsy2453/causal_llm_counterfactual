"""Posterior interventional contrastive reranking for CBO.

This package implements the first, conservative version of the proposed
method: LLM-guided feasible contrast selection inside a near-tie CBO
candidate pool. It intentionally does not claim full SCM counterfactual
inference. The numerical object is a posterior interventional contrast,
typically g(a) - g(b), where g(x) = E[Y | do(X=x)].
"""

from .candidate_pool import build_near_tie_pool
from .contrast_pool import ContrastPoolBuilder
from .integration import candidates_from_cbo_outputs, selected_source_index
from .llm_selectors import (
    HeuristicContrastSelector,
    LLMContrastSelector,
    OpenAICompatibleContrastSelector,
    dashscope_contrast_selector,
    kimi_contrast_selector,
)
from .reranker import ContrastiveDecisionLayer
from .types import (
    ContrastCandidate,
    ContrastResult,
    DecisionLayerResult,
    InterventionCandidate,
    InterventionSpace,
)
from .validators import ContrastValidator

__all__ = [
    "ContrastCandidate",
    "ContrastPoolBuilder",
    "ContrastResult",
    "ContrastValidator",
    "ContrastiveDecisionLayer",
    "DecisionLayerResult",
    "HeuristicContrastSelector",
    "InterventionCandidate",
    "InterventionSpace",
    "LLMContrastSelector",
    "OpenAICompatibleContrastSelector",
    "build_near_tie_pool",
    "candidates_from_cbo_outputs",
    "dashscope_contrast_selector",
    "kimi_contrast_selector",
    "selected_source_index",
]
