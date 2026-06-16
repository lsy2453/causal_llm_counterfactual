from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .candidate_pool import build_near_tie_pool
from .contrast_pool import ContrastPoolBuilder
from .llm_selectors import HeuristicContrastSelector, LLMContrastSelector
from .posterior import PosteriorProvider, posterior_interventional_contrast
from .scoring import ContrastScoreConfig, raw_score
from .types import ContrastResult, DecisionLayerResult, InterventionCandidate
from .validators import ContrastValidator


@dataclass
class ContrastiveDecisionLayer:
    """Conservative first-version decision layer for CBO.

    This layer reranks only candidates that are already competitive under the
    base CBO acquisition. It computes posterior interventional contrasts, not
    full individualized SCM counterfactuals.
    """

    posterior: PosteriorProvider
    pool_builder: ContrastPoolBuilder
    validator: ContrastValidator
    selector: LLMContrastSelector = field(default_factory=HeuristicContrastSelector)
    score_config: ContrastScoreConfig = field(default_factory=ContrastScoreConfig)
    delta: Optional[float] = None
    relative_epsilon: Optional[float] = None
    top_k: Optional[int] = 5
    acquisition_mode: str = "max"

    def choose(
        self,
        candidates: Sequence[InterventionCandidate],
        historical_x: Optional[Sequence[Sequence[float]]] = None,
        historical_y: Optional[Sequence[float]] = None,
        posterior_reference_x: Optional[Sequence[Sequence[float]]] = None,
        posterior_uncertainty: Optional[Sequence[float]] = None,
        lower_cost_direction: Optional[Mapping[str, float]] = None,
        context: Optional[Mapping[str, object]] = None,
    ) -> DecisionLayerResult:
        near_tie = build_near_tie_pool(
            candidates,
            delta=self.delta,
            relative_epsilon=self.relative_epsilon,
            top_k=self.top_k,
            mode=self.acquisition_mode,  # type: ignore[arg-type]
        )
        base_best = near_tie[0]
        all_results = []
        best_candidate = base_best
        best_score = -np.inf

        for candidate in near_tie:
            pool = self.pool_builder.build(
                candidate,
                historical_x=historical_x,
                historical_y=historical_y,
                posterior_reference_x=posterior_reference_x,
                posterior_uncertainty=posterior_uncertainty,
                lower_cost_direction=lower_cost_direction,
            )
            valid_pool = self.validator.filter(candidate, pool)
            selected_contrasts = self.selector.select(candidate, valid_pool, context=context)
            candidate_results = []
            for contrast in selected_contrasts:
                mean_delta, var_delta, p_positive = posterior_interventional_contrast(
                    self.posterior,
                    candidate.x,
                    contrast.x,
                )
                disc, unc, score = raw_score(mean_delta, var_delta, self.score_config)
                result = ContrastResult(
                    candidate=candidate,
                    contrast=contrast,
                    mean_delta=mean_delta,
                    variance_delta=var_delta,
                    probability_positive=p_positive,
                    discriminability=disc,
                    uncertainty=unc,
                    score=score,
                )
                candidate_results.append(result)
                all_results.append(result)

            candidate_score = max((r.score for r in candidate_results), default=-np.inf)
            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = candidate

        if not np.isfinite(best_score):
            best_candidate = base_best
            best_score = 0.0

        return DecisionLayerResult(
            selected=best_candidate,
            base_best=base_best,
            near_tie_candidates=near_tie,
            contrast_results=tuple(all_results),
            selected_score=float(best_score),
            explanation_payload=self._build_explanation_payload(best_candidate, base_best, all_results),
        )

    def _build_explanation_payload(
        self,
        selected: InterventionCandidate,
        base_best: InterventionCandidate,
        results: Sequence[ContrastResult],
    ) -> dict:
        selected_results = [r for r in results if r.candidate is selected]
        selected_results = sorted(selected_results, key=lambda r: r.score, reverse=True)
        return {
            "method": "posterior_interventional_contrastive_reranking",
            "selected_candidate_id": selected.candidate_id,
            "base_best_candidate_id": base_best.candidate_id,
            "selected_x": selected.x.tolist(),
            "base_best_x": base_best.x.tolist(),
            "selected_score": selected_results[0].score if selected_results else 0.0,
            "top_contrasts": [
                {
                    "contrast_x": r.contrast.x.tolist(),
                    "contrast_type": r.contrast.contrast_type,
                    "mean_delta": r.mean_delta,
                    "variance_delta": r.variance_delta,
                    "probability_positive": r.probability_positive,
                    "discriminability": r.discriminability,
                    "rationale": r.contrast.rationale,
                }
                for r in selected_results[:3]
            ],
        }
