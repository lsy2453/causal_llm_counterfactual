from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import ContrastCandidate, InterventionCandidate, InterventionSpace


@dataclass
class ContrastPoolBuilder:
    """Construct feasible numerical contrast pools before LLM selection.

    The LLM should select from this pool rather than freely inventing values.
    This keeps the first version reproducible and compatible with CBO domains.
    """

    space: InterventionSpace
    local_step_fraction: float = 0.15
    include_local: bool = True
    include_history: bool = True
    include_posterior: bool = True
    include_cost_reduction: bool = True
    max_pool_size: int = 64

    def build(
        self,
        candidate: InterventionCandidate,
        historical_x: Optional[Sequence[Sequence[float]]] = None,
        historical_y: Optional[Sequence[float]] = None,
        posterior_reference_x: Optional[Sequence[Sequence[float]]] = None,
        posterior_uncertainty: Optional[Sequence[float]] = None,
        lower_cost_direction: Optional[Mapping[str, float]] = None,
    ) -> Tuple[ContrastCandidate, ...]:
        contrasts = []
        if self.include_local:
            contrasts.extend(self._local_perturbations(candidate))
        if self.include_history and historical_x is not None:
            contrasts.extend(self._history_contrasts(candidate, historical_x, historical_y))
        if self.include_posterior and posterior_reference_x is not None:
            contrasts.extend(
                self._posterior_contrasts(candidate, posterior_reference_x, posterior_uncertainty)
            )
        if self.include_cost_reduction and lower_cost_direction:
            contrasts.extend(self._cost_reduction_contrasts(candidate, lower_cost_direction))

        unique = {}
        for contrast in contrasts:
            projected = self.space.project(contrast.x)
            if np.allclose(projected, candidate.x):
                continue
            key = tuple(np.round(projected, 10))
            if key not in unique:
                unique[key] = ContrastCandidate(
                    x=projected,
                    contrast_type=contrast.contrast_type,
                    changed_variables=contrast.changed_variables,
                    source=contrast.source,
                    rationale=contrast.rationale,
                    metadata=contrast.metadata,
                )
        return tuple(list(unique.values())[: self.max_pool_size])

    def _local_perturbations(self, candidate: InterventionCandidate) -> Iterable[ContrastCandidate]:
        span = self.space.upper_bounds - self.space.lower_bounds
        step = np.maximum(span * self.local_step_fraction, 1e-12)
        for i, name in enumerate(self.space.variable_names):
            for sign, label in [(-1.0, "decrease"), (1.0, "increase")]:
                x = candidate.x.copy()
                x[i] += sign * step[i]
                yield ContrastCandidate(
                    x=self.space.project(x),
                    contrast_type=f"single_variable_{label}",
                    changed_variables=(name,),
                    source="local",
                    rationale=f"Isolate the marginal role of {name}.",
                )

    def _history_contrasts(
        self,
        candidate: InterventionCandidate,
        historical_x: Sequence[Sequence[float]],
        historical_y: Optional[Sequence[float]],
    ) -> Iterable[ContrastCandidate]:
        xs = np.asarray(historical_x, dtype=float)
        if xs.size == 0:
            return []
        if historical_y is not None and len(historical_y) == len(xs):
            order = np.argsort(np.asarray(historical_y, dtype=float).reshape(-1))
        else:
            distance = np.linalg.norm(xs - candidate.x.reshape(1, -1), axis=1)
            order = np.argsort(distance)
        result = []
        for idx in order[: min(5, len(order))]:
            x = self.space.project(xs[idx])
            result.append(
                ContrastCandidate(
                    x=x,
                    contrast_type="historical_comparison",
                    changed_variables=self._changed_variables(candidate.x, x),
                    source="history",
                    rationale="Compare against a historically informative intervention.",
                )
            )
        return result

    def _posterior_contrasts(
        self,
        candidate: InterventionCandidate,
        posterior_reference_x: Sequence[Sequence[float]],
        posterior_uncertainty: Optional[Sequence[float]],
    ) -> Iterable[ContrastCandidate]:
        xs = np.asarray(posterior_reference_x, dtype=float)
        if xs.size == 0:
            return []
        if posterior_uncertainty is not None and len(posterior_uncertainty) == len(xs):
            order = np.argsort(-np.asarray(posterior_uncertainty, dtype=float).reshape(-1))
        else:
            distance = np.linalg.norm(xs - candidate.x.reshape(1, -1), axis=1)
            order = np.argsort(distance)
        result = []
        for idx in order[: min(5, len(order))]:
            x = self.space.project(xs[idx])
            result.append(
                ContrastCandidate(
                    x=x,
                    contrast_type="uncertainty_driven_contrast",
                    changed_variables=self._changed_variables(candidate.x, x),
                    source="posterior",
                    rationale="Probe a posterior-relevant alternative intervention.",
                )
            )
        return result

    def _cost_reduction_contrasts(
        self,
        candidate: InterventionCandidate,
        lower_cost_direction: Mapping[str, float],
    ) -> Iterable[ContrastCandidate]:
        span = self.space.upper_bounds - self.space.lower_bounds
        result = []
        for i, name in enumerate(self.space.variable_names):
            if name not in lower_cost_direction:
                continue
            x = candidate.x.copy()
            x[i] += np.sign(lower_cost_direction[name]) * self.local_step_fraction * span[i]
            x = self.space.project(x)
            result.append(
                ContrastCandidate(
                    x=x,
                    contrast_type="dose_or_cost_reduction",
                    changed_variables=(name,),
                    source="cost",
                    rationale=f"Check whether a lower-cost setting of {name} is sufficient.",
                )
            )
        return result

    def _changed_variables(self, a: np.ndarray, b: np.ndarray) -> Tuple[str, ...]:
        return tuple(
            name
            for i, name in enumerate(self.space.variable_names)
            if not np.isclose(float(a[i]), float(b[i]))
        )
