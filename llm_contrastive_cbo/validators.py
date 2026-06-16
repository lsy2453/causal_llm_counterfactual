from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .types import ContrastCandidate, InterventionCandidate, InterventionSpace


ConstraintFn = Callable[[np.ndarray], bool]


@dataclass
class ContrastValidator:
    """Deterministic guardrail for LLM-selected contrasts."""

    space: InterventionSpace
    min_l2_distance: float = 1e-8
    max_l2_distance: Optional[float] = None
    domain_constraint: Optional[ConstraintFn] = None

    def validate(
        self,
        candidate: InterventionCandidate,
        contrast: ContrastCandidate,
    ) -> Tuple[bool, str]:
        if contrast.x.shape != candidate.x.shape:
            return False, "dimension_mismatch"
        if not self.space.contains(contrast.x):
            return False, "outside_intervention_space"
        if self.domain_constraint is not None and not self.domain_constraint(contrast.x):
            return False, "violates_domain_constraint"
        distance = float(np.linalg.norm(candidate.x - contrast.x))
        if distance < self.min_l2_distance:
            return False, "identical_or_too_close"
        if self.max_l2_distance is not None and distance > self.max_l2_distance:
            return False, "too_far_from_candidate"
        actual_changed = {
            name
            for i, name in enumerate(self.space.variable_names)
            if not np.isclose(candidate.x[i], contrast.x[i])
        }
        if set(contrast.changed_variables) and not set(contrast.changed_variables).issubset(actual_changed):
            return False, "rationale_changed_variables_inconsistent"
        return True, "ok"

    def filter(
        self,
        candidate: InterventionCandidate,
        contrasts: Tuple[ContrastCandidate, ...],
    ) -> Tuple[ContrastCandidate, ...]:
        return tuple(c for c in contrasts if self.validate(candidate, c)[0])
