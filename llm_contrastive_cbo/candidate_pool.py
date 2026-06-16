from __future__ import annotations

from typing import Iterable, Literal, Sequence, Tuple

import numpy as np

from .types import InterventionCandidate


def build_near_tie_pool(
    candidates: Sequence[InterventionCandidate],
    delta: float | None = None,
    relative_epsilon: float | None = None,
    top_k: int | None = None,
    mode: Literal["max", "min"] = "max",
) -> Tuple[InterventionCandidate, ...]:
    """Return candidates close to the best base CBO acquisition value.

    Use ``delta`` for an additive near-tie rule:
    max mode: alpha(x) >= alpha* - delta
    min mode: alpha(x) <= alpha* + delta

    ``relative_epsilon`` supports the paper-style multiplicative rule. ``top_k``
    is applied last as a practical cap.
    """

    if not candidates:
        raise ValueError("At least one candidate is required.")
    if delta is None and relative_epsilon is None and top_k is None:
        top_k = min(5, len(candidates))
    reverse = mode == "max"
    ordered = sorted(candidates, key=lambda c: c.acquisition_value, reverse=reverse)
    best_value = ordered[0].acquisition_value

    pool: Iterable[InterventionCandidate] = ordered
    if delta is not None:
        if delta < 0:
            raise ValueError("delta must be non-negative.")
        if mode == "max":
            pool = [c for c in pool if c.acquisition_value >= best_value - delta]
        else:
            pool = [c for c in pool if c.acquisition_value <= best_value + delta]

    if relative_epsilon is not None:
        if relative_epsilon < 0:
            raise ValueError("relative_epsilon must be non-negative.")
        if np.isclose(best_value, 0.0):
            threshold = relative_epsilon
            if mode == "max":
                pool = [c for c in pool if c.acquisition_value >= best_value - threshold]
            else:
                pool = [c for c in pool if c.acquisition_value <= best_value + threshold]
        elif mode == "max":
            pool = [c for c in pool if c.acquisition_value >= (1.0 - relative_epsilon) * best_value]
        else:
            pool = [c for c in pool if c.acquisition_value <= (1.0 + relative_epsilon) * best_value]

    result = tuple(pool)
    if top_k is not None:
        result = result[:top_k]
    return result
