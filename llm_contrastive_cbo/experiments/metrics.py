from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

from ..benchmarks.base import CausalBenchmark, EvaluationRecord
from ..types import ContrastResult


def best_found_curve(records: Sequence[EvaluationRecord], task: str) -> Tuple[float, ...]:
    curve = []
    best = np.inf if task == "min" else -np.inf
    for record in records:
        if task == "min":
            best = min(best, record.y)
        else:
            best = max(best, record.y)
        curve.append(float(best))
    return tuple(curve)


def contrast_validity_rate(num_valid: int, num_total: int) -> float:
    if num_total == 0:
        return 0.0
    return float(num_valid / num_total)


def mechanism_coverage(
    benchmark: CausalBenchmark,
    records: Sequence[EvaluationRecord],
) -> float:
    covered = set()
    possible = set()
    sets = {item.name: item for item in benchmark.intervention_sets()}
    for item in sets.values():
        possible.update(item.mechanism_tags)
    for record in records:
        covered.update(benchmark.mechanism_tags_for(sets[record.intervention_set], record.x))
    if not possible:
        return 0.0
    return float(len(covered.intersection(possible)) / len(possible))


def explanation_fidelity_proxy(results: Iterable[ContrastResult]) -> float:
    """Proxy: fraction of explanation contrasts with finite computed stats."""

    items = list(results)
    if not items:
        return 0.0
    finite = [
        np.isfinite(r.mean_delta)
        and np.isfinite(r.variance_delta)
        and np.isfinite(r.discriminability)
        for r in items
    ]
    return float(np.mean(finite))
