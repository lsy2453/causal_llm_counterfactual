from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .types import InterventionCandidate


def candidates_from_cbo_outputs(
    x_new_list: Sequence[np.ndarray],
    acquisition_values: Sequence[float],
    exploration_sets: Sequence[Sequence[str]] | None = None,
) -> Tuple[InterventionCandidate, ...]:
    """Build decision-layer candidates from a standard CBO loop.

    This is designed for code like the original
    ``VirgiAgl/CausalBayesianOptimization`` implementation, where each
    exploration set has one optimized acquisition value and one proposed point.
    """

    if len(x_new_list) != len(acquisition_values):
        raise ValueError("x_new_list and acquisition_values must have the same length.")
    candidates = []
    for i, (x, value) in enumerate(zip(x_new_list, acquisition_values)):
        exploration_set = tuple(exploration_sets[i]) if exploration_sets is not None else tuple()
        candidates.append(
            InterventionCandidate(
                x=np.asarray(x, dtype=float).reshape(-1),
                acquisition_value=float(np.asarray(value).reshape(-1)[0]),
                exploration_set=exploration_set,
                candidate_id=f"cbo_candidate_{i}",
                metadata={"source_index": i},
            )
        )
    return tuple(candidates)


def selected_source_index(selected: InterventionCandidate) -> int:
    """Return the original CBO list index stored by ``candidates_from_cbo_outputs``."""

    if "source_index" not in selected.metadata:
        raise KeyError("Selected candidate does not contain a source_index metadata field.")
    return int(selected.metadata["source_index"])
