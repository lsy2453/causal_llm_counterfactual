from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class InterventionSpace:
    """Numerical intervention domain for one exploration set.

    The first version assumes a numeric box domain. Discrete values can be
    represented through ``allowed_values`` and are projected by nearest value.
    """

    variable_names: Tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    allowed_values: Optional[Mapping[str, Sequence[float]]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower_bounds", np.asarray(self.lower_bounds, dtype=float))
        object.__setattr__(self, "upper_bounds", np.asarray(self.upper_bounds, dtype=float))
        if len(self.variable_names) != len(self.lower_bounds):
            raise ValueError("variable_names and bounds must have the same dimension.")
        if len(self.upper_bounds) != len(self.lower_bounds):
            raise ValueError("lower_bounds and upper_bounds must have the same dimension.")

    @property
    def dim(self) -> int:
        return len(self.variable_names)

    def contains(self, x: ArrayLike, atol: float = 1e-9) -> bool:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.shape[0] != self.dim:
            return False
        if np.any(arr < self.lower_bounds - atol) or np.any(arr > self.upper_bounds + atol):
            return False
        if not self.allowed_values:
            return True
        for i, name in enumerate(self.variable_names):
            if name in self.allowed_values:
                vals = np.asarray(self.allowed_values[name], dtype=float)
                if not np.any(np.isclose(arr[i], vals, atol=atol)):
                    return False
        return True

    def project(self, x: ArrayLike) -> np.ndarray:
        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.shape[0] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {arr.shape[0]}.")
        arr = np.minimum(np.maximum(arr, self.lower_bounds), self.upper_bounds)
        if self.allowed_values:
            arr = arr.copy()
            for i, name in enumerate(self.variable_names):
                if name in self.allowed_values:
                    vals = np.asarray(self.allowed_values[name], dtype=float)
                    arr[i] = vals[np.argmin(np.abs(vals - arr[i]))]
        return arr

    def to_dict(self, x: ArrayLike) -> Dict[str, float]:
        arr = np.asarray(x, dtype=float).reshape(-1)
        return {name: float(arr[i]) for i, name in enumerate(self.variable_names)}


@dataclass(frozen=True)
class InterventionCandidate:
    """A CBO candidate with its base acquisition value."""

    x: np.ndarray
    acquisition_value: float
    exploration_set: Tuple[str, ...] = field(default_factory=tuple)
    candidate_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", np.asarray(self.x, dtype=float).reshape(-1))


@dataclass(frozen=True)
class ContrastCandidate:
    """A feasible numerical contrast for a candidate intervention."""

    x: np.ndarray
    contrast_type: str
    changed_variables: Tuple[str, ...]
    source: str
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", np.asarray(self.x, dtype=float).reshape(-1))


@dataclass(frozen=True)
class ContrastResult:
    """Posterior contrast statistics for g(a) - g(b)."""

    candidate: InterventionCandidate
    contrast: ContrastCandidate
    mean_delta: float
    variance_delta: float
    probability_positive: Optional[float]
    discriminability: float
    uncertainty: float
    score: float
    probability_candidate_better: Optional[float] = None
    hardness_weight: float = 1.0
    semantic_relevance_weight: float = 1.0


@dataclass(frozen=True)
class DecisionLayerResult:
    """Output of the contrastive decision layer."""

    selected: InterventionCandidate
    base_best: InterventionCandidate
    near_tie_candidates: Tuple[InterventionCandidate, ...]
    contrast_results: Tuple[ContrastResult, ...]
    selected_score: float
    explanation_payload: Dict[str, Any]


def as_2d(x: ArrayLike | Iterable[ArrayLike]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr
