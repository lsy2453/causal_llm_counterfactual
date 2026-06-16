from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from ..types import InterventionSpace


@dataclass(frozen=True)
class InterventionSet:
    """One CBO exploration/intervention set."""

    name: str
    variables: Tuple[str, ...]
    space: InterventionSpace
    mechanism_tags: Tuple[str, ...] = field(default_factory=tuple)
    semantic_description: str = ""


@dataclass(frozen=True)
class EvaluationRecord:
    """One evaluated intervention in a benchmark run."""

    step: int
    intervention_set: str
    x: np.ndarray
    y: float
    selected_by: str


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_name: str
    method_name: str
    best_y: float
    best_x: np.ndarray
    best_intervention_set: str
    simple_regret: float
    best_found_curve: Tuple[float, ...]
    records: Tuple[EvaluationRecord, ...]
    extra_metrics: Mapping[str, object] = field(default_factory=dict)


class CausalBenchmark:
    """Base interface for first-version CBO benchmarks."""

    name: str
    task: str = "min"
    observational_variables: Tuple[str, ...] = tuple()

    def intervention_sets(self) -> Tuple[InterventionSet, ...]:
        raise NotImplementedError

    def sample_observational(self, n: int, seed: int = 0) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def target_mean(self, intervention_set: InterventionSet, x: Sequence[float]) -> float:
        """Return g(x)=E[Y | do(intervention_set=x)]."""
        raise NotImplementedError

    def mechanism_tags_for(self, intervention_set: InterventionSet, x: Sequence[float]) -> Tuple[str, ...]:
        return intervention_set.mechanism_tags

    def explanation_context(self) -> Mapping[str, object]:
        return {
            "benchmark": self.name,
            "task": self.task,
            "target": "Y",
            "target_objective": "minimize Y" if self.task == "min" else "maximize Y",
        }

    def grid(self, intervention_set: InterventionSet, points_per_dim: int = 9) -> np.ndarray:
        axes = [
            np.linspace(intervention_set.space.lower_bounds[i], intervention_set.space.upper_bounds[i], points_per_dim)
            for i in range(intervention_set.space.dim)
        ]
        mesh = np.meshgrid(*axes, indexing="xy")
        return np.vstack([m.reshape(-1) for m in mesh]).T

    def global_optimum(self, points_per_dim: int = 41) -> Tuple[str, np.ndarray, float]:
        best_name = ""
        best_x = np.array([])
        best_y = np.inf if self.task == "min" else -np.inf
        for intervention_set in self.intervention_sets():
            xs = self.grid(intervention_set, points_per_dim=points_per_dim)
            ys = np.array([self.target_mean(intervention_set, x) for x in xs], dtype=float)
            if self.task == "min":
                idx = int(np.argmin(ys))
                better = ys[idx] < best_y
            else:
                idx = int(np.argmax(ys))
                better = ys[idx] > best_y
            if better:
                best_name = intervention_set.name
                best_x = xs[idx]
                best_y = float(ys[idx])
        return best_name, best_x, best_y

    def simple_regret(self, best_found_y: float, optimum_y: float) -> float:
        if self.task == "min":
            return float(best_found_y - optimum_y)
        return float(optimum_y - best_found_y)
