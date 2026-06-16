from __future__ import annotations

import numpy as np

from llm_contrastive_cbo.benchmarks import CompleteGraphBenchmark, PSAHealthBenchmark, ToyGraphBenchmark
from llm_contrastive_cbo.experiments import ExperimentConfig, run_first_version_benchmark


def test_benchmark_do_means_are_finite() -> None:
    for benchmark in (ToyGraphBenchmark(), CompleteGraphBenchmark(), PSAHealthBenchmark()):
        for intervention_set in benchmark.intervention_sets():
            x = (intervention_set.space.lower_bounds + intervention_set.space.upper_bounds) / 2.0
            assert np.isfinite(benchmark.target_mean(intervention_set, x))


def test_first_version_runner_smoke() -> None:
    result = run_first_version_benchmark(
        ExperimentConfig(
            benchmark="toy",
            method="contrastive",
            num_trials=2,
            initial_points_per_set=2,
            candidate_grid_points=5,
            optimum_grid_points=11,
        )
    )
    assert result.records
    assert np.isfinite(result.simple_regret)
