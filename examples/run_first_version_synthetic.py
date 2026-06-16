"""Minimal example for the first-version contrastive CBO decision layer.

This example uses a lightweight analytic posterior so it can run without the
original CBO benchmark data. In an actual CBO loop, replace the posterior
provider with ``BoTorchPosteriorProvider(model)`` and pass the CBO acquisition
candidates from the base optimizer.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_contrastive_cbo import (
    ContrastPoolBuilder,
    ContrastValidator,
    ContrastiveDecisionLayer,
    HeuristicContrastSelector,
    InterventionCandidate,
    InterventionSpace,
)
from llm_contrastive_cbo.posterior import CallablePosteriorProvider


def mean_fn(x: np.ndarray) -> np.ndarray:
    return -((x[:, 0] - 0.7) ** 2) - 0.5 * ((x[:, 1] - 0.2) ** 2)


def cov_fn(x: np.ndarray) -> np.ndarray:
    sqdist = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
    return 0.05 * np.exp(-sqdist / 0.2) + 1e-6 * np.eye(len(x))


def main() -> None:
    space = InterventionSpace(
        variable_names=("temperature", "catalyst_ratio"),
        lower_bounds=np.array([0.0, 0.0]),
        upper_bounds=np.array([1.0, 1.0]),
    )
    posterior = CallablePosteriorProvider(mean_fn=mean_fn, covariance_fn=cov_fn)
    layer = ContrastiveDecisionLayer(
        posterior=posterior,
        pool_builder=ContrastPoolBuilder(space),
        validator=ContrastValidator(space),
        selector=HeuristicContrastSelector(max_selected=2),
        delta=0.02,
        top_k=4,
    )
    candidates = (
        InterventionCandidate(np.array([0.68, 0.22]), 1.00, candidate_id="A"),
        InterventionCandidate(np.array([0.75, 0.20]), 0.99, candidate_id="B"),
        InterventionCandidate(np.array([0.30, 0.70]), 0.50, candidate_id="C"),
    )
    result = layer.choose(
        candidates,
        historical_x=np.array([[0.5, 0.2], [0.7, 0.4], [0.2, 0.8]]),
        historical_y=np.array([-0.04, -0.02, -0.80]),
        lower_cost_direction={"temperature": -1.0, "catalyst_ratio": -1.0},
    )
    print("Base CBO best:", result.base_best.candidate_id, result.base_best.x)
    print("Selected:", result.selected.candidate_id, result.selected.x)
    print("Explanation payload:", result.explanation_payload)


if __name__ == "__main__":
    main()
