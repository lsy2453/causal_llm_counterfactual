from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class PosteriorProvider(ABC):
    """Provides joint posterior statistics of g(x)=E[Y|do(X=x)]."""

    @abstractmethod
    def mean_and_covariance(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return posterior mean vector and covariance matrix at x."""
        raise NotImplementedError


@dataclass
class CallablePosteriorProvider(PosteriorProvider):
    """Small adapter for tests and custom surrogate implementations."""

    mean_fn: object
    covariance_fn: object

    def mean_and_covariance(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        mean = np.asarray(self.mean_fn(x), dtype=float).reshape(-1)
        cov = np.asarray(self.covariance_fn(x), dtype=float)
        return mean, cov


def posterior_interventional_contrast(
    posterior: PosteriorProvider,
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, float, Optional[float]]:
    """Compute posterior stats for Delta = g(a) - g(b)."""

    points = np.vstack([np.asarray(a, dtype=float).reshape(1, -1), np.asarray(b, dtype=float).reshape(1, -1)])
    mean, cov = posterior.mean_and_covariance(points)
    if mean.shape[0] != 2 or cov.shape != (2, 2):
        raise ValueError("Posterior provider must return joint stats for exactly two points.")
    mean_delta = float(mean[0] - mean[1])
    variance_delta = float(max(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1], 0.0))
    probability_positive = None
    if variance_delta > 1e-14:
        try:
            from math import erf, sqrt

            z = mean_delta / sqrt(variance_delta)
            probability_positive = 0.5 * (1.0 + erf(z / sqrt(2.0)))
        except Exception:
            probability_positive = None
    elif mean_delta > 0:
        probability_positive = 1.0
    else:
        probability_positive = 0.0
    return mean_delta, variance_delta, probability_positive
