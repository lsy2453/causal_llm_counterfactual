from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..posterior import PosteriorProvider


@dataclass
class NumpyRBFPosterior(PosteriorProvider):
    """Small GP posterior used when BoTorch is unavailable."""

    train_x: np.ndarray
    train_y: np.ndarray
    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-4

    def __post_init__(self) -> None:
        self.train_x = np.asarray(self.train_x, dtype=float)
        self.train_y = np.asarray(self.train_y, dtype=float).reshape(-1)
        kxx = self._kernel(self.train_x, self.train_x)
        kxx = kxx + self.noise * np.eye(len(self.train_x))
        self._chol = np.linalg.cholesky(kxx + 1e-10 * np.eye(len(kxx)))
        self._alpha = np.linalg.solve(self._chol.T, np.linalg.solve(self._chol, self.train_y))

    def mean_and_covariance(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        k_x_train = self._kernel(x, self.train_x)
        mean = k_x_train @ self._alpha
        v = np.linalg.solve(self._chol, k_x_train.T)
        cov = self._kernel(x, x) - v.T @ v
        cov = 0.5 * (cov + cov.T)
        cov = cov + 1e-9 * np.eye(len(x))
        return mean.reshape(-1), cov

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, cov = self.mean_and_covariance(x)
        return mean, np.maximum(np.diag(cov), 1e-12)

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sqdist = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        return self.variance * np.exp(-0.5 * sqdist / (self.lengthscale**2))


class BoTorchOrNumpySurrogate:
    """Fit BoTorch SingleTaskGP if available; otherwise use NumpyRBFPosterior."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self.train_x = np.asarray(train_x, dtype=float)
        self.train_y = np.asarray(train_y, dtype=float).reshape(-1, 1)
        self.backend = "numpy"
        self.provider: PosteriorProvider
        self._fit()

    def _fit(self) -> None:
        try:
            import torch
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from gpytorch.mlls import ExactMarginalLogLikelihood

            train_x = torch.as_tensor(self.train_x, dtype=torch.double)
            train_y = torch.as_tensor(self.train_y, dtype=torch.double)
            model = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            from ..adapters import BoTorchPosteriorProvider

            self.provider = BoTorchPosteriorProvider(model)
            self.backend = "botorch"
        except Exception:
            self.provider = NumpyRBFPosterior(self.train_x, self.train_y)
            self.backend = "numpy"

    def mean_and_covariance(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.provider.mean_and_covariance(x)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean, cov = self.mean_and_covariance(x)
        return mean, np.maximum(np.diag(cov), 1e-12)


def expected_improvement(
    mean: np.ndarray,
    variance: np.ndarray,
    best_y: float,
    task: str = "min",
) -> np.ndarray:
    from math import erf, sqrt

    sigma = np.sqrt(np.maximum(variance, 1e-12))
    if task == "min":
        improvement_mean = best_y - mean
    else:
        improvement_mean = mean - best_y
    z = improvement_mean / sigma
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    cdf = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))(z)
    return improvement_mean * cdf + sigma * pdf
