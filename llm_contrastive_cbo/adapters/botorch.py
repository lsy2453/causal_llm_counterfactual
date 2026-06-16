from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..posterior import PosteriorProvider


@dataclass
class BoTorchPosteriorProvider(PosteriorProvider):
    """PosteriorProvider adapter for BoTorch models.

    The wrapped model should expose ``posterior(X)`` where X is a torch tensor
    of shape ``n x d``. The posterior must provide ``mean`` and ``covariance_matrix``.
    """

    model: object
    dtype: object | None = None
    device: object | None = None

    def mean_and_covariance(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        dtype = self.dtype or torch.double
        tensor = torch.as_tensor(np.asarray(x, dtype=float), dtype=dtype, device=self.device)
        posterior = self.model.posterior(tensor)
        mean = posterior.mean.detach().cpu().numpy().reshape(-1)
        cov = posterior.mvn.covariance_matrix.detach().cpu().numpy()
        if cov.ndim == 3:
            cov = cov[0]
        return mean, cov
