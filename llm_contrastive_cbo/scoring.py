from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContrastScoreConfig:
    lambda_disc: float = 1.0
    lambda_unc: float = 0.25
    eta: float = 1e-9
    use_hard_weighting: bool = True
    tau_x: float = 0.35
    tau_y: float = 0.25
    historical_weight: float = 0.45
    local_weight: float = 1.0
    cost_weight: float = 0.9
    posterior_weight: float = 0.8


def discriminability(mean_delta: float, variance_delta: float, eta: float = 1e-9) -> float:
    return float(abs(mean_delta) / np.sqrt(max(variance_delta, 0.0) + eta))


def raw_score(
    mean_delta: float,
    variance_delta: float,
    config: ContrastScoreConfig = ContrastScoreConfig(),
) -> tuple[float, float, float]:
    disc = discriminability(mean_delta, variance_delta, config.eta)
    unc = float(max(variance_delta, 0.0))
    score = config.lambda_disc * disc + config.lambda_unc * unc
    return disc, unc, float(score)


def probability_candidate_better(
    mean_delta: float,
    variance_delta: float,
    task: str,
) -> float:
    """Return P(candidate a is better than contrast b).

    Delta is defined as g(a)-g(b). For minimization, a is better when Delta < 0.
    For maximization, a is better when Delta > 0.
    """

    from math import erf, sqrt

    if variance_delta <= 1e-14:
        if task == "min":
            return 1.0 if mean_delta < 0 else 0.0
        return 1.0 if mean_delta > 0 else 0.0
    z = mean_delta / sqrt(variance_delta)
    p_delta_positive = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    if task == "min":
        return float(1.0 - p_delta_positive)
    return float(p_delta_positive)


def semantic_relevance_weight(contrast_type: str, config: ContrastScoreConfig) -> float:
    if contrast_type.startswith("single_variable"):
        return config.local_weight
    if contrast_type == "historical_comparison":
        return config.historical_weight
    if contrast_type == "dose_or_cost_reduction":
        return config.cost_weight
    if contrast_type == "uncertainty_driven_contrast":
        return config.posterior_weight
    return 0.75


def hardness_weight(
    candidate_x: np.ndarray,
    contrast_x: np.ndarray,
    mean_delta: float,
    config: ContrastScoreConfig,
) -> float:
    """Reward local and performance-competitive contrasts."""

    if not config.use_hard_weighting:
        return 1.0
    distance = float(np.linalg.norm(np.asarray(candidate_x) - np.asarray(contrast_x)))
    w_x = np.exp(-distance / max(config.tau_x, config.eta))
    w_y = np.exp(-abs(mean_delta) / max(config.tau_y, config.eta))
    return float(w_x * w_y)


def hard_weighted_score(
    candidate_x: np.ndarray,
    contrast_x: np.ndarray,
    contrast_type: str,
    mean_delta: float,
    variance_delta: float,
    task: str,
    config: ContrastScoreConfig = ContrastScoreConfig(),
) -> tuple[float, float, float, float, float, float]:
    """Direction-aware hard-contrast-weighted CFScore."""

    disc = discriminability(mean_delta, variance_delta, config.eta)
    unc = float(max(variance_delta, 0.0))
    p_better = probability_candidate_better(mean_delta, variance_delta, task)
    w_hard = hardness_weight(candidate_x, contrast_x, mean_delta, config)
    w_rel = semantic_relevance_weight(contrast_type, config)
    score = w_rel * w_hard * p_better * (config.lambda_disc * disc + config.lambda_unc * unc)
    return disc, unc, p_better, w_hard, w_rel, float(score)
