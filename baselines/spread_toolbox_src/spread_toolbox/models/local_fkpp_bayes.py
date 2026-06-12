"""Empirical-Bayes helpers for Chaggar-style local FKPP."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fkpp import LocalFKPPModel


def require_pymc() -> None:
    try:
        import pymc  # noqa: F401
        import pytensor  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyMC is required for Bayesian local FKPP. Install with: "
            "python -m pip install 'pymc>=5.16' 'arviz>=0.17'"
        ) from exc


def group_indices_by_rid(pairs: list[dict[str, str]], indices: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index in indices:
        grouped.setdefault(pairs[int(index)]["RID"], []).append(int(index))
    return {rid: np.asarray(values, dtype=int) for rid, values in grouped.items()}


def sample_subject_posterior(
    model: LocalFKPPModel,
    baseline: np.ndarray,
    observed: np.ndarray,
    time_years: np.ndarray,
    *,
    prior_rho: float,
    prior_alpha: float,
    prior_scale: float,
    sigma: float,
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
) -> dict[str, float]:
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.compile.ops import as_op

    log_prior_rho = math.log(max(float(prior_rho), 1.0e-8))
    log_prior_alpha = math.log(max(float(prior_alpha), 1.0e-8))
    sigma = max(float(sigma), 1.0e-8)

    @as_op(itypes=[pt.dscalar, pt.dscalar], otypes=[pt.dscalar])
    def log_likelihood(log_rho: float, log_alpha: float) -> np.ndarray:
        rho = float(np.exp(log_rho))
        alpha = float(np.exp(log_alpha))
        predicted = model.predict(baseline, time_years, rho=rho, alpha=alpha)
        residual = observed - predicted
        value = -0.5 * np.sum((residual / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))
        return np.asarray(value, dtype=float)

    with pm.Model() as pymc_model:
        log_rho = pm.Normal("log_rho", mu=log_prior_rho, sigma=prior_scale)
        log_alpha = pm.Normal("log_alpha", mu=log_prior_alpha, sigma=prior_scale)
        rho = pm.Deterministic("rho", pm.math.exp(log_rho))
        alpha = pm.Deterministic("alpha", pm.math.exp(log_alpha))
        pm.Potential("ode_likelihood", log_likelihood(log_rho, log_alpha))
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            step=pm.Metropolis(),
            random_seed=random_seed,
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
        )

    rho_draws = trace.posterior["rho"].values.reshape(-1)
    alpha_draws = trace.posterior["alpha"].values.reshape(-1)
    return {
        "rho_median": float(np.median(rho_draws)),
        "rho_mean": float(np.mean(rho_draws)),
        "rho_q025": float(np.quantile(rho_draws, 0.025)),
        "rho_q975": float(np.quantile(rho_draws, 0.975)),
        "alpha_median": float(np.median(alpha_draws)),
        "alpha_mean": float(np.mean(alpha_draws)),
        "alpha_q025": float(np.quantile(alpha_draws, 0.025)),
        "alpha_q975": float(np.quantile(alpha_draws, 0.975)),
    }


def predict_by_subject(
    model: LocalFKPPModel,
    pairs: list[dict[str, str]],
    baseline: np.ndarray,
    time_years: np.ndarray,
    output_shape: tuple[int, int],
    subject_parameters: dict[str, tuple[float, float]],
    *,
    fallback_rho: float,
    fallback_alpha: float,
) -> np.ndarray:
    predicted = np.empty(output_shape, dtype=float)
    for index, pair in enumerate(pairs):
        rho, alpha = subject_parameters.get(pair["RID"], (fallback_rho, fallback_alpha))
        predicted[index] = model.predict(
            baseline[index : index + 1],
            time_years[index : index + 1],
            rho=rho,
            alpha=alpha,
        )[0]
    return predicted


def median_or(default: float, values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float(default)
    return float(np.median(finite))


def summarize_subject_posteriors(
    subject_rows: list[dict[str, Any]],
    name: str,
    *,
    prefix: str | None = None,
) -> dict[str, float]:
    output_prefix = prefix or name
    values = np.asarray([row[name] for row in subject_rows if np.isfinite(row.get(name, np.nan))], dtype=float)
    if values.size == 0:
        return {
            f"{output_prefix}_min": float("nan"),
            f"{output_prefix}_median": float("nan"),
            f"{output_prefix}_max": float("nan"),
        }
    return {
        f"{output_prefix}_min": float(values.min()),
        f"{output_prefix}_median": float(np.median(values)),
        f"{output_prefix}_max": float(values.max()),
    }
