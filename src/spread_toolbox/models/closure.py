"""Feature-conditioned residual closure for local FKPP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ClosureFeatureLibrary:
    names: list[str]
    values: np.ndarray


@dataclass
class LinearClosureFit:
    feature_names: list[str]
    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficient_draws: np.ndarray
    intercept_draws: np.ndarray
    sigma_draws: np.ndarray
    inclusion_threshold: float
    train_mse_rate: float
    train_r2_rate: float
    sampling_report: dict[str, Any]

    def predict_rate(self, feature_values: np.ndarray) -> np.ndarray:
        feature_values = np.asarray(feature_values, dtype=float)
        if feature_values.shape[-1] != self.coefficients.size:
            raise ValueError("Feature count does not match fitted closure.")
        flat = feature_values.reshape(-1, feature_values.shape[-1])
        predicted = self.intercept + flat @ self.coefficients
        return predicted.reshape(feature_values.shape[:-1])

    @property
    def inclusion_probabilities(self) -> np.ndarray:
        return np.mean(np.abs(self.coefficient_draws) > float(self.inclusion_threshold), axis=0)

    @property
    def selected_count(self) -> int:
        return int(np.sum(self.inclusion_probabilities >= 0.95))

    def term_rows(self) -> list[dict[str, Any]]:
        rows = [
            {
                "term": "1",
                "coefficient_mean": self.intercept,
                "coefficient_sd": float(np.std(self.intercept_draws)),
                "coefficient_q025": float(np.quantile(self.intercept_draws, 0.025)),
                "coefficient_q975": float(np.quantile(self.intercept_draws, 0.975)),
                "inclusion_probability": 1.0,
                "selected": abs(self.intercept) > self.inclusion_threshold,
            }
        ]
        inclusion = self.inclusion_probabilities
        for index, name in enumerate(self.feature_names):
            draws = self.coefficient_draws[:, index]
            rows.append(
                {
                    "term": name,
                    "coefficient_mean": float(self.coefficients[index]),
                    "coefficient_sd": float(np.std(draws)),
                    "coefficient_q025": float(np.quantile(draws, 0.025)),
                    "coefficient_q975": float(np.quantile(draws, 0.975)),
                    "inclusion_probability": float(inclusion[index]),
                    "selected": bool(inclusion[index] >= 0.95),
                }
            )
        return rows


def build_closure_feature_library(
    baseline: np.ndarray,
    *,
    u0: np.ndarray,
    cc: np.ndarray,
    pair_covariates: dict[str, np.ndarray] | None = None,
    regional_covariates: dict[str, np.ndarray] | None = None,
) -> ClosureFeatureLibrary:
    baseline = np.asarray(baseline, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    cc = np.asarray(cc, dtype=float)
    if baseline.ndim != 2:
        raise ValueError("baseline must be two-dimensional.")
    if baseline.shape[1] != u0.size or u0.shape != cc.shape:
        raise ValueError("baseline, u0, and cc region counts must match.")

    state = np.clip(baseline, u0, cc)
    tau = state - u0
    gap = cc - state
    growth_shape = tau * gap

    names = ["tau", "tau^2", "K_minus_tau", "tau*(K_minus_tau)"]
    features = [tau, tau**2, gap, growth_shape]

    for covariate_name, covariate in sorted((pair_covariates or {}).items()):
        values = np.asarray(covariate, dtype=float)
        if values.shape != (baseline.shape[0],):
            raise ValueError(f"Pair covariate {covariate_name!r} must have one value per pair.")
        expanded = values[:, None]
        names.extend([f"{covariate_name}*tau", f"{covariate_name}*tau*(K_minus_tau)"])
        features.extend([expanded * tau, expanded * growth_shape])

    for covariate_name, covariate in sorted((regional_covariates or {}).items()):
        values = np.asarray(covariate, dtype=float)
        if values.shape != baseline.shape:
            raise ValueError(f"Regional covariate {covariate_name!r} must match baseline shape.")
        names.extend(
            [
                f"{covariate_name}*tau",
                f"{covariate_name}*tau^2",
                f"{covariate_name}*tau*(K_minus_tau)",
            ]
        )
        features.extend([values * tau, values * tau**2, values * growth_shape])

    return ClosureFeatureLibrary(names=names, values=np.stack(features, axis=-1))


def fit_horseshoe_linear_closure(
    feature_library: ClosureFeatureLibrary,
    target_rate: np.ndarray,
    *,
    row_indices: np.ndarray,
    draws: int = 300,
    tune: int = 300,
    chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 20260507,
    inclusion_threshold: float = 1.0e-4,
    max_train_rows: int = 10000,
) -> LinearClosureFit:
    """Fit a Bayesian global-local shrinkage linear closure with PyMC."""

    import pymc as pm

    features = np.asarray(feature_library.values, dtype=float)
    target_rate = np.asarray(target_rate, dtype=float)
    if features.shape[:-1] != target_rate.shape:
        raise ValueError("Feature values and target_rate must have matching pair-region dimensions.")

    row_mask = np.zeros(target_rate.shape[0], dtype=bool)
    row_mask[np.asarray(row_indices, dtype=int)] = True
    flat_features = features.reshape(-1, features.shape[-1])
    flat_target = target_rate.reshape(-1)
    flat_row_mask = np.broadcast_to(row_mask[:, None], target_rate.shape).reshape(-1)
    finite_mask = flat_row_mask & np.isfinite(flat_target) & np.all(np.isfinite(flat_features), axis=1)
    x_train = flat_features[finite_mask]
    y_train = flat_target[finite_mask]
    if x_train.shape[0] < max(10, x_train.shape[1] + 2):
        raise ValueError("Not enough finite training rows to fit closure.")
    available_train_rows = int(x_train.shape[0])
    if max_train_rows > 0 and x_train.shape[0] > int(max_train_rows):
        rng = np.random.default_rng(int(random_seed))
        selected = rng.choice(x_train.shape[0], size=int(max_train_rows), replace=False)
        x_train = x_train[selected]
        y_train = y_train[selected]

    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.std(x_train, axis=0)
    feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
    x_scaled = (x_train - feature_mean) / feature_scale
    y_scale = float(np.std(y_train))
    y_scale = y_scale if y_scale > 1.0e-8 else 1.0

    with pm.Model() as pymc_model:
        sigma = pm.HalfNormal("sigma", sigma=2.0 * y_scale)
        global_scale = pm.HalfCauchy("global_scale", beta=1.0)
        local_scale = pm.HalfCauchy("local_scale", beta=1.0, shape=x_scaled.shape[1])
        beta_scaled = pm.Normal("beta_scaled", mu=0.0, sigma=global_scale * local_scale, shape=x_scaled.shape[1])
        intercept_scaled = pm.Normal("intercept_scaled", mu=float(np.mean(y_train)), sigma=2.0 * y_scale)
        mean = intercept_scaled + pm.math.dot(x_scaled, beta_scaled)
        pm.Normal("target_rate", mu=mean, sigma=sigma, observed=y_train)
        trace = pm.sample(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            cores=1,
            target_accept=float(target_accept),
            random_seed=int(random_seed),
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
        )

    beta_scaled_draws = trace.posterior["beta_scaled"].values.reshape(-1, x_scaled.shape[1])
    intercept_scaled_draws = trace.posterior["intercept_scaled"].values.reshape(-1)
    sigma_draws = trace.posterior["sigma"].values.reshape(-1)
    coefficient_draws = beta_scaled_draws / feature_scale[None, :]
    intercept_draws = intercept_scaled_draws - np.sum(beta_scaled_draws * feature_mean[None, :] / feature_scale[None, :], axis=1)
    coefficients = np.mean(coefficient_draws, axis=0)
    intercept = float(np.mean(intercept_draws))

    prediction = intercept + x_train @ coefficients
    residual = y_train - prediction
    train_mse = float(np.mean(residual**2))
    total = float(np.sum((y_train - np.mean(y_train)) ** 2))
    train_r2 = 0.0 if total <= 0.0 else float(1.0 - np.sum(residual**2) / total)

    sampling_report = {
        "draws": int(draws),
        "tune": int(tune),
        "chains": int(chains),
        "posterior_draws": int(coefficient_draws.shape[0]),
        "target_accept": float(target_accept),
        "available_train_rows": available_train_rows,
        "used_train_rows": int(x_train.shape[0]),
        "sigma_mean": float(np.mean(sigma_draws)),
        "global_scale_mean": float(np.mean(trace.posterior["global_scale"].values)),
        "divergences": int(np.sum(trace.sample_stats.get("diverging", 0).values)),
    }
    return LinearClosureFit(
        feature_names=list(feature_library.names),
        coefficients=coefficients,
        intercept=intercept,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficient_draws=coefficient_draws,
        intercept_draws=intercept_draws,
        sigma_draws=sigma_draws,
        inclusion_threshold=float(inclusion_threshold),
        train_mse_rate=train_mse,
        train_r2_rate=train_r2,
        sampling_report=sampling_report,
    )


def apply_closure_delta(
    backbone_prediction: np.ndarray,
    time_years: np.ndarray,
    closure_rate: np.ndarray,
    *,
    u0: np.ndarray,
    cc: np.ndarray,
) -> np.ndarray:
    safe_time = np.maximum(np.asarray(time_years, dtype=float), 0.0)[:, None]
    return np.clip(np.asarray(backbone_prediction, dtype=float) + safe_time * closure_rate, u0, cc)
