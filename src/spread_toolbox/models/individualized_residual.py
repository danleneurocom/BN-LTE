"""Individualized residual correction on top of a physics backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ResidualFeatureLibrary:
    names: list[str]
    values: np.ndarray


@dataclass
class RidgeResidualFit:
    feature_names: list[str]
    coefficients: np.ndarray
    intercept: float
    alpha: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    train_mse_rate: float
    train_r2_rate: float
    cv_report: list[dict[str, Any]]
    used_train_rows: int
    available_train_rows: int

    def predict_rate(self, feature_values: np.ndarray) -> np.ndarray:
        values = np.asarray(feature_values, dtype=float)
        if values.shape[-1] != self.coefficients.size:
            raise ValueError("Feature count does not match fitted residual model.")
        flat = values.reshape(-1, values.shape[-1])
        scaled = (flat - self.feature_mean[None, :]) / self.feature_scale[None, :]
        predicted = self.intercept + scaled @ self.coefficients
        return predicted.reshape(values.shape[:-1])

    def term_rows(self) -> list[dict[str, Any]]:
        rows = [{"term": "1", "coefficient": self.intercept, "abs_coefficient": abs(self.intercept)}]
        for name, coefficient in zip(self.feature_names, self.coefficients, strict=True):
            rows.append(
                {
                    "term": name,
                    "coefficient": float(coefficient),
                    "abs_coefficient": abs(float(coefficient)),
                }
            )
        return sorted(rows, key=lambda row: float(row["abs_coefficient"]), reverse=True)


def build_individualized_residual_features(
    *,
    baseline: np.ndarray,
    backbone_prediction: np.ndarray,
    time_years: np.ndarray,
    laplacian: np.ndarray,
    region_labels: list[str],
    pair_covariates: dict[str, np.ndarray] | None = None,
    regional_covariates: dict[str, np.ndarray] | None = None,
    include_region_bias: bool = True,
    backbone_name: str = "fkpp",
) -> ResidualFeatureLibrary:
    """Build physics, subject, and regional features for residual forecasting."""

    baseline = np.asarray(baseline, dtype=float)
    backbone_prediction = np.asarray(backbone_prediction, dtype=float)
    time_years = np.asarray(time_years, dtype=float)
    laplacian = np.asarray(laplacian, dtype=float)
    if baseline.ndim != 2 or backbone_prediction.shape != baseline.shape:
        raise ValueError("baseline and backbone_prediction must be matching two-dimensional arrays.")
    if time_years.shape != (baseline.shape[0],):
        raise ValueError("time_years must contain one value per pair.")
    if laplacian.shape != (baseline.shape[1], baseline.shape[1]):
        raise ValueError("laplacian must match the region count.")
    if len(region_labels) != baseline.shape[1]:
        raise ValueError("region_labels must match the region count.")

    safe_time = np.maximum(time_years, 1.0e-6)[:, None]
    fkpp_delta = backbone_prediction - baseline
    fkpp_delta_rate = fkpp_delta / safe_time
    growth_drive = baseline * (1.0 - baseline)
    diffusion_drive = -(baseline @ laplacian.T)
    degree = np.diag(laplacian).astype(float)
    degree = standardize_vector(degree)

    names = [
        "baseline_tau",
        "baseline_tau^2",
        f"{backbone_name}_prediction",
        f"{backbone_name}_delta",
        f"{backbone_name}_delta_rate",
        f"{backbone_name}_growth_drive",
        f"{backbone_name}_diffusion_drive",
        "time_years*baseline_tau",
        f"time_years*{backbone_name}_delta_rate",
        "connectome_degree",
        "connectome_degree*baseline_tau",
    ]
    features = [
        baseline,
        baseline**2,
        backbone_prediction,
        fkpp_delta,
        fkpp_delta_rate,
        growth_drive,
        diffusion_drive,
        time_years[:, None] * baseline,
        time_years[:, None] * fkpp_delta_rate,
        np.broadcast_to(degree[None, :], baseline.shape),
        degree[None, :] * baseline,
    ]

    for covariate_name, covariate in sorted((pair_covariates or {}).items()):
        values = np.asarray(covariate, dtype=float)
        if values.shape != (baseline.shape[0],):
            raise ValueError(f"Pair covariate {covariate_name!r} must have one value per pair.")
        expanded = values[:, None]
        names.extend(
            [
                covariate_name,
                f"{covariate_name}*baseline_tau",
                f"{covariate_name}*{backbone_name}_delta",
                f"{covariate_name}*{backbone_name}_growth_drive",
            ]
        )
        features.extend([np.broadcast_to(expanded, baseline.shape), expanded * baseline, expanded * fkpp_delta, expanded * growth_drive])

    for covariate_name, covariate in sorted((regional_covariates or {}).items()):
        values = np.asarray(covariate, dtype=float)
        if values.shape != baseline.shape:
            raise ValueError(f"Regional covariate {covariate_name!r} must match baseline shape.")
        names.extend(
            [
                covariate_name,
                f"{covariate_name}*baseline_tau",
                f"{covariate_name}*{backbone_name}_delta",
                f"{covariate_name}*{backbone_name}_growth_drive",
            ]
        )
        features.extend([values, values * baseline, values * fkpp_delta, values * growth_drive])

    if include_region_bias:
        for region_index, label in enumerate(region_labels):
            bias = np.zeros_like(baseline)
            bias[:, region_index] = 1.0
            names.append(f"region_bias:{label}")
            features.append(bias)

    return ResidualFeatureLibrary(names=names, values=np.stack(features, axis=-1))


def fit_ridge_residual_model(
    feature_library: ResidualFeatureLibrary,
    target_rate: np.ndarray,
    *,
    row_indices: np.ndarray,
    pair_groups: np.ndarray,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    cv_folds: int = 5,
    max_train_rows: int = 60000,
    random_seed: int = 20260507,
) -> RidgeResidualFit:
    """Fit a train-only ridge residual model with subject-group CV for alpha."""

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    features = np.asarray(feature_library.values, dtype=float)
    target_rate = np.asarray(target_rate, dtype=float)
    if features.shape[:-1] != target_rate.shape:
        raise ValueError("Feature values and target_rate must have matching pair-region dimensions.")
    if pair_groups.shape != (target_rate.shape[0],):
        raise ValueError("pair_groups must have one group label per pair.")

    row_mask = np.zeros(target_rate.shape[0], dtype=bool)
    row_mask[np.asarray(row_indices, dtype=int)] = True
    flat_features = features.reshape(-1, features.shape[-1])
    flat_target = target_rate.reshape(-1)
    flat_row_mask = np.broadcast_to(row_mask[:, None], target_rate.shape).reshape(-1)
    flat_groups = np.repeat(pair_groups.astype(str), target_rate.shape[1])
    finite_mask = flat_row_mask & np.isfinite(flat_target) & np.all(np.isfinite(flat_features), axis=1)

    x_train = flat_features[finite_mask]
    y_train = flat_target[finite_mask]
    groups_train = flat_groups[finite_mask]
    if x_train.shape[0] < max(20, x_train.shape[1] + 2):
        raise ValueError("Not enough finite training rows to fit individualized residual model.")

    available_train_rows = int(x_train.shape[0])
    if max_train_rows > 0 and x_train.shape[0] > int(max_train_rows):
        rng = np.random.default_rng(int(random_seed))
        selected = rng.choice(x_train.shape[0], size=int(max_train_rows), replace=False)
        x_train = x_train[selected]
        y_train = y_train[selected]
        groups_train = groups_train[selected]

    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.std(x_train, axis=0)
    feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 1.0e-12), feature_scale, 1.0)
    x_scaled = np.nan_to_num((x_train - feature_mean[None, :]) / feature_scale[None, :], nan=0.0)

    alpha_values = tuple(float(alpha) for alpha in alphas if float(alpha) > 0.0)
    if not alpha_values:
        raise ValueError("At least one positive alpha is required.")
    unique_groups = np.unique(groups_train)
    fold_count = min(int(cv_folds), unique_groups.size)
    cv_report: list[dict[str, Any]] = []
    if fold_count >= 2:
        splitter = GroupKFold(n_splits=fold_count)
        for alpha in alpha_values:
            fold_mse = []
            for train_position, validation_position in splitter.split(x_scaled, y_train, groups_train):
                model = Ridge(alpha=alpha, fit_intercept=True)
                model.fit(x_scaled[train_position], y_train[train_position])
                predicted = model.predict(x_scaled[validation_position])
                fold_mse.append(float(np.mean((predicted - y_train[validation_position]) ** 2)))
            cv_report.append({"alpha": alpha, "cv_mse": float(np.mean(fold_mse)), "folds": fold_count})
        selected_alpha = min(cv_report, key=lambda row: float(row["cv_mse"]))["alpha"]
    else:
        selected_alpha = alpha_values[len(alpha_values) // 2]
        cv_report.append({"alpha": selected_alpha, "cv_mse": float("nan"), "folds": 0})

    final_model = Ridge(alpha=float(selected_alpha), fit_intercept=True)
    final_model.fit(x_scaled, y_train)
    train_prediction = final_model.predict(x_scaled)
    residual = y_train - train_prediction
    train_mse = float(np.mean(residual**2))
    total = float(np.sum((y_train - np.mean(y_train)) ** 2))
    train_r2 = 0.0 if total <= 0.0 else float(1.0 - np.sum(residual**2) / total)

    return RidgeResidualFit(
        feature_names=list(feature_library.names),
        coefficients=np.asarray(final_model.coef_, dtype=float),
        intercept=float(final_model.intercept_),
        alpha=float(selected_alpha),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        train_mse_rate=train_mse,
        train_r2_rate=train_r2,
        cv_report=cv_report,
        used_train_rows=int(x_train.shape[0]),
        available_train_rows=available_train_rows,
    )


def choose_residual_shrinkage(
    *,
    backbone_prediction: np.ndarray,
    observed: np.ndarray,
    time_years: np.ndarray,
    residual_rate: np.ndarray,
    row_indices: np.ndarray,
    candidates: tuple[float, ...],
    lower: float = 0.0,
    upper: float = 1.0,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for candidate in candidates:
        shrinkage = float(candidate)
        predicted = apply_individualized_residual_correction(
            backbone_prediction,
            time_years,
            residual_rate,
            shrinkage=shrinkage,
            lower=lower,
            upper=upper,
        )
        mse = float(np.mean((predicted[np.asarray(row_indices, dtype=int)] - observed[np.asarray(row_indices, dtype=int)]) ** 2))
        rows.append({"shrinkage": shrinkage, "train_mse": mse})
    selected = min(rows, key=lambda row: float(row["train_mse"]))
    return float(selected["shrinkage"]), rows


def apply_individualized_residual_correction(
    backbone_prediction: np.ndarray,
    time_years: np.ndarray,
    residual_rate: np.ndarray,
    *,
    shrinkage: float = 1.0,
    lower: float = 0.0,
    upper: float = 1.0,
    max_abs_delta: float | None = None,
) -> np.ndarray:
    delta = float(shrinkage) * np.maximum(np.asarray(time_years, dtype=float), 0.0)[:, None] * np.asarray(residual_rate, dtype=float)
    if max_abs_delta is not None and float(max_abs_delta) > 0.0:
        delta = np.clip(delta, -float(max_abs_delta), float(max_abs_delta))
    return np.clip(np.asarray(backbone_prediction, dtype=float) + delta, float(lower), float(upper))


def standardize_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    return np.nan_to_num((values - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)
