"""Regional Ridge Regression — pure statistical baseline with no physics.

Predicts observed tau directly from baseline tau + time + biology:

  S1_hat_ij = f(S0_ij, t_i, S0_ij*t_i, S0_ij^2, degree_j, amyloid_ij, thickness_ij, ...)

Each (pair, region) is an independent observation.  No ODE, no connectome
dynamics — this is the "what does a competent statistician get without physics?"
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RegionalRidgeFit:
    coefficients: np.ndarray   # (n_features,)
    intercept: float
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    ridge_alpha: float
    train_mse: float
    train_r2: float
    cv_report: list[dict[str, Any]]
    n_features: int
    used_train_rows: int
    ptau_train_median: float

    def predict_flat(self, X: np.ndarray) -> np.ndarray:
        """Predict from a pre-built (N, n_features) feature matrix."""
        X = np.asarray(X, dtype=float)
        scaled = (X - self.feature_mean[None, :]) / self.feature_scale[None, :]
        return self.intercept + scaled @ self.coefficients


def build_regression_features(
    baseline: np.ndarray,
    time_years: np.ndarray,
    laplacian: np.ndarray,
    *,
    amyloid: np.ndarray | None = None,
    thickness: np.ndarray | None = None,
    apoe4_dose: np.ndarray | None = None,
    plasma_ptau181: np.ndarray | None = None,
    ptau_train_median: float = 0.0,
) -> tuple[np.ndarray, list[str]]:
    """Build flat (n_pairs * n_regions, n_features) feature matrix.

    Each row corresponds to one (pair, region) observation.
    """
    baseline = np.asarray(baseline, dtype=float)
    time_years = np.asarray(time_years, dtype=float)
    n_pairs, n_regions = baseline.shape

    t = time_years[:, None]            # (n_pairs, 1) -> broadcast to regions
    s0 = baseline                       # (n_pairs, n_regions)

    # Degree of each region (standardized)
    raw_degree = np.diag(laplacian).astype(float)
    degree_mean = raw_degree.mean()
    degree_std = raw_degree.std()
    degree_std = degree_std if degree_std > 1.0e-8 else 1.0
    degree = (raw_degree - degree_mean) / degree_std  # (n_regions,)

    cols: list[np.ndarray] = []
    names: list[str] = []

    # Core features
    cols += [s0, s0 ** 2, np.broadcast_to(t, s0.shape), s0 * t, s0 ** 2 * t]
    names += ["baseline_tau", "baseline_tau^2", "time_years",
              "baseline_tau*time_years", "baseline_tau^2*time_years"]

    # Connectivity
    deg_broadcast = np.broadcast_to(degree[None, :], s0.shape)
    cols += [deg_broadcast, deg_broadcast * s0]
    names += ["degree", "degree*baseline_tau"]

    # Regional covariates
    if amyloid is not None:
        a = np.asarray(amyloid, dtype=float)
        cols += [a, a * s0]
        names += ["amyloid_suvr", "amyloid_suvr*baseline_tau"]

    if thickness is not None:
        th = np.asarray(thickness, dtype=float)
        cols += [th, th * s0]
        names += ["cortical_thickness", "cortical_thickness*baseline_tau"]

    # Pair-level covariates — broadcast across regions
    if apoe4_dose is not None:
        ap = np.asarray(apoe4_dose, dtype=float).ravel()[:, None]
        ap_broad = np.broadcast_to(ap, s0.shape)
        cols += [ap_broad, ap_broad * s0]
        names += ["apoe4_dose", "apoe4_dose*baseline_tau"]

    if plasma_ptau181 is not None:
        ptau = np.asarray(plasma_ptau181, dtype=float).ravel()
        obs_mask = np.isfinite(ptau).astype(float)
        ptau_imp = np.where(obs_mask > 0, ptau, ptau_train_median)
        ptau_broad = np.broadcast_to(ptau_imp[:, None], s0.shape)
        obs_broad = np.broadcast_to(obs_mask[:, None], s0.shape)
        cols += [ptau_broad, obs_broad, ptau_broad * s0]
        names += ["plasma_ptau181", "plasma_ptau181_observed", "plasma_ptau181*baseline_tau"]

    # Stack and flatten to (n_pairs * n_regions, n_features)
    stacked = np.stack([np.asarray(c) for c in cols], axis=-1)  # (n_pairs, n_regions, n_feat)
    return stacked.reshape(-1, len(names)), names


def fit_regional_ridge(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    *,
    train_row_indices: np.ndarray,
    pair_groups: np.ndarray,
    n_regions: int,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    cv_folds: int = 5,
    max_train_rows: int = 120000,
    random_seed: int = 20260507,
) -> RegionalRidgeFit:
    """Fit ridge on flat (pair*region, features) → target with subject-grouped CV."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    X = np.asarray(feature_matrix, dtype=float)
    y = np.asarray(target, dtype=float).ravel()

    # Build flat masks: row_i belongs to training if its pair index is in train_row_indices
    n_total = X.shape[0]
    n_pairs = n_total // n_regions
    pair_idx = np.repeat(np.arange(n_pairs), n_regions)
    train_pair_set = set(int(i) for i in train_row_indices)
    train_mask = np.array([int(p) in train_pair_set for p in pair_idx])

    groups_flat = np.repeat(pair_groups, n_regions)
    finite_mask = train_mask & np.isfinite(y) & np.all(np.isfinite(X), axis=1)

    X_train = X[finite_mask]
    y_train = y[finite_mask]
    groups_train = groups_flat[finite_mask]
    available = int(X_train.shape[0])

    if max_train_rows > 0 and X_train.shape[0] > int(max_train_rows):
        rng = np.random.default_rng(int(random_seed))
        sel = rng.choice(X_train.shape[0], size=int(max_train_rows), replace=False)
        X_train = X_train[sel]
        y_train = y_train[sel]
        groups_train = groups_train[sel]

    feat_mean = X_train.mean(axis=0)
    feat_scale = X_train.std(axis=0)
    feat_scale[feat_scale < 1.0e-8] = 1.0
    X_scaled = (X_train - feat_mean) / feat_scale

    alpha_values = [float(a) for a in alphas if float(a) > 0]
    unique_groups = np.unique(groups_train)
    n_folds = min(int(cv_folds), unique_groups.size)
    cv_report: list[dict[str, Any]] = []

    if n_folds >= 2:
        splitter = GroupKFold(n_splits=n_folds)
        for alpha in alpha_values:
            fold_mses = []
            for tr, va in splitter.split(X_scaled, y_train, groups_train):
                m = Ridge(alpha=alpha, fit_intercept=True)
                m.fit(X_scaled[tr], y_train[tr])
                fold_mses.append(float(np.mean((m.predict(X_scaled[va]) - y_train[va]) ** 2)))
            cv_report.append({"alpha": alpha, "cv_mse": float(np.mean(fold_mses)), "folds": n_folds})
        best_alpha = min(cv_report, key=lambda r: r["cv_mse"])["alpha"]
    else:
        best_alpha = alpha_values[len(alpha_values) // 2]
        cv_report.append({"alpha": best_alpha, "cv_mse": float("nan"), "folds": 0})

    final = Ridge(alpha=float(best_alpha), fit_intercept=True)
    final.fit(X_scaled, y_train)
    y_hat = final.predict(X_scaled)
    residuals = y_train - y_hat
    train_mse = float(np.mean(residuals ** 2))
    ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
    train_r2 = 0.0 if ss_tot <= 0.0 else float(1.0 - np.sum(residuals ** 2) / ss_tot)

    ptau_idx = None  # unused here but kept for run script
    return RegionalRidgeFit(
        coefficients=np.asarray(final.coef_, dtype=float),
        intercept=float(final.intercept_),
        feature_names=list(range(X.shape[1])),  # overwritten by caller
        feature_mean=feat_mean,
        feature_scale=feat_scale,
        ridge_alpha=float(best_alpha),
        train_mse=train_mse,
        train_r2=train_r2,
        cv_report=cv_report,
        n_features=X.shape[1],
        used_train_rows=int(X_train.shape[0]),
        ptau_train_median=0.0,
    )
