"""Amortized Subject-Specific FKPP (AS-FKPP).

Personalises FKPP parameters (rho_i, alpha_i) per subject by:
  Stage 1: Global FKPP backbone — fit population (rho, alpha).
  Stage 2a: Per-pair fitting — find (Delta_rho_i, Delta_alpha_i) that minimises
            FKPP MSE for each training pair independently.
  Stage 2b: Amortisation — ridge regression from feature matrix to individual
            offsets, enabling parameter prediction for held-out subjects.

The amortisation feature matrix contains:
  - Baseline tau spatial fingerprint via Laplacian eigenmode projections
  - Tau burden scalars (mean, max, CV, hub-weighted)
  - Biological covariates (amyloid SUVR, cortical thickness, APOE4, p-tau181)

Key difference from global_fkpp_individualized_residual:
  - Correction modulates ODE parameters (rho_i, alpha_i) directly, not a
    post-hoc additive rate.  Personalised FKPP is then integrated forward, so
    the correction propagates correctly over arbitrary forecast horizons.
  - Baseline tau *spatial fingerprint* is used as a predictor of individual
    spreading speed, not just as an ODE initial condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from .fkpp import GraphFKPPModel, normalize_laplacian


@dataclass
class ASFKPPFitResult:
    rho: float
    alpha: float
    # Amortisation weights — stored as plain arrays so no sklearn dependency at prediction time
    rho_coef: np.ndarray        # (n_features,) in standardised feature space
    rho_intercept: float
    alpha_coef: np.ndarray      # (n_features,)
    alpha_intercept: float
    feature_names: list[str]
    feature_mean: np.ndarray    # for standardisation at test time
    feature_scale: np.ndarray
    # Per-pair offsets from Stage 2a (training set only — for diagnostics)
    delta_rho_train: np.ndarray
    delta_alpha_train: np.ndarray
    # Diagnostics
    stage1_train_mse: float
    stage2a_train_mse: float    # MSE using individually-fitted params
    stage2b_train_mse: float    # MSE using amortised params
    amortisation_rho_r2: float  # how well features predict Delta_rho
    amortisation_alpha_r2: float
    ridge_alpha_rho: float
    ridge_alpha_alpha: float
    n_features: int
    n_eigenmodes: int
    per_pair_success_frac: float
    ptau_train_median: float
    backbone_laplacian_scale: float
    backbone_laplacian_normalization: str


class ASFKPPModel:
    """Amortized Subject-Specific FKPP.

    Personalises spreading parameters per subject using baseline tau spatial
    fingerprint (Laplacian eigenmode projections) plus biological covariates.
    """

    def __init__(
        self,
        laplacian: np.ndarray,
        *,
        steps_per_year: int = 12,
        laplacian_normalization: str = "spectral",
        n_eigenmodes: int = 10,
    ):
        laplacian = np.asarray(laplacian, dtype=float)
        if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
            raise ValueError(f"Laplacian must be square, got {laplacian.shape}.")
        if steps_per_year < 1:
            raise ValueError("steps_per_year must be >= 1.")
        self.original_laplacian = laplacian
        self.normalized_laplacian, self.laplacian_scale = normalize_laplacian(
            laplacian, laplacian_normalization
        )
        self.laplacian_normalization = laplacian_normalization
        self.steps_per_year = int(steps_per_year)
        n_regions = laplacian.shape[0]
        self.n_eigenmodes = min(int(n_eigenmodes), n_regions)

        # Laplacian eigenmodes — eigenvectors of the original Laplacian (ascending eigenvalue)
        # v_0 ≈ constant vector (global burden); v_1..v_K capture local spread patterns.
        eigenvalues, eigenvectors = eigh(laplacian)
        self.eigenvalues = eigenvalues[: self.n_eigenmodes].copy()
        self.eigenvectors = eigenvectors[:, : self.n_eigenmodes].copy()  # (n_regions, K)

        # Degree for hub-weighted tau feature (diagonal of combinatorial Laplacian)
        degree = np.diag(laplacian).astype(float)
        self.degree = degree if degree.sum() > 0.0 else np.ones(n_regions)

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def build_features(
        self,
        baseline: np.ndarray,
        amyloid: np.ndarray | None,
        thickness: np.ndarray | None,
        apoe4_dose: np.ndarray | None,
        plasma_ptau181: np.ndarray | None,
        *,
        ptau_train_median: float = 0.0,
    ) -> tuple[np.ndarray, list[str]]:
        """Build (n_pairs, n_features) amortisation feature matrix."""
        baseline = np.asarray(baseline, dtype=float)
        n = baseline.shape[0]
        cols: list[np.ndarray] = []
        names: list[str] = []

        # --- Tau burden scalars ---
        tau_mean = baseline.mean(axis=1)
        tau_max = baseline.max(axis=1)
        tau_cv = baseline.std(axis=1) / np.maximum(tau_mean, 1.0e-6)
        cols += [tau_mean, tau_max, tau_cv]
        names += ["tau_burden_mean", "tau_burden_max", "tau_burden_cv"]

        # Hub-connectivity-weighted tau burden
        hub_tau = (baseline * self.degree[None, :]).sum(axis=1) / self.degree.sum()
        cols.append(hub_tau)
        names.append("tau_hub_weighted")

        # --- Laplacian eigenmode projections: S0 @ V → (n, K) ---
        # Captures which network modes are "loaded" at baseline.
        projections = baseline @ self.eigenvectors  # (n, K)
        for k in range(self.n_eigenmodes):
            cols.append(projections[:, k])
            names.append(f"eigenmode_{k}")

        # --- Biological covariates (scalar per pair) ---
        if amyloid is not None:
            amyloid = np.asarray(amyloid, dtype=float)
            cols.append(amyloid.mean(axis=1))
            names.append("amyloid_mean")

        if thickness is not None:
            thickness = np.asarray(thickness, dtype=float)
            cols.append(thickness.mean(axis=1))
            names.append("thickness_mean")

        if apoe4_dose is not None:
            cols.append(np.asarray(apoe4_dose, dtype=float).ravel())
            names.append("apoe4_dose")

        if plasma_ptau181 is not None:
            ptau = np.asarray(plasma_ptau181, dtype=float).ravel()
            observed_mask = np.isfinite(ptau).astype(float)
            imputed = np.where(observed_mask > 0, ptau, ptau_train_median)
            cols += [imputed, observed_mask]
            names += ["plasma_ptau181", "plasma_ptau181_observed"]

        return np.column_stack(cols), names  # (n, n_features)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        baseline: np.ndarray,
        observed: np.ndarray,
        time_years: np.ndarray,
        *,
        amyloid: np.ndarray | None,
        thickness: np.ndarray | None,
        apoe4_dose: np.ndarray | None = None,
        plasma_ptau181: np.ndarray | None = None,
        train_indices: np.ndarray,
        rho_bounds: tuple[float, float] = (0.0, 10.0),
        alpha_bounds: tuple[float, float] = (0.0, 10.0),
        per_pair_delta_scale: float = 5.0,
        per_pair_maxiter: int = 40,
        ridge_cv_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
        backbone_maxiter: int = 80,
    ) -> ASFKPPFitResult:
        baseline = np.asarray(baseline, dtype=float)
        observed = np.asarray(observed, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        train = np.asarray(train_indices, dtype=int)

        # ---- Stage 1: global FKPP backbone ----
        backbone = GraphFKPPModel(
            self.original_laplacian,
            steps_per_year=self.steps_per_year,
            laplacian_normalization=self.laplacian_normalization,
        )
        stage1 = backbone.fit_global_parameters(
            baseline[train], observed[train], time_years[train],
            rho_bounds=rho_bounds, alpha_bounds=alpha_bounds,
            maxiter=backbone_maxiter,
        )
        rho, alpha = stage1.rho, stage1.alpha

        # Bounds for individual offsets: keep rho_i > 0, reasonable range
        dr_lo = -rho * 0.99
        dr_hi = rho * float(per_pair_delta_scale)
        da_lo = -alpha * 0.99
        da_hi = alpha * float(per_pair_delta_scale)

        # ---- Stage 2a: per-pair parameter fitting ----
        n_train = len(train)
        delta_rho = np.zeros(n_train)
        delta_alpha = np.zeros(n_train)
        n_success = 0

        for idx, i in enumerate(train):
            dr, da, ok = self._fit_one_pair(
                baseline[i], observed[i], float(time_years[i]),
                rho, alpha,
                (dr_lo, dr_hi), (da_lo, da_hi),
                backbone, per_pair_maxiter,
            )
            delta_rho[idx] = dr
            delta_alpha[idx] = da
            n_success += int(ok)

        stage2a_mse = self._eval_mse(
            baseline[train], observed[train], time_years[train],
            rho, alpha, delta_rho, delta_alpha, backbone,
        )

        # ---- Stage 2b: amortisation ----
        ptau_median = 0.0
        if plasma_ptau181 is not None:
            ptau_train = np.asarray(plasma_ptau181, dtype=float).ravel()[train]
            finite_ptau = ptau_train[np.isfinite(ptau_train)]
            ptau_median = float(np.median(finite_ptau)) if finite_ptau.size > 0 else 0.0

        def _slice(arr: np.ndarray | None) -> np.ndarray | None:
            return arr[train] if arr is not None else None

        X_train, feat_names = self.build_features(
            baseline[train],
            _slice(amyloid), _slice(thickness),
            _slice(apoe4_dose), _slice(plasma_ptau181),
            ptau_train_median=ptau_median,
        )

        feat_mean = X_train.mean(axis=0)
        feat_scale = X_train.std(axis=0)
        feat_scale[feat_scale < 1.0e-8] = 1.0
        X_scaled = (X_train - feat_mean) / feat_scale

        rho_coef, rho_intercept, ridge_alpha_rho = _ridge_cv(
            X_scaled, delta_rho, list(ridge_cv_alphas)
        )
        alpha_coef, alpha_intercept, ridge_alpha_alpha = _ridge_cv(
            X_scaled, delta_alpha, list(ridge_cv_alphas)
        )

        dr_hat = X_scaled @ rho_coef + rho_intercept
        da_hat = X_scaled @ alpha_coef + alpha_intercept

        stage2b_mse = self._eval_mse(
            baseline[train], observed[train], time_years[train],
            rho, alpha, dr_hat, da_hat, backbone,
        )

        r2_rho = _r2(delta_rho, dr_hat)
        r2_alpha = _r2(delta_alpha, da_hat)

        return ASFKPPFitResult(
            rho=rho, alpha=alpha,
            rho_coef=rho_coef.copy(),
            rho_intercept=float(rho_intercept),
            alpha_coef=alpha_coef.copy(),
            alpha_intercept=float(alpha_intercept),
            feature_names=feat_names,
            feature_mean=feat_mean.copy(),
            feature_scale=feat_scale.copy(),
            delta_rho_train=delta_rho.copy(),
            delta_alpha_train=delta_alpha.copy(),
            stage1_train_mse=float(stage1.train_mse),
            stage2a_train_mse=float(stage2a_mse),
            stage2b_train_mse=float(stage2b_mse),
            amortisation_rho_r2=float(r2_rho),
            amortisation_alpha_r2=float(r2_alpha),
            ridge_alpha_rho=float(ridge_alpha_rho),
            ridge_alpha_alpha=float(ridge_alpha_alpha),
            n_features=X_train.shape[1],
            n_eigenmodes=self.n_eigenmodes,
            per_pair_success_frac=float(n_success / max(n_train, 1)),
            ptau_train_median=float(ptau_median),
            backbone_laplacian_scale=float(self.laplacian_scale),
            backbone_laplacian_normalization=self.laplacian_normalization,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        baseline: np.ndarray,
        time_years: np.ndarray,
        fit: ASFKPPFitResult,
        *,
        amyloid: np.ndarray | None,
        thickness: np.ndarray | None,
        apoe4_dose: np.ndarray | None = None,
        plasma_ptau181: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict using amortised per-pair parameters."""
        baseline = np.asarray(baseline, dtype=float)
        time_years = np.asarray(time_years, dtype=float)

        X, _ = self.build_features(
            baseline, amyloid, thickness, apoe4_dose, plasma_ptau181,
            ptau_train_median=fit.ptau_train_median,
        )
        X_scaled = (X - fit.feature_mean) / fit.feature_scale
        delta_rho = X_scaled @ fit.rho_coef + fit.rho_intercept
        delta_alpha = X_scaled @ fit.alpha_coef + fit.alpha_intercept

        backbone = GraphFKPPModel(
            self.original_laplacian,
            steps_per_year=self.steps_per_year,
            laplacian_normalization=self.laplacian_normalization,
        )
        n = baseline.shape[0]
        predicted = np.empty_like(baseline)
        for i in range(n):
            rho_i = max(float(fit.rho + delta_rho[i]), 0.0)
            alpha_i = float(fit.alpha + delta_alpha[i])
            predicted[i] = backbone.predict(
                baseline[[i]], time_years[[i]], rho=rho_i, alpha=alpha_i
            )[0]
        return predicted

    def predict_offsets(
        self,
        baseline: np.ndarray,
        fit: ASFKPPFitResult,
        *,
        amyloid: np.ndarray | None,
        thickness: np.ndarray | None,
        apoe4_dose: np.ndarray | None = None,
        plasma_ptau181: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (delta_rho, delta_alpha) for inspection without running FKPP."""
        baseline = np.asarray(baseline, dtype=float)
        X, _ = self.build_features(
            baseline, amyloid, thickness, apoe4_dose, plasma_ptau181,
            ptau_train_median=fit.ptau_train_median,
        )
        X_scaled = (X - fit.feature_mean) / fit.feature_scale
        return (
            X_scaled @ fit.rho_coef + fit.rho_intercept,
            X_scaled @ fit.alpha_coef + fit.alpha_intercept,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_one_pair(
        self,
        s0: np.ndarray,
        s1: np.ndarray,
        t: float,
        rho: float,
        alpha: float,
        dr_bounds: tuple[float, float],
        da_bounds: tuple[float, float],
        backbone: GraphFKPPModel,
        maxiter: int,
    ) -> tuple[float, float, bool]:
        """Stage 2a: find (delta_rho, delta_alpha) for a single pair."""
        s0_row = s0.reshape(1, -1)
        t_arr = np.array([t])

        def obj(params: np.ndarray) -> float:
            rho_i = max(rho + float(params[0]), 0.0)
            alpha_i = alpha + float(params[1])
            pred = backbone.predict(s0_row, t_arr, rho=rho_i, alpha=alpha_i)
            return float(np.mean((pred[0] - s1) ** 2))

        res = minimize(
            obj, np.zeros(2), method="L-BFGS-B",
            bounds=[dr_bounds, da_bounds],
            options={"maxiter": maxiter, "ftol": 1.0e-10, "gtol": 1.0e-6},
        )
        return float(res.x[0]), float(res.x[1]), bool(res.success)

    def _eval_mse(
        self,
        baseline: np.ndarray,
        observed: np.ndarray,
        time_years: np.ndarray,
        rho: float,
        alpha: float,
        delta_rho: np.ndarray,
        delta_alpha: np.ndarray,
        backbone: GraphFKPPModel,
    ) -> float:
        total_se, total_n = 0.0, 0
        for i in range(baseline.shape[0]):
            rho_i = max(rho + float(delta_rho[i]), 0.0)
            alpha_i = alpha + float(delta_alpha[i])
            pred = backbone.predict(
                baseline[[i]], time_years[[i]], rho=rho_i, alpha=alpha_i
            )
            total_se += float(np.sum((pred[0] - observed[i]) ** 2))
            total_n += observed.shape[1]
        return total_se / max(total_n, 1)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, float, float]:
    """Fit ridge with leave-one-out CV alpha selection. Returns (coef, intercept, best_alpha)."""
    from sklearn.linear_model import RidgeCV
    model = RidgeCV(alphas=alphas, fit_intercept=True, scoring="neg_mean_squared_error")
    model.fit(X, y)
    return np.asarray(model.coef_, dtype=float), float(model.intercept_), float(model.alpha_)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1.0e-12 else 0.0


def amortisation_term_rows(fit: ASFKPPFitResult) -> list[dict[str, Any]]:
    """Return sorted term rows for both amortisation models (for inspection/CSV)."""
    rows = []
    for name, cr, ca in zip(fit.feature_names, fit.rho_coef, fit.alpha_coef, strict=True):
        rows.append({
            "term": name,
            "rho_coefficient": float(cr),
            "alpha_coefficient": float(ca),
            "rho_abs": abs(float(cr)),
            "alpha_abs": abs(float(ca)),
        })
    return sorted(rows, key=lambda r: r["rho_abs"] + r["alpha_abs"], reverse=True)
