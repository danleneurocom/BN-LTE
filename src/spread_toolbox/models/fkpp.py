"""Graph Fisher-Kolmogorov-Petrovskii-Piskunov forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


@dataclass
class FKPPFitResult:
    rho: float
    alpha: float
    train_mse: float
    optimizer_success: bool
    optimizer_message: str
    optimizer_iterations: int
    optimizer_evaluations: int


@dataclass
class LocalFKPPComponentParameters:
    u0: np.ndarray
    cc: np.ndarray
    low_component_mean: np.ndarray
    high_component_mean: np.ndarray
    low_component_std: np.ndarray
    high_component_std: np.ndarray
    carrying_capacity_quantile: float

    def clip(self, values: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(values, dtype=float), self.u0, self.cc)


class GraphFKPPModel:
    """Graph FKPP model ``dS/dt = -rho L S + alpha S(1 - S)``."""

    def __init__(
        self,
        laplacian: np.ndarray,
        *,
        steps_per_year: int = 12,
        laplacian_normalization: str = "spectral",
    ):
        laplacian = np.asarray(laplacian, dtype=float)
        if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
            raise ValueError(f"Laplacian must be square, got shape {laplacian.shape}.")
        if steps_per_year < 1:
            raise ValueError("steps_per_year must be at least 1.")

        self.original_laplacian = laplacian
        self.laplacian, self.laplacian_scale = normalize_laplacian(laplacian, laplacian_normalization)
        self.steps_per_year = int(steps_per_year)
        self.laplacian_normalization = laplacian_normalization

    def predict(self, baseline: np.ndarray, time_years: np.ndarray, *, rho: float, alpha: float) -> np.ndarray:
        baseline = np.asarray(baseline, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        if baseline.ndim == 1:
            baseline = baseline.reshape(1, -1)
        if baseline.shape[1] != self.laplacian.shape[0]:
            raise ValueError(f"Baseline vectors have {baseline.shape[1]} regions, expected {self.laplacian.shape[0]}.")
        if time_years.shape[0] != baseline.shape[0]:
            raise ValueError("time_years must have one value per baseline vector.")
        if np.any(time_years < 0):
            raise ValueError("time_years must be non-negative.")

        states = np.clip(baseline, 0.0, 1.0)
        if float(rho) == 0.0 and float(alpha) == 0.0:
            return states

        remaining = time_years.astype(float).copy()
        step_dt = 1.0 / self.steps_per_year
        while np.any(remaining > 0.0):
            active = remaining > 0.0
            dt = np.minimum(step_dt, remaining[active])[:, None]
            active_states = states[active]
            states[active] = self._rk4_step(active_states, dt, float(rho), float(alpha))
            remaining[active] -= dt[:, 0]
        return states

    def fit_global_parameters(
        self,
        baseline: np.ndarray,
        observed: np.ndarray,
        time_years: np.ndarray,
        *,
        rho_bounds: tuple[float, float],
        alpha_bounds: tuple[float, float],
        maxiter: int = 80,
    ) -> FKPPFitResult:
        baseline = np.asarray(baseline, dtype=float)
        observed = np.asarray(observed, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        if baseline.shape != observed.shape:
            raise ValueError("baseline and observed arrays must have the same shape.")
        if baseline.shape[0] == 0:
            raise ValueError("Cannot fit FKPP parameters with zero training rows.")

        bounds = [rho_bounds, alpha_bounds]
        initial = np.asarray(
            [
                midpoint(rho_bounds),
                midpoint(alpha_bounds),
            ],
            dtype=float,
        )

        def objective(parameters: np.ndarray) -> float:
            rho, alpha = parameters
            predicted = self.predict(baseline, time_years, rho=float(rho), alpha=float(alpha))
            return float(np.mean((predicted - observed) ** 2))

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter)},
        )
        rho, alpha = result.x
        return FKPPFitResult(
            rho=float(rho),
            alpha=float(alpha),
            train_mse=float(result.fun),
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
            optimizer_iterations=int(getattr(result, "nit", 0)),
            optimizer_evaluations=int(getattr(result, "nfev", 0)),
        )

    def _rk4_step(self, states: np.ndarray, dt: np.ndarray, rho: float, alpha: float) -> np.ndarray:
        k1 = self._derivative(states, rho, alpha)
        k2 = self._derivative(np.clip(states + 0.5 * dt * k1, 0.0, 1.0), rho, alpha)
        k3 = self._derivative(np.clip(states + 0.5 * dt * k2, 0.0, 1.0), rho, alpha)
        k4 = self._derivative(np.clip(states + dt * k3, 0.0, 1.0), rho, alpha)
        return np.clip(states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0)

    def _derivative(self, states: np.ndarray, rho: float, alpha: float) -> np.ndarray:
        diffusion = -float(rho) * (states @ self.laplacian.T)
        growth = float(alpha) * states * (1.0 - states)
        return diffusion + growth


class LocalFKPPModel:
    """Chaggar-style local FKPP with regional ``u0`` and ``cc``."""

    def __init__(
        self,
        laplacian: np.ndarray,
        *,
        u0: np.ndarray,
        cc: np.ndarray,
        steps_per_year: int = 12,
        laplacian_normalization: str = "spectral",
    ):
        laplacian = np.asarray(laplacian, dtype=float)
        u0 = np.asarray(u0, dtype=float)
        cc = np.asarray(cc, dtype=float)
        if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
            raise ValueError(f"Laplacian must be square, got shape {laplacian.shape}.")
        if u0.ndim != 1 or cc.ndim != 1:
            raise ValueError("u0 and cc must be one-dimensional regional vectors.")
        if laplacian.shape[0] != u0.size or u0.shape != cc.shape:
            raise ValueError("Laplacian, u0, and cc region counts must match.")
        if np.any(cc <= u0):
            raise ValueError("Every local FKPP carrying capacity must be greater than u0.")
        if steps_per_year < 1:
            raise ValueError("steps_per_year must be at least 1.")

        self.original_laplacian = laplacian
        self.laplacian, self.laplacian_scale = normalize_laplacian(laplacian, laplacian_normalization)
        self.u0 = u0
        self.cc = cc
        self.capacity = cc - u0
        self.steps_per_year = int(steps_per_year)
        self.laplacian_normalization = laplacian_normalization

    def predict(self, baseline: np.ndarray, time_years: np.ndarray, *, rho: float, alpha: float) -> np.ndarray:
        baseline = np.asarray(baseline, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        if baseline.ndim == 1:
            baseline = baseline.reshape(1, -1)
        if baseline.shape[1] != self.laplacian.shape[0]:
            raise ValueError(f"Baseline vectors have {baseline.shape[1]} regions, expected {self.laplacian.shape[0]}.")
        if time_years.shape[0] != baseline.shape[0]:
            raise ValueError("time_years must have one value per baseline vector.")
        if np.any(time_years < 0):
            raise ValueError("time_years must be non-negative.")

        states = np.clip(baseline, self.u0, self.cc)
        if float(rho) == 0.0 and float(alpha) == 0.0:
            return states

        remaining = time_years.astype(float).copy()
        step_dt = 1.0 / self.steps_per_year
        while np.any(remaining > 0.0):
            active = remaining > 0.0
            dt = np.minimum(step_dt, remaining[active])[:, None]
            active_states = states[active]
            states[active] = self._rk4_step(active_states, dt, float(rho), float(alpha))
            remaining[active] -= dt[:, 0]
        return states

    def fit_global_parameters(
        self,
        baseline: np.ndarray,
        observed: np.ndarray,
        time_years: np.ndarray,
        *,
        rho_bounds: tuple[float, float],
        alpha_bounds: tuple[float, float],
        maxiter: int = 80,
    ) -> FKPPFitResult:
        baseline = np.asarray(baseline, dtype=float)
        observed = np.asarray(observed, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        if baseline.shape != observed.shape:
            raise ValueError("baseline and observed arrays must have the same shape.")
        if baseline.shape[0] == 0:
            raise ValueError("Cannot fit local FKPP parameters with zero training rows.")

        bounds = [rho_bounds, alpha_bounds]
        initial = np.asarray([midpoint(rho_bounds), midpoint(alpha_bounds)], dtype=float)

        def objective(parameters: np.ndarray) -> float:
            rho, alpha = parameters
            predicted = self.predict(baseline, time_years, rho=float(rho), alpha=float(alpha))
            return float(np.mean((predicted - observed) ** 2))

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter)},
        )
        rho, alpha = result.x
        return FKPPFitResult(
            rho=float(rho),
            alpha=float(alpha),
            train_mse=float(result.fun),
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
            optimizer_iterations=int(getattr(result, "nit", 0)),
            optimizer_evaluations=int(getattr(result, "nfev", 0)),
        )

    def _rk4_step(self, states: np.ndarray, dt: np.ndarray, rho: float, alpha: float) -> np.ndarray:
        k1 = self._derivative(states, rho, alpha)
        k2 = self._derivative(np.clip(states + 0.5 * dt * k1, self.u0, self.cc), rho, alpha)
        k3 = self._derivative(np.clip(states + 0.5 * dt * k2, self.u0, self.cc), rho, alpha)
        k4 = self._derivative(np.clip(states + dt * k3, self.u0, self.cc), rho, alpha)
        return np.clip(states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), self.u0, self.cc)

    def _derivative(self, states: np.ndarray, rho: float, alpha: float) -> np.ndarray:
        relative_state = states - self.u0
        diffusion = -float(rho) * (relative_state @ self.laplacian.T)
        growth = float(alpha) * relative_state * (self.capacity - relative_state)
        return diffusion + growth


def fit_local_fkpp_components(
    *arrays: np.ndarray,
    carrying_capacity_quantile: float = 0.99,
    random_seed: int = 20260507,
    min_std: float = 1.0e-4,
    min_capacity: float = 1.0e-3,
) -> LocalFKPPComponentParameters:
    if not arrays:
        raise ValueError("At least one array is required to estimate local FKPP components.")
    if not 0.5 < carrying_capacity_quantile < 1.0:
        raise ValueError("carrying_capacity_quantile must be between 0.5 and 1.0.")

    normalized_arrays = [np.asarray(array, dtype=float) for array in arrays]
    region_count = normalized_arrays[0].shape[1]
    for array in normalized_arrays:
        if array.ndim != 2:
            raise ValueError("Component inputs must be two-dimensional arrays.")
        if array.shape[1] != region_count:
            raise ValueError("All component inputs must have the same number of regions.")

    values_by_region = np.vstack(normalized_arrays)
    u0 = np.empty(region_count, dtype=float)
    cc = np.empty(region_count, dtype=float)
    low_mean = np.empty(region_count, dtype=float)
    high_mean = np.empty(region_count, dtype=float)
    low_std = np.empty(region_count, dtype=float)
    high_std = np.empty(region_count, dtype=float)

    z_score = float(norm.ppf(carrying_capacity_quantile))
    for region_index in range(region_count):
        values = values_by_region[:, region_index]
        values = values[np.isfinite(values)]
        if values.size < 4 or np.unique(values).size < 2:
            low, high, low_s, high_s = fallback_components(values, min_std)
        else:
            model = GaussianMixture(
                n_components=2,
                covariance_type="diag",
                reg_covar=min_std**2,
                random_state=int(random_seed) + region_index,
            )
            model.fit(values.reshape(-1, 1))
            means = model.means_.reshape(-1)
            variances = model.covariances_.reshape(-1)
            order = np.argsort(means)
            low_index, high_index = int(order[0]), int(order[1])
            low = float(means[low_index])
            high = float(means[high_index])
            low_s = float(np.sqrt(max(variances[low_index], min_std**2)))
            high_s = float(np.sqrt(max(variances[high_index], min_std**2)))

        capacity = high + z_score * high_s
        if not np.isfinite(capacity) or capacity <= low + min_capacity:
            capacity = max(float(np.quantile(values, carrying_capacity_quantile)), low + min_capacity)

        u0[region_index] = low
        cc[region_index] = capacity
        low_mean[region_index] = low
        high_mean[region_index] = high
        low_std[region_index] = low_s
        high_std[region_index] = high_s

    return LocalFKPPComponentParameters(
        u0=u0,
        cc=cc,
        low_component_mean=low_mean,
        high_component_mean=high_mean,
        low_component_std=low_std,
        high_component_std=high_std,
        carrying_capacity_quantile=carrying_capacity_quantile,
    )


def fallback_components(values: np.ndarray, min_std: float) -> tuple[float, float, float, float]:
    if values.size == 0:
        return 0.0, min_std, min_std, min_std
    low = float(np.quantile(values, 0.25))
    high = float(np.quantile(values, 0.75))
    std = float(max(np.std(values), min_std))
    if high <= low:
        high = low + std
    return low, high, std, std


def normalize_laplacian(laplacian: np.ndarray, method: str) -> tuple[np.ndarray, float]:
    laplacian = np.asarray(laplacian, dtype=float)
    if method == "none":
        return laplacian, 1.0
    if method != "spectral":
        raise ValueError(f"Unsupported laplacian_normalization: {method}")

    symmetric = 0.5 * (laplacian + laplacian.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = float(np.max(np.abs(eigenvalues)))
    if scale <= 0.0:
        return laplacian, 1.0
    return laplacian / scale, scale


def midpoint(bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return 0.5 * (float(lower) + float(upper))
