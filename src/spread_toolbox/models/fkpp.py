"""Graph Fisher-Kolmogorov-Petrovskii-Piskunov forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class FKPPFitResult:
    rho: float
    alpha: float
    train_mse: float
    optimizer_success: bool
    optimizer_message: str
    optimizer_iterations: int
    optimizer_evaluations: int


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
