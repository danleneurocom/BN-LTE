"""Network diffusion model forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class NDMFitResult:
    rho: float
    train_mse: float
    optimizer_success: bool
    optimizer_message: str


class NetworkDiffusionModel:
    """NDM with exact solution ``S(t) = exp(-rho * t * L) S(0)``."""

    def __init__(self, laplacian: np.ndarray):
        laplacian = np.asarray(laplacian, dtype=float)
        if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
            raise ValueError(f"Laplacian must be square, got shape {laplacian.shape}.")
        self.laplacian = laplacian
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        self.eigenvalues = np.clip(eigenvalues, 0.0, None)
        self.eigenvectors = eigenvectors

    def predict(self, baseline: np.ndarray, time_years: np.ndarray, rho: float) -> np.ndarray:
        baseline = np.asarray(baseline, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        if baseline.ndim == 1:
            baseline = baseline.reshape(1, -1)
        if baseline.shape[1] != self.laplacian.shape[0]:
            raise ValueError(
                f"Baseline vectors have {baseline.shape[1]} regions, expected {self.laplacian.shape[0]}."
            )
        if time_years.shape[0] != baseline.shape[0]:
            raise ValueError("time_years must have one value per baseline vector.")

        coefficients = baseline @ self.eigenvectors
        decay = np.exp(-float(rho) * time_years[:, None] * self.eigenvalues[None, :])
        return (coefficients * decay) @ self.eigenvectors.T

    def fit_global_rho(
        self,
        baseline: np.ndarray,
        observed: np.ndarray,
        time_years: np.ndarray,
        bounds: tuple[float, float],
    ) -> NDMFitResult:
        baseline = np.asarray(baseline, dtype=float)
        observed = np.asarray(observed, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        if baseline.shape != observed.shape:
            raise ValueError("baseline and observed arrays must have the same shape.")
        if baseline.shape[0] == 0:
            raise ValueError("Cannot fit NDM rho with zero training rows.")

        def objective(rho: float) -> float:
            predicted = self.predict(baseline, time_years, rho)
            return float(np.mean((predicted - observed) ** 2))

        result = minimize_scalar(objective, bounds=bounds, method="bounded")
        return NDMFitResult(
            rho=float(result.x),
            train_mse=float(result.fun),
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
        )
