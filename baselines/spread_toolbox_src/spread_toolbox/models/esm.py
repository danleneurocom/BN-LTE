"""Epidemic spreading model forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class ESMFitResult:
    beta: float
    train_mse: float
    optimizer_success: bool
    optimizer_message: str


class EpidemicSpreadingModel:
    """Saturating epidemic model ``dS/dt = beta * (1 - S) * W S``."""

    def __init__(self, adjacency: np.ndarray, *, steps_per_year: int = 12):
        adjacency = np.asarray(adjacency, dtype=float)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Adjacency must be square, got shape {adjacency.shape}.")
        if np.any(adjacency < 0):
            raise ValueError("Adjacency matrix must not contain negative edge weights.")
        if steps_per_year < 1:
            raise ValueError("steps_per_year must be at least 1.")

        self.adjacency = adjacency
        self.spread_matrix = row_normalize(adjacency)
        self.steps_per_year = int(steps_per_year)

    def predict(self, baseline: np.ndarray, time_years: np.ndarray, beta: float) -> np.ndarray:
        baseline = np.asarray(baseline, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        if baseline.ndim == 1:
            baseline = baseline.reshape(1, -1)
        if baseline.shape[1] != self.adjacency.shape[0]:
            raise ValueError(
                f"Baseline vectors have {baseline.shape[1]} regions, expected {self.adjacency.shape[0]}."
            )
        if time_years.shape[0] != baseline.shape[0]:
            raise ValueError("time_years must have one value per baseline vector.")
        if np.any(time_years < 0):
            raise ValueError("time_years must be non-negative.")

        predicted = np.empty_like(baseline, dtype=float)
        for index, (state, elapsed_years) in enumerate(zip(baseline, time_years)):
            predicted[index] = self._integrate_state(state, float(elapsed_years), float(beta))
        return predicted

    def fit_global_beta(
        self,
        baseline: np.ndarray,
        observed: np.ndarray,
        time_years: np.ndarray,
        bounds: tuple[float, float],
    ) -> ESMFitResult:
        baseline = np.asarray(baseline, dtype=float)
        observed = np.asarray(observed, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        if baseline.shape != observed.shape:
            raise ValueError("baseline and observed arrays must have the same shape.")
        if baseline.shape[0] == 0:
            raise ValueError("Cannot fit ESM beta with zero training rows.")

        def objective(beta: float) -> float:
            predicted = self.predict(baseline, time_years, beta)
            return float(np.mean((predicted - observed) ** 2))

        result = minimize_scalar(objective, bounds=bounds, method="bounded")
        return ESMFitResult(
            beta=float(result.x),
            train_mse=float(result.fun),
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
        )

    def _integrate_state(self, initial_state: np.ndarray, elapsed_years: float, beta: float) -> np.ndarray:
        state = np.clip(np.asarray(initial_state, dtype=float), 0.0, 1.0)
        if elapsed_years == 0.0 or beta == 0.0:
            return state

        step_count = max(1, int(np.ceil(elapsed_years * self.steps_per_year)))
        dt = elapsed_years / step_count
        for _ in range(step_count):
            k1 = self._derivative(state, beta)
            k2 = self._derivative(np.clip(state + 0.5 * dt * k1, 0.0, 1.0), beta)
            k3 = self._derivative(np.clip(state + 0.5 * dt * k2, 0.0, 1.0), beta)
            k4 = self._derivative(np.clip(state + dt * k3, 0.0, 1.0), beta)
            state = np.clip(state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0)
        return state

    def _derivative(self, state: np.ndarray, beta: float) -> np.ndarray:
        network_drive = self.spread_matrix @ state
        return float(beta) * (1.0 - state) * network_drive


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1)
    normalized = np.zeros_like(matrix, dtype=float)
    np.divide(matrix, row_sums[:, None], out=normalized, where=row_sums[:, None] > 0.0)
    return normalized
