"""Full graph SIR forecasting model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .esm import row_normalize
from .fkpp import midpoint


@dataclass
class SIRFitResult:
    beta: float
    gamma: float
    train_mse: float
    optimizer_success: bool
    optimizer_message: str
    optimizer_iterations: int
    optimizer_evaluations: int


class GraphSIRModel:
    """Graph SIR model with tau represented by the infected compartment.

    Equations per region are:

    dS_i/dt = -beta * S_i * (W I)_i
    dI_i/dt =  beta * S_i * (W I)_i - gamma * I_i
    dR_i/dt =  gamma * I_i

    We observe only tau, so baseline scaled tau initializes ``I`` and ``R`` is
    initialized to zero. The predicted tau is the future ``I`` compartment.
    """

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

    def predict(self, baseline: np.ndarray, time_years: np.ndarray, *, beta: float, gamma: float) -> np.ndarray:
        baseline = np.asarray(baseline, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        if baseline.ndim == 1:
            baseline = baseline.reshape(1, -1)
        if baseline.shape[1] != self.adjacency.shape[0]:
            raise ValueError(f"Baseline vectors have {baseline.shape[1]} regions, expected {self.adjacency.shape[0]}.")
        if time_years.shape[0] != baseline.shape[0]:
            raise ValueError("time_years must have one value per baseline vector.")
        if np.any(time_years < 0):
            raise ValueError("time_years must be non-negative.")

        predicted = np.empty_like(baseline, dtype=float)
        for index, (state, elapsed_years) in enumerate(zip(baseline, time_years, strict=True)):
            _, infected, _ = self._integrate_state(state, float(elapsed_years), float(beta), float(gamma))
            predicted[index] = infected
        return predicted

    def fit_global_parameters(
        self,
        baseline: np.ndarray,
        observed: np.ndarray,
        time_years: np.ndarray,
        *,
        beta_bounds: tuple[float, float],
        gamma_bounds: tuple[float, float],
        maxiter: int = 80,
    ) -> SIRFitResult:
        baseline = np.asarray(baseline, dtype=float)
        observed = np.asarray(observed, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        if baseline.shape != observed.shape:
            raise ValueError("baseline and observed arrays must have the same shape.")
        if baseline.shape[0] == 0:
            raise ValueError("Cannot fit SIR parameters with zero training rows.")

        bounds = [beta_bounds, gamma_bounds]
        initial = np.asarray([midpoint(beta_bounds), midpoint(gamma_bounds)], dtype=float)

        def objective(parameters: np.ndarray) -> float:
            beta, gamma = parameters
            predicted = self.predict(baseline, time_years, beta=float(beta), gamma=float(gamma))
            return float(np.mean((predicted - observed) ** 2))

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter)},
        )
        beta, gamma = result.x
        return SIRFitResult(
            beta=float(beta),
            gamma=float(gamma),
            train_mse=float(result.fun),
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
            optimizer_iterations=int(getattr(result, "nit", 0)),
            optimizer_evaluations=int(getattr(result, "nfev", 0)),
        )

    def _integrate_state(
        self,
        initial_infected: np.ndarray,
        elapsed_years: float,
        beta: float,
        gamma: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        infected = np.clip(np.asarray(initial_infected, dtype=float), 0.0, 1.0)
        susceptible = 1.0 - infected
        removed = np.zeros_like(infected, dtype=float)
        if elapsed_years == 0.0 or (float(beta) == 0.0 and float(gamma) == 0.0):
            return susceptible, infected, removed

        step_count = max(1, int(np.ceil(elapsed_years * self.steps_per_year)))
        dt = elapsed_years / step_count
        for _ in range(step_count):
            k1 = self._derivative(susceptible, infected, removed, beta, gamma)
            k2 = self._derivative(
                *self._bounded_state(susceptible, infected, removed, k1, 0.5 * dt),
                beta,
                gamma,
            )
            k3 = self._derivative(
                *self._bounded_state(susceptible, infected, removed, k2, 0.5 * dt),
                beta,
                gamma,
            )
            k4 = self._derivative(*self._bounded_state(susceptible, infected, removed, k3, dt), beta, gamma)
            susceptible = susceptible + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            infected = infected + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            removed = removed + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            susceptible, infected, removed = normalize_compartments(susceptible, infected, removed)
        return susceptible, infected, removed

    def _bounded_state(
        self,
        susceptible: np.ndarray,
        infected: np.ndarray,
        removed: np.ndarray,
        derivative: tuple[np.ndarray, np.ndarray, np.ndarray],
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return normalize_compartments(
            susceptible + float(dt) * derivative[0],
            infected + float(dt) * derivative[1],
            removed + float(dt) * derivative[2],
        )

    def _derivative(
        self,
        susceptible: np.ndarray,
        infected: np.ndarray,
        removed: np.ndarray,
        beta: float,
        gamma: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del removed
        infection_drive = float(beta) * susceptible * (self.spread_matrix @ infected)
        clearance = float(gamma) * infected
        return -infection_drive, infection_drive - clearance, clearance


def normalize_compartments(
    susceptible: np.ndarray,
    infected: np.ndarray,
    removed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    susceptible = np.clip(np.asarray(susceptible, dtype=float), 0.0, 1.0)
    infected = np.clip(np.asarray(infected, dtype=float), 0.0, 1.0)
    removed = np.clip(np.asarray(removed, dtype=float), 0.0, 1.0)
    total = susceptible + infected + removed
    excess = total > 1.0
    if np.any(excess):
        susceptible[excess] /= total[excess]
        infected[excess] /= total[excess]
        removed[excess] /= total[excess]
    return susceptible, infected, removed
