"""Biologically-Modulated FKPP (Bio-FKPP) — full generalised equation.

Implements the three-term modulated FKPP from the SINDy residual analysis:

  dS_i/dt = -rho_eff_i * (L S)_i                                  [modulated diffusion]
             + alpha_eff_i * S_i*(1-S_i)                           [modulated growth]
             - c_eff_i * S_i                                        [local clearance]
             + seeding_i                                            [static seeding from S0]

where all modulated fields are linear in biological covariates:

  rho_eff_i   = rho   + delta_rho * thickness_i
  alpha_eff_i = alpha + beta_g * amyloid_i + beta_t * thickness_i + beta_a * apoe4_s
  c_eff_i     = lambda_c * thickness_i
  seeding_i   = gamma_a * amyloid_i * S0_i + gamma_t * thickness_i * S0_i

Residual rank → biological motivation:
  Rank 1  amyloid*S(1-S)  +0.00611  beta_g    > 0  amyloid-driven autocatalytic growth
  Rank 2  thick*S(1-S)    -0.00606  beta_t    < 0  cortical integrity buffers growth
  Rank 3  thick*S0        +0.00538  gamma_t   > 0  high-tau thick regions escape buffer
  Rank 4  amyloid*S0      -0.00355  gamma_a   ?    amyloid×tau seeding (sign TBD by ODE)
  Rank 8  diffusion_drive +0.00149  delta_rho > 0  FKPP under-predicts diffusion globally
  PDF §2  c(x,b)*S        —         lambda_c  > 0  thickness-dependent clearance

Two-stage:
  Stage 1: global FKPP backbone -> freeze rho, alpha
  Stage 2: jointly optimize (delta_rho, beta_g, beta_t, beta_a, lambda_c, gamma_a, gamma_t)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .fkpp import GraphFKPPModel, normalize_laplacian


@dataclass
class BioFKPPFitResult:
    rho: float
    alpha: float
    # ρ(x,b): diffusion modulation
    delta_rho_thickness: float     # rho_eff = rho + delta_rho * thickness; expected > 0
    # α(x,b): growth modulation
    beta_growth_amyloid: float     # expected > 0
    beta_growth_thickness: float   # expected < 0
    beta_growth_apoe4: float       # expected > 0
    # c(x,b): clearance modulation
    lambda_clearance_thickness: float  # c = lambda_c * thickness; expected > 0
    # static seeding: gamma * cov * S0
    gamma_seeding_amyloid: float
    gamma_seeding_thickness: float
    # connectivity-biology seeding (PySR-discovered): amyloid * thickness * S0 * neighbour_tau
    gamma_seeding_connectivity: float = 0.0
    stage1_train_mse: float = 0.0
    stage2_train_mse: float = 0.0
    mse_reduction_pct: float = 0.0
    optimizer_success: bool = False
    optimizer_message: str = ""
    optimizer_iterations: int = 0
    fitted_terms: list[str] = None

    def __post_init__(self):
        if self.fitted_terms is None:
            self.fitted_terms = []


class BioFKPPModel:
    """Full generalised Bio-FKPP with modulated diffusion, growth, clearance and seeding."""

    def __init__(
        self,
        laplacian: np.ndarray,
        *,
        adjacency: np.ndarray | None = None,
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
        self.laplacian_normalization = laplacian_normalization
        self.steps_per_year = int(steps_per_year)
        # Row-stochastic adjacency for mean-neighbour-tau seeding (PySR-discovered term)
        if adjacency is not None:
            adj = np.asarray(adjacency, dtype=float)
            degree = adj.sum(axis=1)
            degree = np.where(degree > 0, degree, 1.0)
            self.adj_norm: np.ndarray | None = adj / degree[:, None]
        else:
            self.adj_norm = None

    def build_rho_eff(
        self,
        n_pairs: int,
        rho: float,
        thickness: np.ndarray | None,
        *,
        delta_rho_thickness: float,
    ) -> np.ndarray:
        """(n_pairs, n_regions) effective diffusion rate, clipped to ≥ 0."""
        n_regions = self.laplacian.shape[0]
        rho_eff = np.full((n_pairs, n_regions), float(rho), dtype=float)
        if thickness is not None:
            rho_eff = rho_eff + float(delta_rho_thickness) * np.asarray(thickness, dtype=float)
        return np.maximum(rho_eff, 0.0)

    def build_alpha_eff(
        self,
        n_pairs: int,
        alpha: float,
        amyloid: np.ndarray | None,
        thickness: np.ndarray | None,
        apoe4_dose: np.ndarray | None,
        *,
        beta_growth_amyloid: float,
        beta_growth_thickness: float,
        beta_growth_apoe4: float,
    ) -> np.ndarray:
        """(n_pairs, n_regions) effective growth rate."""
        n_regions = self.laplacian.shape[0]
        alpha_eff = np.full((n_pairs, n_regions), float(alpha), dtype=float)
        if amyloid is not None:
            alpha_eff = alpha_eff + float(beta_growth_amyloid) * np.asarray(amyloid, dtype=float)
        if thickness is not None:
            alpha_eff = alpha_eff + float(beta_growth_thickness) * np.asarray(thickness, dtype=float)
        if apoe4_dose is not None:
            alpha_eff = alpha_eff + float(beta_growth_apoe4) * np.asarray(apoe4_dose, dtype=float)[:, None]
        return alpha_eff

    def build_clearance(
        self,
        n_pairs: int,
        thickness: np.ndarray | None,
        *,
        lambda_clearance_thickness: float,
    ) -> np.ndarray:
        """(n_pairs, n_regions) clearance rate c_eff. Subtracted as c_eff * S during integration."""
        n_regions = self.laplacian.shape[0]
        clearance = np.zeros((n_pairs, n_regions), dtype=float)
        if thickness is not None:
            clearance = clearance + float(lambda_clearance_thickness) * np.asarray(thickness, dtype=float)
        return clearance

    def build_seeding(
        self,
        baseline: np.ndarray,
        amyloid: np.ndarray | None,
        thickness: np.ndarray | None,
        *,
        gamma_seeding_amyloid: float,
        gamma_seeding_thickness: float,
        gamma_seeding_connectivity: float = 0.0,
    ) -> np.ndarray:
        """(n_pairs, n_regions) constant additive seeding rate: gamma * cov * S0.

        Includes the PySR-discovered connectivity-biology term:
            gamma_seeding_connectivity * amyloid * thickness * S0 * mean_neighbour_tau
        where mean_neighbour_tau = (A_row_norm @ S0) uses the HCP adjacency.
        Only active when adjacency was provided at construction and both amyloid
        and thickness are available.
        """
        baseline = np.asarray(baseline, dtype=float)
        seeding = np.zeros_like(baseline)
        if amyloid is not None:
            seeding = seeding + float(gamma_seeding_amyloid) * np.asarray(amyloid, dtype=float) * baseline
        if thickness is not None:
            seeding = seeding + float(gamma_seeding_thickness) * np.asarray(thickness, dtype=float) * baseline
        # Connectivity-biology interaction (PySR-discovered from HCP residuals)
        if (
            self.adj_norm is not None
            and amyloid is not None
            and thickness is not None
            and float(gamma_seeding_connectivity) != 0.0
        ):
            neighbour_tau = baseline @ self.adj_norm.T        # (n_pairs, n_regions)
            seeding = seeding + (
                float(gamma_seeding_connectivity)
                * np.asarray(amyloid, dtype=float)
                * np.asarray(thickness, dtype=float)
                * baseline
                * neighbour_tau
            )
        return seeding

    def predict(
        self,
        baseline: np.ndarray,
        time_years: np.ndarray,
        *,
        rho_eff: np.ndarray,
        alpha_eff: np.ndarray,
        clearance: np.ndarray,
        seeding: np.ndarray,
    ) -> np.ndarray:
        baseline = np.asarray(baseline, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        if baseline.ndim == 1:
            baseline = baseline.reshape(1, -1)
        if baseline.shape[1] != self.laplacian.shape[0]:
            raise ValueError(f"Baseline has {baseline.shape[1]} regions, expected {self.laplacian.shape[0]}.")
        if time_years.shape[0] != baseline.shape[0]:
            raise ValueError("time_years must have one value per baseline row.")
        if np.any(time_years < 0):
            raise ValueError("time_years must be non-negative.")

        states = np.clip(baseline, 0.0, 1.0)
        remaining = time_years.astype(float).copy()
        step_dt = 1.0 / self.steps_per_year
        while np.any(remaining > 0.0):
            active = remaining > 0.0
            dt = np.minimum(step_dt, remaining[active])[:, None]
            states[active] = self._rk4_step(
                states[active], dt,
                rho_eff[active], alpha_eff[active], clearance[active], seeding[active],
            )
            remaining[active] -= dt[:, 0]
        return states

    def fit(
        self,
        baseline: np.ndarray,
        observed: np.ndarray,
        time_years: np.ndarray,
        *,
        amyloid: np.ndarray | None,
        thickness: np.ndarray | None,
        apoe4_dose: np.ndarray | None = None,
        train_indices: np.ndarray,
        rho_bounds: tuple[float, float] = (0.0, 10.0),
        alpha_bounds: tuple[float, float] = (0.0, 10.0),
        beta_bounds: tuple[float, float] = (-5.0, 5.0),
        gamma_bounds: tuple[float, float] = (-5.0, 5.0),
        delta_rho_bounds: tuple[float, float] = (-2.0, 2.0),
        lambda_c_bounds: tuple[float, float] = (-5.0, 5.0),
        maxiter: int = 120,
    ) -> BioFKPPFitResult:
        baseline = np.asarray(baseline, dtype=float)
        observed = np.asarray(observed, dtype=float)
        time_years = np.asarray(time_years, dtype=float)
        train = np.asarray(train_indices, dtype=int)
        n_pairs = baseline.shape[0]

        # Stage 1: global FKPP backbone — rho and alpha frozen for Stage 2
        backbone = GraphFKPPModel(
            self.original_laplacian,
            steps_per_year=self.steps_per_year,
            laplacian_normalization=self.laplacian_normalization,
        )
        stage1 = backbone.fit_global_parameters(
            baseline[train], observed[train], time_years[train],
            rho_bounds=rho_bounds, alpha_bounds=alpha_bounds, maxiter=maxiter,
        )
        rho = stage1.rho
        alpha = stage1.alpha

        # Build parameter registry: name → (index, bounds)
        param_names: list[str] = []
        bounds: list[tuple[float, float]] = []
        fitted_terms: list[str] = []

        if thickness is not None:
            param_names.append("dr"); bounds.append(delta_rho_bounds)
            fitted_terms.append("rho*thickness*(LS)")
        if amyloid is not None:
            param_names.append("bg"); bounds.append(beta_bounds)
            fitted_terms.append("amyloid_suvr*S*(1-S)")
        if thickness is not None:
            param_names.append("bt"); bounds.append(beta_bounds)
            fitted_terms.append("thickness*S*(1-S)")
        if apoe4_dose is not None:
            param_names.append("ba"); bounds.append(beta_bounds)
            fitted_terms.append("apoe4_dose*S*(1-S)")
        if thickness is not None:
            param_names.append("lc"); bounds.append(lambda_c_bounds)
            fitted_terms.append("thickness*S")
        if amyloid is not None:
            param_names.append("ga"); bounds.append(gamma_bounds)
            fitted_terms.append("amyloid_suvr*S0")
        if thickness is not None:
            param_names.append("gt"); bounds.append(gamma_bounds)
            fitted_terms.append("thickness*S0")
        # PySR-discovered connectivity-biology term (only when adjacency provided)
        if self.adj_norm is not None and amyloid is not None and thickness is not None:
            param_names.append("gc"); bounds.append(gamma_bounds)
            fitted_terms.append("amyloid*thickness*S0*mean_neighbour_tau")

        if not param_names:
            ae = np.full((n_pairs, self.laplacian.shape[0]), alpha, dtype=float)
            re = np.full_like(ae, rho)
            cl = np.zeros_like(ae)
            sd = np.zeros_like(baseline)
            pred = self.predict(baseline[train], time_years[train],
                                rho_eff=re[train], alpha_eff=ae[train],
                                clearance=cl[train], seeding=sd[train])
            mse = float(np.mean((pred - observed[train]) ** 2))
            return _zero_result(rho, alpha, float(stage1.train_mse), mse)

        def unpack(p: np.ndarray) -> dict[str, float]:
            return dict(zip(param_names, p.tolist(), strict=True))

        # Stage 2: jointly optimise all modulation parameters
        def objective(params: np.ndarray) -> float:
            pv = unpack(params)
            re = self.build_rho_eff(n_pairs, rho, thickness, delta_rho_thickness=pv.get("dr", 0.0))
            ae = self.build_alpha_eff(n_pairs, alpha, amyloid, thickness, apoe4_dose,
                                      beta_growth_amyloid=pv.get("bg", 0.0),
                                      beta_growth_thickness=pv.get("bt", 0.0),
                                      beta_growth_apoe4=pv.get("ba", 0.0))
            cl = self.build_clearance(n_pairs, thickness, lambda_clearance_thickness=pv.get("lc", 0.0))
            sd = self.build_seeding(baseline, amyloid, thickness,
                                    gamma_seeding_amyloid=pv.get("ga", 0.0),
                                    gamma_seeding_thickness=pv.get("gt", 0.0),
                                    gamma_seeding_connectivity=pv.get("gc", 0.0))
            pred = self.predict(baseline[train], time_years[train],
                                rho_eff=re[train], alpha_eff=ae[train],
                                clearance=cl[train], seeding=sd[train])
            return float(np.mean((pred - observed[train]) ** 2))

        result = minimize(
            objective, np.zeros(len(param_names)),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": int(maxiter)},
        )
        pv = unpack(result.x)
        s2_mse = float(result.fun)
        reduction = 100.0 * (1.0 - s2_mse / stage1.train_mse) if stage1.train_mse > 0 else 0.0

        return BioFKPPFitResult(
            rho=rho, alpha=alpha,
            delta_rho_thickness=pv.get("dr", 0.0),
            beta_growth_amyloid=pv.get("bg", 0.0),
            beta_growth_thickness=pv.get("bt", 0.0),
            beta_growth_apoe4=pv.get("ba", 0.0),
            lambda_clearance_thickness=pv.get("lc", 0.0),
            gamma_seeding_amyloid=pv.get("ga", 0.0),
            gamma_seeding_thickness=pv.get("gt", 0.0),
            gamma_seeding_connectivity=pv.get("gc", 0.0),
            stage1_train_mse=float(stage1.train_mse),
            stage2_train_mse=s2_mse,
            mse_reduction_pct=reduction,
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
            optimizer_iterations=int(getattr(result, "nit", 0)),
            fitted_terms=fitted_terms,
        )

    def _rk4_step(
        self,
        states: np.ndarray,
        dt: np.ndarray,
        rho_eff: np.ndarray,
        alpha_eff: np.ndarray,
        clearance: np.ndarray,
        seeding: np.ndarray,
    ) -> np.ndarray:
        k1 = self._derivative(states, rho_eff, alpha_eff, clearance, seeding)
        k2 = self._derivative(np.clip(states + 0.5 * dt * k1, 0.0, 1.0), rho_eff, alpha_eff, clearance, seeding)
        k3 = self._derivative(np.clip(states + 0.5 * dt * k2, 0.0, 1.0), rho_eff, alpha_eff, clearance, seeding)
        k4 = self._derivative(np.clip(states + dt * k3, 0.0, 1.0), rho_eff, alpha_eff, clearance, seeding)
        return np.clip(states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0)

    def _derivative(
        self,
        states: np.ndarray,
        rho_eff: np.ndarray,
        alpha_eff: np.ndarray,
        clearance: np.ndarray,
        seeding: np.ndarray,
    ) -> np.ndarray:
        # rho_eff modulates diffusion per region: -rho_eff_i * (LS)_i
        diffusion = -(states @ self.laplacian.T) * rho_eff
        growth = alpha_eff * states * (1.0 - states)
        damping = clearance * states          # -c_eff * S
        return diffusion + growth - damping + seeding


def _zero_result(rho: float, alpha: float, s1_mse: float, s2_mse: float) -> BioFKPPFitResult:
    return BioFKPPFitResult(
        rho=rho, alpha=alpha,
        delta_rho_thickness=0.0,
        beta_growth_amyloid=0.0, beta_growth_thickness=0.0, beta_growth_apoe4=0.0,
        lambda_clearance_thickness=0.0,
        gamma_seeding_amyloid=0.0, gamma_seeding_thickness=0.0,
        gamma_seeding_connectivity=0.0,
        stage1_train_mse=s1_mse, stage2_train_mse=s2_mse, mse_reduction_pct=0.0,
        optimizer_success=True, optimizer_message="no covariates provided",
        optimizer_iterations=0, fitted_terms=[],
    )
