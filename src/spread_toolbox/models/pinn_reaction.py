"""PINN-style constrained reaction discovery for graph tau spreading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .fkpp import FKPPFitResult, midpoint, normalize_laplacian
from .ude import activation_function, mlp_forward, mlp_parameter_count


@dataclass
class KPPReactionNetwork:
    """Scalar reaction network with hard KPP boundary constraints.

    The reaction is parameterized as

    f(c) = c(1-c) exp(g_phi(c)) / (4 max_x c(1-c) exp(g_phi(x)))

    so f(0)=f(1)=0 and max f is normalized to 1/4, matching the
    constrained parameterization used in the PINN+SR tau paper.
    """

    parameters: np.ndarray
    hidden_layers: tuple[int, ...]
    activation: str = "tanh"
    grid_size: int = 201
    epsilon: float = 1.0e-8

    def predict(self, c: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
        raw = self._raw(values)
        return raw / self.normalizer

    @property
    def parameter_count(self) -> int:
        return int(self.parameters.size)

    @property
    def normalizer(self) -> float:
        grid = np.linspace(0.0, 1.0, int(self.grid_size))
        maximum = float(np.max(self._raw(grid)))
        if not np.isfinite(maximum) or maximum <= self.epsilon:
            maximum = 0.25
        return 4.0 * maximum

    def _raw(self, c: np.ndarray) -> np.ndarray:
        flat = np.asarray(c, dtype=float).reshape(-1, 1)
        g = mlp_forward(
            flat,
            self.parameters,
            input_dim=1,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
        ).reshape(np.asarray(c).shape)
        g = np.clip(g, -20.0, 20.0)
        return np.clip(c, 0.0, 1.0) * (1.0 - np.clip(c, 0.0, 1.0)) * np.exp(g)


@dataclass
class SymbolicKPPReaction:
    expression: str
    callable: Callable[[np.ndarray], np.ndarray]

    def predict(self, c: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
        predicted = np.asarray(self.callable(values), dtype=float)
        if predicted.shape == ():
            predicted = np.full_like(values, float(predicted))
        return np.clip(predicted, 0.0, 0.25)


@dataclass
class PINNReactionFit:
    rho: float
    alpha: float
    reaction: KPPReactionNetwork
    report: dict[str, Any]


class GraphKPPReactionModel:
    """Graph reaction-diffusion with learned scalar KPP reaction ``f(c)``."""

    def __init__(
        self,
        laplacian: np.ndarray,
        *,
        u0: np.ndarray,
        cc: np.ndarray,
        reaction: KPPReactionNetwork | SymbolicKPPReaction,
        steps_per_year: int = 12,
        laplacian_normalization: str = "spectral",
    ):
        laplacian = np.asarray(laplacian, dtype=float)
        u0 = np.asarray(u0, dtype=float)
        cc = np.asarray(cc, dtype=float)
        if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
            raise ValueError(f"Laplacian must be square, got shape {laplacian.shape}.")
        if u0.ndim != 1 or cc.ndim != 1 or u0.shape != cc.shape:
            raise ValueError("u0 and cc must be one-dimensional vectors with the same shape.")
        if laplacian.shape[0] != u0.size:
            raise ValueError("Laplacian, u0, and cc region counts must match.")
        if np.any(cc <= u0):
            raise ValueError("Every carrying capacity must be greater than u0.")
        if steps_per_year < 1:
            raise ValueError("steps_per_year must be at least 1.")

        self.original_laplacian = laplacian
        self.laplacian, self.laplacian_scale = normalize_laplacian(laplacian, laplacian_normalization)
        self.u0 = u0
        self.cc = cc
        self.capacity = cc - u0
        self.reaction = reaction
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

        states = np.clip(baseline, self.u0, self.cc)
        remaining = time_years.astype(float).copy()
        step_dt = 1.0 / self.steps_per_year
        while np.any(remaining > 0.0):
            active = remaining > 0.0
            dt = np.minimum(step_dt, remaining[active])[:, None]
            states[active] = self._rk4_step(states[active], dt, float(rho), float(alpha))
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
        from scipy.optimize import minimize

        bounds = [rho_bounds, alpha_bounds]
        initial = np.asarray([midpoint(rho_bounds), midpoint(alpha_bounds)], dtype=float)

        def objective(parameters: np.ndarray) -> float:
            rho, alpha = parameters
            predicted = self.predict(baseline, time_years, rho=float(rho), alpha=float(alpha))
            return float(np.mean((predicted - observed) ** 2))

        result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(maxiter)})
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
        concentration = np.clip(relative_state / self.capacity[None, :], 0.0, 1.0)
        diffusion = -float(rho) * (relative_state @ self.laplacian.T)
        reaction = float(alpha) * self.capacity[None, :] * self.reaction.predict(concentration)
        return diffusion + reaction


def fisher_reaction(hidden_layers: tuple[int, ...] = (8,), activation: str = "tanh") -> KPPReactionNetwork:
    return KPPReactionNetwork(
        parameters=np.zeros(mlp_parameter_count(1, hidden_layers), dtype=float),
        hidden_layers=tuple(hidden_layers),
        activation=activation,
    )


def fit_pinn_kpp_reaction(
    *,
    laplacian: np.ndarray,
    baseline: np.ndarray,
    observed: np.ndarray,
    time_years: np.ndarray,
    row_indices: np.ndarray,
    u0: np.ndarray,
    cc: np.ndarray,
    initial_rho: float,
    initial_alpha: float,
    rho_bounds: tuple[float, float],
    alpha_bounds: tuple[float, float],
    hidden_layers: tuple[int, ...] = (8,),
    activation: str = "tanh",
    steps_per_year: int = 12,
    laplacian_normalization: str = "spectral",
    iterations: int = 160,
    learning_rate: float = 0.04,
    perturbation: float = 0.08,
    gradient_clip: float = 1.0,
    l2_weight: float = 1.0e-4,
    aux_weight: float = 1.0e-3,
    objective_max_pairs: int = 256,
    random_seed: int = 20260507,
) -> PINNReactionFit:
    """Fit rho, alpha, and hard-constrained reaction NN through ODE projection."""

    rng = np.random.default_rng(int(random_seed))
    row_indices = np.asarray(row_indices, dtype=int)
    if objective_max_pairs > 0 and row_indices.size > int(objective_max_pairs):
        objective_indices = np.sort(rng.choice(row_indices, size=int(objective_max_pairs), replace=False))
    else:
        objective_indices = row_indices

    nn_count = mlp_parameter_count(1, hidden_layers)
    theta = np.zeros(2 + nn_count, dtype=float)
    theta[0] = value_to_logit(initial_rho, rho_bounds)
    theta[1] = value_to_logit(initial_alpha, alpha_bounds)
    theta[2:] = rng.normal(0.0, 0.02, size=nn_count)

    best_theta = theta.copy()
    best_loss = kpp_projection_objective(
        theta,
        laplacian=laplacian,
        baseline=baseline,
        observed=observed,
        time_years=time_years,
        objective_indices=objective_indices,
        u0=u0,
        cc=cc,
        rho_bounds=rho_bounds,
        alpha_bounds=alpha_bounds,
        hidden_layers=hidden_layers,
        activation=activation,
        steps_per_year=steps_per_year,
        laplacian_normalization=laplacian_normalization,
        l2_weight=l2_weight,
        aux_weight=aux_weight,
    )
    history = [{"iteration": 0, "loss": best_loss}]
    stability = max(10.0, 0.1 * float(iterations))

    for iteration in range(1, int(iterations) + 1):
        ak = float(learning_rate) / ((iteration + stability) ** 0.602)
        ck = float(perturbation) / (iteration**0.101)
        direction = rng.choice(np.array([-1.0, 1.0]), size=theta.size)
        plus = kpp_projection_objective(
            theta + ck * direction,
            laplacian=laplacian,
            baseline=baseline,
            observed=observed,
            time_years=time_years,
            objective_indices=objective_indices,
            u0=u0,
            cc=cc,
            rho_bounds=rho_bounds,
            alpha_bounds=alpha_bounds,
            hidden_layers=hidden_layers,
            activation=activation,
            steps_per_year=steps_per_year,
            laplacian_normalization=laplacian_normalization,
            l2_weight=l2_weight,
            aux_weight=aux_weight,
        )
        minus = kpp_projection_objective(
            theta - ck * direction,
            laplacian=laplacian,
            baseline=baseline,
            observed=observed,
            time_years=time_years,
            objective_indices=objective_indices,
            u0=u0,
            cc=cc,
            rho_bounds=rho_bounds,
            alpha_bounds=alpha_bounds,
            hidden_layers=hidden_layers,
            activation=activation,
            steps_per_year=steps_per_year,
            laplacian_normalization=laplacian_normalization,
            l2_weight=l2_weight,
            aux_weight=aux_weight,
        )
        gradient = ((plus - minus) / (2.0 * ck)) * direction
        norm = float(np.linalg.norm(gradient))
        if norm > float(gradient_clip):
            gradient = gradient * (float(gradient_clip) / norm)
        theta = theta - ak * gradient

        if iteration == int(iterations) or iteration % max(1, int(iterations) // 10) == 0:
            loss = kpp_projection_objective(
                theta,
                laplacian=laplacian,
                baseline=baseline,
                observed=observed,
                time_years=time_years,
                objective_indices=objective_indices,
                u0=u0,
                cc=cc,
                rho_bounds=rho_bounds,
                alpha_bounds=alpha_bounds,
                hidden_layers=hidden_layers,
                activation=activation,
                steps_per_year=steps_per_year,
                laplacian_normalization=laplacian_normalization,
                l2_weight=l2_weight,
                aux_weight=aux_weight,
            )
            history.append({"iteration": float(iteration), "loss": loss})
            if loss < best_loss:
                best_loss = loss
                best_theta = theta.copy()

    rho, alpha, reaction = reaction_from_theta(best_theta, rho_bounds, alpha_bounds, hidden_layers, activation)
    final_loss = kpp_projection_objective(
        best_theta,
        laplacian=laplacian,
        baseline=baseline,
        observed=observed,
        time_years=time_years,
        objective_indices=objective_indices,
        u0=u0,
        cc=cc,
        rho_bounds=rho_bounds,
        alpha_bounds=alpha_bounds,
        hidden_layers=hidden_layers,
        activation=activation,
        steps_per_year=steps_per_year,
        laplacian_normalization=laplacian_normalization,
        l2_weight=0.0,
        aux_weight=0.0,
    )
    return PINNReactionFit(
        rho=rho,
        alpha=alpha,
        reaction=reaction,
        report={
            "training_objective": "integrated graph KPP reaction-diffusion projection MSE",
            "optimizer": "SPSA",
            "iterations": int(iterations),
            "objective_pairs": int(objective_indices.size),
            "available_train_pairs": int(row_indices.size),
            "parameter_count": int(best_theta.size),
            "hidden_layers": list(hidden_layers),
            "activation": activation,
            "initial_rho": float(initial_rho),
            "initial_alpha": float(initial_alpha),
            "rho": float(rho),
            "alpha": float(alpha),
            "l2_weight": float(l2_weight),
            "aux_weight": float(aux_weight),
            "best_objective_with_l2": float(best_loss),
            "final_objective_no_l2": float(final_loss),
            "history": history,
        },
    )


def kpp_projection_objective(theta: np.ndarray, **kwargs: Any) -> float:
    rho, alpha, reaction = reaction_from_theta(
        theta,
        kwargs["rho_bounds"],
        kwargs["alpha_bounds"],
        kwargs["hidden_layers"],
        kwargs["activation"],
    )
    objective_indices = kwargs["objective_indices"]
    model = GraphKPPReactionModel(
        kwargs["laplacian"],
        u0=kwargs["u0"],
        cc=kwargs["cc"],
        reaction=reaction,
        steps_per_year=kwargs["steps_per_year"],
        laplacian_normalization=kwargs["laplacian_normalization"],
    )
    predicted = model.predict(
        kwargs["baseline"][objective_indices],
        kwargs["time_years"][objective_indices],
        rho=rho,
        alpha=alpha,
    )
    residual = predicted - kwargs["observed"][objective_indices]
    return float(
        np.mean(residual**2)
        + float(kwargs["l2_weight"]) * np.mean(theta[2:] ** 2)
        + float(kwargs.get("aux_weight", 0.0)) * kpp_auxiliary_loss(reaction, alpha)
    )


def kpp_auxiliary_loss(reaction: KPPReactionNetwork, alpha: float, *, grid_size: int = 101) -> float:
    """Softly enforce the KPP slope condition used in the PINN+SR paper."""

    grid = np.linspace(0.0, 1.0, int(grid_size))
    values = reaction.predict(grid)
    derivatives = np.gradient(values, grid, edge_order=1)
    excess = np.maximum(0.0, float(alpha) * (derivatives - derivatives[0]))
    return float(np.mean(np.abs(excess)))


def reaction_from_theta(
    theta: np.ndarray,
    rho_bounds: tuple[float, float],
    alpha_bounds: tuple[float, float],
    hidden_layers: tuple[int, ...],
    activation: str,
) -> tuple[float, float, KPPReactionNetwork]:
    theta = np.asarray(theta, dtype=float)
    rho = logit_to_value(theta[0], rho_bounds)
    alpha = logit_to_value(theta[1], alpha_bounds)
    reaction = KPPReactionNetwork(parameters=theta[2:].copy(), hidden_layers=tuple(hidden_layers), activation=activation)
    return rho, alpha, reaction


def value_to_logit(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    scaled = (float(value) - lower) / max(upper - lower, 1.0e-12)
    scaled = float(np.clip(scaled, 1.0e-6, 1.0 - 1.0e-6))
    return float(np.log(scaled / (1.0 - scaled)))


def logit_to_value(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    sigmoid = 1.0 / (1.0 + np.exp(-float(np.clip(value, -40.0, 40.0))))
    return float(lower + (upper - lower) * sigmoid)


def symbolic_reaction_from_expression(expression: str) -> SymbolicKPPReaction:
    import sympy

    c = sympy.Symbol("c")
    parsed = sympy.sympify(expression)
    func = sympy.lambdify(c, parsed, modules=["numpy"])
    return SymbolicKPPReaction(expression=str(parsed), callable=func)


def distill_kpp_reaction_with_pysr(
    reaction: KPPReactionNetwork,
    *,
    grid_size: int = 512,
    niterations: int = 80,
    populations: int = 12,
    population_size: int = 24,
    maxsize: int = 24,
    parsimony: float = 0.0032,
    timeout_seconds: int = 180,
    random_seed: int = 20260507,
    output_directory: str | None = None,
    binary_operators: tuple[str, ...] = ("+", "-", "*", "/"),
    unary_operators: tuple[str, ...] = ("square", "exp"),
) -> tuple[str, list[dict[str, Any]]]:
    """Distill a hard-constrained reaction by fitting the KPP shape factor.

    PySR learns q(c) and the reported reaction is f(c)=c(1-c)q(c). This
    guarantees the symbolic equation keeps f(0)=f(1)=0.
    """

    import sympy
    from pysr import PySRRegressor

    c_grid = np.linspace(1.0e-4, 1.0 - 1.0e-4, int(grid_size))
    target_f = reaction.predict(c_grid)
    shape_target = target_f / (c_grid * (1.0 - c_grid))
    model = PySRRegressor(
        niterations=int(niterations),
        populations=int(populations),
        population_size=int(population_size),
        maxsize=int(maxsize),
        timeout_in_seconds=int(timeout_seconds),
        binary_operators=list(binary_operators),
        unary_operators=list(unary_operators),
        complexity_of_operators={"/": 2, "exp": 2},
        nested_constraints={"exp": {"exp": 0}},
        parsimony=float(parsimony),
        model_selection="best",
        deterministic=True,
        random_state=int(random_seed),
        tournament_selection_n=min(15, max(2, int(population_size) // 2)),
        topn=min(12, int(population_size)),
        batching=True,
        batch_size=min(256, int(grid_size)),
        ncycles_per_iteration=100,
        parallelism="serial",
        progress=False,
        verbosity=0,
        output_directory=output_directory,
    )
    model.fit(c_grid.reshape(-1, 1), shape_target, variable_names=["c"])

    c = sympy.Symbol("c")
    rows: list[dict[str, Any]] = []
    best_shape_loss = float("inf")
    for row in model.equations_.to_dict("records"):
        shape_expr = str(row.get("sympy_format", row.get("equation", "")))
        final_expr = sympy.simplify(c * (1 - c) * sympy.sympify(shape_expr))
        predicted_f = evaluate_expression(str(final_expr), c_grid)
        f_loss = float(np.mean((target_f - predicted_f) ** 2))
        best_shape_loss = min(best_shape_loss, float(row.get("loss", np.inf)))
        rows.append(
            {
                "complexity": int(row.get("complexity", 0)),
                "shape_loss": float(row.get("loss", np.nan)),
                "reaction_loss": f_loss,
                "score": float(row.get("score", 0.0)),
                "shape_expression": shape_expr,
                "reaction_expression": str(final_expr),
            }
        )
    if not rows:
        raise ValueError("PySR returned no equations for KPP reaction distillation.")

    threshold = 1.5 * best_shape_loss
    candidates = [row for row in rows if row["shape_loss"] <= threshold]
    selected = max(candidates or rows, key=lambda row: row["score"])
    return str(selected["reaction_expression"]), rows


def evaluate_expression(expression: str, c_values: np.ndarray) -> np.ndarray:
    reaction = symbolic_reaction_from_expression(expression)
    return reaction.predict(c_values)
