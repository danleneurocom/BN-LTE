"""Universal differential-equation closures for local FKPP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BASE_FEATURES = {"tau", "gap", "tau_gap"}


@dataclass
class UDEFeatureTensor:
    names: list[str]
    values: np.ndarray


@dataclass
class NeuralODEClosureFit:
    """Small MLP closure trained through the ODE forecast objective."""

    feature_names: list[str]
    parameters: np.ndarray
    hidden_layers: tuple[int, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    rate_scale: float
    activation: str
    report: dict[str, Any]

    def predict_rate(self, feature_values: np.ndarray) -> np.ndarray:
        values = np.asarray(feature_values, dtype=float)
        if values.shape[-1] != len(self.feature_names):
            raise ValueError("Feature count does not match fitted neural closure.")
        flat = values.reshape(-1, values.shape[-1])
        scaled = (flat - self.feature_mean[None, :]) / self.feature_scale[None, :]
        predicted = mlp_forward(
            scaled,
            self.parameters,
            input_dim=len(self.feature_names),
            hidden_layers=self.hidden_layers,
            activation=self.activation,
        )
        return (float(self.rate_scale) * predicted).reshape(values.shape[:-1])

    @property
    def parameter_count(self) -> int:
        return int(self.parameters.size)


@dataclass
class PySRResidual:
    """Symbolic closure distilled from the neural ODE closure with real PySR."""

    feature_names: list[str]
    regressor: Any
    expression: str
    pareto_rows: list[dict[str, Any]]
    report: dict[str, Any]

    def predict_rate(self, feature_values: np.ndarray) -> np.ndarray:
        values = np.asarray(feature_values, dtype=float)
        if values.shape[-1] != len(self.feature_names):
            raise ValueError("Feature count does not match fitted PySR closure.")
        flat = values.reshape(-1, values.shape[-1])
        predicted = self.regressor.predict(flat)
        return np.asarray(predicted, dtype=float).reshape(values.shape[:-1])

    def term_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "term": self.expression,
                "coefficient": "",
                "selected": True,
                "distiller": "pysr",
            }
        ]


class LocalFKPPUDEModel:
    """Local FKPP with a learned residual rate inside the ODE."""

    def __init__(
        self,
        laplacian: np.ndarray,
        *,
        u0: np.ndarray,
        cc: np.ndarray,
        residual_model: NeuralODEClosureFit | PySRResidual | Any,
        pair_covariates: dict[str, np.ndarray] | None = None,
        regional_covariates: dict[str, np.ndarray] | None = None,
        steps_per_year: int = 12,
        laplacian_normalization: str = "spectral",
    ):
        from .fkpp import normalize_laplacian

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
            raise ValueError("Every local FKPP carrying capacity must be greater than u0.")
        if steps_per_year < 1:
            raise ValueError("steps_per_year must be at least 1.")

        self.original_laplacian = laplacian
        self.laplacian, self.laplacian_scale = normalize_laplacian(laplacian, laplacian_normalization)
        self.u0 = u0
        self.cc = cc
        self.capacity = cc - u0
        self.residual_model = residual_model
        self.pair_covariates = {name: np.asarray(values, dtype=float) for name, values in (pair_covariates or {}).items()}
        self.regional_covariates = {
            name: np.asarray(values, dtype=float) for name, values in (regional_covariates or {}).items()
        }
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
        remaining = time_years.astype(float).copy()
        step_dt = 1.0 / self.steps_per_year
        while np.any(remaining > 0.0):
            active = remaining > 0.0
            active_indices = np.where(active)[0]
            dt = np.minimum(step_dt, remaining[active])[:, None]
            active_states = states[active]
            states[active] = self._rk4_step(active_states, active_indices, dt, float(rho), float(alpha))
            remaining[active] -= dt[:, 0]
        return states

    def _rk4_step(
        self,
        states: np.ndarray,
        row_indices: np.ndarray,
        dt: np.ndarray,
        rho: float,
        alpha: float,
    ) -> np.ndarray:
        k1 = self._derivative(states, row_indices, rho, alpha)
        k2 = self._derivative(np.clip(states + 0.5 * dt * k1, self.u0, self.cc), row_indices, rho, alpha)
        k3 = self._derivative(np.clip(states + 0.5 * dt * k2, self.u0, self.cc), row_indices, rho, alpha)
        k4 = self._derivative(np.clip(states + dt * k3, self.u0, self.cc), row_indices, rho, alpha)
        return np.clip(states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), self.u0, self.cc)

    def _derivative(self, states: np.ndarray, row_indices: np.ndarray, rho: float, alpha: float) -> np.ndarray:
        relative_state = states - self.u0
        diffusion = -float(rho) * (relative_state @ self.laplacian.T)
        growth = float(alpha) * relative_state * (self.capacity - relative_state)
        pair_covariates = {name: values[row_indices] for name, values in self.pair_covariates.items()}
        regional_covariates = {name: values[row_indices] for name, values in self.regional_covariates.items()}
        feature_tensor = build_ude_feature_tensor(
            states,
            u0=self.u0,
            cc=self.cc,
            pair_covariates=pair_covariates,
            regional_covariates=regional_covariates,
        )
        residual = self.residual_model.predict_rate(feature_tensor.values)
        return diffusion + growth + residual


def build_ude_feature_tensor(
    states: np.ndarray,
    *,
    u0: np.ndarray,
    cc: np.ndarray,
    pair_covariates: dict[str, np.ndarray] | None = None,
    regional_covariates: dict[str, np.ndarray] | None = None,
) -> UDEFeatureTensor:
    states = np.asarray(states, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    cc = np.asarray(cc, dtype=float)
    if states.ndim != 2:
        raise ValueError("states must be two-dimensional.")
    if states.shape[1] != u0.size or u0.shape != cc.shape:
        raise ValueError("states, u0, and cc region counts must match.")

    capacity = np.maximum(cc - u0, 1.0e-8)
    tau = np.clip((states - u0[None, :]) / capacity[None, :], 0.0, 1.0)
    gap = 1.0 - tau
    tau_gap = tau * gap

    names = ["tau", "gap", "tau_gap"]
    features = [tau, gap, tau_gap]

    for covariate_name, covariate in sorted((pair_covariates or {}).items()):
        values = np.asarray(covariate, dtype=float)
        if values.shape != (states.shape[0],):
            raise ValueError(f"Pair covariate {covariate_name!r} must have one value per pair.")
        names.append(covariate_name)
        features.append(np.broadcast_to(values[:, None], states.shape))

    for covariate_name, covariate in sorted((regional_covariates or {}).items()):
        values = np.asarray(covariate, dtype=float)
        if values.shape != states.shape:
            raise ValueError(f"Regional covariate {covariate_name!r} must match states shape.")
        names.append(covariate_name)
        features.append(values)

    return UDEFeatureTensor(names=names, values=np.stack(features, axis=-1))


def fit_neural_ode_closure(
    *,
    laplacian: np.ndarray,
    baseline: np.ndarray,
    observed: np.ndarray,
    time_years: np.ndarray,
    row_indices: np.ndarray,
    u0: np.ndarray,
    cc: np.ndarray,
    rho: float,
    alpha: float,
    pair_covariates: dict[str, np.ndarray],
    regional_covariates: dict[str, np.ndarray],
    steps_per_year: int,
    laplacian_normalization: str,
    hidden_layers: tuple[int, ...] = (8,),
    activation: str = "tanh",
    rate_scale: float = 0.05,
    iterations: int = 120,
    learning_rate: float = 0.04,
    perturbation: float = 0.08,
    gradient_clip: float = 1.0,
    l2_weight: float = 1.0e-4,
    objective_max_pairs: int = 256,
    random_seed: int = 20260507,
) -> NeuralODEClosureFit:
    """Train a neural closure by minimizing ODE forecast error.

    This uses SPSA so the loss is the actual integrated forecast MSE without
    requiring Torch/JAX adjoints in the project environment.
    """

    baseline = np.asarray(baseline, dtype=float)
    observed = np.asarray(observed, dtype=float)
    time_years = np.asarray(time_years, dtype=float)
    row_indices = np.asarray(row_indices, dtype=int)
    rng = np.random.default_rng(int(random_seed))
    if objective_max_pairs > 0 and row_indices.size > int(objective_max_pairs):
        objective_indices = np.sort(rng.choice(row_indices, size=int(objective_max_pairs), replace=False))
    else:
        objective_indices = row_indices

    reference_features = build_ude_feature_tensor(
        baseline,
        u0=u0,
        cc=cc,
        pair_covariates=pair_covariates,
        regional_covariates=regional_covariates,
    )
    feature_mean, feature_scale = feature_standardization(reference_features.values, objective_indices)
    parameter_count = mlp_parameter_count(len(reference_features.names), hidden_layers)
    initial_scale = 0.05
    parameters = rng.normal(0.0, initial_scale, size=parameter_count)
    best_parameters = parameters.copy()
    best_loss = integrated_forecast_loss(
        parameters,
        laplacian=laplacian,
        baseline=baseline,
        observed=observed,
        time_years=time_years,
        objective_indices=objective_indices,
        u0=u0,
        cc=cc,
        rho=rho,
        alpha=alpha,
        pair_covariates=pair_covariates,
        regional_covariates=regional_covariates,
        steps_per_year=steps_per_year,
        laplacian_normalization=laplacian_normalization,
        feature_names=reference_features.names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        hidden_layers=hidden_layers,
        activation=activation,
        rate_scale=rate_scale,
        l2_weight=l2_weight,
    )
    history: list[dict[str, float]] = [{"iteration": 0, "loss": best_loss}]
    stability = max(10.0, 0.1 * float(iterations))

    for iteration in range(1, int(iterations) + 1):
        ak = float(learning_rate) / ((iteration + stability) ** 0.602)
        ck = float(perturbation) / (iteration**0.101)
        direction = rng.choice(np.array([-1.0, 1.0]), size=parameter_count)
        loss_plus = integrated_forecast_loss(
            parameters + ck * direction,
            laplacian=laplacian,
            baseline=baseline,
            observed=observed,
            time_years=time_years,
            objective_indices=objective_indices,
            u0=u0,
            cc=cc,
            rho=rho,
            alpha=alpha,
            pair_covariates=pair_covariates,
            regional_covariates=regional_covariates,
            steps_per_year=steps_per_year,
            laplacian_normalization=laplacian_normalization,
            feature_names=reference_features.names,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            hidden_layers=hidden_layers,
            activation=activation,
            rate_scale=rate_scale,
            l2_weight=l2_weight,
        )
        loss_minus = integrated_forecast_loss(
            parameters - ck * direction,
            laplacian=laplacian,
            baseline=baseline,
            observed=observed,
            time_years=time_years,
            objective_indices=objective_indices,
            u0=u0,
            cc=cc,
            rho=rho,
            alpha=alpha,
            pair_covariates=pair_covariates,
            regional_covariates=regional_covariates,
            steps_per_year=steps_per_year,
            laplacian_normalization=laplacian_normalization,
            feature_names=reference_features.names,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            hidden_layers=hidden_layers,
            activation=activation,
            rate_scale=rate_scale,
            l2_weight=l2_weight,
        )
        gradient = ((loss_plus - loss_minus) / (2.0 * ck)) * direction
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > float(gradient_clip):
            gradient = gradient * (float(gradient_clip) / gradient_norm)
        parameters = parameters - ak * gradient

        if iteration == int(iterations) or iteration % max(1, int(iterations) // 12) == 0:
            current_loss = integrated_forecast_loss(
                parameters,
                laplacian=laplacian,
                baseline=baseline,
                observed=observed,
                time_years=time_years,
                objective_indices=objective_indices,
                u0=u0,
                cc=cc,
                rho=rho,
                alpha=alpha,
                pair_covariates=pair_covariates,
                regional_covariates=regional_covariates,
                steps_per_year=steps_per_year,
                laplacian_normalization=laplacian_normalization,
                feature_names=reference_features.names,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                hidden_layers=hidden_layers,
                activation=activation,
                rate_scale=rate_scale,
                l2_weight=l2_weight,
            )
            history.append({"iteration": float(iteration), "loss": current_loss})
            if current_loss < best_loss:
                best_loss = current_loss
                best_parameters = parameters.copy()

    final_loss = integrated_forecast_loss(
        best_parameters,
        laplacian=laplacian,
        baseline=baseline,
        observed=observed,
        time_years=time_years,
        objective_indices=objective_indices,
        u0=u0,
        cc=cc,
        rho=rho,
        alpha=alpha,
        pair_covariates=pair_covariates,
        regional_covariates=regional_covariates,
        steps_per_year=steps_per_year,
        laplacian_normalization=laplacian_normalization,
        feature_names=reference_features.names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        hidden_layers=hidden_layers,
        activation=activation,
        rate_scale=rate_scale,
        l2_weight=0.0,
    )
    return NeuralODEClosureFit(
        feature_names=list(reference_features.names),
        parameters=best_parameters,
        hidden_layers=tuple(hidden_layers),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        rate_scale=float(rate_scale),
        activation=str(activation),
        report={
            "training_objective": "integrated local-FKPP+neural-closure forecast MSE",
            "optimizer": "SPSA",
            "iterations": int(iterations),
            "objective_pairs": int(objective_indices.size),
            "available_train_pairs": int(row_indices.size),
            "feature_count": len(reference_features.names),
            "parameter_count": int(best_parameters.size),
            "hidden_layers": list(hidden_layers),
            "activation": str(activation),
            "rate_scale": float(rate_scale),
            "learning_rate": float(learning_rate),
            "perturbation": float(perturbation),
            "gradient_clip": float(gradient_clip),
            "l2_weight": float(l2_weight),
            "initial_objective": float(history[0]["loss"]),
            "best_objective_with_l2": float(best_loss),
            "final_objective_no_l2": float(final_loss),
            "history": history,
        },
    )


def integrated_forecast_loss(
    parameters: np.ndarray,
    *,
    laplacian: np.ndarray,
    baseline: np.ndarray,
    observed: np.ndarray,
    time_years: np.ndarray,
    objective_indices: np.ndarray,
    u0: np.ndarray,
    cc: np.ndarray,
    rho: float,
    alpha: float,
    pair_covariates: dict[str, np.ndarray],
    regional_covariates: dict[str, np.ndarray],
    steps_per_year: int,
    laplacian_normalization: str,
    feature_names: list[str],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    hidden_layers: tuple[int, ...],
    activation: str,
    rate_scale: float,
    l2_weight: float,
) -> float:
    closure = NeuralODEClosureFit(
        feature_names=list(feature_names),
        parameters=np.asarray(parameters, dtype=float),
        hidden_layers=tuple(hidden_layers),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        rate_scale=float(rate_scale),
        activation=str(activation),
        report={},
    )
    subset_pair_covariates = {name: values[objective_indices] for name, values in pair_covariates.items()}
    subset_regional_covariates = {name: values[objective_indices] for name, values in regional_covariates.items()}
    model = LocalFKPPUDEModel(
        laplacian,
        u0=u0,
        cc=cc,
        residual_model=closure,
        pair_covariates=subset_pair_covariates,
        regional_covariates=subset_regional_covariates,
        steps_per_year=steps_per_year,
        laplacian_normalization=laplacian_normalization,
    )
    predicted = model.predict(baseline[objective_indices], time_years[objective_indices], rho=rho, alpha=alpha)
    residual = predicted - observed[objective_indices]
    mse = float(np.mean(residual**2))
    penalty = float(l2_weight) * float(np.mean(np.asarray(parameters, dtype=float) ** 2))
    return mse + penalty


def distill_neural_closure_with_pysr(
    feature_tensor: UDEFeatureTensor,
    neural_fit: NeuralODEClosureFit,
    *,
    row_indices: np.ndarray,
    max_distill_rows: int = 20000,
    niterations: int = 200,
    populations: int = 20,
    population_size: int = 33,
    maxsize: int = 24,
    parsimony: float = 0.0032,
    timeout_seconds: int | None = None,
    batching: bool = True,
    batch_size: int = 512,
    ncycles_per_iteration: int = 100,
    binary_operators: list[str] | None = None,
    unary_operators: list[str] | None = None,
    output_directory: str | None = None,
    random_seed: int = 20260507,
) -> PySRResidual:
    """Run real PySR and keep the full Pareto front."""

    from pysr import PySRRegressor

    dense_features = sample_distillation_features(
        feature_tensor.values,
        feature_tensor.names,
        row_indices=row_indices,
        max_rows=max_distill_rows,
        random_seed=random_seed,
    )
    dense_target = neural_fit.predict_rate(dense_features).reshape(-1)
    flat_features = dense_features.reshape(-1, dense_features.shape[-1])
    binary_operators = binary_operators or ["+", "-", "*", "/"]
    unary_operators = unary_operators or ["square", "exp"]
    complexity = {"/": 2}
    if "exp" in unary_operators:
        complexity["exp"] = 2
    if "log1p_abs" in unary_operators:
        complexity["log1p_abs"] = 2
    nested_constraints = {"exp": {"exp": 0}} if "exp" in unary_operators else None

    model = PySRRegressor(
        niterations=int(niterations),
        populations=int(populations),
        population_size=int(population_size),
        timeout_in_seconds=int(timeout_seconds) if timeout_seconds else None,
        batching=bool(batching),
        batch_size=int(batch_size),
        ncycles_per_iteration=int(ncycles_per_iteration),
        binary_operators=list(binary_operators),
        unary_operators=prepare_pysr_unary_operators(unary_operators),
        extra_sympy_mappings=extra_sympy_mappings(unary_operators),
        complexity_of_operators=complexity,
        nested_constraints=nested_constraints,
        maxsize=int(maxsize),
        parsimony=float(parsimony),
        model_selection="best",
        random_state=int(random_seed),
        deterministic=True,
        tournament_selection_n=min(15, max(2, int(population_size) // 2)),
        topn=min(12, int(population_size)),
        parallelism="serial",
        progress=False,
        verbosity=0,
        output_directory=output_directory,
    )
    model.fit(flat_features, dense_target, variable_names=list(feature_tensor.names))
    prediction = np.asarray(model.predict(flat_features), dtype=float)
    expression = safe_pysr_expression(model)
    pareto_rows = pysr_pareto_rows(model)
    return PySRResidual(
        feature_names=list(feature_tensor.names),
        regressor=model,
        expression=expression,
        pareto_rows=pareto_rows,
        report={
            "distiller": "pysr",
            "expression": expression,
            "distillation_rows": int(flat_features.shape[0]),
            "distillation_mse": float(np.mean((dense_target - prediction) ** 2)),
            "distillation_r2": scalar_r2(dense_target, prediction),
            "niterations": int(niterations),
            "populations": int(populations),
            "population_size": int(population_size),
            "maxsize": int(maxsize),
            "parsimony": float(parsimony),
            "timeout_seconds": int(timeout_seconds) if timeout_seconds else None,
            "batching": bool(batching),
            "batch_size": int(batch_size),
            "ncycles_per_iteration": int(ncycles_per_iteration),
            "binary_operators": list(binary_operators),
            "unary_operators": list(unary_operators),
            "pareto_front_size": int(len(pareto_rows)),
        },
    )


def feature_standardization(feature_values: np.ndarray, row_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(feature_values, dtype=float)
    selected = values[np.asarray(row_indices, dtype=int)].reshape(-1, values.shape[-1])
    finite = np.all(np.isfinite(selected), axis=1)
    selected = selected[finite]
    mean = np.mean(selected, axis=0)
    scale = np.std(selected, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return mean, scale


def sample_distillation_features(
    feature_values: np.ndarray,
    feature_names: list[str],
    *,
    row_indices: np.ndarray,
    max_rows: int,
    random_seed: int,
) -> np.ndarray:
    values = np.asarray(feature_values, dtype=float)
    row_mask = np.zeros(values.shape[0], dtype=bool)
    row_mask[np.asarray(row_indices, dtype=int)] = True
    flat = values.reshape(-1, values.shape[-1])
    flat_row_mask = np.broadcast_to(row_mask[:, None], values.shape[:-1]).reshape(-1)
    finite = flat_row_mask & np.all(np.isfinite(flat), axis=1)
    source = flat[finite]
    if source.shape[0] == 0:
        raise ValueError("No finite rows are available for symbolic distillation.")

    rng = np.random.default_rng(int(random_seed))
    sample_count = min(int(max_rows), source.shape[0]) if max_rows > 0 else source.shape[0]
    selected = rng.choice(source.shape[0], size=sample_count, replace=False)
    dense = source[selected].copy()

    name_to_index = {name: index for index, name in enumerate(feature_names)}
    missing = BASE_FEATURES - set(name_to_index)
    if missing:
        raise ValueError(f"Missing base UDE features: {sorted(missing)}")
    tau = rng.uniform(0.0, 1.0, size=dense.shape[0])
    dense[:, name_to_index["tau"]] = tau
    dense[:, name_to_index["gap"]] = 1.0 - tau
    dense[:, name_to_index["tau_gap"]] = tau * (1.0 - tau)
    return dense.reshape(dense.shape[0], 1, dense.shape[1])


def mlp_parameter_count(input_dim: int, hidden_layers: tuple[int, ...]) -> int:
    dimensions = [int(input_dim), *[int(value) for value in hidden_layers], 1]
    total = 0
    for left, right in zip(dimensions[:-1], dimensions[1:]):
        total += left * right + right
    return int(total)


def unpack_mlp_parameters(
    parameters: np.ndarray,
    *,
    input_dim: int,
    hidden_layers: tuple[int, ...],
) -> list[tuple[np.ndarray, np.ndarray]]:
    parameters = np.asarray(parameters, dtype=float)
    dimensions = [int(input_dim), *[int(value) for value in hidden_layers], 1]
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    offset = 0
    for left, right in zip(dimensions[:-1], dimensions[1:]):
        weight_size = left * right
        weights = parameters[offset : offset + weight_size].reshape(left, right)
        offset += weight_size
        bias = parameters[offset : offset + right]
        offset += right
        layers.append((weights, bias))
    if offset != parameters.size:
        raise ValueError("MLP parameter vector has the wrong length.")
    return layers


def mlp_forward(
    values: np.ndarray,
    parameters: np.ndarray,
    *,
    input_dim: int,
    hidden_layers: tuple[int, ...],
    activation: str,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    layers = unpack_mlp_parameters(parameters, input_dim=input_dim, hidden_layers=hidden_layers)
    for index, (weights, bias) in enumerate(layers):
        x = x @ weights + bias[None, :]
        if index < len(layers) - 1:
            x = activation_function(x, activation)
    return x[:, 0]


def activation_function(values: np.ndarray, activation: str) -> np.ndarray:
    if activation == "tanh":
        return np.tanh(values)
    if activation == "relu":
        return np.maximum(values, 0.0)
    if activation == "identity":
        return values
    raise ValueError(f"Unsupported UDE activation: {activation}")


def prepare_pysr_unary_operators(unary_operators: list[str]) -> list[str]:
    prepared: list[str] = []
    for operator in unary_operators:
        if operator == "log1p_abs":
            prepared.append("log1p_abs(x) = log(1 + abs(x))")
        else:
            prepared.append(operator)
    return prepared


def extra_sympy_mappings(unary_operators: list[str]) -> dict[str, Any]:
    if "log1p_abs" not in unary_operators:
        return {}
    import sympy

    return {"log1p_abs": lambda x: sympy.log(1 + sympy.Abs(x))}


def safe_pysr_expression(model: Any) -> str:
    try:
        return str(model.sympy())
    except Exception:
        try:
            best = model.get_best()
            return str(best)
        except Exception:
            return ""


def pysr_pareto_rows(model: Any) -> list[dict[str, Any]]:
    equations = getattr(model, "equations_", None)
    if equations is None:
        return []
    rows: list[dict[str, Any]] = []
    for row in equations.to_dict("records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, np.integer)):
                clean[key] = value.item()
            elif isinstance(value, np.ndarray):
                clean[key] = value.tolist()
            else:
                clean[key] = str(value) if key in {"equation", "sympy_format", "lambda_format"} else value
        rows.append(clean)
    return rows


def scalar_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(mask):
        return float("nan")
    y = observed[mask]
    yhat = predicted[mask]
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / total)
