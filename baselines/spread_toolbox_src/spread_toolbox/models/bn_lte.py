"""Prototype Bayesian-network latent-time embedding utilities.

This module implements an empirical BN-LTE prototype for the data that are
currently available in the toolbox. It is intentionally conservative: the full
proposal calls for joint Bayesian MCMC over pseudotime, graph structure, and
spline coefficients, but the present ADNI forecasting tables contain regional
tau trajectories rather than the complete ADNI + UKB multimodal matrix. The
helpers below therefore provide train-only pseudotime, smooth time-varying
edge functions, root-node covariates, and bootstrap inclusion probabilities as
an auditable stepping stone toward the full model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PseudotimeEmbedding:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    component: np.ndarray
    score_lower: float
    score_upper: float
    explained_variance_ratio: float
    burden_correlation: float

    def transform(self, values: np.ndarray, *, clip: bool = True) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.component.size:
            raise ValueError("values must be a two-dimensional matrix with the fitted feature count.")
        scaled = (matrix - self.feature_mean[None, :]) / self.feature_scale[None, :]
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.einsum("ij,j->i", scaled, self.component)
        denom = max(float(self.score_upper - self.score_lower), 1.0e-12)
        z = (scores - self.score_lower) / denom
        if clip:
            return np.clip(z, 0.0, 1.0)
        return z


@dataclass
class ConstantSplineBasis:
    n_basis: int = 1
    degree: int = 0
    n_knots: int = 1

    def transform(self, z: np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=float).reshape(-1)
        return np.ones((values.size, 1), dtype=float)


@dataclass
class FittedSplineBasis:
    transformer: Any
    n_basis: int
    degree: int
    n_knots: int

    def transform(self, z: np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=float).reshape(-1, 1)
        basis = self.transformer.transform(values)
        if hasattr(basis, "toarray"):
            basis = basis.toarray()
        return np.asarray(basis, dtype=float)


@dataclass
class DesignMatrix:
    values: np.ndarray
    feature_names: list[str]
    parent_effect_columns: dict[str, list[int]]
    parent_source_types: dict[str, str]


@dataclass
class RidgeRateFit:
    feature_names: list[str]
    coefficients: np.ndarray
    intercept: float
    alpha: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    train_mse: float
    train_r2: float
    cv_report: list[dict[str, Any]]
    used_train_rows: int

    def predict(self, feature_values: np.ndarray) -> np.ndarray:
        values = np.asarray(feature_values, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.coefficients.size:
            raise ValueError("feature_values must have the fitted feature count.")
        scaled = (values - self.feature_mean[None, :]) / self.feature_scale[None, :]
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.intercept + np.einsum("ij,j->i", scaled, self.coefficients)


@dataclass
class BNLTEChildFit:
    child_label: str
    child_index: int
    parent_indices: list[int]
    parent_labels: list[str]
    root_names: list[str]
    ridge: RidgeRateFit
    parent_effect_columns: dict[str, list[int]]
    parent_source_types: dict[str, str]

    def effect_curve(self, parent_label: str, basis_values: np.ndarray) -> np.ndarray:
        columns = self.parent_effect_columns.get(parent_label, [])
        if not columns:
            return np.zeros(basis_values.shape[0], dtype=float)
        if len(columns) != basis_values.shape[1]:
            raise ValueError("basis_values do not match the fitted parent effect dimension.")
        raw_coefficients = np.asarray(
            [
                self.ridge.coefficients[column] / self.ridge.feature_scale[column]
                for column in columns
            ],
            dtype=float,
        )
        return np.einsum("ij,j->i", basis_values, raw_coefficients)


@dataclass
class BNLTEFit:
    model_name: str
    region_indices: np.ndarray
    region_labels: list[str]
    pseudotime: PseudotimeEmbedding
    spline_basis: ConstantSplineBasis | FittedSplineBasis
    child_fits: list[BNLTEChildFit]
    train_indices: np.ndarray
    parent_mode: str
    include_roots: bool
    include_self_history: bool
    prediction_lower: np.ndarray
    prediction_upper: np.ndarray
    edge_effect_threshold: float
    z_grid: np.ndarray

    def transform_pseudotime(self, baseline: np.ndarray) -> np.ndarray:
        return self.pseudotime.transform(np.asarray(baseline, dtype=float)[:, self.region_indices])

    def predict(
        self,
        baseline: np.ndarray,
        time_years: np.ndarray,
        *,
        root_values: np.ndarray | None = None,
    ) -> np.ndarray:
        selected_baseline = np.asarray(baseline, dtype=float)[:, self.region_indices]
        z = self.pseudotime.transform(selected_baseline)
        root_matrix = empty_root_values(selected_baseline.shape[0]) if root_values is None else np.asarray(root_values, dtype=float)
        rates = np.zeros_like(selected_baseline, dtype=float)
        for child_fit in self.child_fits:
            design = build_design_matrix(
                baseline=selected_baseline,
                state=scale_state(selected_baseline, self.prediction_lower, self.prediction_upper),
                z=z,
                spline_basis=self.spline_basis,
                child_index=child_fit.child_index,
                parent_indices=child_fit.parent_indices,
                parent_labels=child_fit.parent_labels,
                root_names=child_fit.root_names,
                root_values=root_matrix,
                include_self_history=self.include_self_history,
            )
            rates[:, child_fit.child_index] = child_fit.ridge.predict(design.values)
        predicted = selected_baseline + np.maximum(np.asarray(time_years, dtype=float), 0.0)[:, None] * rates
        return np.clip(predicted, self.prediction_lower[None, :], self.prediction_upper[None, :])

    def edge_rows(self) -> list[dict[str, Any]]:
        basis = self.spline_basis.transform(self.z_grid)
        rows: list[dict[str, Any]] = []
        for child_fit in self.child_fits:
            for parent_label in child_fit.parent_effect_columns:
                effect = child_fit.effect_curve(parent_label, basis)
                max_index = int(np.argmax(np.abs(effect))) if effect.size else 0
                max_abs = float(np.max(np.abs(effect))) if effect.size else 0.0
                rows.append(
                    {
                        "model": self.model_name,
                        "parent": parent_label,
                        "child": child_fit.child_label,
                        "source_type": child_fit.parent_source_types.get(parent_label, "regional"),
                        "max_abs_effect": max_abs,
                        "mean_abs_effect": float(np.mean(np.abs(effect))) if effect.size else 0.0,
                        "z_at_max_abs_effect": float(self.z_grid[max_index]) if effect.size else float("nan"),
                        "effect_at_z_min": float(effect[0]) if effect.size else float("nan"),
                        "effect_at_z_mid": float(effect[effect.size // 2]) if effect.size else float("nan"),
                        "effect_at_z_max": float(effect[-1]) if effect.size else float("nan"),
                        "included_by_effect_threshold": bool(max_abs >= self.edge_effect_threshold),
                        "effect_threshold": float(self.edge_effect_threshold),
                    }
                )
        return sorted(rows, key=lambda row: float(row["max_abs_effect"]), reverse=True)

    def report(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "region_count": len(self.region_labels),
            "regions": self.region_labels,
            "parent_mode": self.parent_mode,
            "include_roots": self.include_roots,
            "include_self_history": self.include_self_history,
            "pseudotime": {
                "explained_variance_ratio": self.pseudotime.explained_variance_ratio,
                "burden_correlation": self.pseudotime.burden_correlation,
                "score_lower": self.pseudotime.score_lower,
                "score_upper": self.pseudotime.score_upper,
            },
            "spline": {
                "n_basis": self.spline_basis.n_basis,
                "degree": self.spline_basis.degree,
                "n_knots": self.spline_basis.n_knots,
            },
            "children": [
                {
                    "child": fit.child_label,
                    "alpha": fit.ridge.alpha,
                    "train_mse_rate": fit.ridge.train_mse,
                    "train_r2_rate": fit.ridge.train_r2,
                    "regional_parent_count": len(fit.parent_labels),
                    "root_parent_count": len(fit.root_names),
                }
                for fit in self.child_fits
            ],
        }


def fit_pseudotime_embedding(
    baseline: np.ndarray,
    *,
    train_indices: np.ndarray,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> PseudotimeEmbedding:
    matrix = np.asarray(baseline, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("baseline must be two-dimensional.")
    train = matrix[np.asarray(train_indices, dtype=int)]
    if train.shape[0] < 3:
        raise ValueError("At least three training rows are required for pseudotime.")

    feature_mean = np.mean(train, axis=0)
    feature_scale = np.std(train, axis=0)
    feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 1.0e-12), feature_scale, 1.0)
    scaled_train = np.nan_to_num(
        (train - feature_mean[None, :]) / feature_scale[None, :],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    _, singular_values, vt = np.linalg.svd(scaled_train, full_matrices=False)
    component = np.asarray(vt[0], dtype=float)
    scores = np.einsum("ij,j->i", scaled_train, component)
    burden = np.mean(train, axis=1)
    burden_correlation = safe_correlation(scores, burden)
    if np.isfinite(burden_correlation) and burden_correlation < 0.0:
        component = -component
        scores = -scores
        burden_correlation = -burden_correlation

    lower = float(np.quantile(scores, lower_quantile))
    upper = float(np.quantile(scores, upper_quantile))
    if not np.isfinite(upper - lower) or upper <= lower:
        lower = float(np.min(scores))
        upper = float(np.max(scores))
    if not np.isfinite(upper - lower) or upper <= lower:
        upper = lower + 1.0

    variance = singular_values**2
    explained = float(variance[0] / np.sum(variance)) if np.sum(variance) > 0.0 else 0.0
    return PseudotimeEmbedding(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        component=component,
        score_lower=lower,
        score_upper=upper,
        explained_variance_ratio=explained,
        burden_correlation=float(burden_correlation),
    )


def fit_spline_basis(z_train: np.ndarray, *, n_knots: int, degree: int) -> ConstantSplineBasis | FittedSplineBasis:
    z_values = np.asarray(z_train, dtype=float).reshape(-1)
    if n_knots <= 1 or degree <= 0 or np.unique(np.round(z_values, 8)).size < max(3, degree):
        return ConstantSplineBasis()
    from sklearn.preprocessing import SplineTransformer

    knots = np.linspace(0.0, 1.0, int(n_knots), dtype=float).reshape(-1, 1)
    try:
        transformer = SplineTransformer(
            degree=int(degree),
            knots=knots,
            include_bias=True,
            extrapolation="constant",
            sparse_output=False,
        )
    except TypeError:
        transformer = SplineTransformer(
            degree=int(degree),
            knots=knots,
            include_bias=True,
            extrapolation="constant",
            sparse=False,
        )
    basis = transformer.fit_transform(z_values.reshape(-1, 1))
    if hasattr(basis, "toarray"):
        basis = basis.toarray()
    return FittedSplineBasis(
        transformer=transformer,
        n_basis=int(np.asarray(basis).shape[1]),
        degree=int(degree),
        n_knots=int(n_knots),
    )


def select_regions_by_train_variance(
    baseline: np.ndarray,
    region_labels: list[str],
    *,
    train_indices: np.ndarray,
    max_regions: int,
) -> np.ndarray:
    if len(region_labels) != np.asarray(baseline).shape[1]:
        raise ValueError("region_labels must match baseline columns.")
    region_count = len(region_labels)
    if max_regions <= 0 or max_regions >= region_count:
        return np.arange(region_count, dtype=int)
    train = np.asarray(baseline, dtype=float)[np.asarray(train_indices, dtype=int)]
    variance = np.nanvar(train, axis=0)
    order = np.argsort(-variance, kind="mergesort")
    return np.sort(order[: int(max_regions)]).astype(int)


def fit_bn_lte_model(
    *,
    baseline: np.ndarray,
    observed: np.ndarray,
    time_years: np.ndarray,
    region_labels: list[str],
    train_indices: np.ndarray,
    pair_groups: np.ndarray,
    model_name: str,
    parent_mode: str = "none",
    include_roots: bool = False,
    root_names: list[str] | None = None,
    root_values: np.ndarray | None = None,
    allowed_edges: set[tuple[str, str]] | None = None,
    include_self_history: bool = True,
    max_parents_per_child: int = 5,
    n_knots: int = 4,
    spline_degree: int = 3,
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
    cv_folds: int = 5,
    edge_effect_threshold: float = 0.01,
) -> BNLTEFit:
    baseline = np.asarray(baseline, dtype=float)
    observed = np.asarray(observed, dtype=float)
    time_years = np.asarray(time_years, dtype=float)
    train_indices = np.asarray(train_indices, dtype=int)
    if baseline.shape != observed.shape or baseline.ndim != 2:
        raise ValueError("baseline and observed must be matching two-dimensional arrays.")
    if len(region_labels) != baseline.shape[1]:
        raise ValueError("region_labels must match the regional matrix width.")
    if time_years.shape != (baseline.shape[0],):
        raise ValueError("time_years must have one value per row.")
    if pair_groups.shape != (baseline.shape[0],):
        raise ValueError("pair_groups must have one group label per row.")

    root_names = list(root_names or [])
    if include_roots:
        root_matrix = np.asarray(root_values, dtype=float) if root_values is not None else empty_root_values(baseline.shape[0])
        if root_matrix.shape != (baseline.shape[0], len(root_names)):
            raise ValueError("root_values must have shape rows x root_names.")
    else:
        root_names = []
        root_matrix = empty_root_values(baseline.shape[0])

    pseudotime = fit_pseudotime_embedding(baseline, train_indices=train_indices)
    z = pseudotime.transform(baseline)
    spline_basis = fit_spline_basis(z[train_indices], n_knots=n_knots, degree=spline_degree)
    lower, upper = prediction_bounds(baseline[train_indices], observed[train_indices])
    state = scale_state(baseline, lower, upper)
    safe_time = np.maximum(time_years, 1.0e-6)
    target_rate = (observed - baseline) / safe_time[:, None]
    parent_sets = make_parent_sets(
        baseline=baseline,
        observed=observed,
        time_years=time_years,
        train_indices=train_indices,
        region_labels=region_labels,
        parent_mode=parent_mode,
        max_parents_per_child=max_parents_per_child,
        allowed_edges=allowed_edges,
    )

    child_fits: list[BNLTEChildFit] = []
    for child_index, child_label in enumerate(region_labels):
        parent_indices = parent_sets[child_index]
        parent_labels = [region_labels[index] for index in parent_indices]
        design = build_design_matrix(
            baseline=baseline,
            state=state,
            z=z,
            spline_basis=spline_basis,
            child_index=child_index,
            parent_indices=parent_indices,
            parent_labels=parent_labels,
            root_names=root_names,
            root_values=root_matrix,
            include_self_history=include_self_history,
        )
        ridge = fit_ridge_rate(
            design.values,
            target_rate[:, child_index],
            row_indices=train_indices,
            pair_groups=pair_groups,
            feature_names=design.feature_names,
            alphas=ridge_alphas,
            cv_folds=cv_folds,
        )
        child_fits.append(
            BNLTEChildFit(
                child_label=child_label,
                child_index=child_index,
                parent_indices=parent_indices,
                parent_labels=parent_labels,
                root_names=root_names,
                ridge=ridge,
                parent_effect_columns=design.parent_effect_columns,
                parent_source_types=design.parent_source_types,
            )
        )

    return BNLTEFit(
        model_name=model_name,
        region_indices=np.arange(baseline.shape[1], dtype=int),
        region_labels=list(region_labels),
        pseudotime=pseudotime,
        spline_basis=spline_basis,
        child_fits=child_fits,
        train_indices=train_indices,
        parent_mode=parent_mode,
        include_roots=bool(include_roots),
        include_self_history=bool(include_self_history),
        prediction_lower=lower,
        prediction_upper=upper,
        edge_effect_threshold=float(edge_effect_threshold),
        z_grid=np.linspace(0.0, 1.0, 101),
    )


def build_design_matrix(
    *,
    baseline: np.ndarray,
    state: np.ndarray,
    z: np.ndarray,
    spline_basis: ConstantSplineBasis | FittedSplineBasis,
    child_index: int,
    parent_indices: list[int],
    parent_labels: list[str],
    root_names: list[str],
    root_values: np.ndarray,
    include_self_history: bool,
) -> DesignMatrix:
    baseline = np.asarray(baseline, dtype=float)
    state = np.asarray(state, dtype=float)
    basis = spline_basis.transform(z)
    columns: list[np.ndarray] = []
    names: list[str] = []
    parent_effect_columns: dict[str, list[int]] = {}
    parent_source_types: dict[str, str] = {}

    for basis_index in range(basis.shape[1]):
        columns.append(basis[:, basis_index])
        names.append(f"trajectory_spline_{basis_index}")

    if include_self_history:
        child_state = state[:, child_index]
        self_features = [
            ("self_state", child_state),
            ("self_state^2", child_state**2),
            ("self_state^3*(1-self_state)", child_state**3 * (1.0 - child_state)),
        ]
        for feature_name, values in self_features:
            columns.append(values)
            names.append(feature_name)

    for parent_index, parent_label in zip(parent_indices, parent_labels, strict=True):
        parent_effect_columns[parent_label] = []
        parent_source_types[parent_label] = "regional"
        parent_values = baseline[:, parent_index]
        for basis_index in range(basis.shape[1]):
            parent_effect_columns[parent_label].append(len(columns))
            columns.append(parent_values * basis[:, basis_index])
            names.append(f"edge:{parent_label}:spline_{basis_index}")

    root_matrix = np.asarray(root_values, dtype=float)
    if root_matrix.shape[1] != len(root_names):
        raise ValueError("root_values column count must match root_names.")
    for root_index, root_name in enumerate(root_names):
        parent_label = f"root:{root_name}"
        parent_effect_columns[parent_label] = []
        parent_source_types[parent_label] = "root"
        root_column = root_matrix[:, root_index]
        for basis_index in range(basis.shape[1]):
            parent_effect_columns[parent_label].append(len(columns))
            columns.append(root_column * basis[:, basis_index])
            names.append(f"root_edge:{root_name}:spline_{basis_index}")

    return DesignMatrix(
        values=np.column_stack(columns),
        feature_names=names,
        parent_effect_columns=parent_effect_columns,
        parent_source_types=parent_source_types,
    )


def fit_ridge_rate(
    feature_values: np.ndarray,
    target: np.ndarray,
    *,
    row_indices: np.ndarray,
    pair_groups: np.ndarray,
    feature_names: list[str],
    alphas: tuple[float, ...],
    cv_folds: int,
) -> RidgeRateFit:
    from sklearn.model_selection import GroupKFold

    values = np.asarray(feature_values, dtype=float)
    target = np.asarray(target, dtype=float)
    row_indices = np.asarray(row_indices, dtype=int)
    selected_x = values[row_indices]
    selected_y = target[row_indices]
    selected_groups = pair_groups.astype(str)[row_indices]
    finite_mask = np.isfinite(selected_y) & np.all(np.isfinite(selected_x), axis=1)
    x_train = selected_x[finite_mask]
    y_train = selected_y[finite_mask]
    groups_train = selected_groups[finite_mask]
    if x_train.shape[0] < max(8, min(values.shape[1] + 1, 20)):
        raise ValueError("Not enough finite rows to fit BN-LTE child model.")

    feature_mean = np.mean(x_train, axis=0)
    feature_scale = np.std(x_train, axis=0)
    feature_scale = np.where(np.isfinite(feature_scale) & (feature_scale > 1.0e-12), feature_scale, 1.0)
    x_scaled = np.nan_to_num(
        (x_train - feature_mean[None, :]) / feature_scale[None, :],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    alpha_values = tuple(float(alpha) for alpha in alphas if float(alpha) > 0.0)
    if not alpha_values:
        raise ValueError("At least one positive ridge alpha is required.")

    unique_groups = np.unique(groups_train)
    fold_count = min(int(cv_folds), unique_groups.size)
    cv_report: list[dict[str, Any]] = []
    if fold_count >= 2 and x_train.shape[0] >= fold_count:
        splitter = GroupKFold(n_splits=fold_count)
        for alpha in alpha_values:
            fold_mse: list[float] = []
            for train_position, validation_position in splitter.split(x_scaled, y_train, groups_train):
                intercept, coefficients = solve_ridge_closed_form(
                    x_scaled[train_position],
                    y_train[train_position],
                    alpha=alpha,
                )
                predicted = intercept + np.einsum("ij,j->i", x_scaled[validation_position], coefficients)
                fold_mse.append(float(np.mean((predicted - y_train[validation_position]) ** 2)))
            cv_report.append({"alpha": alpha, "cv_mse": float(np.mean(fold_mse)), "folds": fold_count})
        selected_alpha = float(min(cv_report, key=lambda row: float(row["cv_mse"]))["alpha"])
    else:
        selected_alpha = alpha_values[len(alpha_values) // 2]
        cv_report.append({"alpha": selected_alpha, "cv_mse": float("nan"), "folds": 0})

    intercept, coefficients = solve_ridge_closed_form(x_scaled, y_train, alpha=selected_alpha)
    train_prediction = intercept + np.einsum("ij,j->i", x_scaled, coefficients)
    residual = y_train - train_prediction
    train_mse = float(np.mean(residual**2))
    total = float(np.sum((y_train - np.mean(y_train)) ** 2))
    train_r2 = 0.0 if total <= 0.0 else float(1.0 - np.sum(residual**2) / total)
    return RidgeRateFit(
        feature_names=list(feature_names),
        coefficients=np.asarray(coefficients, dtype=float),
        intercept=float(intercept),
        alpha=selected_alpha,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        train_mse=train_mse,
        train_r2=train_r2,
        cv_report=cv_report,
        used_train_rows=int(x_train.shape[0]),
    )


def solve_ridge_closed_form(x: np.ndarray, y: np.ndarray, *, alpha: float) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("x and y have incompatible shapes for ridge fitting.")
    intercept = float(np.mean(y))
    centered_y = y - intercept
    xtx = np.einsum("ni,nj->ij", x, x)
    rhs = np.einsum("ni,n->i", x, centered_y)
    penalty = np.eye(x.shape[1], dtype=float) * float(alpha)
    try:
        coefficients = np.linalg.solve(xtx + penalty, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(xtx + penalty, rhs, rcond=None)[0]
    coefficients = np.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)
    return intercept, np.asarray(coefficients, dtype=float)


def make_parent_sets(
    *,
    baseline: np.ndarray,
    observed: np.ndarray,
    time_years: np.ndarray,
    train_indices: np.ndarray,
    region_labels: list[str],
    parent_mode: str,
    max_parents_per_child: int,
    allowed_edges: set[tuple[str, str]] | None,
) -> list[list[int]]:
    region_count = len(region_labels)
    if parent_mode == "none":
        return [[] for _ in range(region_count)]
    if parent_mode not in {"progression_ordered", "all"}:
        raise ValueError(f"Unknown parent_mode: {parent_mode}")

    train_indices = np.asarray(train_indices, dtype=int)
    baseline = np.asarray(baseline, dtype=float)
    observed = np.asarray(observed, dtype=float)
    safe_time = np.maximum(np.asarray(time_years, dtype=float), 1.0e-6)
    target_rate = (observed - baseline) / safe_time[:, None]
    mean_burden = np.mean(baseline[train_indices], axis=0)
    burden_rank = np.empty(region_count, dtype=int)
    burden_rank[np.argsort(-mean_burden, kind="mergesort")] = np.arange(region_count)

    parent_sets: list[list[int]] = []
    for child_index, child_label in enumerate(region_labels):
        candidates = []
        for parent_index, parent_label in enumerate(region_labels):
            if parent_index == child_index:
                continue
            if allowed_edges is not None and (parent_label, child_label) not in allowed_edges:
                continue
            if parent_mode == "progression_ordered" and burden_rank[parent_index] >= burden_rank[child_index]:
                continue
            association = abs(safe_correlation(baseline[train_indices, parent_index], target_rate[train_indices, child_index]))
            if not np.isfinite(association):
                association = 0.0
            candidates.append((association, burden_rank[parent_index], parent_index))
        candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
        if max_parents_per_child > 0:
            candidates = candidates[: int(max_parents_per_child)]
        parent_sets.append([int(row[2]) for row in candidates])
    return parent_sets


def bootstrap_edge_probabilities(
    *,
    baseline: np.ndarray,
    observed: np.ndarray,
    time_years: np.ndarray,
    region_labels: list[str],
    train_indices: np.ndarray,
    pair_groups: np.ndarray,
    iterations: int,
    random_seed: int,
    model_name: str,
    parent_mode: str,
    include_roots: bool,
    root_names: list[str] | None,
    root_values: np.ndarray | None,
    include_self_history: bool,
    max_parents_per_child: int,
    n_knots: int,
    spline_degree: int,
    ridge_alphas: tuple[float, ...],
    edge_effect_threshold: float,
) -> list[dict[str, Any]]:
    if iterations <= 0:
        return []
    train_indices = np.asarray(train_indices, dtype=int)
    pair_groups = np.asarray(pair_groups).astype(str)
    groups = np.unique(pair_groups[train_indices])
    indices_by_group = {group: train_indices[pair_groups[train_indices] == group] for group in groups}
    rng = np.random.default_rng(int(random_seed))
    edge_counts: dict[tuple[str, str], int] = {}
    edge_effect_sum: dict[tuple[str, str], float] = {}
    edge_source_type: dict[tuple[str, str], str] = {}

    for iteration in range(int(iterations)):
        sampled_groups = rng.choice(groups, size=groups.size, replace=True)
        sampled_indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        fit = fit_bn_lte_model(
            baseline=baseline,
            observed=observed,
            time_years=time_years,
            region_labels=region_labels,
            train_indices=sampled_indices,
            pair_groups=pair_groups,
            model_name=f"{model_name}_bootstrap_{iteration + 1}",
            parent_mode=parent_mode,
            include_roots=include_roots,
            root_names=root_names,
            root_values=root_values,
            include_self_history=include_self_history,
            max_parents_per_child=max_parents_per_child,
            n_knots=n_knots,
            spline_degree=spline_degree,
            ridge_alphas=ridge_alphas,
            cv_folds=0,
            edge_effect_threshold=edge_effect_threshold,
        )
        for row in fit.edge_rows():
            key = (str(row["parent"]), str(row["child"]))
            edge_source_type[key] = str(row["source_type"])
            edge_effect_sum[key] = edge_effect_sum.get(key, 0.0) + float(row["max_abs_effect"])
            if row["included_by_effect_threshold"]:
                edge_counts[key] = edge_counts.get(key, 0) + 1
            else:
                edge_counts.setdefault(key, 0)

    rows = []
    for key in sorted(edge_effect_sum, key=lambda item: (-edge_counts.get(item, 0), item[0], item[1])):
        count = int(edge_counts.get(key, 0))
        rows.append(
            {
                "model": model_name,
                "parent": key[0],
                "child": key[1],
                "source_type": edge_source_type.get(key, "regional"),
                "bootstrap_inclusion_probability": float(count / iterations),
                "included_bootstraps": count,
                "bootstrap_iterations": int(iterations),
                "mean_max_abs_effect": float(edge_effect_sum[key] / iterations),
                "effect_threshold": float(edge_effect_threshold),
            }
        )
    return rows


def stable_edges_from_bootstrap(
    rows: list[dict[str, Any]],
    *,
    pip_threshold: float,
    source_type: str = "regional",
) -> set[tuple[str, str]]:
    return {
        (str(row["parent"]), str(row["child"]))
        for row in rows
        if str(row.get("source_type")) == source_type
        and float(row.get("bootstrap_inclusion_probability", 0.0)) >= float(pip_threshold)
    }


def scale_state(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float), 1.0e-8)
    return np.clip((np.asarray(values, dtype=float) - lower[None, :]) / scale[None, :], 0.0, 1.0)


def prediction_bounds(baseline_train: np.ndarray, observed_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack([np.asarray(baseline_train, dtype=float), np.asarray(observed_train, dtype=float)])
    lower = np.nanpercentile(stacked, 0.5, axis=0)
    upper = np.nanpercentile(stacked, 99.5, axis=0)
    upper = np.where(upper > lower, upper, lower + 1.0)
    margin = 0.05 * (upper - lower)
    return lower - margin, upper + margin


def empty_root_values(row_count: int) -> np.ndarray:
    return np.zeros((int(row_count), 0), dtype=float)


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
