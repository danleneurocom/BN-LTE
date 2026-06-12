"""Constrained dynamic structural causal model for BN-LTE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .constraints import CausalConstraints
from .data import MultimodalPairDataset
from .pseudotime import PseudotimeModel, safe_correlation


@dataclass
class SplineBasis:
    n_basis: int
    n_knots: int
    degree: int
    transformer: Any | None = None

    def transform(self, z: np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=float).reshape(-1, 1)
        if self.transformer is not None:
            basis = self.transformer.transform(values)
            if hasattr(basis, "toarray"):
                basis = basis.toarray()
            return np.asarray(basis, dtype=float)
        flat = values.reshape(-1)
        columns = [np.ones_like(flat)]
        for power in range(1, self.n_basis):
            columns.append(flat**power)
        return np.column_stack(columns)


@dataclass
class DesignInfo:
    values: np.ndarray
    feature_names: list[str]
    parent_columns: dict[str, list[int]]
    self_columns: list[int]


@dataclass
class RidgeFit:
    alpha: float
    intercept: float
    coefficients: np.ndarray
    fill_values: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    train_mse: float
    train_r2: float
    cv_report: list[dict[str, float | int]]

    def predict(self, design: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(design), design, self.fill_values[None, :])
        scaled = (filled - self.center[None, :]) / self.scale[None, :]
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.intercept + np.einsum("ij,j->i", scaled, self.coefficients)


@dataclass
class TargetSCMFit:
    target_name: str
    target_index: int
    parent_names: list[str]
    design_info: DesignInfo
    ridge: RidgeFit

    def parent_effect_curve(self, parent_name: str, basis_values: np.ndarray) -> np.ndarray:
        columns = self.design_info.parent_columns.get(parent_name, [])
        if not columns:
            return np.zeros(basis_values.shape[0], dtype=float)
        effect = np.zeros(basis_values.shape[0], dtype=float)
        for basis_idx, column in enumerate(columns):
            effect += basis_values[:, basis_idx] * self.ridge.coefficients[column] / self.ridge.scale[column]
        return effect

    def self_effect_curve(self, basis_values: np.ndarray) -> np.ndarray:
        if not self.design_info.self_columns:
            return np.zeros(basis_values.shape[0], dtype=float)
        effect = np.zeros(basis_values.shape[0], dtype=float)
        for basis_idx, column in enumerate(self.design_info.self_columns):
            effect += basis_values[:, basis_idx] * self.ridge.coefficients[column] / self.ridge.scale[column]
        return effect


@dataclass
class DynamicSCMFit:
    target_fits: list[TargetSCMFit]
    pseudotime: PseudotimeModel
    spline_basis: SplineBasis
    feature_names: list[str]
    target_names: list[str]
    train_indices: np.ndarray
    z_grid: np.ndarray
    edge_effect_threshold: float
    constraints_report: dict[str, Any]

    def predict_rates(self, dataset: MultimodalPairDataset) -> np.ndarray:
        z = self.pseudotime.transform(dataset.feature_matrix)
        rates = np.full((dataset.pair_count, len(self.target_names)), np.nan, dtype=float)
        for fit in self.target_fits:
            design = build_design_matrix(
                feature_matrix=dataset.feature_matrix,
                target_baseline=dataset.target_baseline[:, fit.target_index],
                z=z,
                spline_basis=self.spline_basis,
                parent_names=fit.parent_names,
                feature_names=dataset.feature_names,
            )
            rates[:, fit.target_index] = fit.ridge.predict(design.values)
        return rates

    def predict_observed(self, dataset: MultimodalPairDataset) -> np.ndarray:
        rates = self.predict_rates(dataset)
        return dataset.target_baseline + dataset.time_years[:, None] * rates

    def edge_effect_rows(self) -> list[dict[str, Any]]:
        basis = self.spline_basis.transform(self.z_grid)
        rows: list[dict[str, Any]] = []
        for fit in self.target_fits:
            self_effect = fit.self_effect_curve(basis)
            if self_effect.size:
                rows.append(effect_summary_row("self_history", fit.target_name, self_effect, self.z_grid, self.edge_effect_threshold))
            for parent in fit.parent_names:
                effect = fit.parent_effect_curve(parent, basis)
                rows.append(effect_summary_row(parent, fit.target_name, effect, self.z_grid, self.edge_effect_threshold))
        return sorted(rows, key=lambda row: float(row["max_abs_effect"]), reverse=True)

    def report(self) -> dict[str, Any]:
        return {
            "target_count": len(self.target_fits),
            "targets": self.target_names,
            "spline": {
                "n_basis": self.spline_basis.n_basis,
                "n_knots": self.spline_basis.n_knots,
                "degree": self.spline_basis.degree,
            },
            "edge_effect_threshold": self.edge_effect_threshold,
            "constraints": self.constraints_report,
            "target_fits": [
                {
                    "target": fit.target_name,
                    "parent_count": len(fit.parent_names),
                    "parents": fit.parent_names,
                    "alpha": fit.ridge.alpha,
                    "train_mse_rate": fit.ridge.train_mse,
                    "train_r2_rate": fit.ridge.train_r2,
                }
                for fit in self.target_fits
            ],
        }


def fit_dynamic_scm(
    dataset: MultimodalPairDataset,
    pseudotime: PseudotimeModel,
    train_indices: np.ndarray,
    *,
    target_names: Iterable[str] | None = None,
    constraints: CausalConstraints | None = None,
    max_parents_per_target: int = 8,
    n_knots: int = 4,
    spline_degree: int = 3,
    ridge_alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    cv_folds: int = 5,
    edge_effect_threshold: float = 0.01,
) -> DynamicSCMFit:
    train_indices = np.asarray(train_indices, dtype=int)
    constraints = constraints or CausalConstraints(dataset.variable_specs)
    selected_targets = list(target_names or dataset.target_names)
    target_indices = [dataset.target_index(name) for name in selected_targets]
    z = pseudotime.transform(dataset.feature_matrix)
    spline_basis = fit_spline_basis(z[train_indices], n_knots=n_knots, degree=spline_degree)
    groups = np.asarray([row["RID"] for row in dataset.metadata_rows], dtype=object)

    target_fits: list[TargetSCMFit] = []
    for target_index in target_indices:
        target_name = dataset.target_names[target_index]
        parent_names = select_candidate_parents(
            dataset,
            constraints,
            target_name,
            target_index,
            train_indices=train_indices,
            max_parents=max_parents_per_target,
        )
        design = build_design_matrix(
            feature_matrix=dataset.feature_matrix,
            target_baseline=dataset.target_baseline[:, target_index],
            z=z,
            spline_basis=spline_basis,
            parent_names=parent_names,
            feature_names=dataset.feature_names,
        )
        ridge = fit_ridge(
            design.values,
            dataset.target_rates[:, target_index],
            train_indices=train_indices,
            groups=groups,
            alphas=ridge_alphas,
            cv_folds=cv_folds,
        )
        target_fits.append(
            TargetSCMFit(
                target_name=target_name,
                target_index=target_index,
                parent_names=parent_names,
                design_info=design,
                ridge=ridge,
            )
        )

    return DynamicSCMFit(
        target_fits=target_fits,
        pseudotime=pseudotime,
        spline_basis=spline_basis,
        feature_names=dataset.feature_names,
        target_names=dataset.target_names,
        train_indices=train_indices,
        z_grid=np.linspace(0.0, 1.0, 101),
        edge_effect_threshold=float(edge_effect_threshold),
        constraints_report=constraints.report(),
    )


def fit_spline_basis(z_train: np.ndarray, *, n_knots: int, degree: int) -> SplineBasis:
    values = np.asarray(z_train, dtype=float).reshape(-1)
    if int(n_knots) <= 1 or int(degree) <= 0 or np.unique(np.round(values, 8)).size < max(3, int(degree)):
        return SplineBasis(n_basis=1, n_knots=1, degree=0, transformer=None)
    try:
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
        basis = transformer.fit_transform(values.reshape(-1, 1))
        if hasattr(basis, "toarray"):
            basis = basis.toarray()
        return SplineBasis(n_basis=int(np.asarray(basis).shape[1]), n_knots=int(n_knots), degree=int(degree), transformer=transformer)
    except Exception:
        return SplineBasis(n_basis=min(4, int(degree) + 1), n_knots=0, degree=int(degree), transformer=None)


def select_candidate_parents(
    dataset: MultimodalPairDataset,
    constraints: CausalConstraints,
    target_name: str,
    target_index: int,
    *,
    train_indices: np.ndarray,
    max_parents: int,
) -> list[str]:
    candidates = constraints.candidate_parents(target_name, dataset.feature_names)
    y = dataset.target_rates[np.asarray(train_indices, dtype=int), target_index]
    priority = priority_parents_for_target(target_name, candidates)
    ranked = []
    for name in candidates:
        idx = dataset.feature_index(name)
        x = dataset.feature_matrix[np.asarray(train_indices, dtype=int), idx]
        coverage = float(np.mean(np.isfinite(x)))
        if coverage < 0.25:
            continue
        association = abs(safe_correlation(x, y))
        if not np.isfinite(association):
            association = 0.0
        ranked.append((association, coverage, name))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    usable_candidate_names = {name for _, _, name in ranked}
    output: list[str] = []
    for name in priority:
        if name in usable_candidate_names and name not in output:
            output.append(name)
    for _, _, name in ranked:
        if name not in output:
            output.append(name)
        if len(output) >= int(max_parents):
            break
    return output[: int(max_parents)]


def priority_parents_for_target(target_name: str, candidates: Iterable[str]) -> list[str]:
    candidate_set = set(candidates)
    text = target_name.lower()
    amyloid_priority = (
        "amyloid_centiloids",
        "amyloid_summary_suvr",
        "amyloid_positive",
        "plasma_ab42_ab40",
        "apoe4_dose",
        "age_years",
    )
    tau_priority = (
        "plasma_pt217",
        "amyloid_centiloids",
        "amyloid_summary_suvr",
        "interaction:amyloid_centiloids_x_mri_hippocampus_vulnerability",
        "interaction:amyloid_centiloids_x_mri_temporal_thickness_vulnerability",
        "interaction:amyloid_centiloids_x_tau_meta_temporal",
        "mri_hippocampus_vulnerability",
        "mri_temporal_cortical_thickness",
        "plasma_ab42_ab40",
        "apoe4_dose",
        "age_years",
    )
    neuro_priority = (
        "tau_meta_temporal",
        "plasma_pt217",
        "amyloid_centiloids",
        "amyloid_summary_suvr",
        "plasma_nfl",
        "plasma_gfap",
        "mri_hippocampus_vulnerability",
        "mri_temporal_cortical_thickness",
        "age_years",
        "apoe4_dose",
    )
    if "amyloid_rate" in text:
        return [name for name in amyloid_priority if name in candidate_set]
    if "tau_rate" in text:
        return [name for name in tau_priority if name in candidate_set]
    if any(token in text for token in ("atrophy_rate", "ashs_rate", "mri_thickness_rate", "mri_volume_rate")):
        return [name for name in neuro_priority if name in candidate_set]
    if "cognitive_rate" in text:
        return [
            name
            for name in (
                "tau_meta_temporal",
                "mri_hippocampus_vulnerability",
                "mri_temporal_cortical_thickness",
                "amyloid_centiloids",
                "amyloid_summary_suvr",
                "plasma_nfl",
                "plasma_gfap",
                "age_years",
            )
            if name in candidate_set
        ]
    return []


def build_design_matrix(
    *,
    feature_matrix: np.ndarray,
    target_baseline: np.ndarray,
    z: np.ndarray,
    spline_basis: SplineBasis,
    parent_names: list[str],
    feature_names: list[str],
) -> DesignInfo:
    basis = spline_basis.transform(z)
    columns: list[np.ndarray] = []
    names: list[str] = []
    parent_columns: dict[str, list[int]] = {}
    self_columns: list[int] = []

    for basis_idx in range(basis.shape[1]):
        columns.append(basis[:, basis_idx])
        names.append(f"trajectory_spline_{basis_idx}")

    baseline = np.asarray(target_baseline, dtype=float)
    for basis_idx in range(basis.shape[1]):
        self_columns.append(len(columns))
        columns.append(baseline * basis[:, basis_idx])
        names.append(f"self_history:spline_{basis_idx}")

    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    for parent in parent_names:
        parent_values = feature_matrix[:, name_to_index[parent]]
        parent_columns[parent] = []
        for basis_idx in range(basis.shape[1]):
            parent_columns[parent].append(len(columns))
            columns.append(parent_values * basis[:, basis_idx])
            names.append(f"edge:{parent}:spline_{basis_idx}")

    return DesignInfo(
        values=np.column_stack(columns),
        feature_names=names,
        parent_columns=parent_columns,
        self_columns=self_columns,
    )


def fit_ridge(
    design: np.ndarray,
    target: np.ndarray,
    *,
    train_indices: np.ndarray,
    groups: np.ndarray,
    alphas: tuple[float, ...],
    cv_folds: int,
) -> RidgeFit:
    values = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    train_indices = np.asarray(train_indices, dtype=int)
    train_values_raw = values[train_indices]
    train_target_raw = target[train_indices]
    train_groups_raw = np.asarray(groups, dtype=object)[train_indices]
    y_finite = np.isfinite(train_target_raw)
    if int(np.sum(y_finite)) < max(8, min(values.shape[1] + 1, 25)):
        raise ValueError("Not enough finite target rows to fit dynamic SCM target.")

    fill_values = np.nanmedian(train_values_raw[y_finite], axis=0)
    global_fill = float(np.nanmedian(train_values_raw[y_finite])) if np.any(np.isfinite(train_values_raw[y_finite])) else 0.0
    fill_values = np.where(np.isfinite(fill_values), fill_values, global_fill)
    train_values = np.where(np.isfinite(train_values_raw), train_values_raw, fill_values[None, :])[y_finite]
    train_target = train_target_raw[y_finite]
    train_groups = train_groups_raw[y_finite].astype(str)

    center = np.mean(train_values, axis=0)
    scale = np.std(train_values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    train_scaled = np.nan_to_num((train_values - center[None, :]) / scale[None, :], nan=0.0)

    alpha_values = tuple(float(alpha) for alpha in alphas if float(alpha) > 0.0)
    if not alpha_values:
        raise ValueError("At least one positive ridge alpha is required.")

    selected_alpha, cv_report = select_alpha(train_scaled, train_target, train_groups, alpha_values, cv_folds)
    intercept, coefficients = solve_ridge(train_scaled, train_target, alpha=selected_alpha)
    prediction = intercept + np.einsum("ij,j->i", train_scaled, coefficients)
    residual = train_target - prediction
    mse = float(np.mean(residual**2))
    total = float(np.sum((train_target - np.mean(train_target)) ** 2))
    r2 = 0.0 if total <= 0.0 else float(1.0 - np.sum(residual**2) / total)
    return RidgeFit(
        alpha=float(selected_alpha),
        intercept=float(intercept),
        coefficients=np.asarray(coefficients, dtype=float),
        fill_values=np.asarray(fill_values, dtype=float),
        center=np.asarray(center, dtype=float),
        scale=np.asarray(scale, dtype=float),
        train_mse=mse,
        train_r2=r2,
        cv_report=cv_report,
    )


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: tuple[float, ...],
    cv_folds: int,
) -> tuple[float, list[dict[str, float | int]]]:
    unique_groups = np.unique(groups)
    folds = min(int(cv_folds), unique_groups.size)
    report: list[dict[str, float | int]] = []
    if folds >= 2:
        try:
            from sklearn.model_selection import GroupKFold

            splitter = GroupKFold(n_splits=folds)
            for alpha in alphas:
                losses = []
                for train_pos, val_pos in splitter.split(x, y, groups):
                    intercept, coef = solve_ridge(x[train_pos], y[train_pos], alpha=alpha)
                    pred = intercept + np.einsum("ij,j->i", x[val_pos], coef)
                    losses.append(float(np.mean((pred - y[val_pos]) ** 2)))
                report.append({"alpha": float(alpha), "cv_mse": float(np.mean(losses)), "folds": int(folds)})
            selected = min(report, key=lambda row: float(row["cv_mse"]))["alpha"]
            return float(selected), report
        except Exception:
            pass

    for alpha in alphas:
        intercept, coef = solve_ridge(x, y, alpha=alpha)
        pred = intercept + np.einsum("ij,j->i", x, coef)
        report.append({"alpha": float(alpha), "cv_mse": float(np.mean((pred - y) ** 2)), "folds": 0})
    selected = min(report, key=lambda row: float(row["cv_mse"]))["alpha"]
    return float(selected), report


def solve_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float) -> tuple[float, np.ndarray]:
    intercept = float(np.mean(y))
    centered_y = y - intercept
    xtx = np.einsum("ni,nj->ij", x, x)
    rhs = np.einsum("ni,n->i", x, centered_y)
    penalty = np.eye(x.shape[1], dtype=float) * float(alpha)
    try:
        coef = np.linalg.solve(xtx + penalty, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx + penalty, rhs, rcond=None)[0]
    return intercept, np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)


def effect_summary_row(parent: str, target: str, effect: np.ndarray, z_grid: np.ndarray, threshold: float) -> dict[str, Any]:
    max_idx = int(np.argmax(np.abs(effect))) if effect.size else 0
    max_abs = float(np.max(np.abs(effect))) if effect.size else 0.0
    return {
        "parent": parent,
        "target": target,
        "max_abs_effect": max_abs,
        "mean_abs_effect": float(np.mean(np.abs(effect))) if effect.size else 0.0,
        "z_at_max_abs_effect": float(z_grid[max_idx]) if effect.size else float("nan"),
        "effect_at_z_min": float(effect[0]) if effect.size else float("nan"),
        "effect_at_z_mid": float(effect[effect.size // 2]) if effect.size else float("nan"),
        "effect_at_z_max": float(effect[-1]) if effect.size else float("nan"),
        "included_by_effect_threshold": bool(max_abs >= float(threshold)),
        "effect_threshold": float(threshold),
    }


def bootstrap_edge_stability(
    dataset: MultimodalPairDataset,
    pseudotime: PseudotimeModel,
    train_indices: np.ndarray,
    *,
    target_names: Iterable[str],
    iterations: int,
    random_seed: int = 20260519,
    max_parents_per_target: int = 8,
    edge_effect_threshold: float = 0.01,
) -> list[dict[str, Any]]:
    if int(iterations) <= 0:
        return []
    train_indices = np.asarray(train_indices, dtype=int)
    groups = np.asarray([row["RID"] for row in dataset.metadata_rows], dtype=object)
    unique_groups = np.unique(groups[train_indices])
    indices_by_group = {group: train_indices[groups[train_indices] == group] for group in unique_groups}
    rng = np.random.default_rng(int(random_seed))
    counts: dict[tuple[str, str], int] = {}
    effects: dict[tuple[str, str], float] = {}

    for _ in range(int(iterations)):
        sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        sampled_indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        fit = fit_dynamic_scm(
            dataset,
            pseudotime,
            sampled_indices,
            target_names=target_names,
            max_parents_per_target=max_parents_per_target,
            cv_folds=0,
            edge_effect_threshold=edge_effect_threshold,
        )
        for row in fit.edge_effect_rows():
            if row["parent"] == "self_history":
                continue
            key = (str(row["parent"]), str(row["target"]))
            effects[key] = effects.get(key, 0.0) + float(row["max_abs_effect"])
            if bool(row["included_by_effect_threshold"]):
                counts[key] = counts.get(key, 0) + 1
            else:
                counts.setdefault(key, 0)

    rows = []
    for key in sorted(effects, key=lambda item: (-counts.get(item, 0), item[0], item[1])):
        rows.append(
            {
                "parent": key[0],
                "target": key[1],
                "bootstrap_inclusion_probability": float(counts.get(key, 0) / int(iterations)),
                "included_bootstraps": int(counts.get(key, 0)),
                "bootstrap_iterations": int(iterations),
                "mean_max_abs_effect": float(effects[key] / int(iterations)),
                "effect_threshold": float(edge_effect_threshold),
            }
        )
    return rows
