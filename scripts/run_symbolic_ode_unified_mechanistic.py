#!/usr/bin/env python3
"""Unified mechanistic symbolic ODE candidates with connectivity ablations.

This script is deliberately not another per-subject coefficient model.  It fits
one global ODE at a time and chooses among biologically interpretable candidate
equations by held-out validation delta-spreading performance.

Candidate terms are assembled from the same standardised feature space used by
SymbolicODEModel:
    amyloid_saturation = amyloid_x_tau * (C - tau)
    fickian            = neighbour_tau - tau
    fickian_x_tau      = fickian * tau
    autonomous_tau     = tau^2 * tau_logistic
    tau_decay          = -tau
    thickness_clearance= -max(thickness_z, 0) * tau
    atrophy_growth     = max(-thickness_z, 0) * tau

All non-intercept coefficients are constrained non-negative.  Negative regional
rates can still arise from positive Fickian diffusion terms when a region has
more tau than its connected neighbours, or from globally shared damping terms.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.adni_features import build_closure_covariates  # noqa: E402
from spread_toolbox.forecasting import (  # noqa: E402
    ForecastDataset,
    MinMaxStateScaler,
    SubjectSplit,
    SubjectTrainValidationTestSplit,
    compute_aggregate_metrics,
    compute_pair_metrics,
    load_forecast_dataset,
    load_labeled_matrix,
    make_subject_split,
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.models.symbolic_ode import SymbolicODEModel  # noqa: E402


MODEL_NAME = "symbolic_ode_unified_mechanistic"


@dataclass
class UnifiedFit:
    terms: tuple[str, ...]
    coefficients: np.ndarray
    cap: float
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    train_r2: float
    train_mse: float


def default_config_path() -> Path:
    exp = PROJECT_ROOT / "experiments" / "group_average_enigma"
    hcp = exp / "config_hcp.yaml"
    local = exp / "config.yaml"
    return hcp if hcp.exists() else local if local.exists() else exp / "config.example.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})
    seed = int(config.get("experiment", {}).get("random_seed", 20260507))

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=seed,
    )
    selection_split = make_selection_split(
        dataset,
        split,
        validation_fraction=float(args.validation_fraction),
        random_seed=seed + 101,
    )

    _, adjacency = load_labeled_matrix(
        output_dir / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv")
    )
    scaler = MinMaxStateScaler.fit(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
    )
    bl_s = scaler.transform(dataset.baseline)
    ob_s = scaler.transform(dataset.observed)

    pair_cov, reg_cov, _ = build_closure_covariates(dataset, split, config, PROJECT_ROOT)
    amyloid = reg_cov.get("amyloid_suvr")
    thickness = reg_cov.get("cortical_thickness")

    model = SymbolicODEModel(adjacency, steps_per_year=12)

    candidates: list[tuple[str, ...]] = [
        ("amyloid_saturation",),
        ("amyloid_saturation", "autonomous_tau"),
        ("amyloid_saturation", "fickian"),
        ("amyloid_saturation", "fickian_x_tau"),
        ("amyloid_saturation", "fickian", "fickian_x_tau"),
        ("amyloid_saturation", "fickian_x_tau", "autonomous_tau"),
        ("amyloid_saturation", "fickian", "fickian_x_tau", "autonomous_tau"),
        ("amyloid_saturation", "fickian", "tau_decay"),
        ("amyloid_saturation", "fickian_x_tau", "tau_decay"),
        ("amyloid_saturation", "fickian", "fickian_x_tau", "tau_decay"),
        ("amyloid_saturation", "fickian", "fickian_x_tau", "autonomous_tau", "tau_decay"),
        ("amyloid_saturation", "fickian", "fickian_x_tau", "thickness_clearance"),
        ("amyloid_saturation", "fickian", "fickian_x_tau", "atrophy_growth", "thickness_clearance"),
        ("amyloid_saturation", "fickian", "fickian_x_tau", "autonomous_tau", "atrophy_growth", "thickness_clearance"),
    ]

    print("\n[1/3] Candidate selection on train/validation split")
    print(f"      train pairs={selection_split.train_indices.size}  "
          f"validation pairs={selection_split.validation_indices.size}  "
          f"test pairs={selection_split.test_indices.size}")

    selection_rows: list[dict[str, Any]] = []
    candidate_fits: dict[str, UnifiedFit] = {}
    for terms in candidates:
        name = "+".join(terms)
        fit = fit_unified_candidate(
            model,
            bl_s,
            ob_s,
            dataset.time_years,
            train_indices=selection_split.train_indices,
            terms=terms,
            amyloid=amyloid,
            thickness=thickness,
            random_seed=seed,
        )
        candidate_fits[name] = fit
        pred_s = predict_unified_candidate(
            model,
            bl_s,
            dataset.time_years,
            fit,
            amyloid=amyloid,
            thickness=thickness,
        )
        pred = scaler.inverse_transform(pred_s)
        pair_metrics = compute_pair_metrics(
            dataset.pairs, dataset.baseline, dataset.observed, pred,
            selection_split, name,
        )
        summary = summarize_split(pair_metrics, "validation")
        row = {
            "candidate": name,
            "terms": list(terms),
            "train_rate_r2": fit.train_r2,
            "train_rate_mse": fit.train_mse,
            **summary,
            "coefficients": fit.coefficients.tolist(),
            "cap": fit.cap,
        }
        selection_rows.append(row)
        print_candidate(row)

    best = choose_candidate(selection_rows)
    print("\n[2/3] Selected unified equation")
    print_candidate(best, prefix="      ")

    selected_terms = tuple(best["terms"])
    final_fit = fit_unified_candidate(
        model,
        bl_s,
        ob_s,
        dataset.time_years,
        train_indices=split.train_indices,
        terms=selected_terms,
        amyloid=amyloid,
        thickness=thickness,
        random_seed=seed,
    )
    final_pred_s = predict_unified_candidate(
        model,
        bl_s,
        dataset.time_years,
        final_fit,
        amyloid=amyloid,
        thickness=thickness,
    )
    final_pred = scaler.inverse_transform(final_pred_s)
    pair_metrics = compute_pair_metrics(
        dataset.pairs, dataset.baseline, dataset.observed, final_pred, split, MODEL_NAME
    )
    summary_rows = compute_aggregate_metrics(pair_metrics)

    print("\n[3/3] Final held-out test performance")
    for row in summary_rows:
        if row["split"] == "test" and row["metric"] in {
            "mae", "rmse", "subject_spearman", "delta_spearman", "top10_overlap"
        }:
            print(f"      {row['metric']:<18} mean={row['mean']:.4f} "
                  f"median={row['median']:.4f}")

    report = {
        "model": MODEL_NAME,
        "selection_metric": "validation mean delta_spearman",
        "selected_terms": list(selected_terms),
        "equation": equation_string(final_fit),
        "candidate_selection": selection_rows,
        "final_fit": {
            "terms": list(final_fit.terms),
            "coefficients": final_fit.coefficients.tolist(),
            "cap": final_fit.cap,
            "feature_names": final_fit.feature_names,
            "train_rate_r2": final_fit.train_r2,
            "train_rate_mse": final_fit.train_mse,
        },
    }

    if not args.no_write:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv_rows(output_dir / f"{MODEL_NAME}_pair_metrics.csv", pair_metrics)
        write_csv_rows(output_dir / f"{MODEL_NAME}_metrics_summary.csv", summary_rows)
        write_json(output_dir / f"{MODEL_NAME}_report.json", report)
        print(f"\nWrote {MODEL_NAME} outputs to {output_dir}")
    return 0


def make_selection_split(
    dataset: ForecastDataset,
    split: SubjectSplit,
    *,
    validation_fraction: float,
    random_seed: int,
) -> SubjectTrainValidationTestSplit:
    train_rids = sorted({dataset.pairs[int(i)]["RID"] for i in split.train_indices})
    rng = np.random.default_rng(random_seed)
    shuffled = np.asarray(train_rids, dtype=object)
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_rids = sorted(str(v) for v in shuffled[:validation_count])
    subtrain_rids = sorted(str(v) for v in shuffled[validation_count:])
    validation_set = set(validation_rids)
    subtrain_set = set(subtrain_rids)

    train_indices, validation_indices = [], []
    for i in split.train_indices:
        rid = dataset.pairs[int(i)]["RID"]
        if rid in validation_set:
            validation_indices.append(int(i))
        elif rid in subtrain_set:
            train_indices.append(int(i))
        else:
            raise ValueError(f"RID {rid} missing from selection split.")

    return SubjectTrainValidationTestSplit(
        train_indices=np.asarray(train_indices, dtype=int),
        validation_indices=np.asarray(validation_indices, dtype=int),
        test_indices=split.test_indices,
        train_rids=subtrain_rids,
        validation_rids=validation_rids,
        test_rids=split.test_rids,
    )


def fit_unified_candidate(
    model: SymbolicODEModel,
    baseline_scaled: np.ndarray,
    observed_scaled: np.ndarray,
    time_years: np.ndarray,
    *,
    train_indices: np.ndarray,
    terms: tuple[str, ...],
    amyloid: np.ndarray | None,
    thickness: np.ndarray | None,
    random_seed: int,
) -> UnifiedFit:
    bl = np.asarray(baseline_scaled, dtype=float)
    obs = np.asarray(observed_scaled, dtype=float)
    t = np.asarray(time_years, dtype=float)
    X_flat, feat_names, fm, fs = model.build_features(bl, amyloid=amyloid, thickness=thickness)
    y_flat = ((obs - bl) / np.maximum(t, 1.0e-6)[:, None]).reshape(-1)
    n_reg = bl.shape[1]

    flat_mask = np.zeros(y_flat.shape[0], dtype=bool)
    for i in np.asarray(train_indices, dtype=int):
        flat_mask[i * n_reg:(i + 1) * n_reg] = True
    finite = flat_mask & np.isfinite(y_flat) & np.all(np.isfinite(X_flat), axis=1)
    X_tr = X_flat[finite]
    y_tr = y_flat[finite]

    n_coeff = 1 + len(terms)
    x0 = np.zeros(n_coeff + 1, dtype=float)
    x0[0] = 0.0
    x0[1:n_coeff] = 0.001
    x0[-1] = 3.9
    bounds = [(-0.05, 0.05)] + [(0.0, 1.0)] * len(terms) + [(1.5, 8.0)]

    def objective(params: np.ndarray) -> float:
        coeffs = params[:n_coeff]
        cap = float(params[-1])
        state_tr = bl.reshape(-1)[finite]
        thick_tr = _fill_nan(np.asarray(thickness, dtype=float), *bl.shape).reshape(-1)[finite] \
            if thickness is not None else None
        design = design_matrix(X_tr, terms, cap, state_flat=state_tr, thickness_flat=thick_tr)
        pred = design @ coeffs
        residual = pred - y_tr
        return float(np.mean(residual**2) + 1.0e-6 * np.sum(coeffs[1:] ** 2))

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1.0e-12},
    )
    params = result.x if result.success else x0
    coeffs = params[:n_coeff]
    cap = float(params[-1])
    state_tr = bl.reshape(-1)[finite]
    thick_tr = _fill_nan(np.asarray(thickness, dtype=float), *bl.shape).reshape(-1)[finite] \
        if thickness is not None else None
    pred = design_matrix(X_tr, terms, cap, state_flat=state_tr, thickness_flat=thick_tr) @ coeffs
    ss_res = float(np.sum((y_tr - pred) ** 2))
    ss_tot = float(np.sum((y_tr - y_tr.mean()) ** 2))
    train_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    train_mse = float(np.mean((y_tr - pred) ** 2))
    return UnifiedFit(
        terms=terms,
        coefficients=coeffs,
        cap=cap,
        feature_names=feat_names,
        feature_mean=fm,
        feature_scale=fs,
        train_r2=train_r2,
        train_mse=train_mse,
    )


def predict_unified_candidate(
    model: SymbolicODEModel,
    baseline_scaled: np.ndarray,
    time_years: np.ndarray,
    fit: UnifiedFit,
    *,
    amyloid: np.ndarray | None,
    thickness: np.ndarray | None,
) -> np.ndarray:
    bl = np.asarray(baseline_scaled, dtype=float)
    t = np.asarray(time_years, dtype=float)
    n_pairs, n_reg = bl.shape
    amy = _fill_nan(np.asarray(amyloid, dtype=float), n_pairs, n_reg) if amyloid is not None else None
    thick = _fill_nan(np.asarray(thickness, dtype=float), n_pairs, n_reg) if thickness is not None else None
    states = np.clip(bl, 0.0, 1.0).copy()
    remaining = t.copy()
    step_dt = 1.0 / model.steps_per_year

    def rate_fn(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
        X, _, _, _ = model.build_features(
            S,
            amyloid=amy[idx] if amy is not None else None,
            thickness=thick[idx] if thick is not None else None,
            feature_mean=fit.feature_mean,
            feature_scale=fit.feature_scale,
        )
        thickness_flat = thick[idx].reshape(-1) if thick is not None else None
        rate = design_matrix(
            X,
            fit.terms,
            fit.cap,
            state_flat=S.reshape(-1),
            thickness_flat=thickness_flat,
        ) @ fit.coefficients
        return np.clip(np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).reshape(S.shape)

    while np.any(remaining > 0.0):
        active = remaining > 0.0
        idx = np.where(active)[0]
        dt = np.minimum(step_dt, remaining[active])[:, None]
        s = states[active]
        k1 = rate_fn(s, idx)
        k2 = rate_fn(np.clip(s + 0.5 * dt * k1, 0.0, 1.0), idx)
        k3 = rate_fn(np.clip(s + 0.5 * dt * k2, 0.0, 1.0), idx)
        k4 = rate_fn(np.clip(s + dt * k3, 0.0, 1.0), idx)
        states[active] = np.clip(s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, 1.0)
        remaining[active] -= dt[:, 0]
    return states


def design_matrix(
    X: np.ndarray,
    terms: tuple[str, ...],
    cap: float,
    *,
    state_flat: np.ndarray | None = None,
    thickness_flat: np.ndarray | None = None,
) -> np.ndarray:
    columns = [np.ones(X.shape[0], dtype=float)]
    tau = X[:, 0]
    tau_logistic = X[:, 1]
    fickian = X[:, 2]
    fickian_x_tau = X[:, 3]
    amyloid_x_tau = X[:, 4] if X.shape[1] > 4 else np.zeros_like(tau)
    state = np.asarray(state_flat, dtype=float) if state_flat is not None else tau
    thick = np.asarray(thickness_flat, dtype=float) if thickness_flat is not None else np.zeros_like(tau)
    for term in terms:
        if term == "amyloid_saturation":
            columns.append(amyloid_x_tau * (cap - tau))
        elif term == "fickian":
            columns.append(fickian)
        elif term == "fickian_x_tau":
            columns.append(fickian_x_tau)
        elif term == "autonomous_tau":
            columns.append((tau**2) * tau_logistic)
        elif term == "tau_decay":
            columns.append(-state)
        elif term == "thickness_clearance":
            columns.append(-np.maximum(thick, 0.0) * state)
        elif term == "atrophy_growth":
            columns.append(np.maximum(-thick, 0.0) * state)
        else:
            raise ValueError(f"Unknown term {term!r}.")
    return np.column_stack(columns)


def summarize_split(pair_metrics: list[dict[str, Any]], split_name: str) -> dict[str, float]:
    rows = [r for r in pair_metrics if r["split"] == split_name]
    out: dict[str, float] = {}
    for metric in ["mae", "rmse", "subject_spearman", "delta_spearman", "delta_pearson", "top10_overlap"]:
        vals = np.asarray([r[metric] for r in rows if r[metric] == r[metric]], dtype=float)
        out[f"validation_{metric}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
        out[f"validation_{metric}_median"] = float(np.median(vals)) if vals.size else float("nan")
    return out


def choose_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda r: (
            np.nan_to_num(r["validation_delta_spearman_mean"], nan=-999.0),
            np.nan_to_num(r["validation_top10_overlap_mean"], nan=-999.0),
            -np.nan_to_num(r["validation_mae_mean"], nan=999.0),
        ),
    )


def print_candidate(row: dict[str, Any], *, prefix: str = "") -> None:
    print(
        f"{prefix}{row['candidate']:<78} "
        f"val_delta_mean={row['validation_delta_spearman_mean']:+.4f} "
        f"val_top10={row['validation_top10_overlap_mean']:.4f} "
        f"val_mae={row['validation_mae_mean']:.4f} "
        f"rate_R2={row['train_rate_r2']:.4f}"
    )


def equation_string(fit: UnifiedFit) -> str:
    pieces = [f"{fit.coefficients[0]:+.6g}"]
    for coeff, term in zip(fit.coefficients[1:], fit.terms):
        pieces.append(f"{coeff:+.6g}*{term}")
    return "dS/dt = " + " ".join(pieces) + f"  (C={fit.cap:.3f})"


def _fill_nan(values: np.ndarray, n_pairs: int, n_reg: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n_pairs, n_reg):
        raise ValueError(f"Expected {(n_pairs, n_reg)}, got {arr.shape}.")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    raise SystemExit(main())
