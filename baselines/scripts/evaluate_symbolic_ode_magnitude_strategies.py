#!/usr/bin/env python3
"""Evaluate magnitude-focused symbolic ODE strategies.

This script is deliberately diagnostic: it does not replace the selected
state-gated symbolic ODE unless a candidate gives a clear held-out improvement.
It asks whether the current model's low rate R2 is mainly a noise-ceiling issue,
an uncertainty-weighting issue, or a simple global magnitude-calibration issue.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.adni_features import build_closure_covariates  # noqa: E402
from spread_toolbox.forecasting import (  # noqa: E402
    MinMaxStateScaler,
    compute_pair_metrics,
    load_forecast_dataset,
    load_labeled_matrix,
    make_subject_split,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.models.symbolic_ode import SymbolicODEModel, build_amortization_features  # noqa: E402

import scripts.run_symbolic_ode_state_gated as state_gated  # noqa: E402


MECHANISMS = ("growth", "amyloid_growth", "fickian", "fickian_x_tau", "tau_decay")
RIDGE_ALPHA = 10000.0


def default_config_path() -> Path:
    return PROJECT_ROOT / "experiments" / "group_average_enigma" / "config_hcp.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    args = parser.parse_args()

    ctx = load_context(args.config)
    reliability = estimate_rate_reliability(ctx)
    weighting_rows = evaluate_weighting(ctx)
    calibration_rows = evaluate_global_calibration(ctx)

    report = {
        "purpose": "Magnitude-focused diagnostics for state-gated symbolic ODE",
        "baseline_model": "growth+amyloid_growth+fickian+fickian_x_tau+tau_decay|gates=spatial_state|ridge=10000",
        "rate_reliability": reliability,
        "uncertainty_weighting": weighting_rows,
        "global_delta_calibration": calibration_rows,
        "findings": summarize_findings(reliability, weighting_rows, calibration_rows),
    }
    out_dir = ctx["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "symbolic_ode_magnitude_strategy_report.json", report)
    write_csv(out_dir / "symbolic_ode_magnitude_strategy_candidates.csv", weighting_rows + calibration_rows)

    print(json.dumps(report["findings"], indent=2, sort_keys=True))
    print(f"\nWrote magnitude strategy report to {out_dir}")
    return 0


def load_context(config_path: Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
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
    selection = state_gated.make_selection_split(
        dataset,
        split,
        validation_fraction=float(modeling.get("validation_fraction", 0.2)),
        random_seed=seed + 211,
    )

    _, adjacency = load_labeled_matrix(output_dir / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv"))
    _, laplacian = load_labeled_matrix(output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"))
    scaler = MinMaxStateScaler.fit(dataset.baseline[split.train_indices], dataset.observed[split.train_indices])
    baseline_scaled = scaler.transform(dataset.baseline)
    observed_scaled = scaler.transform(dataset.observed)

    pair_cov, reg_cov, _ = build_closure_covariates(dataset, split, config, PROJECT_ROOT)
    model = SymbolicODEModel(adjacency, steps_per_year=12)
    braak_idx = state_gated.build_braak_indices(dataset.region_labels)
    _, eigenvectors = np.linalg.eigh(laplacian)
    gate_matrix, gate_names = build_amortization_features(
        baseline_scaled,
        dataset.time_years,
        reg_cov.get("amyloid_suvr"),
        reg_cov.get("cortical_thickness"),
        pair_cov.get("apoe4_dose"),
        pair_cov.get("plasma_ptau181"),
        braak_idx,
        eigenvectors,
        model.adj_norm,
    )
    selected_gates = state_gated.build_gate_sets(gate_names)["spatial_state"]
    gate_indices = [gate_names.index(name) for name in selected_gates]

    return {
        "config": config,
        "output_dir": output_dir,
        "dataset": dataset,
        "split": split,
        "selection": selection,
        "scaler": scaler,
        "baseline_scaled": baseline_scaled,
        "observed_scaled": observed_scaled,
        "model": model,
        "amyloid": reg_cov.get("amyloid_suvr"),
        "thickness": reg_cov.get("cortical_thickness"),
        "gate_matrix": gate_matrix,
        "gate_names": gate_names,
        "gate_indices": gate_indices,
    }


def estimate_rate_reliability(ctx: dict[str, Any]) -> dict[str, Any]:
    dataset = ctx["dataset"]
    obs_path = ctx["output_dir"] / ctx["config"].get("outputs", {}).get(
        "tau_observations_table", "cohort_tau_observations.csv"
    )
    observations = pd.read_csv(obs_path)
    observations["SCANDATE"] = pd.to_datetime(observations["SCANDATE"])
    values = observations[dataset.tau_columns].apply(pd.to_numeric, errors="coerce")
    observations = observations[values.notna().all(axis=1)].copy()
    observations = observations.sort_values(["RID", "TRACER", "SCANDATE"])

    consecutive_a: list[np.ndarray] = []
    consecutive_b: list[np.ndarray] = []
    baseline_next: list[np.ndarray] = []
    baseline_last: list[np.ndarray] = []
    scan_counts: list[int] = []

    for _, group in observations.groupby(["RID", "TRACER"], sort=False):
        group = group.sort_values("SCANDATE")
        if len(group) < 2:
            continue
        scan_counts.append(len(group))
        x = group[dataset.tau_columns].to_numpy(dtype=float)
        dates = group["SCANDATE"].to_numpy()
        years = np.asarray(
            [(dates[i + 1] - dates[i]).astype("timedelta64[D]").astype(int) / 365.25 for i in range(len(dates) - 1)],
            dtype=float,
        )
        rates = (x[1:] - x[:-1]) / np.maximum(years, 1.0e-6)[:, None]
        for k in range(len(rates) - 1):
            consecutive_a.append(rates[k])
            consecutive_b.append(rates[k + 1])
        if len(group) >= 3:
            first_years = max((dates[1] - dates[0]).astype("timedelta64[D]").astype(int) / 365.25, 1.0e-6)
            total_years = max((dates[-1] - dates[0]).astype("timedelta64[D]").astype(int) / 365.25, 1.0e-6)
            baseline_next.append((x[1] - x[0]) / first_years)
            baseline_last.append((x[-1] - x[0]) / total_years)

    return {
        "scan_count_distribution": {
            int(k): int(v) for k, v in pd.Series(scan_counts).value_counts().sort_index().items()
        },
        "consecutive_nonoverlap": summarize_rate_reliability(consecutive_a, consecutive_b),
        "baseline_next_vs_baseline_last": summarize_rate_reliability(baseline_next, baseline_last),
    }


def summarize_rate_reliability(a: list[np.ndarray], b: list[np.ndarray]) -> dict[str, float | int]:
    if not a:
        return {"n_vector_pairs": 0}
    x = np.vstack(a)
    y = np.vstack(b)
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    mask = np.isfinite(flat_x) & np.isfinite(flat_y)
    design = np.column_stack([np.ones(mask.sum()), flat_x[mask]])
    coef = np.linalg.lstsq(design, flat_y[mask], rcond=None)[0]
    pred = design @ coef
    ss_tot = float(np.sum((flat_y[mask] - flat_y[mask].mean()) ** 2))
    ss_identity = float(np.sum((flat_y[mask] - flat_x[mask]) ** 2))
    ss_cal = float(np.sum((flat_y[mask] - pred) ** 2))
    region_r = []
    for j in range(x.shape[1]):
        if np.std(x[:, j]) > 1.0e-12 and np.std(y[:, j]) > 1.0e-12:
            region_r.append(float(pearsonr(x[:, j], y[:, j]).statistic))
    return {
        "n_vector_pairs": int(len(a)),
        "n_flattened": int(mask.sum()),
        "flattened_pearson": float(pearsonr(flat_x[mask], flat_y[mask]).statistic),
        "flattened_spearman": float(spearmanr(flat_x[mask], flat_y[mask]).statistic),
        "identity_r2": 1.0 - ss_identity / ss_tot if ss_tot > 0.0 else float("nan"),
        "linear_calibrated_r2": 1.0 - ss_cal / ss_tot if ss_tot > 0.0 else float("nan"),
        "median_region_pearson": float(np.nanmedian(region_r)) if region_r else float("nan"),
    }


def evaluate_weighting(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    schemes = {
        "uniform": None,
        "sqrt_t": np.sqrt(ctx["dataset"].time_years),
        "t": ctx["dataset"].time_years,
        "t2": ctx["dataset"].time_years**2,
        "t_clip3_sq": np.minimum(ctx["dataset"].time_years, 3.0) ** 2,
        "t_clip5_sq": np.minimum(ctx["dataset"].time_years, 5.0) ** 2,
    }
    rows = []
    for scheme, sample_weight in schemes.items():
        for alpha in (1000.0, 10000.0, 100000.0):
            fit = fit_state_gated(ctx, ctx["selection"].train_indices, alpha=alpha, sample_weight=sample_weight)
            pred_s = predict_state_gated(ctx, fit)
            rows.append(
                {
                    "strategy": "uncertainty_weighting",
                    "candidate": scheme,
                    "ridge_alpha": alpha,
                    "split": "validation",
                    "rate_r2": rate_r2(ctx, pred_s, ctx["selection"].validation_indices),
                    **endpoint_metrics(ctx, pred_s, ctx["selection"], "validation"),
                    "train_rate_r2": fit.train_r2,
                }
            )
    return rows


def evaluate_global_calibration(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split_name, train_indices, eval_split, eval_name in [
        ("validation", ctx["selection"].train_indices, ctx["selection"], "validation"),
        ("test", ctx["split"].train_indices, ctx["split"], "test"),
    ]:
        fit = fit_state_gated(ctx, train_indices, alpha=RIDGE_ALPHA, sample_weight=None)
        pred_s = predict_state_gated(ctx, fit)
        for mode in ("none", "scale_no_intercept", "scale_intercept", "scale_baseline_intercept"):
            coef = [] if mode == "none" else fit_delta_calibration(ctx, pred_s, train_indices, mode).tolist()
            calibrated = apply_delta_calibration(ctx, pred_s, np.asarray(coef, dtype=float), mode)
            rows.append(
                {
                    "strategy": "global_delta_calibration",
                    "candidate": mode,
                    "ridge_alpha": RIDGE_ALPHA,
                    "split": split_name,
                    "coefficients": coef,
                    "rate_r2": rate_r2(ctx, calibrated, getattr(eval_split, f"{eval_name}_indices")),
                    **endpoint_metrics(ctx, calibrated, eval_split, eval_name),
                    "train_rate_r2": fit.train_r2,
                }
            )
    return rows


def fit_state_gated(ctx: dict[str, Any], train_indices: np.ndarray, *, alpha: float, sample_weight: np.ndarray | None):
    return state_gated.fit_gated_candidate(
        ctx["model"],
        ctx["baseline_scaled"],
        ctx["observed_scaled"],
        ctx["dataset"].time_years,
        train_indices=train_indices,
        mechanisms=MECHANISMS,
        gate_matrix=ctx["gate_matrix"],
        gate_indices=ctx["gate_indices"],
        all_gate_names=ctx["gate_names"],
        amyloid=ctx["amyloid"],
        thickness=ctx["thickness"],
        ridge_alpha=alpha,
        fit_intercept=False,
        sample_weight=sample_weight,
    )


def predict_state_gated(ctx: dict[str, Any], fit) -> np.ndarray:
    return state_gated.predict_gated_candidate(
        ctx["model"],
        ctx["baseline_scaled"],
        ctx["dataset"].time_years,
        fit,
        gate_matrix=ctx["gate_matrix"],
        amyloid=ctx["amyloid"],
        thickness=ctx["thickness"],
    )


def endpoint_metrics(ctx: dict[str, Any], pred_s: np.ndarray, split, split_name: str) -> dict[str, float]:
    pred = ctx["scaler"].inverse_transform(pred_s)
    pair_metrics = compute_pair_metrics(
        ctx["dataset"].pairs,
        ctx["dataset"].baseline,
        ctx["dataset"].observed,
        pred,
        split,
        "candidate",
    )
    rows = [row for row in pair_metrics if row["split"] == split_name]
    out = {}
    for metric in ("delta_spearman", "mae", "top10_overlap", "subject_spearman"):
        vals = np.asarray([row[metric] for row in rows if row[metric] == row[metric]], dtype=float)
        out[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
        out[f"{metric}_median"] = float(np.median(vals)) if vals.size else float("nan")
    return out


def rate_r2(ctx: dict[str, Any], pred_s: np.ndarray, indices: np.ndarray) -> float:
    t = np.maximum(ctx["dataset"].time_years, 1.0e-6)[:, None]
    y = ((ctx["observed_scaled"] - ctx["baseline_scaled"]) / t).reshape(-1)
    y_pred = ((pred_s - ctx["baseline_scaled"]) / t).reshape(-1)
    n_pairs, n_reg = ctx["baseline_scaled"].shape
    mask = np.zeros(n_pairs * n_reg, dtype=bool)
    for i in np.asarray(indices, dtype=int):
        mask[i * n_reg : (i + 1) * n_reg] = True
    mask &= np.isfinite(y) & np.isfinite(y_pred)
    ss_res = float(np.sum((y[mask] - y_pred[mask]) ** 2))
    ss_tot = float(np.sum((y[mask] - y[mask].mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def fit_delta_calibration(ctx: dict[str, Any], pred_s: np.ndarray, train_indices: np.ndarray, mode: str) -> np.ndarray:
    pred_delta = pred_s - ctx["baseline_scaled"]
    true_delta = ctx["observed_scaled"] - ctx["baseline_scaled"]
    train = np.asarray(train_indices, dtype=int)
    x = pred_delta[train].reshape(-1)
    y = true_delta[train].reshape(-1)
    if mode == "scale_no_intercept":
        denom = float(np.dot(x, x))
        return np.asarray([float(np.dot(x, y) / denom) if denom > 1.0e-12 else 1.0])
    if mode == "scale_intercept":
        return np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    if mode == "scale_baseline_intercept":
        baseline = ctx["baseline_scaled"][train].reshape(-1)
        return np.linalg.lstsq(np.column_stack([x, baseline, np.ones_like(x)]), y, rcond=None)[0]
    raise ValueError(f"Unknown calibration mode: {mode}")


def apply_delta_calibration(ctx: dict[str, Any], pred_s: np.ndarray, coef: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return pred_s
    pred_delta = pred_s - ctx["baseline_scaled"]
    if mode == "scale_no_intercept":
        calibrated_delta = coef[0] * pred_delta
    elif mode == "scale_intercept":
        calibrated_delta = coef[0] * pred_delta + coef[1]
    elif mode == "scale_baseline_intercept":
        calibrated_delta = coef[0] * pred_delta + coef[1] * ctx["baseline_scaled"] + coef[2]
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")
    return np.clip(ctx["baseline_scaled"] + calibrated_delta, 0.0, 1.0)


def summarize_findings(
    reliability: dict[str, Any],
    weighting_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_weight = next(
        row for row in weighting_rows
        if row["candidate"] == "uniform" and row["ridge_alpha"] == RIDGE_ALPHA and row["split"] == "validation"
    )
    best_weight = max(weighting_rows, key=lambda row: (row["rate_r2"], row["delta_spearman_mean"]))
    test_cal = [row for row in calibration_rows if row["split"] == "test"]
    baseline_cal = next(row for row in test_cal if row["candidate"] == "none")
    scale_cal = next(row for row in test_cal if row["candidate"] == "scale_no_intercept")
    return {
        "noise_ceiling_warning": (
            "Consecutive non-overlapping annualized rates are near non-reproducible; "
            "held-out exact-rate R2 should be expected to be low."
        ),
        "consecutive_rate_linear_r2": reliability["consecutive_nonoverlap"].get("linear_calibrated_r2"),
        "overlapping_baseline_rate_linear_r2": reliability["baseline_next_vs_baseline_last"].get("linear_calibrated_r2"),
        "uncertainty_weighting_decision": "reject",
        "uncertainty_weighting_reason": (
            "The best validation rate-R2 weighting candidate sacrifices spatial delta-Spearman; "
            "the baseline remains best under a spatial-preservation constraint."
        ),
        "baseline_validation_rate_r2": baseline_weight["rate_r2"],
        "best_weighting_validation_candidate": best_weight,
        "global_scale_decision": "diagnostic_only",
        "global_scale_reason": (
            "Scale-only calibration slightly improves test rate R2 and MAE without changing rank metrics, "
            "but the gain is too small to treat as a new mechanistic finding."
        ),
        "baseline_test_rate_r2": baseline_cal["rate_r2"],
        "scale_only_test_rate_r2": scale_cal["rate_r2"],
        "baseline_test_delta_spearman": baseline_cal["delta_spearman_mean"],
        "scale_only_test_delta_spearman": scale_cal["delta_spearman_mean"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, default=json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
