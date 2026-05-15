#!/usr/bin/env python3
"""Symbolic ODE Discovery — backbone-free, parsimonious, universally generalisable.

Discovers a unified tau-spreading equation directly from the raw observed rate,
using ONLY universally-available features. No FKPP backbone assumed. No AHBA or
dataset-specific biomarkers required.

Features used (4 core + 2 optional):
  Core (always):  S0, S0*(1-S0), fickian_gradient, fickian_x_state
  Optional:       amyloid*S0, thickness*S0

Fickian gradient = Σ_j C_ji (S_j - S_i) — raw concentration-gradient diffusion
using HCP tractography weights, not the normalised graph Laplacian.

The discovered expression is a standalone ODE integrated dynamically via RK4.
It requires no retraining on new datasets — just re-evaluate the features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.adni_features import build_closure_covariates           # noqa: E402
from spread_toolbox.forecasting import (                                      # noqa: E402
    ForecastDataset, MinMaxStateScaler, SubjectSplit,
    compute_aggregate_metrics, compute_gaussian_likelihood_metrics,
    compute_pair_metrics, load_forecast_dataset, load_labeled_matrix,
    make_subject_split, read_csv_rows, write_csv_rows, write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path   # noqa: E402
from spread_toolbox.models.symbolic_ode import SymbolicODEModel            # noqa: E402

MODEL_NAME = "symbolic_ode"


def default_config_path() -> Path:
    exp = PROJECT_ROOT / "experiments" / "group_average_enigma"
    local = exp / "config.yaml"
    return local if local.exists() else exp / "config.example.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config     = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs    = config.get("outputs", {})
    modeling   = config.get("modeling", {})
    seed       = int(config.get("experiment", {}).get("random_seed", 20260507))

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    split   = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=seed,
    )

    # Adjacency (HCP) — required for Fickian gradient
    adj_path = output_dir / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv")
    if not adj_path.exists():
        raise FileNotFoundError(
            f"HCP adjacency not found: {adj_path}\n"
            "Run prepare_hcp_connectome.py first."
        )
    _, adjacency = load_labeled_matrix(adj_path)

    # Scaled arrays
    scaler = MinMaxStateScaler.fit(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
    )
    bl_s = scaler.transform(dataset.baseline)
    ob_s = scaler.transform(dataset.observed)

    # Biology covariates (optional — model degrades gracefully if absent)
    pair_cov, reg_cov, cov_report = build_closure_covariates(
        dataset, split, config, PROJECT_ROOT
    )
    amyloid   = reg_cov.get("amyloid_suvr")
    thickness = reg_cov.get("cortical_thickness")

    print(f"\nSymbolic ODE Discovery — HCP connectome, backbone-free")
    print(f"Train: {split.train_indices.size}  |  Test: {split.test_indices.size}")
    print(f"Biology: amyloid={'yes' if amyloid is not None else 'no'}  "
          f"thickness={'yes' if thickness is not None else 'no'}")

    # Build and fit
    model = SymbolicODEModel(
        adjacency,
        steps_per_year=int(modeling.get("symbolic_ode_steps_per_year", 12)),
    )
    fit = model.fit(
        bl_s, ob_s, dataset.time_years,
        train_indices=split.train_indices,
        amyloid=amyloid,
        thickness=thickness,
        pysr_niterations      =int(modeling.get("symbolic_ode_pysr_niterations",   300)),
        pysr_populations      =int(modeling.get("symbolic_ode_pysr_populations",    20)),
        pysr_population_size  =int(modeling.get("symbolic_ode_pysr_population_size",33)),
        pysr_maxsize          =int(modeling.get("symbolic_ode_pysr_maxsize",        15)),
        pysr_parsimony        =float(modeling.get("symbolic_ode_pysr_parsimony",  0.01)),
        pysr_binary_operators =list(modeling.get("symbolic_ode_pysr_binary_operators",
                                                  ["+", "-", "*", "/"])),
        pysr_unary_operators  =list(modeling.get("symbolic_ode_pysr_unary_operators",
                                                  ["square"])),
        pysr_batching         =bool(modeling.get("symbolic_ode_pysr_batching",    True)),
        pysr_batch_size       =int(modeling.get("symbolic_ode_pysr_batch_size",   2048)),
        pysr_timeout_seconds  =int(modeling.get("symbolic_ode_pysr_timeout_seconds",300)),
        max_train_rows        =int(modeling.get("symbolic_ode_max_train_rows",   40000)),
        random_seed=seed,
    )

    # Predict with dynamic RK4 integration of discovered ODE
    predicted_s = model.predict(
        bl_s, dataset.time_years, fit,
        amyloid=amyloid, thickness=thickness,
    )
    predicted = scaler.inverse_transform(predicted_s)

    pair_metrics    = compute_pair_metrics(
        dataset.pairs, dataset.baseline, dataset.observed, predicted, split, MODEL_NAME
    )
    metrics_summary = compute_aggregate_metrics(pair_metrics)
    likelihood_metrics = compute_gaussian_likelihood_metrics(
        dataset.observed, predicted, split, MODEL_NAME,
        n_parameters=fit.n_parameters,
        min_sigma=float(modeling.get("likelihood_min_sigma", 1e-6)),
    )

    # R² diagnostics
    r2_s0    = _scalar_r2(dataset.observed[split.test_indices],
                           dataset.baseline[split.test_indices])
    r2_final = _scalar_r2(dataset.observed[split.test_indices],
                           predicted[split.test_indices])

    report = {
        "model": MODEL_NAME,
        "description": (
            "Backbone-free symbolic ODE discovery. PySR fits the raw observed rate "
            "(S1-S0)/dt using only universally-available features: baseline tau, "
            "Fickian gradient (HCP connectivity), and optional amyloid/thickness. "
            "No FKPP backbone, no AHBA gene expression, no dataset-specific biomarkers."
        ),
        "discovered_equation":    fit.symbolic_expression,
        "features_used":          fit.feature_names,
        "n_parameters":           fit.n_parameters,
        "rate_train_r2":          fit.residual_train_r2,
        "rate_train_mse":         fit.residual_train_mse,
        "test_r2_gain_over_s0":   r2_final - r2_s0,
        "pareto_front":           fit.pareto_front,
        "covariates":             cov_report,
    }

    print("\n" + "=" * 65)
    print("DISCOVERED UNIFIED EQUATION:")
    print(f"  dS/dt = {fit.symbolic_expression}")
    print()
    print(f"Rate R² (how well equation fits raw dS/dt): {fit.residual_train_r2:.4f}")
    print(f"Expression complexity: {fit.n_parameters} nodes")
    print(f"Test R² gain over S0: {r2_final - r2_s0:+.4f}")
    print()
    _print_comparison(metrics_summary, output_dir, outputs)
    print()
    _print_pareto(fit.pareto_front[:12])

    if not args.no_write:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv_rows(
            output_dir / outputs.get("symbolic_ode_pair_metrics",
                                      "symbolic_ode_pair_metrics.csv"),
            pair_metrics,
        )
        write_csv_rows(
            output_dir / outputs.get("symbolic_ode_metrics_summary",
                                      "symbolic_ode_metrics_summary.csv"),
            metrics_summary,
        )
        write_csv_rows(
            output_dir / outputs.get("symbolic_ode_likelihood_metrics",
                                      "symbolic_ode_likelihood_metrics.csv"),
            likelihood_metrics,
        )
        write_json(
            output_dir / outputs.get("symbolic_ode_report",
                                      "symbolic_ode_report.json"),
            report,
        )
        print("Wrote symbolic ODE outputs.")
    return 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scalar_r2(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred)
    y, yh = obs[mask], pred[mask]
    tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - np.sum((y - yh) ** 2) / tot) if tot > 0 else float("nan")


def _print_comparison(
    metrics_summary: list[dict],
    output_dir: Path,
    outputs: dict,
) -> None:
    comparison_files = {
        "FKPP+IR":        "individualized_residual_metrics_summary.csv",
        "NDM+IR":         "ndm_individualized_residual_metrics_summary.csv",
        "Bio-FKPP+conn":  "bio_fkpp_metrics_summary.csv",
        "S0 persistence": "persistence_metrics_summary.csv",
    }
    all_rows = list(metrics_summary)
    for name, default in comparison_files.items():
        key  = {v: k for k, v in outputs.items()}.get(default, default)
        path = output_dir / outputs.get(key, default)
        if path.exists():
            for r in read_csv_rows(path):
                r["_display_model"] = name
                all_rows.append(r)

    by_model: dict[str, dict] = {}
    for r in all_rows:
        m = r.get("_display_model", r.get("model", "?"))
        if r.get("split") == "test":
            by_model.setdefault(m, {})[r.get("metric", "")] = r

    def _med(m: dict, k: str) -> float:
        v = m.get(k, {})
        return float(v.get("median", float("nan"))) if isinstance(v, dict) else float("nan")

    print(f"{'Model':<35} {'subj_ρ':>8} {'delta_ρ':>9} {'MAE':>8}")
    print("-" * 62)
    order = [MODEL_NAME, "FKPP+IR", "NDM+IR", "Bio-FKPP+conn", "S0 persistence"]
    for m in order:
        if m not in by_model:
            continue
        flag = "  ← this model" if m == MODEL_NAME else ""
        print(f"  {m:<33} "
              f"{_med(by_model[m],'subject_spearman'):>8.4f} "
              f"{_med(by_model[m],'delta_spearman'):>9.4f} "
              f"{_med(by_model[m],'mae'):>8.4f}{flag}")


def _print_pareto(rows: list[dict]) -> None:
    if not rows:
        return
    print("PySR Pareto front — complexity vs loss on raw rate:")
    print(f"  {'complexity':>10}  {'loss':>12}  equation")
    print("  " + "-" * 62)
    for row in sorted(rows, key=lambda r: int(r.get("complexity", 0))):
        c  = int(row.get("complexity", 0))
        lo = float(row.get("loss", float("nan")))
        eq = str(row.get("equation", ""))[:58]
        print(f"  {c:>10}  {lo:>12.6f}  {eq}")


if __name__ == "__main__":
    raise SystemExit(main())
