#!/usr/bin/env python3
"""Fit AS-FKPP: Amortized Subject-Specific FKPP.

Personalises FKPP parameters (rho_i, alpha_i) per subject using:
  - Baseline tau spatial fingerprint (Laplacian eigenmode projections)
  - Tau burden scalars (mean, max, CV, hub-weighted)
  - Biological covariates (amyloid SUVR, cortical thickness, APOE4, p-tau181)

Three stages:
  Stage 1: Global FKPP backbone -> freeze (rho, alpha)
  Stage 2a: Per-pair optimization of (Delta_rho, Delta_alpha)
  Stage 2b: Ridge amortisation onto feature matrix
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

from spread_toolbox.adni_features import build_closure_covariates  # noqa: E402
from spread_toolbox.forecasting import (  # noqa: E402
    ForecastDataset,
    MinMaxStateScaler,
    SubjectSplit,
    compute_aggregate_metrics,
    compute_gaussian_likelihood_metrics,
    compute_pair_metrics,
    load_forecast_dataset,
    load_labeled_matrix,
    make_subject_split,
    read_csv_rows,
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.models.as_fkpp import ASFKPPModel, amortisation_term_rows  # noqa: E402


MODEL_NAME = "as_fkpp"


def default_config_path() -> Path:
    experiment_dir = PROJECT_ROOT / "experiments" / "group_average_enigma"
    local_config = experiment_dir / "config.yaml"
    if local_config.exists():
        return local_config
    return experiment_dir / "config.example.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    laplacian = load_required_matrix(
        output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"), dataset
    )
    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=int(config.get("experiment", {}).get("random_seed", 20260507)),
    )

    result = run_as_fkpp_model(dataset, split, laplacian, modeling, config)
    pair_metrics = result["pair_metrics"]
    metrics_summary = compute_aggregate_metrics(pair_metrics)
    likelihood_metrics = compute_gaussian_likelihood_metrics(
        dataset.observed,
        result["predicted"],
        split,
        MODEL_NAME,
        n_parameters=int(result["fit_report"]["n_likelihood_parameters"]),
        min_sigma=float(modeling.get("likelihood_min_sigma", 1.0e-6)),
    )

    baseline_summary = load_optional_rows(
        output_dir / outputs.get("baseline_comparison_metrics_summary", "baseline_comparison_metrics_summary.csv")
    )
    ir_summary = load_optional_rows(
        output_dir / outputs.get("individualized_residual_metrics_summary", "individualized_residual_metrics_summary.csv")
    )
    comparison_rows = test_comparison_rows(baseline_summary + ir_summary + metrics_summary)

    report = {
        "model": MODEL_NAME,
        "description": (
            "Amortized Subject-Specific FKPP: personalises (rho_i, alpha_i) per subject "
            "using baseline tau spatial fingerprint (Laplacian eigenmode projections) "
            "and biological covariates."
        ),
        "equation": (
            "S1_i = FKPP(S0_i; rho+Delta_rho_i, alpha+Delta_alpha_i); "
            "Delta_rho_i, Delta_alpha_i ~ ridge(eigenmode_projections, biology)"
        ),
        "split": split_summary(split),
        "fit_report": result["fit_report"],
        "r2_report": result["r2_report"],
        "test_comparison": comparison_rows,
    }

    print(json.dumps(report["split"], indent=2, sort_keys=True))
    print("\nAS-FKPP fit report:")
    print(json.dumps(
        {k: v for k, v in report["fit_report"].items() if k not in ("covariates", "feature_names")},
        indent=2,
    ))
    print("\nR2 report:")
    print(json.dumps(report["r2_report"], indent=2))
    print("\nAS-FKPP test comparison:")
    print_comparison_table(comparison_rows)
    print("\nTop amortisation terms (rho + alpha |coef| combined):")
    print_term_rows(result["term_rows"][:20])

    if not args.no_write:
        write_csv_rows(
            output_dir / outputs.get("as_fkpp_pair_metrics", "as_fkpp_pair_metrics.csv"),
            pair_metrics,
        )
        write_csv_rows(
            output_dir / outputs.get("as_fkpp_metrics_summary", "as_fkpp_metrics_summary.csv"),
            metrics_summary,
        )
        write_csv_rows(
            output_dir / outputs.get("as_fkpp_likelihood_metrics", "as_fkpp_likelihood_metrics.csv"),
            likelihood_metrics,
        )
        write_csv_rows(
            output_dir / outputs.get("as_fkpp_terms", "as_fkpp_terms.csv"),
            result["term_rows"],
        )
        write_json(
            output_dir / outputs.get("as_fkpp_report", "as_fkpp_report.json"),
            {**report, "likelihood_metrics": likelihood_metrics},
        )
        print("\nWrote AS-FKPP outputs.")

    return 0


def run_as_fkpp_model(
    dataset: ForecastDataset,
    split: SubjectSplit,
    laplacian: np.ndarray,
    modeling: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    scaler = MinMaxStateScaler.fit(dataset.baseline[split.train_indices], dataset.observed[split.train_indices])
    baseline_scaled = scaler.transform(dataset.baseline)
    observed_scaled = scaler.transform(dataset.observed)

    parameter_bounds = modeling.get("parameter_bounds", {})
    rho_bounds = tuple(float(v) for v in parameter_bounds.get("rho", [0.0, 10.0]))
    alpha_bounds = tuple(float(v) for v in parameter_bounds.get("alpha", [0.0, 10.0]))

    model = ASFKPPModel(
        laplacian,
        steps_per_year=int(modeling.get("as_fkpp_steps_per_year", 12)),
        laplacian_normalization=str(modeling.get("as_fkpp_laplacian_normalization", "spectral")),
        n_eigenmodes=int(modeling.get("as_fkpp_n_eigenmodes", 10)),
    )

    pair_covariates, regional_covariates, covariate_report = build_closure_covariates(
        dataset, split, config, PROJECT_ROOT
    )
    amyloid = regional_covariates.get("amyloid_suvr")
    thickness = regional_covariates.get("cortical_thickness")
    apoe4_dose = pair_covariates.get("apoe4_dose")
    plasma_ptau181 = pair_covariates.get("plasma_ptau181")

    ridge_alphas = tuple(
        float(v) for v in modeling.get("as_fkpp_ridge_cv_alphas", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    )

    print(f"AS-FKPP Stage 2a: fitting {len(split.train_indices)} per-pair parameter offsets...")
    fit = model.fit(
        baseline_scaled, observed_scaled, dataset.time_years,
        amyloid=amyloid,
        thickness=thickness,
        apoe4_dose=apoe4_dose,
        plasma_ptau181=plasma_ptau181,
        train_indices=split.train_indices,
        rho_bounds=(rho_bounds[0], rho_bounds[1]),
        alpha_bounds=(alpha_bounds[0], alpha_bounds[1]),
        per_pair_delta_scale=float(modeling.get("as_fkpp_per_pair_delta_scale", 5.0)),
        per_pair_maxiter=int(modeling.get("as_fkpp_per_pair_maxiter", 40)),
        ridge_cv_alphas=ridge_alphas,
        backbone_maxiter=int(modeling.get("as_fkpp_backbone_maxiter", 80)),
    )
    print(
        f"  backbone rho={fit.rho:.4f}, alpha={fit.alpha:.4f} | "
        f"stage1 MSE={fit.stage1_train_mse:.6f} | "
        f"stage2a MSE={fit.stage2a_train_mse:.6f} | "
        f"stage2b MSE={fit.stage2b_train_mse:.6f}"
    )
    print(
        f"  amortisation R2: rho={fit.amortisation_rho_r2:.4f}  "
        f"alpha={fit.amortisation_alpha_r2:.4f} | "
        f"per-pair success: {fit.per_pair_success_frac:.1%}"
    )

    print("AS-FKPP: generating predictions...")
    predicted_scaled = model.predict(
        baseline_scaled, dataset.time_years, fit,
        amyloid=amyloid,
        thickness=thickness,
        apoe4_dose=apoe4_dose,
        plasma_ptau181=plasma_ptau181,
    )
    predicted = scaler.inverse_transform(predicted_scaled)

    # Build backbone predictions for R2 comparison
    from spread_toolbox.models.fkpp import GraphFKPPModel
    backbone_model = GraphFKPPModel(
        laplacian,
        steps_per_year=int(modeling.get("as_fkpp_steps_per_year", 12)),
        laplacian_normalization=str(modeling.get("as_fkpp_laplacian_normalization", "spectral")),
    )
    backbone_scaled = backbone_model.predict(
        baseline_scaled, dataset.time_years, rho=fit.rho, alpha=fit.alpha
    )
    backbone_predicted = scaler.inverse_transform(backbone_scaled)

    pair_metrics = compute_pair_metrics(
        dataset.pairs, dataset.baseline, dataset.observed, predicted, split, MODEL_NAME
    )

    # Delta offsets summary
    dr_train = fit.delta_rho_train
    da_train = fit.delta_alpha_train

    # n_likelihood_parameters: backbone (rho, alpha) + n_features per offset model × 2 + sigma
    n_likelihood_parameters = 3 + 2 * fit.n_features

    term_rows = amortisation_term_rows(fit)

    return {
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "term_rows": term_rows,
        "r2_report": _build_r2_report(dataset, split, backbone_predicted, predicted),
        "fit_report": {
            "model": MODEL_NAME,
            "backbone": "global_fkpp",
            "backbone_rho": fit.rho,
            "backbone_alpha": fit.alpha,
            "n_eigenmodes": fit.n_eigenmodes,
            "n_features": fit.n_features,
            "feature_names": fit.feature_names,
            "stage1_train_mse_scaled": fit.stage1_train_mse,
            "stage2a_train_mse_scaled": fit.stage2a_train_mse,
            "stage2b_train_mse_scaled": fit.stage2b_train_mse,
            "mse_reduction_stage2a_pct": 100.0 * (1.0 - fit.stage2a_train_mse / fit.stage1_train_mse) if fit.stage1_train_mse > 0 else 0.0,
            "mse_reduction_stage2b_pct": 100.0 * (1.0 - fit.stage2b_train_mse / fit.stage1_train_mse) if fit.stage1_train_mse > 0 else 0.0,
            "amortisation_rho_r2": fit.amortisation_rho_r2,
            "amortisation_alpha_r2": fit.amortisation_alpha_r2,
            "ridge_alpha_rho": fit.ridge_alpha_rho,
            "ridge_alpha_alpha": fit.ridge_alpha_alpha,
            "delta_rho_train_mean": float(dr_train.mean()),
            "delta_rho_train_std": float(dr_train.std()),
            "delta_alpha_train_mean": float(da_train.mean()),
            "delta_alpha_train_std": float(da_train.std()),
            "per_pair_success_frac": fit.per_pair_success_frac,
            "laplacian_normalization": fit.backbone_laplacian_normalization,
            "laplacian_scale": fit.backbone_laplacian_scale,
            "covariates": covariate_report,
            "n_likelihood_parameters": n_likelihood_parameters,
        },
    }


def _build_r2_report(
    dataset: ForecastDataset,
    split: SubjectSplit,
    backbone_predicted: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    train, test = split.train_indices, split.test_indices
    return {
        "train_scalar_r2_backbone": _scalar_r2(dataset.observed[train], backbone_predicted[train]),
        "train_scalar_r2_as_fkpp": _scalar_r2(dataset.observed[train], predicted[train]),
        "test_scalar_r2_backbone": _scalar_r2(dataset.observed[test], backbone_predicted[test]),
        "test_scalar_r2_as_fkpp": _scalar_r2(dataset.observed[test], predicted[test]),
        "test_r2_gain": (
            _scalar_r2(dataset.observed[test], predicted[test])
            - _scalar_r2(dataset.observed[test], backbone_predicted[test])
        ),
    }


def _scalar_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    y, yhat = observed[mask], predicted[mask]
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 0.0 else float("nan")


def load_required_matrix(path: Path, dataset: ForecastDataset) -> np.ndarray:
    labels, matrix = load_labeled_matrix(path)
    if labels != dataset.region_labels:
        raise ValueError(f"Matrix labels do not match dataset region labels: {path}")
    return matrix


def load_optional_rows(path: Path) -> list[dict[str, Any]]:
    return read_csv_rows(path) if path.exists() else []


def test_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {"mae", "rmse", "subject_spearman", "delta_spearman", "top5_overlap", "top10_overlap"}
    return [row for row in rows if row.get("split") == "test" and row.get("metric") in wanted]


def split_summary(split: SubjectSplit) -> dict[str, int]:
    return {
        "train_pairs": int(split.train_indices.size),
        "test_pairs": int(split.test_indices.size),
        "train_subjects": len(split.train_rids),
        "test_subjects": len(split.test_rids),
    }


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), {})[str(row["metric"])] = row
    print(f"{'model':<45} {'MAE':>8} {'RMSE':>8} {'subj rho':>10} {'delta rho':>10} {'top5':>8} {'top10':>8}")
    for model_name, metrics in sorted(by_model.items()):
        if metrics:
            print(
                f"{model_name:<45} "
                f"{_med(metrics, 'mae'):>8.4f} "
                f"{_med(metrics, 'rmse'):>8.4f} "
                f"{_med(metrics, 'subject_spearman'):>10.4f} "
                f"{_med(metrics, 'delta_spearman'):>10.4f} "
                f"{_med(metrics, 'top5_overlap'):>8.4f} "
                f"{_med(metrics, 'top10_overlap'):>8.4f}"
            )


def print_term_rows(rows: list[dict[str, Any]]) -> None:
    print(f"{'term':<40} {'rho_coef':>12} {'alpha_coef':>12}")
    for row in rows:
        print(
            f"{str(row['term'])[:40]:<40} "
            f"{float(row['rho_coefficient']):>12.5g} "
            f"{float(row['alpha_coefficient']):>12.5g}"
        )


def _med(metrics: dict[str, dict[str, Any]], name: str) -> float:
    return float(metrics[name]["median"])


if __name__ == "__main__":
    raise SystemExit(main())
