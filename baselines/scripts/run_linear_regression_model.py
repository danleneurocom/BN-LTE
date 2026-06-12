#!/usr/bin/env python3
"""Fit regional ridge regression — pure statistical baseline (no physics).

  S1_hat_ij = f(S0_ij, t_i, S0_ij*t_i, S0_ij^2, amyloid_ij, thickness_ij, ...)

No ODE, no connectome dynamics.  Answers: what does a competent statistician
get without any physics knowledge?
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
from spread_toolbox.models.linear_regression import (  # noqa: E402
    build_regression_features,
    fit_regional_ridge,
)


MODEL_NAME = "linear_regression"


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

    result = run_linear_regression_model(dataset, split, laplacian, modeling, config)
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
            "Pure statistical baseline: regional ridge regression predicting observed tau "
            "directly from baseline tau, time, and biology. No ODE or physics."
        ),
        "equation": "S1_hat_ij = ridge(S0_ij, t_i, S0_ij*t_i, S0_ij^2, amyloid_ij, thickness_ij, ...)",
        "split": split_summary(split),
        "fit_report": result["fit_report"],
        "test_comparison": comparison_rows,
    }

    print(json.dumps(report["split"], indent=2, sort_keys=True))
    print("\nLinear regression fit report:")
    print(json.dumps(
        {k: v for k, v in report["fit_report"].items() if k != "covariates"},
        indent=2,
    ))
    print("\nTest comparison (linear regression vs physics models):")
    print_comparison_table(comparison_rows)
    print("\nTop regression terms:")
    print_term_rows(result["term_rows"][:15])

    if not args.no_write:
        write_csv_rows(
            output_dir / outputs.get("linear_regression_pair_metrics", "linear_regression_pair_metrics.csv"),
            pair_metrics,
        )
        write_csv_rows(
            output_dir / outputs.get("linear_regression_metrics_summary", "linear_regression_metrics_summary.csv"),
            metrics_summary,
        )
        write_csv_rows(
            output_dir / outputs.get("linear_regression_likelihood_metrics", "linear_regression_likelihood_metrics.csv"),
            likelihood_metrics,
        )
        write_csv_rows(
            output_dir / outputs.get("linear_regression_terms", "linear_regression_terms.csv"),
            result["term_rows"],
        )
        write_json(
            output_dir / outputs.get("linear_regression_report", "linear_regression_report.json"),
            {**report, "likelihood_metrics": likelihood_metrics},
        )
        print("\nWrote linear regression outputs.")

    return 0


def run_linear_regression_model(
    dataset: ForecastDataset,
    split: SubjectSplit,
    laplacian: np.ndarray,
    modeling: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    random_seed = int(config.get("experiment", {}).get("random_seed", 20260507))

    scaler = MinMaxStateScaler.fit(dataset.baseline[split.train_indices], dataset.observed[split.train_indices])
    baseline_scaled = scaler.transform(dataset.baseline)
    observed_scaled = scaler.transform(dataset.observed)

    pair_covariates, regional_covariates, covariate_report = build_closure_covariates(
        dataset, split, config, PROJECT_ROOT
    )
    amyloid = regional_covariates.get("amyloid_suvr")
    thickness = regional_covariates.get("cortical_thickness")
    apoe4_dose = pair_covariates.get("apoe4_dose")
    plasma_ptau181 = pair_covariates.get("plasma_ptau181")

    # Compute training median for p-tau imputation from training pairs only
    ptau_median = 0.0
    if plasma_ptau181 is not None:
        train_ptau = np.asarray(plasma_ptau181, dtype=float)[split.train_indices]
        finite = train_ptau[np.isfinite(train_ptau)]
        ptau_median = float(np.median(finite)) if finite.size > 0 else 0.0

    X, feat_names = build_regression_features(
        baseline_scaled, dataset.time_years, laplacian,
        amyloid=amyloid,
        thickness=thickness,
        apoe4_dose=apoe4_dose,
        plasma_ptau181=plasma_ptau181,
        ptau_train_median=ptau_median,
    )
    # Target: observed scaled, flattened to (n_pairs * n_regions,)
    target = observed_scaled.ravel()
    pair_groups = np.array([pair["RID"] for pair in dataset.pairs])

    alphas = tuple(
        float(v) for v in modeling.get("linear_regression_ridge_alphas", [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    )
    fit = fit_regional_ridge(
        X, target,
        train_row_indices=split.train_indices,
        pair_groups=pair_groups,
        n_regions=baseline_scaled.shape[1],
        alphas=alphas,
        cv_folds=int(modeling.get("linear_regression_cv_folds", 5)),
        max_train_rows=int(modeling.get("linear_regression_max_train_rows", 120000)),
        random_seed=random_seed,
    )
    # Attach proper feature names
    fit.feature_names = feat_names  # type: ignore[assignment]
    fit.ptau_train_median = ptau_median  # type: ignore[assignment]

    # Predict: reshape back to (n_pairs, n_regions)
    pred_flat = fit.predict_flat(X)
    predicted_scaled = np.clip(pred_flat.reshape(baseline_scaled.shape), 0.0, 1.0)
    predicted = scaler.inverse_transform(predicted_scaled)

    pair_metrics = compute_pair_metrics(
        dataset.pairs, dataset.baseline, dataset.observed, predicted, split, MODEL_NAME
    )
    n_likelihood_parameters = 2 + len(feat_names)  # intercept + features + sigma

    # Term rows sorted by |coefficient|
    term_rows = sorted(
        [
            {"term": name, "coefficient": float(c), "abs_coefficient": abs(float(c))}
            for name, c in zip(feat_names, fit.coefficients, strict=True)
        ],
        key=lambda r: r["abs_coefficient"],
        reverse=True,
    )

    return {
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "term_rows": term_rows,
        "fit_report": {
            "model": MODEL_NAME,
            "n_features": fit.n_features,
            "feature_names": feat_names,
            "ridge_alpha": fit.ridge_alpha,
            "train_mse_scaled": fit.train_mse,
            "train_r2_scaled": fit.train_r2,
            "ridge_cv": fit.cv_report,
            "used_train_rows": fit.used_train_rows,
            "covariates": covariate_report,
            "n_likelihood_parameters": n_likelihood_parameters,
        },
    }


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
    print(f"{'term':<40} {'coefficient':>14}")
    for r in rows:
        print(f"{str(r['term'])[:40]:<40} {float(r['coefficient']):>14.5g}")


def _med(metrics: dict[str, dict[str, Any]], name: str) -> float:
    return float(metrics[name]["median"])


if __name__ == "__main__":
    raise SystemExit(main())
