#!/usr/bin/env python3
"""Fit global-FKPP backbone plus individualized residual correction."""

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
from spread_toolbox.models.fkpp import GraphFKPPModel  # noqa: E402
from spread_toolbox.models.individualized_residual import (  # noqa: E402
    apply_individualized_residual_correction,
    build_individualized_residual_features,
    choose_residual_shrinkage,
    fit_ridge_residual_model,
)


MODEL_NAME = "global_fkpp_individualized_residual"


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
    laplacian = load_required_matrix(output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"), dataset)
    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=int(config.get("experiment", {}).get("random_seed", 20260507)),
    )

    result = run_individualized_residual_model(dataset, split, laplacian, modeling, config)
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
    closure_summary = load_optional_rows(
        output_dir / outputs.get("global_fkpp_closure_metrics_summary", "global_fkpp_closure_metrics_summary.csv")
    )
    comparison_rows = test_comparison_rows(baseline_summary + closure_summary + metrics_summary)
    report = {
        "model": MODEL_NAME,
        "purpose": (
            "Global FKPP backbone plus train-only individualized residual correction using FKPP, subject, "
            "regional, and region-bias features."
        ),
        "equation": "S1_hat = global_FKPP(S0, connectome, dt) + gamma * dt * ridge_residual_rate(features)",
        "split": split_summary(split),
        "fit_report": result["fit_report"],
        "r2_report": result["r2_report"],
        "test_comparison": comparison_rows,
    }

    print(json.dumps(report["split"], indent=2, sort_keys=True))
    print("\nGlobal FKPP + individualized residual comparison:")
    print_comparison_table(comparison_rows)
    print("\nR2 report:")
    print(json.dumps(report["r2_report"], indent=2, sort_keys=True))
    print("\nTop residual terms:")
    print_terms(result["term_rows"][:20])

    if not args.no_write:
        write_csv_rows(
            output_dir / outputs.get("individualized_residual_pair_metrics", "individualized_residual_pair_metrics.csv"),
            pair_metrics,
        )
        write_csv_rows(
            output_dir
            / outputs.get("individualized_residual_metrics_summary", "individualized_residual_metrics_summary.csv"),
            metrics_summary,
        )
        write_csv_rows(
            output_dir
            / outputs.get("individualized_residual_likelihood_metrics", "individualized_residual_likelihood_metrics.csv"),
            likelihood_metrics,
        )
        write_csv_rows(
            output_dir / outputs.get("individualized_residual_terms", "individualized_residual_terms.csv"),
            result["term_rows"],
        )
        write_json(
            output_dir / outputs.get("individualized_residual_report", "individualized_residual_report.json"),
            {**report, "likelihood_metrics": likelihood_metrics},
        )
        print("\nWrote individualized residual outputs.")

    return 0


def run_individualized_residual_model(
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

    parameter_bounds = modeling.get("parameter_bounds", {})
    rho_bounds = tuple(float(value) for value in parameter_bounds.get("rho", [0.0, 10.0]))
    alpha_bounds = tuple(float(value) for value in parameter_bounds.get("alpha", [0.0, 10.0]))
    fkpp_model = GraphFKPPModel(
        laplacian,
        steps_per_year=int(modeling.get("individualized_residual_fkpp_steps_per_year", modeling.get("fkpp_steps_per_year", 12))),
        laplacian_normalization=str(
            modeling.get("individualized_residual_laplacian_normalization", modeling.get("fkpp_laplacian_normalization", "spectral"))
        ),
    )
    backbone_fit = fkpp_model.fit_global_parameters(
        baseline_scaled[split.train_indices],
        observed_scaled[split.train_indices],
        dataset.time_years[split.train_indices],
        rho_bounds=(rho_bounds[0], rho_bounds[1]),
        alpha_bounds=(alpha_bounds[0], alpha_bounds[1]),
        maxiter=int(modeling.get("individualized_residual_fkpp_optimizer_maxiter", modeling.get("fkpp_optimizer_maxiter", 80))),
    )
    backbone_scaled = fkpp_model.predict(baseline_scaled, dataset.time_years, rho=backbone_fit.rho, alpha=backbone_fit.alpha)
    backbone_predicted = scaler.inverse_transform(backbone_scaled)

    pair_covariates, regional_covariates, covariate_report = build_closure_covariates(dataset, split, config, PROJECT_ROOT)
    features = build_individualized_residual_features(
        baseline=baseline_scaled,
        backbone_prediction=backbone_scaled,
        time_years=dataset.time_years,
        laplacian=fkpp_model.laplacian,
        region_labels=dataset.region_labels,
        pair_covariates=pair_covariates,
        regional_covariates=regional_covariates,
        include_region_bias=bool(modeling.get("individualized_residual_region_bias", True)),
    )
    target_rate = (observed_scaled - backbone_scaled) / np.maximum(dataset.time_years, 1.0e-6)[:, None]
    fit = fit_ridge_residual_model(
        features,
        target_rate,
        row_indices=split.train_indices,
        pair_groups=np.asarray([pair["RID"] for pair in dataset.pairs]),
        alphas=tuple(float(value) for value in modeling.get("individualized_residual_ridge_alphas", [0.1, 1, 10, 100, 1000])),
        cv_folds=int(modeling.get("individualized_residual_cv_folds", 5)),
        max_train_rows=int(modeling.get("individualized_residual_max_train_rows", 60000)),
        random_seed=random_seed,
    )
    residual_rate = fit.predict_rate(features.values)
    shrinkage_candidates = tuple(
        float(value) for value in modeling.get("individualized_residual_shrinkage_candidates", [0, 0.25, 0.5, 0.75, 1.0])
    )
    shrinkage, shrinkage_report = choose_residual_shrinkage(
        backbone_prediction=backbone_scaled,
        observed=observed_scaled,
        time_years=dataset.time_years,
        residual_rate=residual_rate,
        row_indices=split.train_indices,
        candidates=shrinkage_candidates,
    )
    predicted_scaled = apply_individualized_residual_correction(
        backbone_scaled,
        dataset.time_years,
        residual_rate,
        shrinkage=shrinkage,
        max_abs_delta=modeling.get("individualized_residual_max_abs_scaled_delta", None),
    )
    predicted = scaler.inverse_transform(predicted_scaled)
    pair_metrics = compute_pair_metrics(dataset.pairs, dataset.baseline, dataset.observed, predicted, split, MODEL_NAME)
    n_likelihood_parameters = 4 + len(fit.feature_names)
    return {
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "term_rows": fit.term_rows(),
        "r2_report": build_r2_report(dataset, split, backbone_predicted, predicted),
        "fit_report": {
            "model": MODEL_NAME,
            "backbone": "global_fkpp",
            "backbone_rho": backbone_fit.rho,
            "backbone_alpha": backbone_fit.alpha,
            "backbone_train_mse_scaled": backbone_fit.train_mse,
            "laplacian_normalization": fkpp_model.laplacian_normalization,
            "laplacian_scale": fkpp_model.laplacian_scale,
            "residual_target": "scaled residual rate = (observed_scaled - global_fkpp_scaled) / dt",
            "ridge_alpha": fit.alpha,
            "residual_shrinkage": shrinkage,
            "shrinkage_report": shrinkage_report,
            "residual_train_mse_rate": fit.train_mse_rate,
            "residual_train_r2_rate": fit.train_r2_rate,
            "feature_count": len(fit.feature_names),
            "used_train_rows": fit.used_train_rows,
            "available_train_rows": fit.available_train_rows,
            "ridge_cv": fit.cv_report,
            "covariates": covariate_report,
            "n_likelihood_parameters": n_likelihood_parameters,
        },
    }


def build_r2_report(
    dataset: ForecastDataset,
    split: SubjectSplit,
    backbone_predicted: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    train = split.train_indices
    test = split.test_indices
    return {
        "train_scalar_r2_backbone": scalar_r2(dataset.observed[train], backbone_predicted[train]),
        "train_scalar_r2_individualized_residual": scalar_r2(dataset.observed[train], predicted[train]),
        "test_scalar_r2_backbone": scalar_r2(dataset.observed[test], backbone_predicted[test]),
        "test_scalar_r2_individualized_residual": scalar_r2(dataset.observed[test], predicted[test]),
        "test_scalar_r2_gain": scalar_r2(dataset.observed[test], predicted[test])
        - scalar_r2(dataset.observed[test], backbone_predicted[test]),
    }


def scalar_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    y = observed[mask]
    yhat = predicted[mask]
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / total)


def load_required_matrix(path: Path, dataset: ForecastDataset) -> np.ndarray:
    labels, matrix = load_labeled_matrix(path)
    if labels != dataset.region_labels:
        raise ValueError(f"Matrix labels do not match forecast dataset region labels: {path}")
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
    order = ["global_fkpp", "global_fkpp_horseshoe_closure", MODEL_NAME]
    for row in rows:
        by_model.setdefault(str(row["model"]), {})[str(row["metric"])] = row
    print(f"{'model':<38} {'MAE':>8} {'RMSE':>8} {'subj rho':>10} {'delta rho':>10} {'top5':>8} {'top10':>8}")
    for model in order:
        metrics = by_model.get(model, {})
        if metrics:
            print(
                f"{model:<38} "
                f"{median(metrics, 'mae'):>8.4f} "
                f"{median(metrics, 'rmse'):>8.4f} "
                f"{median(metrics, 'subject_spearman'):>10.4f} "
                f"{median(metrics, 'delta_spearman'):>10.4f} "
                f"{median(metrics, 'top5_overlap'):>8.4f} "
                f"{median(metrics, 'top10_overlap'):>8.4f}"
            )


def print_terms(rows: list[dict[str, Any]]) -> None:
    print(f"{'term':<52} {'coef':>12}")
    for row in rows:
        print(f"{str(row['term'])[:52]:<52} {float(row['coefficient']):>12.5g}")


def median(metrics: dict[str, dict[str, Any]], name: str) -> float:
    return float(metrics[name]["median"])


if __name__ == "__main__":
    raise SystemExit(main())
