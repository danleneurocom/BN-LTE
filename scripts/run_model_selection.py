#!/usr/bin/env python3
"""Train NDM, ESM, and FKPP with train/validation/test model selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.forecasting import (  # noqa: E402
    ForecastDataset,
    MinMaxStateScaler,
    SubjectTrainValidationTestSplit,
    build_prediction_rows,
    compute_aggregate_metrics,
    compute_pair_metrics,
    load_forecast_dataset,
    load_labeled_matrix,
    make_subject_train_validation_test_split,
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.models.esm import EpidemicSpreadingModel  # noqa: E402
from spread_toolbox.models.fkpp import GraphFKPPModel  # noqa: E402
from spread_toolbox.models.ndm import NetworkDiffusionModel  # noqa: E402


HIGHER_IS_BETTER = {
    "subject_spearman",
    "subject_pearson",
    "delta_spearman",
    "delta_pearson",
    "top5_overlap",
    "top10_overlap",
}
LOWER_IS_BETTER = {"mae", "rmse"}


def default_config_path() -> Path:
    experiment_dir = PROJECT_ROOT / "experiments" / "group_average_enigma"
    local_config = experiment_dir / "config.yaml"
    if local_config.exists():
        return local_config
    return experiment_dir / "config.example.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to config YAML. Defaults to config.yaml if present, otherwise config.example.yaml.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run model selection without writing outputs.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})
    evaluation = config.get("evaluation", {})

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    adjacency = load_required_matrix(output_dir / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv"), dataset)
    laplacian = load_required_matrix(output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"), dataset)

    split = make_subject_train_validation_test_split(
        dataset.pairs,
        validation_fraction=float(modeling.get("validation_fraction", 0.2)),
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=int(config.get("experiment", {}).get("random_seed", 20260507)),
    )

    model_results = [
        run_ndm(dataset, split, laplacian, modeling),
        run_esm(dataset, split, adjacency, modeling),
        run_fkpp(dataset, split, laplacian, modeling),
    ]

    primary_metric = str(evaluation.get("primary_metric", "subject_spearman"))
    selection_stat = str(modeling.get("selection_stat", "median"))
    comparison_pair_metrics = [
        row
        for result in model_results
        for row in result["pair_metrics"]
        if row["split"] in {"train", "validation"}
    ]
    comparison_summary = compute_aggregate_metrics(comparison_pair_metrics)
    selection_rows = validation_selection_rows(comparison_summary, primary_metric, selection_stat)
    selected_model = choose_best_model(selection_rows, primary_metric, selection_stat)

    selected_result = next(result for result in model_results if result["model"] == selected_model)
    selected_pair_metrics = selected_result["pair_metrics"]
    selected_summary = compute_aggregate_metrics(selected_pair_metrics)
    selected_predictions = build_prediction_rows(dataset, selected_result["predicted"], split, selected_model)
    final_test_rows = [
        row
        for row in selected_summary
        if row["split"] == "test" and row["metric"] in {"mae", "rmse", "subject_spearman", "delta_spearman"}
    ]

    report = {
        "selection_rule": {
            "split": "validation",
            "metric": primary_metric,
            "stat": selection_stat,
            "direction": metric_direction(primary_metric),
        },
        "selected_model": selected_model,
        "split": split_report(split),
        "model_fit_reports": [result["fit_report"] for result in model_results],
        "validation_selection": selection_rows,
        "selected_model_test_metrics": final_test_rows,
    }

    print(json.dumps(report["selection_rule"], indent=2, sort_keys=True))
    print("\nSplit:")
    print(json.dumps(report["split"], indent=2, sort_keys=True))
    print("\nValidation comparison:")
    for row in selection_rows:
        marker = "*" if row["model"] == selected_model else " "
        print(f"{marker} {row['model']}: validation {primary_metric} {selection_stat}={row[selection_stat]:.4f}")
    print(f"\nSelected model: {selected_model}")
    print("\nFinal held-out test result for selected model:")
    for row in final_test_rows:
        print(f"test {row['metric']}: median={row['median']:.4f}, mean={row['mean']:.4f}, n={row['n']}")

    if not args.no_write:
        comparison_pair_metrics_path = output_dir / outputs.get(
            "model_selection_pair_metrics",
            "model_selection_pair_metrics_train_validation.csv",
        )
        comparison_summary_path = output_dir / outputs.get(
            "model_selection_metrics_summary",
            "model_selection_metrics_summary.csv",
        )
        selected_predictions_path = output_dir / outputs.get(
            "model_selection_selected_predictions",
            "model_selection_selected_predictions.csv",
        )
        selected_metrics_path = output_dir / outputs.get(
            "model_selection_selected_metrics",
            "model_selection_selected_metrics.csv",
        )
        selected_summary_path = output_dir / outputs.get(
            "model_selection_selected_metrics_summary",
            "model_selection_selected_metrics_summary.csv",
        )
        report_path = output_dir / outputs.get("model_selection_report", "model_selection_report.json")

        write_csv_rows(comparison_pair_metrics_path, comparison_pair_metrics)
        write_csv_rows(comparison_summary_path, comparison_summary)
        write_csv_rows(selected_predictions_path, selected_predictions)
        write_csv_rows(selected_metrics_path, selected_pair_metrics)
        write_csv_rows(selected_summary_path, selected_summary)
        write_json(report_path, report)

        print("\nWrote model-selection outputs:")
        print(f"train_validation_pair_metrics: {comparison_pair_metrics_path}")
        print(f"train_validation_metrics_summary: {comparison_summary_path}")
        print(f"selected_predictions: {selected_predictions_path}")
        print(f"selected_pair_metrics: {selected_metrics_path}")
        print(f"selected_metrics_summary: {selected_summary_path}")
        print(f"model_selection_report: {report_path}")

    return 0


def load_required_matrix(path: Path, dataset: ForecastDataset) -> np.ndarray:
    labels, matrix = load_labeled_matrix(path)
    if labels != dataset.region_labels:
        raise ValueError(f"Matrix labels do not match forecast dataset region labels: {path}")
    return matrix


def run_ndm(
    dataset: ForecastDataset,
    split: SubjectTrainValidationTestSplit,
    laplacian: np.ndarray,
    modeling: dict[str, Any],
) -> dict[str, Any]:
    bounds = tuple(float(value) for value in modeling.get("parameter_bounds", {}).get("rho", [0.0, 10.0]))
    model = NetworkDiffusionModel(laplacian)
    fit = model.fit_global_rho(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
        dataset.time_years[split.train_indices],
        bounds=(bounds[0], bounds[1]),
    )
    predicted = model.predict(dataset.baseline, dataset.time_years, fit.rho)
    pair_metrics = compute_pair_metrics(dataset.pairs, dataset.baseline, dataset.observed, predicted, split, "ndm")
    return {
        "model": "ndm",
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "fit_report": {
            "model": "ndm",
            "equation": "dS/dt = -rho L S",
            "rho": fit.rho,
            "rho_bounds": list(bounds),
            "train_mse": fit.train_mse,
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
        },
    }


def run_esm(
    dataset: ForecastDataset,
    split: SubjectTrainValidationTestSplit,
    adjacency: np.ndarray,
    modeling: dict[str, Any],
) -> dict[str, Any]:
    scaler = MinMaxStateScaler.fit(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
    )
    baseline_scaled = scaler.transform(dataset.baseline)
    observed_scaled = scaler.transform(dataset.observed)
    bounds = tuple(float(value) for value in modeling.get("parameter_bounds", {}).get("beta", [0.0, 10.0]))
    model = EpidemicSpreadingModel(adjacency, steps_per_year=int(modeling.get("esm_steps_per_year", 12)))
    fit = model.fit_global_beta(
        baseline_scaled[split.train_indices],
        observed_scaled[split.train_indices],
        dataset.time_years[split.train_indices],
        bounds=(bounds[0], bounds[1]),
    )
    predicted = scaler.inverse_transform(model.predict(baseline_scaled, dataset.time_years, fit.beta))
    pair_metrics = compute_pair_metrics(dataset.pairs, dataset.baseline, dataset.observed, predicted, split, "esm")
    return {
        "model": "esm",
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "fit_report": {
            "model": "esm",
            "equation": "dS/dt = beta * (1 - S) * W S",
            "state_scaling": "per-region min-max fit on training baseline and target tau",
            "beta": fit.beta,
            "beta_bounds": list(bounds),
            "train_mse_scaled": fit.train_mse,
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
        },
    }


def run_fkpp(
    dataset: ForecastDataset,
    split: SubjectTrainValidationTestSplit,
    laplacian: np.ndarray,
    modeling: dict[str, Any],
) -> dict[str, Any]:
    scaler = MinMaxStateScaler.fit(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
    )
    baseline_scaled = scaler.transform(dataset.baseline)
    observed_scaled = scaler.transform(dataset.observed)
    parameter_bounds = modeling.get("parameter_bounds", {})
    rho_bounds = tuple(float(value) for value in parameter_bounds.get("rho", [0.0, 10.0]))
    alpha_bounds = tuple(float(value) for value in parameter_bounds.get("alpha", [0.0, 10.0]))
    model = GraphFKPPModel(
        laplacian,
        steps_per_year=int(modeling.get("fkpp_steps_per_year", 12)),
        laplacian_normalization=str(modeling.get("fkpp_laplacian_normalization", "spectral")),
    )
    fit = model.fit_global_parameters(
        baseline_scaled[split.train_indices],
        observed_scaled[split.train_indices],
        dataset.time_years[split.train_indices],
        rho_bounds=(rho_bounds[0], rho_bounds[1]),
        alpha_bounds=(alpha_bounds[0], alpha_bounds[1]),
        maxiter=int(modeling.get("fkpp_optimizer_maxiter", 80)),
    )
    predicted = scaler.inverse_transform(model.predict(baseline_scaled, dataset.time_years, rho=fit.rho, alpha=fit.alpha))
    pair_metrics = compute_pair_metrics(
        dataset.pairs,
        dataset.baseline,
        dataset.observed,
        predicted,
        split,
        "global_fkpp",
    )
    return {
        "model": "global_fkpp",
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "fit_report": {
            "model": "global_fkpp",
            "equation": "dS/dt = -rho L S + alpha S(1 - S)",
            "state_scaling": "per-region min-max fit on training baseline and target tau",
            "laplacian_normalization": model.laplacian_normalization,
            "laplacian_scale": model.laplacian_scale,
            "rho": fit.rho,
            "alpha": fit.alpha,
            "rho_bounds": list(rho_bounds),
            "alpha_bounds": list(alpha_bounds),
            "train_mse_scaled": fit.train_mse,
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
            "optimizer_iterations": fit.optimizer_iterations,
            "optimizer_evaluations": fit.optimizer_evaluations,
        },
    }


def validation_selection_rows(
    aggregate_metrics: list[dict[str, Any]],
    primary_metric: str,
    selection_stat: str,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in aggregate_metrics
        if row["split"] == "validation" and row["metric"] == primary_metric and selection_stat in row
    ]
    if not rows:
        raise ValueError(f"No validation rows found for metric {primary_metric!r}.")
    return sorted(rows, key=lambda row: row["model"])


def choose_best_model(selection_rows: list[dict[str, Any]], primary_metric: str, selection_stat: str) -> str:
    direction = metric_direction(primary_metric)
    if direction == "higher":
        best = max(selection_rows, key=lambda row: float(row[selection_stat]))
    else:
        best = min(selection_rows, key=lambda row: float(row[selection_stat]))
    return str(best["model"])


def metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    raise ValueError(f"Unknown metric direction for {metric!r}.")


def split_report(split: SubjectTrainValidationTestSplit) -> dict[str, int]:
    return {
        "train_pairs": int(split.train_indices.size),
        "validation_pairs": int(split.validation_indices.size),
        "test_pairs": int(split.test_indices.size),
        "train_subjects": len(split.train_rids),
        "validation_subjects": len(split.validation_rids),
        "test_subjects": len(split.test_rids),
    }


if __name__ == "__main__":
    raise SystemExit(main())
