#!/usr/bin/env python3
"""Train and compare baseline forecasting models on one train/test split."""

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
    SubjectSplit,
    compute_aggregate_metrics,
    compute_pair_metrics,
    load_forecast_dataset,
    load_labeled_matrix,
    make_subject_split,
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.models.esm import EpidemicSpreadingModel  # noqa: E402
from spread_toolbox.models.fkpp import GraphFKPPModel  # noqa: E402
from spread_toolbox.models.ndm import NetworkDiffusionModel  # noqa: E402


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
        help="Run the comparison without writing outputs.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    adjacency = load_required_matrix(output_dir / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv"), dataset)
    laplacian = load_required_matrix(output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"), dataset)
    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=int(config.get("experiment", {}).get("random_seed", 20260507)),
    )

    model_results = [
        run_ndm(dataset, split, laplacian, modeling),
        run_esm(dataset, split, adjacency, modeling),
        run_fkpp(dataset, split, laplacian, modeling),
    ]
    pair_metrics = [row for result in model_results for row in result["pair_metrics"]]
    metrics_summary = compute_aggregate_metrics(pair_metrics)
    test_comparison = test_comparison_rows(metrics_summary)

    report = {
        "purpose": "Baseline comparison only: fit each baseline on train and compare all baselines on the same test set.",
        "split": {
            "train_pairs": int(split.train_indices.size),
            "test_pairs": int(split.test_indices.size),
            "train_subjects": len(split.train_rids),
            "test_subjects": len(split.test_rids),
        },
        "model_fit_reports": [result["fit_report"] for result in model_results],
        "test_comparison": test_comparison,
    }

    print(json.dumps(report["split"], indent=2, sort_keys=True))
    print("\nBaseline test comparison:")
    print_comparison_table(test_comparison)

    if not args.no_write:
        pair_metrics_path = output_dir / outputs.get(
            "baseline_comparison_pair_metrics",
            "baseline_comparison_pair_metrics.csv",
        )
        metrics_summary_path = output_dir / outputs.get(
            "baseline_comparison_metrics_summary",
            "baseline_comparison_metrics_summary.csv",
        )
        report_path = output_dir / outputs.get("baseline_comparison_report", "baseline_comparison_report.json")

        write_csv_rows(pair_metrics_path, pair_metrics)
        write_csv_rows(metrics_summary_path, metrics_summary)
        write_json(report_path, report)

        print("\nWrote baseline-comparison outputs:")
        print(f"pair_metrics: {pair_metrics_path}")
        print(f"metrics_summary: {metrics_summary_path}")
        print(f"baseline_comparison_report: {report_path}")

    return 0


def load_required_matrix(path: Path, dataset: ForecastDataset) -> np.ndarray:
    labels, matrix = load_labeled_matrix(path)
    if labels != dataset.region_labels:
        raise ValueError(f"Matrix labels do not match forecast dataset region labels: {path}")
    return matrix


def run_ndm(
    dataset: ForecastDataset,
    split: SubjectSplit,
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
    split: SubjectSplit,
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
        "pair_metrics": pair_metrics,
        "fit_report": {
            "model": "esm",
            "equation": "dS/dt = beta * (1 - S) * W S",
            "beta": fit.beta,
            "beta_bounds": list(bounds),
            "train_mse_scaled": fit.train_mse,
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
        },
    }


def run_fkpp(
    dataset: ForecastDataset,
    split: SubjectSplit,
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
        "pair_metrics": pair_metrics,
        "fit_report": {
            "model": "global_fkpp",
            "equation": "dS/dt = -rho L S + alpha S(1 - S)",
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


def test_comparison_rows(metrics_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted_metrics = ["mae", "rmse", "subject_spearman", "delta_spearman", "top5_overlap", "top10_overlap"]
    return [
        row
        for row in metrics_summary
        if row["split"] == "test" and row["metric"] in wanted_metrics
    ]


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), {})[str(row["metric"])] = row
    header = (
        "model",
        "MAE med",
        "RMSE med",
        "subject Spearman med",
        "delta Spearman med",
        "top5 med",
        "top10 med",
    )
    print(
        f"{header[0]:<14} {header[1]:>9} {header[2]:>9} {header[3]:>21} "
        f"{header[4]:>19} {header[5]:>9} {header[6]:>10}"
    )
    for model in ["ndm", "esm", "global_fkpp"]:
        metrics = by_model.get(model, {})
        print(
            f"{model:<14} "
            f"{median(metrics, 'mae'):>9.4f} "
            f"{median(metrics, 'rmse'):>9.4f} "
            f"{median(metrics, 'subject_spearman'):>21.4f} "
            f"{median(metrics, 'delta_spearman'):>19.4f} "
            f"{median(metrics, 'top5_overlap'):>9.4f} "
            f"{median(metrics, 'top10_overlap'):>10.4f}"
        )


def median(metrics: dict[str, dict[str, Any]], name: str) -> float:
    return float(metrics[name]["median"])


if __name__ == "__main__":
    raise SystemExit(main())
