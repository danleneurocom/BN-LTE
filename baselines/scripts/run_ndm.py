#!/usr/bin/env python3
"""Fit and run the first NDM forecasting baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.forecasting import (  # noqa: E402
    build_prediction_rows,
    compute_aggregate_metrics,
    compute_pair_metrics,
    load_forecast_dataset,
    load_labeled_matrix,
    make_subject_split,
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
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
        help="Fit and evaluate NDM without writing outputs.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    laplacian_path = output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv")
    laplacian_labels, laplacian = load_labeled_matrix(laplacian_path)
    if laplacian_labels != dataset.region_labels:
        raise ValueError("Laplacian labels do not match forecast dataset region labels.")

    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=int(config.get("experiment", {}).get("random_seed", 20260507)),
    )

    rho_bounds = tuple(float(value) for value in modeling.get("parameter_bounds", {}).get("rho", [0.0, 10.0]))
    model = NetworkDiffusionModel(laplacian)
    fit = model.fit_global_rho(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
        dataset.time_years[split.train_indices],
        bounds=(rho_bounds[0], rho_bounds[1]),
    )
    predicted = model.predict(dataset.baseline, dataset.time_years, fit.rho)

    pair_metrics = compute_pair_metrics(dataset.pairs, dataset.baseline, dataset.observed, predicted, split, "ndm")
    aggregate_metrics = compute_aggregate_metrics(pair_metrics)
    prediction_rows = build_prediction_rows(dataset, predicted, split, "ndm")

    fit_report = {
        "model": "ndm",
        "equation": "dS/dt = -rho L S",
        "solution": "S(t) = exp(-rho * t * L) S(0)",
        "rho": fit.rho,
        "rho_bounds": list(rho_bounds),
        "train_mse": fit.train_mse,
        "optimizer_success": fit.optimizer_success,
        "optimizer_message": fit.optimizer_message,
        "forecast_pairs": len(dataset.pairs),
        "regions": len(dataset.region_labels),
        "train_pairs": int(split.train_indices.size),
        "test_pairs": int(split.test_indices.size),
        "train_subjects": len(split.train_rids),
        "test_subjects": len(split.test_rids),
    }

    print(json.dumps(fit_report, indent=2, sort_keys=True))
    print("\nAggregate metrics:")
    for row in aggregate_metrics:
        if row["split"] == "test" and row["metric"] in {"subject_spearman", "mae", "rmse", "delta_spearman"}:
            print(f"test {row['metric']}: median={row['median']:.4f}, mean={row['mean']:.4f}, n={row['n']}")

    if not args.no_write:
        predictions_path = output_dir / outputs.get("predictions_table", "forecast_predictions.csv")
        metrics_path = output_dir / outputs.get("metrics_table", "forecast_metrics.csv")
        fit_report_path = output_dir / outputs.get("ndm_fit_report", "ndm_fit_report.json")
        aggregate_metrics_path = output_dir / "forecast_metrics_summary.csv"

        write_csv_rows(predictions_path, prediction_rows)
        write_csv_rows(metrics_path, pair_metrics)
        write_csv_rows(aggregate_metrics_path, aggregate_metrics)
        write_json(fit_report_path, fit_report)

        print("\nWrote NDM outputs:")
        print(f"predictions_table: {predictions_path}")
        print(f"metrics_table: {metrics_path}")
        print(f"metrics_summary: {aggregate_metrics_path}")
        print(f"ndm_fit_report: {fit_report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
