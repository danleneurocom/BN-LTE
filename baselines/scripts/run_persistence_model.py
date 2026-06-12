#!/usr/bin/env python3
"""Persistence baseline: predict S1 = S0 (no change).

The most important sanity check. If subject_spearman is already 0.90 just by
predicting no change, then a model with 0.905 is only marginally better than
doing nothing. This establishes the true floor that all models must beat.
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

from spread_toolbox.forecasting import (  # noqa: E402
    compute_aggregate_metrics,
    compute_pair_metrics,
    load_forecast_dataset,
    load_labeled_matrix,
    make_subject_split,
    read_csv_rows,
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402

MODEL_NAME = "persistence_s0"


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
    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=int(config.get("experiment", {}).get("random_seed", 20260507)),
    )

    # Persistence: predict S1 = S0
    predicted = dataset.baseline.copy()

    pair_metrics = compute_pair_metrics(
        dataset.pairs, dataset.baseline, dataset.observed, predicted, split, MODEL_NAME
    )
    metrics_summary = compute_aggregate_metrics(pair_metrics)

    # Load all available model summaries for comparison
    all_summaries: list[dict[str, Any]] = []
    summary_keys = [
        "baseline_comparison_metrics_summary",
        "individualized_residual_metrics_summary",
        "linear_regression_metrics_summary",
        "as_fkpp_metrics_summary",
        "bio_fkpp_metrics_summary",
    ]
    for key in summary_keys:
        path = output_dir / outputs.get(key, key.replace("_metrics_summary", "") + "_metrics_summary.csv")
        if path.exists():
            all_summaries.extend(read_csv_rows(path))

    comparison_rows = test_comparison_rows(all_summaries + metrics_summary)

    print(f"Split: {split.train_indices.size} train pairs / {split.test_indices.size} test pairs")
    print()
    print("Comparison — persistence S0 vs all models (test set):")
    print_comparison_table(comparison_rows)
    print()
    print_gain_analysis(comparison_rows)

    if not args.no_write:
        write_csv_rows(
            output_dir / outputs.get("persistence_pair_metrics", "persistence_pair_metrics.csv"),
            pair_metrics,
        )
        write_csv_rows(
            output_dir / outputs.get("persistence_metrics_summary", "persistence_metrics_summary.csv"),
            metrics_summary,
        )
        write_json(
            output_dir / outputs.get("persistence_report", "persistence_report.json"),
            {
                "model": MODEL_NAME,
                "description": "Persistence baseline: predict S1 = S0, no change.",
                "test_comparison": comparison_rows,
            },
        )
        print("Wrote persistence outputs.")

    return 0


def test_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {"mae", "rmse", "subject_spearman", "delta_spearman", "top5_overlap", "top10_overlap"}
    seen: set[tuple[str, str]] = set()
    out = []
    for row in rows:
        if row.get("split") == "test" and row.get("metric") in wanted:
            key = (str(row["model"]), str(row["metric"]))
            if key not in seen:
                seen.add(key)
                out.append(row)
    return out


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), {})[str(row["metric"])] = row

    # Sort: persistence first, then by descending subject_spearman
    def sort_key(item: tuple[str, dict]) -> tuple[int, float]:
        name, metrics = item
        if name == MODEL_NAME:
            return (0, 0.0)
        return (1, -float(metrics.get("subject_spearman", {}).get("median", 0)))

    hdr = f"{'model':<45} {'MAE':>8} {'RMSE':>8} {'subj rho':>10} {'delta rho':>10} {'top5':>8} {'top10':>8}"
    print(hdr)
    print("-" * len(hdr))
    for model_name, metrics in sorted(by_model.items(), key=sort_key):
        if metrics:
            flag = "  <-- baseline" if model_name == MODEL_NAME else ""
            print(
                f"{model_name:<45} "
                f"{_med(metrics, 'mae'):>8.4f} "
                f"{_med(metrics, 'rmse'):>8.4f} "
                f"{_med(metrics, 'subject_spearman'):>10.4f} "
                f"{_med(metrics, 'delta_spearman'):>10.4f} "
                f"{_med(metrics, 'top5_overlap'):>8.4f} "
                f"{_med(metrics, 'top10_overlap'):>8.4f}"
                f"{flag}"
            )


def print_gain_analysis(rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), {})[str(row["metric"])] = row

    if MODEL_NAME not in by_model:
        return

    base = by_model[MODEL_NAME]
    base_rho = _med(base, "subject_spearman")
    base_delta = _med(base, "delta_spearman")
    base_mae = _med(base, "mae")

    print("Gain over persistence baseline (absolute difference on test set):")
    print(f"  {'model':<45} {'Δ subj rho':>12} {'Δ delta rho':>13} {'Δ MAE':>10}")
    print(f"  {'-'*82}")
    for model_name, metrics in sorted(by_model.items(),
                                       key=lambda x: -_med(x[1], "subject_spearman")):
        if model_name == MODEL_NAME or not metrics:
            continue
        d_rho = _med(metrics, "subject_spearman") - base_rho
        d_delta = _med(metrics, "delta_spearman") - base_delta
        d_mae = _med(metrics, "mae") - base_mae
        print(
            f"  {model_name:<45} "
            f"{d_rho:>+12.4f} "
            f"{d_delta:>+13.4f} "
            f"{d_mae:>+10.4f}"
        )


def _med(metrics: dict[str, Any], name: str) -> float:
    entry = metrics.get(name, {})
    if isinstance(entry, dict):
        return float(entry.get("median", float("nan")))
    return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
