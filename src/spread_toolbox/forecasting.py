"""Forecasting dataset assembly, metrics, and output helpers."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .connectome import read_mapping
from .io_adni import resolve_project_path


@dataclass
class ForecastDataset:
    pairs: list[dict[str, str]]
    region_labels: list[str]
    tau_columns: list[str]
    baseline: np.ndarray
    observed: np.ndarray
    time_years: np.ndarray


@dataclass
class SubjectSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_rids: list[str]
    test_rids: list[str]


def load_labeled_matrix(path: str | Path) -> tuple[list[str], np.ndarray]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        labels = header[1:]
        rows = []
        row_labels = []
        for row in reader:
            row_labels.append(row[0])
            rows.append([float(value) for value in row[1:]])
    if row_labels != labels:
        raise ValueError("Matrix row labels do not match column labels.")
    return labels, np.asarray(rows, dtype=float)


def load_forecast_dataset(config: dict[str, Any], project_root: str | Path) -> ForecastDataset:
    output_dir = resolve_project_path(config["paths"]["output_dir"], project_root)
    outputs = config.get("outputs", {})
    region_config = config.get("region_mapping", {})

    pairs_path = output_dir / outputs.get("forecast_pairs_table", "cohort_forecast_pairs.csv")
    observations_path = output_dir / outputs.get("tau_observations_table", "cohort_tau_observations.csv")
    mapping_path = resolve_project_path(region_config["mapping_file"], project_root)

    mapping_rows = sorted(read_mapping(mapping_path), key=lambda row: int(row["enigma_index"]))
    region_labels = [row["enigma_label"] for row in mapping_rows]
    tau_columns = [row["adni_tau_column"] for row in mapping_rows]

    pairs = read_csv_rows(pairs_path)
    observations = {row["LONIUID"]: row for row in read_csv_rows(observations_path)}

    baseline_vectors: list[list[float]] = []
    observed_vectors: list[list[float]] = []
    time_years: list[float] = []
    usable_pairs: list[dict[str, str]] = []

    for pair in pairs:
        baseline_row = observations.get(pair["baseline_loniuid"])
        target_row = observations.get(pair["target_loniuid"])
        if baseline_row is None or target_row is None:
            continue
        baseline_vector = vector_from_row(baseline_row, tau_columns)
        observed_vector = vector_from_row(target_row, tau_columns)
        if baseline_vector is None or observed_vector is None:
            continue
        baseline_vectors.append(baseline_vector)
        observed_vectors.append(observed_vector)
        time_years.append(float(pair["target_time_years"]))
        usable_pairs.append(pair)

    if not usable_pairs:
        raise ValueError("No usable forecast pairs were found.")

    return ForecastDataset(
        pairs=usable_pairs,
        region_labels=region_labels,
        tau_columns=tau_columns,
        baseline=np.asarray(baseline_vectors, dtype=float),
        observed=np.asarray(observed_vectors, dtype=float),
        time_years=np.asarray(time_years, dtype=float),
    )


def vector_from_row(row: dict[str, str], columns: list[str]) -> list[float] | None:
    values: list[float] = []
    for column in columns:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return values


def make_subject_split(
    pairs: list[dict[str, str]],
    *,
    test_fraction: float,
    random_seed: int,
) -> SubjectSplit:
    unique_rids = sorted({pair["RID"] for pair in pairs}, key=lambda value: int(value) if value.isdigit() else value)
    rng = np.random.default_rng(random_seed)
    shuffled = np.asarray(unique_rids, dtype=object)
    rng.shuffle(shuffled)

    test_count = max(1, int(round(len(shuffled) * test_fraction)))
    test_rids = sorted(str(value) for value in shuffled[:test_count])
    train_rids = sorted(str(value) for value in shuffled[test_count:])
    test_set = set(test_rids)

    train_indices = []
    test_indices = []
    for index, pair in enumerate(pairs):
        if pair["RID"] in test_set:
            test_indices.append(index)
        else:
            train_indices.append(index)

    return SubjectSplit(
        train_indices=np.asarray(train_indices, dtype=int),
        test_indices=np.asarray(test_indices, dtype=int),
        train_rids=train_rids,
        test_rids=test_rids,
    )


def compute_pair_metrics(
    pairs: list[dict[str, str]],
    baseline: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    split: SubjectSplit,
    model_name: str,
) -> list[dict[str, Any]]:
    split_by_index = {int(index): "train" for index in split.train_indices}
    split_by_index.update({int(index): "test" for index in split.test_indices})

    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        y_true = observed[index]
        y_pred = predicted[index]
        base = baseline[index]
        rows.append(
            {
                "model": model_name,
                "split": split_by_index[index],
                "RID": pair["RID"],
                "PTID": pair["PTID"],
                "TRACER": pair["TRACER"],
                "target_role": pair["target_role"],
                "baseline_tau_date": pair["baseline_tau_date"],
                "target_tau_date": pair["target_tau_date"],
                "target_time_years": pair["target_time_years"],
                "mae": float(np.mean(np.abs(y_pred - y_true))),
                "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
                "subject_spearman": safe_spearman(y_true, y_pred),
                "subject_pearson": safe_pearson(y_true, y_pred),
                "delta_spearman": safe_spearman(y_true - base, y_pred - base),
                "delta_pearson": safe_pearson(y_true - base, y_pred - base),
                "top5_overlap": top_k_overlap(y_true - base, y_pred - base, 5),
                "top10_overlap": top_k_overlap(y_true - base, y_pred - base, 10),
            }
        )
    return rows


def compute_aggregate_metrics(pair_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        "mae",
        "rmse",
        "subject_spearman",
        "subject_pearson",
        "delta_spearman",
        "delta_pearson",
        "top5_overlap",
        "top10_overlap",
    ]
    rows: list[dict[str, Any]] = []
    for split in ["train", "test", "all"]:
        split_rows = pair_metrics if split == "all" else [row for row in pair_metrics if row["split"] == split]
        for metric in metric_names:
            values = np.asarray([row[metric] for row in split_rows if row[metric] == row[metric]], dtype=float)
            if values.size == 0:
                continue
            rows.append(
                {
                    "model": split_rows[0]["model"] if split_rows else "",
                    "split": split,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    return rows


def build_prediction_rows(
    dataset: ForecastDataset,
    predicted: np.ndarray,
    split: SubjectSplit,
    model_name: str,
) -> list[dict[str, Any]]:
    split_by_index = {int(index): "train" for index in split.train_indices}
    split_by_index.update({int(index): "test" for index in split.test_indices})

    rows: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(dataset.pairs):
        for region_index, (label, tau_column) in enumerate(zip(dataset.region_labels, dataset.tau_columns)):
            baseline_tau = dataset.baseline[pair_index, region_index]
            observed_tau = dataset.observed[pair_index, region_index]
            predicted_tau = predicted[pair_index, region_index]
            rows.append(
                {
                    "model": model_name,
                    "split": split_by_index[pair_index],
                    "RID": pair["RID"],
                    "PTID": pair["PTID"],
                    "TRACER": pair["TRACER"],
                    "target_role": pair["target_role"],
                    "baseline_tau_date": pair["baseline_tau_date"],
                    "target_tau_date": pair["target_tau_date"],
                    "target_time_years": pair["target_time_years"],
                    "enigma_label": label,
                    "adni_tau_column": tau_column,
                    "baseline_tau": float(baseline_tau),
                    "observed_tau": float(observed_tau),
                    "predicted_tau": float(predicted_tau),
                    "observed_delta": float(observed_tau - baseline_tau),
                    "predicted_delta": float(predicted_tau - baseline_tau),
                }
            )
    return rows


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    result = spearmanr(a, b).statistic
    return float(result) if np.isfinite(result) else float("nan")


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    try:
        result = pearsonr(a, b).statistic
    except ValueError:
        return float("nan")
    return float(result) if np.isfinite(result) else float("nan")


def top_k_overlap(observed_delta: np.ndarray, predicted_delta: np.ndarray, k: int) -> float:
    k = min(k, observed_delta.size)
    observed_top = set(np.argsort(observed_delta)[-k:])
    predicted_top = set(np.argsort(predicted_delta)[-k:])
    return len(observed_top & predicted_top) / k


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = union_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def union_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    return fieldnames
