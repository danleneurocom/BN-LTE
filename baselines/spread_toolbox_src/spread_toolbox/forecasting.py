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


@dataclass
class SubjectTrainValidationTestSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_rids: list[str]
    validation_rids: list[str]
    test_rids: list[str]


@dataclass
class MinMaxStateScaler:
    """Per-region min-max scaler for bounded state-space models."""

    lower: np.ndarray
    upper: np.ndarray
    epsilon: float = 1.0e-8

    @classmethod
    def fit(cls, *arrays: np.ndarray, epsilon: float = 1.0e-8) -> "MinMaxStateScaler":
        if not arrays:
            raise ValueError("At least one array is required to fit MinMaxStateScaler.")
        normalized_arrays = [np.asarray(array, dtype=float) for array in arrays]
        region_count = normalized_arrays[0].shape[1]
        for array in normalized_arrays:
            if array.ndim != 2:
                raise ValueError("Scaler inputs must be two-dimensional arrays.")
            if array.shape[1] != region_count:
                raise ValueError("All scaler inputs must have the same number of regions.")
        stacked = np.vstack(normalized_arrays)
        return cls(lower=np.min(stacked, axis=0), upper=np.max(stacked, axis=0), epsilon=epsilon)

    @property
    def scale(self) -> np.ndarray:
        return np.maximum(self.upper - self.lower, self.epsilon)

    def transform(self, values: np.ndarray, *, clip: bool = True) -> np.ndarray:
        transformed = (np.asarray(values, dtype=float) - self.lower) / self.scale
        if clip:
            return np.clip(transformed, 0.0, 1.0)
        return transformed

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.lower


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


def make_subject_train_validation_test_split(
    pairs: list[dict[str, str]],
    *,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> SubjectTrainValidationTestSplit:
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation_fraction and test_fraction must be positive.")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.")

    unique_rids = sorted({pair["RID"] for pair in pairs}, key=lambda value: int(value) if value.isdigit() else value)
    if len(unique_rids) < 3:
        raise ValueError("At least three unique subjects are required for train/validation/test splitting.")

    rng = np.random.default_rng(random_seed)
    shuffled = np.asarray(unique_rids, dtype=object)
    rng.shuffle(shuffled)

    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    test_count = max(1, int(round(len(shuffled) * test_fraction)))
    if validation_count + test_count >= len(shuffled):
        raise ValueError("Split fractions leave no subjects for training.")

    test_rids = sorted(str(value) for value in shuffled[:test_count])
    validation_rids = sorted(str(value) for value in shuffled[test_count : test_count + validation_count])
    train_rids = sorted(str(value) for value in shuffled[test_count + validation_count :])

    train_set = set(train_rids)
    validation_set = set(validation_rids)
    test_set = set(test_rids)

    train_indices = []
    validation_indices = []
    test_indices = []
    for index, pair in enumerate(pairs):
        rid = pair["RID"]
        if rid in test_set:
            test_indices.append(index)
        elif rid in validation_set:
            validation_indices.append(index)
        elif rid in train_set:
            train_indices.append(index)
        else:
            raise ValueError(f"RID {rid} was not assigned to a split.")

    return SubjectTrainValidationTestSplit(
        train_indices=np.asarray(train_indices, dtype=int),
        validation_indices=np.asarray(validation_indices, dtype=int),
        test_indices=np.asarray(test_indices, dtype=int),
        train_rids=train_rids,
        validation_rids=validation_rids,
        test_rids=test_rids,
    )


def compute_pair_metrics(
    pairs: list[dict[str, str]],
    baseline: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    split: SubjectSplit | SubjectTrainValidationTestSplit,
    model_name: str,
) -> list[dict[str, Any]]:
    split_by_index = split_labels_by_index(split)

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
    models = []
    for row in pair_metrics:
        if row["model"] not in models:
            models.append(row["model"])

    for model in models:
        model_rows = [row for row in pair_metrics if row["model"] == model]
        present_splits = {row["split"] for row in model_rows}
        ordered_splits = [split for split in ["train", "validation", "test"] if split in present_splits] + ["all"]
        for split in ordered_splits:
            split_rows = model_rows if split == "all" else [row for row in model_rows if row["split"] == split]
            for metric in metric_names:
                values = np.asarray([row[metric] for row in split_rows if row[metric] == row[metric]], dtype=float)
                if values.size == 0:
                    continue
                rows.append(
                    {
                        "model": model,
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


def compute_gaussian_likelihood_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    split: SubjectSplit | SubjectTrainValidationTestSplit,
    model_name: str,
    *,
    n_parameters: int,
    min_sigma: float = 1.0e-6,
) -> list[dict[str, Any]]:
    """Compute BIC and an ELPD-style Gaussian log predictive density.

    Sigma is estimated once from training residuals, then reused on test so the
    held-out score is genuinely out of sample.
    """

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted arrays must have the same shape.")
    if observed.ndim != 2:
        raise ValueError("observed and predicted arrays must be two-dimensional.")
    if n_parameters < 1:
        raise ValueError("n_parameters must be at least 1.")

    train_residuals = finite_residuals(observed[split.train_indices], predicted[split.train_indices])
    if train_residuals.size == 0:
        raise ValueError("Cannot estimate likelihood sigma with zero finite training residuals.")
    sigma_train = max(float(np.sqrt(np.mean(train_residuals**2))), float(min_sigma))

    split_indices: list[tuple[str, np.ndarray]] = [("train", split.train_indices)]
    if hasattr(split, "validation_indices"):
        split_indices.append(("validation", split.validation_indices))
    split_indices.append(("test", split.test_indices))
    all_indices = np.concatenate([indices for _, indices in split_indices if indices.size > 0])
    split_indices.append(("all", all_indices))

    rows: list[dict[str, Any]] = []
    for split_name, indices in split_indices:
        residuals = finite_residuals(observed[indices], predicted[indices])
        n_scalar = int(residuals.size)
        if n_scalar == 0:
            continue
        sse = float(np.sum(residuals**2))
        log_likelihood = gaussian_log_likelihood_from_sse(sse, n_scalar, sigma_train)
        bic = float(n_parameters * math.log(n_scalar) - 2.0 * log_likelihood)
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "n_pairs": int(indices.size),
                "n_scalar_observations": n_scalar,
                "n_parameters": int(n_parameters),
                "sigma_train": sigma_train,
                "sse": sse,
                "log_likelihood": log_likelihood,
                "elpd": log_likelihood,
                "elpd_per_observation": float(log_likelihood / n_scalar),
                "bic": bic,
            }
        )
    return rows


def finite_residuals(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    residuals = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    return residuals[np.isfinite(residuals)]


def gaussian_log_likelihood_from_sse(sse: float, n_observations: int, sigma: float) -> float:
    if n_observations < 1:
        raise ValueError("n_observations must be at least 1.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    variance = float(sigma) ** 2
    return float(-0.5 * (n_observations * math.log(2.0 * math.pi * variance) + float(sse) / variance))


def build_prediction_rows(
    dataset: ForecastDataset,
    predicted: np.ndarray,
    split: SubjectSplit | SubjectTrainValidationTestSplit,
    model_name: str,
) -> list[dict[str, Any]]:
    split_by_index = split_labels_by_index(split)

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


def split_labels_by_index(split: SubjectSplit | SubjectTrainValidationTestSplit) -> dict[int, str]:
    split_by_index = {int(index): "train" for index in split.train_indices}
    if hasattr(split, "validation_indices"):
        split_by_index.update({int(index): "validation" for index in split.validation_indices})
    split_by_index.update({int(index): "test" for index in split.test_indices})
    return split_by_index


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
