"""Reporting and evaluation helpers for the dynamic BN-SCM prototype."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SubjectSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_rids: list[str]
    validation_rids: list[str]
    test_rids: list[str]

    def label_for_index(self, index: int) -> str:
        if index in set(int(i) for i in self.train_indices):
            return "train"
        if index in set(int(i) for i in self.validation_indices):
            return "validation"
        if index in set(int(i) for i in self.test_indices):
            return "test"
        return "unknown"

    def report(self) -> dict[str, Any]:
        return {
            "train_pairs": int(self.train_indices.size),
            "validation_pairs": int(self.validation_indices.size),
            "test_pairs": int(self.test_indices.size),
            "train_subjects": len(self.train_rids),
            "validation_subjects": len(self.validation_rids),
            "test_subjects": len(self.test_rids),
        }


def make_subject_split(
    metadata_rows: list[dict[str, Any]],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    random_seed: int = 20260519,
) -> SubjectSplit:
    if validation_fraction <= 0.0 or test_fraction <= 0.0 or validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation/test fractions must be positive and sum to less than one.")
    unique_rids = sorted({str(row["RID"]) for row in metadata_rows}, key=lambda value: int(value) if value.isdigit() else value)
    if len(unique_rids) < 3:
        raise ValueError("At least three subjects are required for subject-level splitting.")
    rng = np.random.default_rng(int(random_seed))
    shuffled = np.asarray(unique_rids, dtype=object)
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * float(test_fraction))))
    validation_count = max(1, int(round(len(shuffled) * float(validation_fraction))))
    if test_count + validation_count >= len(shuffled):
        raise ValueError("Split fractions leave no training subjects.")
    test_rids = sorted(str(value) for value in shuffled[:test_count])
    validation_rids = sorted(str(value) for value in shuffled[test_count : test_count + validation_count])
    train_rids = sorted(str(value) for value in shuffled[test_count + validation_count :])
    train_set = set(train_rids)
    validation_set = set(validation_rids)
    test_set = set(test_rids)
    train_indices = []
    validation_indices = []
    test_indices = []
    for idx, row in enumerate(metadata_rows):
        rid = str(row["RID"])
        if rid in test_set:
            test_indices.append(idx)
        elif rid in validation_set:
            validation_indices.append(idx)
        elif rid in train_set:
            train_indices.append(idx)
        else:
            raise ValueError(f"Unassigned RID in split: {rid}")
    return SubjectSplit(
        train_indices=np.asarray(train_indices, dtype=int),
        validation_indices=np.asarray(validation_indices, dtype=int),
        test_indices=np.asarray(test_indices, dtype=int),
        train_rids=train_rids,
        validation_rids=validation_rids,
        test_rids=test_rids,
    )


def compute_rate_metrics(
    observed_rates: np.ndarray,
    predicted_rates: np.ndarray,
    target_names: list[str],
    split: SubjectSplit,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_items = [
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
        ("all", np.arange(observed_rates.shape[0], dtype=int)),
    ]
    for split_name, indices in split_items:
        for target_idx, target_name in enumerate(target_names):
            y = observed_rates[indices, target_idx]
            pred = predicted_rates[indices, target_idx]
            mask = np.isfinite(y) & np.isfinite(pred)
            if int(np.sum(mask)) == 0:
                rows.append(empty_metric_row(split_name, target_name))
                continue
            residual = pred[mask] - y[mask]
            rows.append(
                {
                    "split": split_name,
                    "target": target_name,
                    "n": int(np.sum(mask)),
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "pearson": safe_correlation(y[mask], pred[mask], rank=False),
                    "spearman": safe_correlation(y[mask], pred[mask], rank=True),
                    "observed_mean": float(np.mean(y[mask])),
                    "predicted_mean": float(np.mean(pred[mask])),
                }
            )
    return rows


def summarize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split_name in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split_name]
        for metric in ("mae", "rmse", "pearson", "spearman"):
            values = np.asarray([float(row[metric]) for row in split_rows if np.isfinite(float(row[metric]))], dtype=float)
            output.append(
                {
                    "split": split_name,
                    "metric": metric,
                    "n_targets": int(values.size),
                    "mean": float(np.mean(values)) if values.size else float("nan"),
                    "median": float(np.median(values)) if values.size else float("nan"),
                }
            )
    return output


def h1_decoupling_summary(edge_rows: list[dict[str, Any]], *, parent: str = "plasma_pt217") -> dict[str, Any]:
    tau_rows = [
        row
        for row in edge_rows
        if row.get("parent") == parent and str(row.get("target", "")).startswith("tau_rate:")
    ]
    if not tau_rows:
        return {"tested": False, "reason": f"No {parent} -> tau_rate edge rows."}
    summaries = []
    for row in tau_rows:
        early = float(row["effect_at_z_min"])
        mid = float(row["effect_at_z_mid"])
        late = float(row["effect_at_z_max"])
        summaries.append(
            {
                "target": row["target"],
                "early_effect": early,
                "mid_effect": mid,
                "late_effect": late,
                "late_abs_less_than_early_abs": abs(late) < abs(early),
                "included_by_threshold": bool(row["included_by_effect_threshold"]),
            }
        )
    decoupling_fraction = float(np.mean([item["late_abs_less_than_early_abs"] for item in summaries]))
    return {
        "tested": True,
        "parent": parent,
        "tau_target_count": len(summaries),
        "late_abs_less_than_early_abs_fraction": decoupling_fraction,
        "interpretation": (
            "Prototype descriptive test only: full posterior thresholding requires Bayesian edge sampling."
        ),
        "targets": summaries,
    }


def empty_metric_row(split_name: str, target_name: str) -> dict[str, Any]:
    return {
        "split": split_name,
        "target": target_name,
        "n": 0,
        "mae": float("nan"),
        "rmse": float("nan"),
        "pearson": float("nan"),
        "spearman": float("nan"),
        "observed_mean": float("nan"),
        "predicted_mean": float("nan"),
    }


def safe_correlation(a: np.ndarray, b: np.ndarray, *, rank: bool) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if rank:
        x = rankdata(x)
        y = rankdata(y)
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_report(report: dict[str, Any]) -> str:
    split = report["split"]
    best = report["metric_summary"].get("validation_tau_rate_mae", {})
    lines = [
        "# Dynamic BN-SCM Report",
        "",
        "## Scope",
        "",
        "This is the revised baseline-to-rate BN-SCM prototype, separate from the existing BN-LTE code.",
        "It uses train-only explainable pseudotime and constrained baseline parents for future biomarker rates.",
        "",
        "## Data",
        "",
        f"- Usable pairs: {report['data']['rows']['usable_pairs']}",
        f"- Unique subjects: {report['data']['rows']['unique_subjects']}",
        f"- Features: {report['data']['feature_count']}; targets: {report['data']['target_count']}",
        f"- Selected tau regions: {', '.join(report['data']['selected_tau_regions'])}",
        "",
        "## Split",
        "",
        f"- Train: {split['train_pairs']} pairs / {split['train_subjects']} subjects",
        f"- Validation: {split['validation_pairs']} pairs / {split['validation_subjects']} subjects",
        f"- Test: {split['test_pairs']} pairs / {split['test_subjects']} subjects",
        "",
        "## Pseudotime",
        "",
        f"- Mode: {report['pseudotime']['mode']}",
        f"- Selected features: {report['pseudotime']['selected_feature_count']}",
        f"- Burden correlation: {format_float(report['pseudotime']['burden_correlation'])}",
        f"- Explained variance ratio: {format_float(report['pseudotime']['explained_variance_ratio'])}",
        "",
        "Diagnosis ordering:",
        "",
    ]
    for label, values in report["pseudotime"].get("diagnosis_ordering", {}).items():
        lines.append(f"- {label}: n={values['n']}, median Z={format_float(values['median_z'])}")
    lines.extend(
        [
            "",
            "## Model",
            "",
            f"- Targets fit: {report['model']['target_count']}",
            f"- Edge effect threshold: {report['model']['edge_effect_threshold']}",
            f"- Validation tau-rate MAE median: {format_float(best.get('median', float('nan')))}",
            "",
            "Top edge effects:",
            "",
        ]
    )
    for row in report.get("top_edges", [])[:12]:
        lines.append(
            f"- {row['parent']} -> {row['target']}: max_abs={format_float(row['max_abs_effect'])}, "
            f"z_max={format_float(row['z_at_max_abs_effect'])}"
        )
    h1 = report.get("h1_decoupling", {})
    lines.extend(["", "## H1 pT217 Decoupling", ""])
    if h1.get("tested"):
        lines.append(
            f"- Tau targets tested: {h1['tau_target_count']}; late<early absolute effect fraction: "
            f"{format_float(h1['late_abs_less_than_early_abs_fraction'])}"
        )
        lines.append("- This is descriptive until full posterior edge sampling is added.")
    else:
        lines.append(f"- Not tested: {h1.get('reason', 'unknown reason')}")
    lines.append("")
    return "\n".join(lines)


def compact_metric_summary(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tau_validation = [
        row
        for row in metric_rows
        if row["split"] == "validation" and str(row["target"]).startswith("tau_rate:") and np.isfinite(float(row["mae"]))
    ]
    tau_test = [
        row
        for row in metric_rows
        if row["split"] == "test" and str(row["target"]).startswith("tau_rate:") and np.isfinite(float(row["mae"]))
    ]
    return {
        "validation_tau_rate_mae": summarize_target_metric(tau_validation, "mae"),
        "test_tau_rate_mae": summarize_target_metric(tau_test, "mae"),
        "validation_tau_rate_spearman": summarize_target_metric(tau_validation, "spearman"),
        "test_tau_rate_spearman": summarize_target_metric(tau_test, "spearman"),
    }


def summarize_target_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, float | int]:
    values = np.asarray([float(row[metric]) for row in rows if np.isfinite(float(row[metric]))], dtype=float)
    return {
        "n_targets": int(values.size),
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "median": float(np.median(values)) if values.size else float("nan"),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.4f}"
