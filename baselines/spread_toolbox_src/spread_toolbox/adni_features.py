"""ADNI-derived covariates for feature-conditioned forecasting models."""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .forecasting import ForecastDataset, SubjectSplit, read_csv_rows
from .io_adni import resolve_project_path


def build_closure_covariates(
    dataset: ForecastDataset,
    split: SubjectSplit,
    config: dict[str, Any],
    project_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Build train-standardized biological covariates for closure models."""

    adni_dir = resolve_project_path(config["paths"]["adni_dir"], project_root)
    adni_files = config.get("adni_files", {})
    modeling = config.get("modeling", {})
    max_days = int(modeling.get("closure_covariate_max_days", 1095))
    report: dict[str, Any] = {"max_date_distance_days": max_days}

    pair_covariates: dict[str, np.ndarray] = {
        "apoe4_dose": standardize_pair_values(
            np.asarray([apoe4_dose(pair.get("apoe_genotype", "")) for pair in dataset.pairs], dtype=float),
            split.train_indices,
        ),
    }
    pair_report: dict[str, Any] = {"apoe4_dose": {"source": "cohort_forecast_pairs.apoe_genotype"}}

    plasma = build_nearest_pair_covariate(
        dataset,
        adni_dir / "Biospecimen_Results" / "UGOTPTAU181_06_18_20_03May2026.csv",
        value_column="PLASMAPTAU181",
        date_column="EXAMDATE",
        max_days=max_days,
    )
    if plasma is not None:
        pair_covariates["plasma_ptau181"] = standardize_pair_values(plasma, split.train_indices)
        pair_report["plasma_ptau181"] = {
            "source": "UGOTPTAU181_06_18_20_03May2026.csv",
            "note": "p-tau217 is not available in the current ADNI files; plasma p-tau181 is used instead.",
            **coverage_report(plasma[:, None]),
        }
    else:
        pair_report["plasma_ptau181"] = {"source": "not_available"}

    regional_covariates: dict[str, np.ndarray] = {}
    regional_report: dict[str, Any] = {}

    amyloid_path = adni_dir / str(adni_files.get("amyloid_analysis", ""))
    amyloid = build_regional_amyloid_covariate(dataset, amyloid_path, max_days=max_days)
    if amyloid is not None:
        regional_covariates["amyloid_suvr"] = train_standardized_regional_matrix(amyloid, split.train_indices)
        regional_report["amyloid_suvr"] = {
            "source": str(amyloid_path.relative_to(adni_dir)),
            "note": "Regional amyloid SUVR is used because regional centiloid columns are not provided.",
            **coverage_report(amyloid),
        }

    thickness, cortical_volume, mri_report = build_regional_mri_covariates(
        dataset,
        adni_dir,
        tuple(str(path) for path in adni_files.get("mri_structural", [])),
        max_days=max_days,
    )
    regional_report["mri_sources"] = mri_report
    if thickness is not None:
        regional_covariates["cortical_thickness"] = train_standardized_regional_matrix(thickness, split.train_indices)
        regional_report["cortical_thickness"] = coverage_report(thickness)
    if cortical_volume is not None:
        regional_covariates["cortical_volume"] = train_standardized_regional_matrix(cortical_volume, split.train_indices)
        regional_report["cortical_volume"] = coverage_report(cortical_volume)

    ahba = load_optional_ahba_eigenmaps(dataset, config, project_root)
    if ahba:
        for name, values in ahba.items():
            regional_covariates[name] = values
        regional_report["ahba"] = {
            "loaded_components": sorted(ahba),
            "source": str(modeling.get("closure_ahba_eigenmaps_file", "")),
        }
    else:
        regional_report["ahba"] = {"loaded_components": [], "source": "not_configured"}

    report["pair_covariates"] = pair_report
    report["regional_covariates"] = regional_report
    return pair_covariates, regional_covariates, report


def train_standardized_regional_matrix(matrix: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    finite = np.isfinite(values)
    train = values[train_indices]
    if np.any(np.isfinite(train)):
        region_fill = np.nanmedian(train, axis=0)
        global_fill = float(np.nanmedian(train))
    else:
        region_fill = np.zeros(values.shape[1], dtype=float)
        global_fill = 0.0
    region_fill = np.where(np.isfinite(region_fill), region_fill, global_fill)
    filled = np.where(finite, values, region_fill[None, :])
    train_filled = filled[train_indices]
    center = np.mean(train_filled, axis=0)
    scale = np.std(train_filled, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    return np.nan_to_num((filled - center[None, :]) / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0)


def standardize_pair_values(values: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    train = values[train_indices]
    fill = float(np.nanmedian(train)) if np.any(np.isfinite(train)) else 0.0
    filled = np.where(np.isfinite(values), values, fill)
    center = float(np.mean(filled[train_indices]))
    scale = float(np.std(filled[train_indices]))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    return np.nan_to_num((filled - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def coverage_report(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    return {
        "pair_coverage": float(np.mean(np.any(finite, axis=1))) if values.ndim == 2 and values.shape[0] else 0.0,
        "region_coverage": float(np.mean(np.any(finite, axis=0))) if values.ndim == 2 and values.shape[1] else 0.0,
        "cell_coverage": float(np.mean(finite)) if values.size else 0.0,
        "missing_cells": int(values.size - np.count_nonzero(finite)),
    }


def build_nearest_pair_covariate(
    dataset: ForecastDataset,
    path: Path,
    *,
    value_column: str,
    date_column: str,
    max_days: int,
) -> np.ndarray | None:
    if not path.exists():
        return None
    rows_by_rid = group_rows_by_rid(read_csv_rows(path), date_column)
    values = np.full(len(dataset.pairs), np.nan, dtype=float)
    for pair_index, pair in enumerate(dataset.pairs):
        row = nearest_dated_row(
            rows_by_rid.get(normalize_rid(pair.get("RID", "")), []),
            parse_date(pair.get("baseline_tau_date", "")),
            max_days=max_days,
        )
        if row is not None:
            values[pair_index] = parse_float(row.get(value_column, ""))
    return values


def build_regional_amyloid_covariate(
    dataset: ForecastDataset,
    amyloid_path: Path,
    *,
    max_days: int,
) -> np.ndarray | None:
    if not amyloid_path.exists():
        return None
    rows_by_rid = group_rows_by_rid(read_csv_rows(amyloid_path), "SCANDATE")
    amyloid = np.full_like(dataset.baseline, np.nan, dtype=float)
    for pair_index, pair in enumerate(dataset.pairs):
        row = nearest_dated_row(
            rows_by_rid.get(normalize_rid(pair.get("RID", "")), []),
            parse_date(pair.get("baseline_tau_date", "")),
            max_days=max_days,
        )
        if row is None:
            continue
        for region_index, column in enumerate(dataset.tau_columns):
            amyloid[pair_index, region_index] = parse_float(row.get(column, ""))
    return amyloid


def build_regional_mri_covariates(
    dataset: ForecastDataset,
    adni_dir: Path,
    structural_files: tuple[str, ...],
    *,
    max_days: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if not structural_files:
        return None, None, {"loaded_files": 0, "records": 0}

    region_keys = [aparc_key_from_tau_column(column) for column in dataset.tau_columns]
    records_by_rid: dict[str, list[dict[str, Any]]] = {}
    source_report: list[dict[str, Any]] = []
    for relative_path in structural_files:
        csv_path = adni_dir / relative_path
        dataset_name = structural_dataset_name(csv_path)
        rd_path = adni_dir / "ADNIMERGE2" / "man" / f"{dataset_name}.Rd"
        if not csv_path.exists() or not rd_path.exists():
            source_report.append({"file": relative_path, "loaded": False, "reason": "missing_csv_or_dictionary"})
            continue

        dictionary = parse_freesurfer_region_dictionary(rd_path)
        thickness_columns = [dictionary.get((*key, "TA"), "") for key in region_keys]
        volume_columns = [dictionary.get((*key, "CV"), "") for key in region_keys]
        usable_records = 0
        for row in read_csv_rows(csv_path):
            if str(row.get("OVERALLQC", "")).strip().lower() == "fail":
                continue
            exam_date = parse_date(row.get("EXAMDATE", ""))
            if exam_date is None:
                continue
            thickness = values_from_columns(row, thickness_columns)
            cortical_volume = values_from_columns(row, volume_columns)
            if not np.any(np.isfinite(thickness)) and not np.any(np.isfinite(cortical_volume)):
                continue
            records_by_rid.setdefault(normalize_rid(row.get("RID", "")), []).append(
                {
                    "date": exam_date,
                    "thickness": thickness,
                    "cortical_volume": cortical_volume,
                }
            )
            usable_records += 1
        source_report.append(
            {
                "file": relative_path,
                "loaded": True,
                "dictionary": str(rd_path.relative_to(adni_dir)),
                "usable_records": usable_records,
                "mapped_thickness_regions": int(sum(bool(column) for column in thickness_columns)),
                "mapped_volume_regions": int(sum(bool(column) for column in volume_columns)),
            }
        )

    for rid_records in records_by_rid.values():
        rid_records.sort(key=lambda record: record["date"])
    if not records_by_rid:
        return None, None, {"loaded_files": 0, "records": 0, "sources": source_report}

    thickness_matrix = np.full_like(dataset.baseline, np.nan, dtype=float)
    volume_matrix = np.full_like(dataset.baseline, np.nan, dtype=float)
    matched_pairs = 0
    for pair_index, pair in enumerate(dataset.pairs):
        record = nearest_dated_row(
            records_by_rid.get(normalize_rid(pair.get("RID", "")), []),
            parse_date(pair.get("baseline_tau_date", "")),
            max_days=max_days,
        )
        if record is None:
            continue
        thickness_matrix[pair_index] = record["thickness"]
        volume_matrix[pair_index] = record["cortical_volume"]
        matched_pairs += 1

    return (
        thickness_matrix,
        volume_matrix,
        {
            "loaded_files": sum(1 for source in source_report if source["loaded"]),
            "records": sum(len(records) for records in records_by_rid.values()),
            "matched_pairs": matched_pairs,
            "sources": source_report,
        },
    )


def load_optional_ahba_eigenmaps(
    dataset: ForecastDataset,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, np.ndarray]:
    configured = str(config.get("modeling", {}).get("closure_ahba_eigenmaps_file", "")).strip()
    if not configured:
        return {}
    path = resolve_project_path(configured, project_root)
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    by_label = {row.get("enigma_label", row.get("label", "")): row for row in rows}
    component_columns = [
        column
        for column in rows[0]
        if column.lower().startswith("ahba") or column.lower().startswith("pc")
    ]
    output: dict[str, np.ndarray] = {}
    for column in component_columns[:5]:
        regional = np.asarray([parse_float(by_label.get(label, {}).get(column, "")) for label in dataset.region_labels])
        if np.all(~np.isfinite(regional)):
            continue
        fill = float(np.nanmedian(regional))
        regional = np.where(np.isfinite(regional), regional, fill)
        scale = float(np.std(regional))
        if not np.isfinite(scale) or scale <= 1.0e-12:
            scale = 1.0
        z = (regional - float(np.mean(regional))) / scale
        output[f"ahba_{column}"] = np.broadcast_to(z[None, :], dataset.baseline.shape)
    return output


def group_rows_by_rid(rows: list[dict[str, str]], date_column: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row_date = parse_date(row.get(date_column, ""))
        if row_date is None:
            continue
        enriched = dict(row)
        enriched["date"] = row_date
        grouped.setdefault(normalize_rid(row.get("RID", "")), []).append(enriched)
    for rid_rows in grouped.values():
        rid_rows.sort(key=lambda row: row["date"])
    return grouped


def nearest_dated_row(
    rows: list[dict[str, Any]],
    target_date: date | None,
    *,
    max_days: int,
) -> dict[str, Any] | None:
    if target_date is None or not rows:
        return None
    best_row = None
    best_delta = math.inf
    for row in rows:
        row_date = row.get("date")
        if row_date is None:
            continue
        delta = abs((row_date - target_date).days)
        if delta < best_delta:
            best_row = row
            best_delta = delta
    if best_row is None or best_delta > max_days:
        return None
    return best_row


def parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y", "%d%b%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def normalize_rid(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def aparc_key_from_tau_column(column: str) -> tuple[str, str]:
    text = column.upper()
    if text.startswith("CTX_LH_"):
        return "left", normalize_region_name(text.removeprefix("CTX_LH_").removesuffix("_SUVR"))
    if text.startswith("CTX_RH_"):
        return "right", normalize_region_name(text.removeprefix("CTX_RH_").removesuffix("_SUVR"))
    raise ValueError(f"Cannot infer hemisphere from tau column: {column}")


def normalize_region_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def structural_dataset_name(path: Path) -> str:
    return path.stem.split("_")[0]


def parse_freesurfer_region_dictionary(path: Path) -> dict[tuple[str, str, str], str]:
    import re

    pattern = re.compile(r"\\strong\{(ST\d+)(TA|CV)\}.* of (Left|Right)([A-Za-z]+)")
    mapping: dict[tuple[str, str, str], str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        stem, measure, hemisphere_text, region_text = match.groups()
        hemisphere = "left" if hemisphere_text == "Left" else "right"
        mapping[(hemisphere, normalize_region_name(region_text), measure)] = f"{stem}{measure}"
    return mapping


def values_from_columns(row: dict[str, str], columns: list[str]) -> np.ndarray:
    return np.asarray([parse_float(row.get(column, "")) if column else float("nan") for column in columns], dtype=float)


def parse_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def apoe4_dose(genotype: str) -> float:
    if not genotype:
        return 0.0
    return float(sum(1 for allele in genotype.replace("|", "/").split("/") if allele.strip() == "4"))
