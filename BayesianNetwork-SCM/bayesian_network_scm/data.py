"""Multimodal ADNI baseline-to-rate dataset assembly."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .constraints import VariableSpec, default_variable_specs, infer_layer, infer_role


DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y", "%d%b%Y")
QC_FAIL_VALUES = {"0", "-2", "fail", "failed", "f"}
DEFAULT_SELECTED_REGIONS = (
    "L_entorhinal",
    "R_entorhinal",
    "L_fusiform",
    "R_fusiform",
    "L_inferiortemporal",
    "R_inferiortemporal",
    "L_middletemporal",
    "R_middletemporal",
    "L_inferiorparietal",
    "R_inferiorparietal",
)
DEFAULT_MRI_FILES = (
    "MR_Image_Analysis/UCSFFSX7_03May2026.csv",
    "MR_Image_Analysis/UCSFFSX6_03May2026.csv",
    "MR_Image_Analysis/UCSFFSX51_11_08_19_03May2026.csv",
    "MR_Image_Analysis/UCSFFSL51_03_01_22_03May2026.csv",
)


@dataclass
class MultimodalPairDataset:
    """Baseline predictors and annualized rate targets for dynamic SCM fitting."""

    metadata_rows: list[dict[str, Any]]
    feature_names: list[str]
    feature_matrix: np.ndarray
    target_names: list[str]
    target_baseline: np.ndarray
    target_observed: np.ndarray
    target_rates: np.ndarray
    time_years: np.ndarray
    variable_specs: list[VariableSpec]
    report: dict[str, Any]
    target_time_years: np.ndarray | None = None

    @property
    def pair_count(self) -> int:
        return len(self.metadata_rows)

    @property
    def feature_layers(self) -> dict[str, str]:
        return {spec.name: spec.layer for spec in self.variable_specs}

    def feature_index(self, name: str) -> int:
        return self.feature_names.index(name)

    def target_index(self, name: str) -> int:
        return self.target_names.index(name)


def build_multimodal_pair_dataset(
    project_root: str | Path,
    *,
    output_dir: str | Path = "experiments/group_average_enigma/output",
    adni_dir: str | Path = "BRAIN DATA/ADNI",
    mapping_file: str | Path = "experiments/group_average_enigma/adni_to_enigma_aparc_mapping.csv",
    selected_tau_regions: Iterable[str] = DEFAULT_SELECTED_REGIONS,
    max_date_distance_days: int = 1095,
) -> MultimodalPairDataset:
    """Build a leakage-controlled ADNI pair table from existing cohort outputs."""

    root = Path(project_root).resolve()
    output_path = resolve_path(output_dir, root)
    adni_path = resolve_path(adni_dir, root)
    mapping_path = resolve_path(mapping_file, root)

    pairs = read_csv_rows(output_path / "cohort_forecast_pairs.csv")
    observations = {row["LONIUID"]: row for row in read_csv_rows(output_path / "cohort_tau_observations.csv")}
    cohort_by_rid = {normalize_rid(row.get("RID", "")): row for row in read_csv_rows(output_path / "cohort_longitudinal_tau.csv")}
    mapping_rows = read_csv_rows(mapping_path)
    region_to_column = {row["enigma_label"]: row["adni_tau_column"] for row in mapping_rows}
    selected_regions = [region for region in selected_tau_regions if region in region_to_column]

    demographics_by_rid = first_row_by_rid(read_csv_rows(adni_path / "Subject_Demographics" / "PTDEMOG_03May2026.csv"))
    plasma_by_rid = group_rows_by_rid(
        read_csv_rows(adni_path / "Biospecimen_Results" / "UPENN_PLASMA_FUJIREBIO_QUANTERIX_03May2026.csv"),
        "EXAMDATE",
    )
    amyloid_by_rid = group_rows_by_rid(
        [row for row in read_csv_rows(adni_path / "PET_Image_Analysis" / "UCBERKELEY_AMY_6MM_03May2026.csv") if amyloid_passes_qc(row)],
        "SCANDATE",
    )
    cognitive_sources = load_cognitive_sources(adni_path)
    mri_by_rid, mri_report = load_mri_summaries(adni_path, DEFAULT_MRI_FILES)

    metadata_rows: list[dict[str, Any]] = []
    feature_rows: list[list[float]] = []
    target_baseline_rows: list[list[float]] = []
    target_observed_rows: list[list[float]] = []
    target_rate_rows: list[list[float]] = []

    feature_names = [
        "age_years",
        "sex_female",
        "education_years",
        "apoe4_dose",
        "plasma_pt217",
        "plasma_ab42_ab40",
        "plasma_nfl",
        "plasma_gfap",
        "amyloid_summary_suvr",
        "amyloid_centiloids",
        "amyloid_positive",
        "tau_meta_temporal",
        "mri_hippocampus_volume",
        "mri_amygdala_volume",
        "mri_temporal_cortical_volume",
        "mri_temporal_cortical_thickness",
        "adas13",
        "mmse",
        "ravlt_immediate",
        "cdrsb",
    ]
    for region in selected_regions:
        feature_names.append(f"tau_region:{region}")

    target_names = ["tau_rate:meta_temporal"]
    target_names.extend(f"tau_rate:{region}" for region in selected_regions)
    target_names.extend(["cognitive_rate:adas13", "cognitive_rate:mmse", "cognitive_rate:ravlt_immediate", "cognitive_rate:cdrsb"])

    for pair in pairs:
        baseline = observations.get(pair.get("baseline_loniuid", ""))
        target = observations.get(pair.get("target_loniuid", ""))
        if baseline is None or target is None:
            continue
        baseline_date = parse_date(pair.get("baseline_tau_date"))
        target_date = parse_date(pair.get("target_tau_date"))
        dt = parse_float(pair.get("target_time_years"))
        if baseline_date is None or target_date is None or not np.isfinite(dt) or dt <= 0.0:
            continue

        rid = normalize_rid(pair.get("RID", ""))
        cohort_row = cohort_by_rid.get(rid, {})
        demographics = demographics_by_rid.get(rid, {})
        plasma = nearest_dated_row(plasma_by_rid.get(rid, []), baseline_date, max_days=max_date_distance_days)
        amyloid = nearest_dated_row(amyloid_by_rid.get(rid, []), baseline_date, max_days=max_date_distance_days)
        mri = nearest_dated_row(mri_by_rid.get(rid, []), baseline_date, max_days=max_date_distance_days)
        cognitive_baseline = nearest_cognitive_values(cognitive_sources, rid, baseline_date, max_date_distance_days)
        cognitive_target = nearest_cognitive_values(cognitive_sources, rid, target_date, max_date_distance_days)

        baseline_tau_values = target_tau_values = None
        baseline_tau_values = [parse_float(baseline.get(region_to_column[region])) for region in selected_regions]
        target_tau_values = [parse_float(target.get(region_to_column[region])) for region in selected_regions]
        baseline_meta = parse_float(baseline.get("META_TEMPORAL_SUVR"))
        target_meta = parse_float(target.get("META_TEMPORAL_SUVR"))
        if not np.isfinite(baseline_meta) or not np.isfinite(target_meta):
            continue

        age_years = age_at_date(demographics.get("PTDOBYY") or demographics.get("PTDOB"), baseline_date)
        sex_female = sex_to_female(demographics.get("PTGENDER") or cohort_row.get("sex_code"))
        education = parse_float(demographics.get("PTEDUCAT") or cohort_row.get("education_years"))
        apoe4 = apoe4_dose(pair.get("apoe_genotype") or cohort_row.get("apoe_genotype"))

        feature_row = [
            age_years,
            sex_female,
            education,
            apoe4,
            parse_float(plasma.get("pT217_F") if plasma else ""),
            parse_float(plasma.get("AB42_AB40_F") if plasma else ""),
            parse_float(plasma.get("NfL_Q") if plasma else ""),
            parse_float(plasma.get("GFAP_Q") if plasma else ""),
            parse_float(amyloid.get("SUMMARY_SUVR") if amyloid else ""),
            parse_float(amyloid.get("CENTILOIDS") if amyloid else ""),
            amyloid_status_to_float(amyloid.get("AMYLOID_STATUS") if amyloid else ""),
            baseline_meta,
            parse_float(mri.get("hippocampus_volume") if mri else ""),
            parse_float(mri.get("amygdala_volume") if mri else ""),
            parse_float(mri.get("temporal_cortical_volume") if mri else ""),
            parse_float(mri.get("temporal_cortical_thickness") if mri else ""),
            cognitive_baseline.get("adas13", float("nan")),
            cognitive_baseline.get("mmse", float("nan")),
            cognitive_baseline.get("ravlt_immediate", float("nan")),
            cognitive_baseline.get("cdrsb", float("nan")),
        ]
        feature_row.extend(baseline_tau_values)

        target_baseline_row = [baseline_meta] + baseline_tau_values
        target_observed_row = [target_meta] + target_tau_values
        target_rate_row = [(target_observed_row[idx] - target_baseline_row[idx]) / dt for idx in range(len(target_baseline_row))]
        for name in ("adas13", "mmse", "ravlt_immediate", "cdrsb"):
            b_value = cognitive_baseline.get(name, float("nan"))
            t_value = cognitive_target.get(name, float("nan"))
            target_baseline_row.append(b_value)
            target_observed_row.append(t_value)
            target_rate_row.append((t_value - b_value) / dt if np.isfinite(b_value) and np.isfinite(t_value) else float("nan"))

        metadata_rows.append(
            {
                "RID": rid,
                "PTID": pair.get("PTID", ""),
                "TRACER": pair.get("TRACER", ""),
                "target_role": pair.get("target_role", ""),
                "baseline_tau_date": baseline_date.isoformat(),
                "target_tau_date": target_date.isoformat(),
                "target_time_years": float(dt),
                "dx_nearest_baseline": pair.get("dx_nearest_baseline", cohort_row.get("dx_nearest_baseline", "")),
                "amyloid_status": amyloid.get("AMYLOID_STATUS", "") if amyloid else "",
            }
        )
        feature_rows.append(feature_row)
        target_baseline_rows.append(target_baseline_row)
        target_observed_rows.append(target_observed_row)
        target_rate_rows.append(target_rate_row)

    feature_matrix = np.asarray(feature_rows, dtype=float)
    target_baseline = np.asarray(target_baseline_rows, dtype=float)
    target_observed = np.asarray(target_observed_rows, dtype=float)
    target_rates = np.asarray(target_rate_rows, dtype=float)
    time_years = np.asarray([row["target_time_years"] for row in metadata_rows], dtype=float)

    variable_specs = default_variable_specs(feature_names, target_names)
    report = {
        "source": {
            "output_dir": str(output_path),
            "adni_dir": str(adni_path),
            "mapping_file": str(mapping_path),
            "max_date_distance_days": int(max_date_distance_days),
        },
        "rows": {
            "input_pairs": len(pairs),
            "usable_pairs": len(metadata_rows),
            "unique_subjects": len({row["RID"] for row in metadata_rows}),
        },
        "selected_tau_regions": selected_regions,
        "feature_coverage": matrix_coverage(feature_matrix, feature_names),
        "target_coverage": matrix_coverage(target_rates, target_names),
        "mri": mri_report,
        "notes": [
            "Predictors are nearest-to-baseline values only.",
            "Targets are annualized baseline-to-target rates.",
            "MRI volume summaries are raw FreeSurfer volumes; ICV normalization is not applied unless an ICV source is added.",
        ],
    }

    return MultimodalPairDataset(
        metadata_rows=metadata_rows,
        feature_names=feature_names,
        feature_matrix=feature_matrix,
        target_names=target_names,
        target_baseline=target_baseline,
        target_observed=target_observed,
        target_rates=target_rates,
        time_years=time_years,
        variable_specs=variable_specs,
        report=report,
    )


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve_path(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def first_row_by_rid(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        rid = normalize_rid(row.get("RID", ""))
        if rid:
            output.setdefault(rid, row)
    return output


def group_rows_by_rid(rows: Iterable[dict[str, str]], date_column: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_date = parse_date(row.get(date_column))
        rid = normalize_rid(row.get("RID", ""))
        if rid and row_date is not None:
            enriched = dict(row)
            enriched["date"] = row_date
            grouped[rid].append(enriched)
    for rid_rows in grouped.values():
        rid_rows.sort(key=lambda item: item["date"])
    return dict(grouped)


def nearest_dated_row(rows: list[dict[str, Any]], target_date: date | None, *, max_days: int) -> dict[str, Any] | None:
    if target_date is None or not rows:
        return None
    best = min(rows, key=lambda row: abs((row["date"] - target_date).days))
    if abs((best["date"] - target_date).days) > max_days:
        return None
    return best


def parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    text = text.split()[0]
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def parse_float(value: Any) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def normalize_rid(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def apoe4_dose(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return float("nan")
    alleles = [token.strip() for token in text.replace("|", "/").split("/") if token.strip()]
    if not alleles:
        return float("nan")
    return float(sum(allele == "4" for allele in alleles))


def sex_to_female(value: Any) -> float:
    text = str(value or "").strip().lower()
    if not text:
        return float("nan")
    if text in {"2", "female", "f"}:
        return 1.0
    if text in {"1", "male", "m"}:
        return 0.0
    return float("nan")


def age_at_date(dob_value: Any, target_date: date) -> float:
    dob = parse_date(dob_value)
    if dob is None:
        return float("nan")
    return (target_date - dob).days / 365.25


def amyloid_status_to_float(value: Any) -> float:
    text = str(value or "").strip().lower()
    if not text:
        return float("nan")
    if text in {"1", "positive", "pos", "a+", "yes"} or "pos" in text:
        return 1.0
    if text in {"0", "negative", "neg", "a-", "no"} or "neg" in text:
        return 0.0
    return parse_float(value)


def amyloid_passes_qc(row: dict[str, str]) -> bool:
    qc_value = str(row.get("qc_flag", "") or "").strip().lower()
    return qc_value not in QC_FAIL_VALUES


def load_cognitive_sources(adni_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    sources = {
        "adas13": (
            adni_dir / "Neuropsychological" / "ADAS_03May2026.csv",
            "VISDATE",
            lambda row: parse_float(row.get("TOTAL13")),
        ),
        "mmse": (
            adni_dir / "Neuropsychological" / "MMSE_03May2026.csv",
            "VISDATE",
            lambda row: parse_float(row.get("MMSCORE")),
        ),
        "ravlt_immediate": (
            adni_dir / "Neuropsychological" / "NEUROBAT_03May2026.csv",
            "VISDATE",
            ravlt_immediate,
        ),
        "cdrsb": (
            adni_dir / "Neuropsychological" / "CDR_03May2026.csv",
            "VISDATE",
            lambda row: parse_float(row.get("CDRSB")),
        ),
    }
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, (path, date_column, value_fn) in sources.items():
        rows = []
        if path.exists():
            for row in read_csv_rows(path):
                value = value_fn(row)
                row_date = parse_date(row.get(date_column))
                rid = normalize_rid(row.get("RID", ""))
                if rid and row_date is not None and np.isfinite(value):
                    rows.append({"RID": rid, "date": row_date, "value": float(value)})
        output[name] = group_value_rows(rows)
    return output


def group_value_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["RID"]].append(row)
    for rid_rows in grouped.values():
        rid_rows.sort(key=lambda item: item["date"])
    return dict(grouped)


def ravlt_immediate(row: dict[str, str]) -> float:
    values = [parse_float(row.get(f"AVTOT{idx}")) for idx in range(1, 6)]
    if all(np.isfinite(value) for value in values):
        return float(sum(values))
    return float("nan")


def nearest_cognitive_values(
    sources: dict[str, dict[str, list[dict[str, Any]]]],
    rid: str,
    target_date: date,
    max_days: int,
) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, by_rid in sources.items():
        row = nearest_dated_row(by_rid.get(rid, []), target_date, max_days=max_days)
        output[name] = parse_float(row.get("value") if row else "")
    return output


def load_mri_summaries(adni_dir: Path, structural_files: Iterable[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records_by_rid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_report = []
    for relative in structural_files:
        path = adni_dir / relative
        dictionary_path = adni_dir / "ADNIMERGE2" / "man" / f"{path.stem.split('_')[0]}.Rd"
        if not path.exists() or not dictionary_path.exists():
            source_report.append({"file": relative, "loaded": False, "reason": "missing_csv_or_dictionary"})
            continue
        dictionary = parse_freesurfer_dictionary(dictionary_path)
        column_sets = mri_column_sets(dictionary)
        usable = 0
        for row in read_csv_rows(path):
            if str(row.get("OVERALLQC", "")).strip().lower() == "fail":
                continue
            row_date = parse_date(row.get("EXAMDATE"))
            rid = normalize_rid(row.get("RID", ""))
            if not rid or row_date is None:
                continue
            summary = summarize_mri_row(row, column_sets)
            if not any(np.isfinite(value) for value in summary.values()):
                continue
            records_by_rid[rid].append({"date": row_date, **summary})
            usable += 1
        source_report.append({"file": relative, "loaded": True, "usable_records": usable, **{k: len(v) for k, v in column_sets.items()}})
    for rid_rows in records_by_rid.values():
        rid_rows.sort(key=lambda item: item["date"])
    return dict(records_by_rid), {"sources": source_report, "subjects": len(records_by_rid)}


def parse_freesurfer_dictionary(path: Path) -> dict[tuple[str, str, str], str]:
    pattern = re.compile(r"\\strong\{(ST\d+)(TA|CV|SV)\}.* of (Left|Right)([A-Za-z]+)")
    mapping: dict[tuple[str, str, str], str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        stem, measure, hemisphere_text, region_text = match.groups()
        hemisphere = "left" if hemisphere_text == "Left" else "right"
        mapping[(hemisphere, normalize_region(region_text), measure)] = f"{stem}{measure}"
    return mapping


def mri_column_sets(dictionary: dict[tuple[str, str, str], str]) -> dict[str, list[str]]:
    temporal_regions = ("entorhinal", "fusiform", "inferiortemporal", "middletemporal")
    hemispheres = ("left", "right")
    return {
        "hippocampus_volume": [dictionary.get((hemi, "hippocampus", "SV"), "") for hemi in hemispheres],
        "amygdala_volume": [dictionary.get((hemi, "amygdala", "SV"), "") for hemi in hemispheres],
        "temporal_cortical_volume": [
            dictionary.get((hemi, region, "CV"), "") for hemi in hemispheres for region in temporal_regions
        ],
        "temporal_cortical_thickness": [
            dictionary.get((hemi, region, "TA"), "") for hemi in hemispheres for region in temporal_regions
        ],
    }


def summarize_mri_row(row: dict[str, str], column_sets: dict[str, list[str]]) -> dict[str, float]:
    output = {}
    for name, columns in column_sets.items():
        values = [parse_float(row.get(column)) for column in columns if column]
        finite = [value for value in values if np.isfinite(value)]
        output[name] = float(np.mean(finite)) if finite else float("nan")
    return output


def normalize_region(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def matrix_coverage(matrix: np.ndarray, names: list[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if matrix.size == 0:
        return output
    finite = np.isfinite(matrix)
    for idx, name in enumerate(names):
        output[name] = {
            "finite_rows": int(np.count_nonzero(finite[:, idx])),
            "total_rows": int(matrix.shape[0]),
            "coverage": float(np.mean(finite[:, idx])),
            "layer": infer_layer(name),
            "role": infer_role(name),
        }
    return output
