"""ADNI Study File loading and longitudinal tau cohort construction."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d")
IDENTIFIER_COLUMNS = (
    "LONIUID",
    "LONI_IMAGE",
    "IMAGEUID",
    "IMAGEUID_AP",
    "ImageUID",
    "image_id",
)
TAU_DATE_COLUMNS = ("SCANDATE", "EXAMDATE", "ScanDate", "StudyDate", "DATE")
QC_FAIL_VALUES = {"0", "-2", "fail", "failed", "f"}
DEFAULT_TAU_PASS_VALUES = {"2"}
TAU_FEATURE_EXCLUDE = {
    "rid",
    "ptid",
    "viscode",
    "viscode2",
    "examdate",
    "scandate",
    "scan_date",
    "tracer",
    "qc_flag",
    "loniuid",
    "imageuid",
    "update_stamp",
    "rundate",
    "siteid",
}


@dataclass
class CohortBuildResult:
    cohort_rows: list[dict[str, Any]]
    forecast_pair_rows: list[dict[str, Any]]
    tau_observation_rows: list[dict[str, Any]]
    row_counts: dict[str, Any]
    tau_feature_columns: list[str]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load config files. Install pyyaml>=6.0.") from exc

    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping at top level: {path}")
    return data


def resolve_project_path(path_value: str | Path, project_root: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (Path(project_root) / path).resolve()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except TimeoutError as exc:
        raise TimeoutError(
            f"Timed out reading {csv_path}. If this file is in OneDrive, mark the ADNI folder "
            "as available offline / Always Keep on This Device, then rerun the cohort build."
        ) from exc


def read_optional_csv(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    csv_path = Path(path)
    if not csv_path.exists() or not csv_path.is_file():
        return []
    return read_csv(csv_path)


def parse_date(value: Any) -> date | None:
    value = str(value or "").strip()
    if not value:
        return None
    value = value.split()[0]
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def strdate(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


def years_between(start: date, end: date) -> float:
    return (end - start).days / 365.25


def diag_label(code: Any) -> str:
    value = str(code or "").strip()
    return {"1": "CN", "2": "MCI", "3": "AD"}.get(value, value)


def apoe4_dose(genotype: Any) -> str:
    value = str(genotype or "").strip()
    if not value:
        return ""
    return str(value.count("4"))


def identifier_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for col in IDENTIFIER_COLUMNS:
        raw = str(row.get(col, "") or "").strip()
        if not raw:
            continue
        values.append(raw)
        if raw.startswith("I") and raw[1:].isdigit():
            values.append(raw[1:])
        elif raw.isdigit():
            values.append(f"I{raw}")
    return list(dict.fromkeys(values))


def index_rows_by_identifier(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        for value in identifier_values(row):
            indexed.setdefault(value, row)
    return indexed


def first_parseable_date(row: dict[str, Any]) -> date | None:
    for col in TAU_DATE_COLUMNS:
        parsed = parse_date(row.get(col))
        if parsed:
            return parsed
    return None


def tau_scan_date(row: dict[str, Any], tau_meta_by_uid: dict[str, dict[str, Any]]) -> date | None:
    direct = first_parseable_date(row)
    if direct:
        return direct
    for uid in identifier_values(row):
        meta_row = tau_meta_by_uid.get(uid)
        if meta_row:
            parsed = first_parseable_date(meta_row)
            if parsed:
                return parsed
    return None


def tau_passes_qc(row: dict[str, Any], pass_values: set[str]) -> bool:
    if "qc_flag" not in row or row.get("qc_flag") in (None, ""):
        return True
    return str(row.get("qc_flag", "")).strip() in pass_values


def amyloid_passes_qc(row: dict[str, Any]) -> bool:
    qc_value = str(row.get("qc_flag", "") or "").strip().lower()
    return qc_value not in QC_FAIL_VALUES


def nearest_row_by_date(rows: list[tuple[date, dict[str, Any]]], target: date) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda item: abs((item[0] - target).days))[1]


def infer_tau_feature_columns(rows: list[dict[str, Any]], sample_limit: int = 200) -> list[str]:
    if not rows:
        return []
    fieldnames = list(rows[0].keys())
    sample = rows[:sample_limit]
    feature_columns: list[str] = []
    for col in fieldnames:
        normalized = col.strip().lower()
        if normalized in TAU_FEATURE_EXCLUDE:
            continue
        numeric_count = sum(1 for row in sample if as_float(row.get(col)) is not None)
        if numeric_count:
            feature_columns.append(col)
    return feature_columns


def build_longitudinal_tau_cohort(config: dict[str, Any], project_root: str | Path) -> CohortBuildResult:
    paths = config.get("paths", {})
    adni_files = config.get("adni_files", {})
    cohort_config = config.get("cohort", {})

    adni_dir = resolve_project_path(paths["adni_dir"], project_root)
    row_counts: dict[str, Any] = {"adni_dir": str(adni_dir)}

    def required_file(key: str) -> Path:
        if key not in adni_files:
            raise KeyError(f"Missing adni_files.{key} in config")
        path = adni_dir / adni_files[key]
        if not path.exists():
            raise FileNotFoundError(f"Missing ADNI Study File for {key}: {path}")
        return path

    def optional_file(key: str) -> Path | None:
        value = adni_files.get(key)
        if not value:
            return None
        path = adni_dir / value
        return path if path.exists() else None

    tau_rows = read_csv(required_file("tau_analysis"))
    tau_meta_rows = read_optional_csv(required_file("tau_metadata"))
    tau_qc_rows = read_optional_csv(required_file("tau_qc"))
    roster_rows = read_optional_csv(optional_file("enrollment_roster"))
    amyloid_rows = read_optional_csv(required_file("amyloid_analysis"))
    diagnosis_rows = read_optional_csv(required_file("diagnosis"))
    demographics_rows = read_optional_csv(required_file("demographics"))
    apoe_rows = read_optional_csv(required_file("apoe"))

    row_counts.update(
        {
            "tau_analysis_rows": len(tau_rows),
            "tau_metadata_rows": len(tau_meta_rows),
            "tau_qc_rows": len(tau_qc_rows),
            "roster_rows": len(roster_rows),
            "amyloid_rows": len(amyloid_rows),
            "diagnosis_rows": len(diagnosis_rows),
            "demographics_rows": len(demographics_rows),
            "apoe_rows": len(apoe_rows),
        }
    )

    tau_feature_columns = infer_tau_feature_columns(tau_rows)
    row_counts["tau_feature_columns"] = len(tau_feature_columns)

    tau_meta_by_uid = index_rows_by_identifier(tau_meta_rows)

    tau_pass_values = {str(value) for value in cohort_config.get("tau_pass_values", DEFAULT_TAU_PASS_VALUES)}
    if cohort_config.get("allow_partial_tau_qc", False):
        tau_pass_values.add("1")

    usable_tau_rows: list[dict[str, Any]] = []
    missing_scan_date = 0
    failed_tau_qc = 0
    for row in tau_rows:
        scan_date = tau_scan_date(row, tau_meta_by_uid)
        if not scan_date:
            missing_scan_date += 1
            continue
        if not tau_passes_qc(row, tau_pass_values):
            failed_tau_qc += 1
            continue
        enriched = dict(row)
        enriched["_scan_date"] = scan_date
        usable_tau_rows.append(enriched)

    row_counts["tau_missing_scan_date"] = missing_scan_date
    row_counts["tau_failed_qc"] = failed_tau_qc
    row_counts["tau_usable_rows"] = len(usable_tau_rows)
    row_counts["tau_tracer_distribution"] = dict(Counter(str(row.get("TRACER", "") or "") for row in usable_tau_rows))

    rid_to_ptid = {str(row.get("RID", "")).strip(): str(row.get("PTID", "")).strip() for row in roster_rows}
    min_tau_timepoints = int(cohort_config.get("min_tau_timepoints", 2))
    require_same_tracer = bool(cohort_config.get("require_same_tracer", True))

    tau_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in usable_tau_rows:
        rid = str(row.get("RID", "") or "").strip()
        if not rid:
            continue
        tracer = str(row.get("TRACER", "") or "").strip() if require_same_tracer else "ANY"
        tau_groups[(rid, tracer)].append(row)

    row_counts["tau_groups"] = len(tau_groups)

    amyloid_by_rid: dict[str, list[date]] = defaultdict(list)
    for row in amyloid_rows:
        rid = str(row.get("RID", "") or "").strip()
        scan_date = first_parseable_date(row)
        if rid and scan_date and amyloid_passes_qc(row):
            amyloid_by_rid[rid].append(scan_date)

    dx_by_rid: dict[str, list[tuple[date, dict[str, Any]]]] = defaultdict(list)
    for row in diagnosis_rows:
        rid = str(row.get("RID", "") or "").strip()
        exam_date = first_parseable_date(row)
        if rid and exam_date:
            dx_by_rid[rid].append((exam_date, row))

    demographics_by_rid = first_row_by_rid(demographics_rows)
    apoe_by_rid = first_row_by_rid(apoe_rows)

    cohort_rows: list[dict[str, Any]] = []
    forecast_pair_rows: list[dict[str, Any]] = []
    tau_observation_rows: list[dict[str, Any]] = []
    skipped_short_groups = 0

    for (rid, tracer), rows in sorted(tau_groups.items(), key=lambda item: (int_or_text(item[0][0]), item[0][1])):
        by_date: dict[date, dict[str, Any]] = {}
        for row in rows:
            by_date.setdefault(row["_scan_date"], row)
        dates = sorted(by_date)
        if len(dates) < min_tau_timepoints:
            skipped_short_groups += 1
            continue

        selected_rows = [by_date[scan_date] for scan_date in dates]
        baseline = selected_rows[0]
        next_scan = selected_rows[1]
        last_scan = selected_rows[-1]
        baseline_date = dates[0]
        last_date = dates[-1]
        ptid = str(baseline.get("PTID", "") or "").strip() or rid_to_ptid.get(rid, "")

        dx = nearest_row_by_date(dx_by_rid.get(rid, []), baseline_date)
        demographics = demographics_by_rid.get(rid, {})
        apoe = apoe_by_rid.get(rid, {})
        amyloid_dates = sorted(set(amyloid_by_rid.get(rid, [])))

        cohort_row = {
            "RID": rid,
            "PTID": ptid,
            "TRACER": tracer,
            "tau_n": len(dates),
            "baseline_tau_date": baseline_date,
            "next_tau_date": dates[1],
            "last_tau_date": last_date,
            "tau_followup_dates": ";".join(strdate(value) for value in dates[1:]),
            "tau_duration_days": (last_date - baseline_date).days,
            "tau_duration_years": round(years_between(baseline_date, last_date), 6),
            "tau_loniuids": ";".join(str(by_date[d].get("LONIUID", "") or "") for d in dates),
            "tau_qc_status": f"qc_flag in {sorted(tau_pass_values)}",
            "tau_feature_count": len(tau_feature_columns),
            "baseline_meta_temporal_suvr": baseline.get("META_TEMPORAL_SUVR", ""),
            "next_meta_temporal_suvr": next_scan.get("META_TEMPORAL_SUVR", ""),
            "last_meta_temporal_suvr": last_scan.get("META_TEMPORAL_SUVR", ""),
            "annualized_meta_temporal_suvr_change_baseline_to_last": annualized_change(
                baseline.get("META_TEMPORAL_SUVR"), last_scan.get("META_TEMPORAL_SUVR"), baseline_date, last_date
            ),
            "dx_nearest_baseline": diag_label(dx.get("DIAGNOSIS", "") if dx else ""),
            "dx_nearest_baseline_date": dx.get("EXAMDATE", "") if dx else "",
            "sex_code": demographics.get("PTGENDER", ""),
            "education_years": demographics.get("PTEDUCAT", ""),
            "apoe_genotype": apoe.get("GENOTYPE", ""),
            "apoe4_dose": apoe4_dose(apoe.get("GENOTYPE", "")),
            "amyloid_scan_count": len(amyloid_dates),
            "amyloid_dates": ";".join(strdate(value) for value in amyloid_dates),
        }
        cohort_rows.append(cohort_row)

        target_dates = [("next", dates[1])]
        if last_date != dates[1]:
            target_dates.append(("last", last_date))
        for target_role, target_date in target_dates:
            target = by_date[target_date]
            forecast_pair_rows.append(
                {
                    "RID": rid,
                    "PTID": ptid,
                    "TRACER": tracer,
                    "target_role": target_role,
                    "baseline_tau_date": baseline_date,
                    "target_tau_date": target_date,
                    "target_time_days": (target_date - baseline_date).days,
                    "target_time_years": round(years_between(baseline_date, target_date), 6),
                    "baseline_loniuid": baseline.get("LONIUID", ""),
                    "target_loniuid": target.get("LONIUID", ""),
                    "baseline_meta_temporal_suvr": baseline.get("META_TEMPORAL_SUVR", ""),
                    "target_meta_temporal_suvr": target.get("META_TEMPORAL_SUVR", ""),
                    "annualized_meta_temporal_suvr_change": annualized_change(
                        baseline.get("META_TEMPORAL_SUVR"), target.get("META_TEMPORAL_SUVR"), baseline_date, target_date
                    ),
                    "dx_nearest_baseline": cohort_row["dx_nearest_baseline"],
                    "apoe_genotype": cohort_row["apoe_genotype"],
                    "amyloid_scan_count": cohort_row["amyloid_scan_count"],
                }
            )

        for sequence, scan_date in enumerate(dates, start=1):
            observation = dict(by_date[scan_date])
            observation.update(
                {
                    "computed_RID": rid,
                    "computed_PTID": ptid,
                    "computed_TRACER": tracer,
                    "computed_scan_date": scan_date,
                    "computed_tau_sequence": sequence,
                    "computed_is_baseline": sequence == 1,
                }
            )
            tau_observation_rows.append(observation)

    row_counts["tau_groups_skipped_too_short"] = skipped_short_groups
    row_counts["longitudinal_tau_groups"] = len(cohort_rows)
    row_counts["forecast_pairs"] = len(forecast_pair_rows)
    row_counts["selected_tau_observations"] = len(tau_observation_rows)

    return CohortBuildResult(
        cohort_rows=cohort_rows,
        forecast_pair_rows=forecast_pair_rows,
        tau_observation_rows=tau_observation_rows,
        row_counts=row_counts,
        tau_feature_columns=tau_feature_columns,
    )


def first_row_by_rid(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_rid: dict[str, dict[str, Any]] = {}
    for row in rows:
        rid = str(row.get("RID", "") or "").strip()
        if rid:
            by_rid.setdefault(rid, row)
    return by_rid


def int_or_text(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def annualized_change(start_value: Any, end_value: Any, start_date: date, end_date: date) -> str:
    start = as_float(start_value)
    end = as_float(end_value)
    if start is None or end is None or end_date <= start_date:
        return ""
    years = years_between(start_date, end_date)
    return f"{(end - start) / years:.8g}"


def write_cohort_outputs(
    result: CohortBuildResult,
    config: dict[str, Any],
    project_root: str | Path,
) -> dict[str, Path]:
    output_dir = resolve_project_path(config["paths"]["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = config.get("outputs", {})

    paths = {
        "cohort_table": output_dir / outputs.get("cohort_table", "cohort_longitudinal_tau.csv"),
        "forecast_pairs_table": output_dir / outputs.get("forecast_pairs_table", "cohort_forecast_pairs.csv"),
        "tau_observations_table": output_dir / outputs.get("tau_observations_table", "cohort_tau_observations.csv"),
        "row_counts": output_dir / outputs.get("row_counts", "cohort_row_counts.json"),
    }

    write_csv_rows(paths["cohort_table"], result.cohort_rows)
    write_csv_rows(paths["forecast_pairs_table"], result.forecast_pair_rows)
    write_csv_rows(paths["tau_observations_table"], result.tau_observation_rows)
    with paths["row_counts"].open("w", encoding="utf-8") as handle:
        json.dump(result.row_counts, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return paths


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = union_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: strdate(row.get(field, "")) for field in fieldnames})


def union_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames and not field.startswith("_"):
                fieldnames.append(field)
    return fieldnames
