"""A/T/N rate-target dataset assembly for causal-ordering experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .constraints import default_variable_specs
from .data import (
    DEFAULT_SELECTED_REGIONS,
    MultimodalPairDataset,
    age_at_date,
    amyloid_passes_qc,
    amyloid_status_to_float,
    apoe4_dose,
    first_row_by_rid,
    group_rows_by_rid,
    load_cognitive_sources,
    matrix_coverage,
    nearest_cognitive_values,
    nearest_dated_row,
    normalize_rid,
    parse_date,
    parse_float,
    parse_freesurfer_dictionary,
    read_csv_rows,
    resolve_path,
    sex_to_female,
)


ATN_PRIMARY_TOLERANCE_DAYS = 365

AMYLOID_REGION_COLUMNS = {
    "entorhinal": "CTX_ENTORHINAL_SUVR",
    "fusiform": "CTX_FUSIFORM_SUVR",
    "inferiortemporal": "CTX_INFERIORTEMPORAL_SUVR",
    "middletemporal": "CTX_MIDDLETEMPORAL_SUVR",
    "inferiorparietal": "CTX_INFERIORPARIETAL_SUVR",
}

PICSL_VOLUME_COLUMNS = {
    "left_hippocampus": "LEFT_HIPP_VOL",
    "right_hippocampus": "RIGHT_HIPP_VOL",
    "left_erc": "LEFT_ERC_VOL",
    "right_erc": "RIGHT_ERC_VOL",
    "left_ba35": "LEFT_BA35_VOL",
    "right_ba35": "RIGHT_BA35_VOL",
    "left_ba36": "LEFT_BA36_VOL",
    "right_ba36": "RIGHT_BA36_VOL",
    "left_phc": "LEFT_PHC_VOL",
    "right_phc": "RIGHT_PHC_VOL",
}

FOXLAB_COLUMNS = {
    "brain_volume": "BRAINVOL",
    "ventricle_volume": "VENTVOL",
}


@dataclass(frozen=True)
class DatedPair:
    baseline: dict[str, Any] | None
    followup: dict[str, Any] | None
    baseline_days_from_tau: float
    followup_days_from_tau: float
    interval_years: float

    @property
    def usable(self) -> bool:
        return self.baseline is not None and self.followup is not None and self.interval_years > 0.0


def build_atn_rate_dataset(
    project_root: str | Path,
    *,
    output_dir: str | Path = "experiments/group_average_enigma/output",
    adni_dir: str | Path = "BRAIN DATA/ADNI",
    mapping_file: str | Path = "experiments/group_average_enigma/adni_to_enigma_aparc_mapping.csv",
    selected_tau_regions: Iterable[str] = DEFAULT_SELECTED_REGIONS,
    max_date_distance_days: int = ATN_PRIMARY_TOLERANCE_DAYS,
) -> MultimodalPairDataset:
    """Build a multimodal A/T/N baseline-to-rate dataset on tau-PET pair anchors.

    Rows are inherited from the existing ADNI tau forecast-pair table. For each
    row, amyloid PET and MRI measurements are paired to the tau baseline and
    tau follow-up dates by nearest scan within ``max_date_distance_days``.
    Targets are annualized rates using the actual dates of the source modality
    whenever both source visits are available.
    """

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
    fox_by_rid = group_rows_by_rid(
        [row for row in read_csv_rows(adni_path / "MR_Image_Analysis" / "FOXLABBSI_03May2026.csv") if foxlab_passes_qc(row)],
        "EXAMDATE",
    )
    picsl_by_rid = group_rows_by_rid(
        read_csv_rows(adni_path / "MR_Image_Analysis" / "ADNI_PICSLASHS_03May2026.csv"),
        "EXAMDATE",
    )
    fsx6_dictionary = parse_freesurfer_dictionary(adni_path / "ADNIMERGE2" / "man" / "UCSFFSX6.Rd")
    fsx6_columns = fsx_region_columns(fsx6_dictionary, selected_regions)
    fsx6_by_rid = group_rows_by_rid(
        [row for row in read_csv_rows(adni_path / "MR_Image_Analysis" / "UCSFFSX6_03May2026.csv") if freesurfer_passes_qc(row)],
        "EXAMDATE",
    )
    cognitive_sources = load_cognitive_sources(adni_path)

    target_names = build_atn_target_names(selected_regions)
    feature_names = build_atn_feature_names(selected_regions, fsx6_columns)

    metadata_rows: list[dict[str, Any]] = []
    feature_rows: list[list[float]] = []
    target_baseline_rows: list[list[float]] = []
    target_observed_rows: list[list[float]] = []
    target_rate_rows: list[list[float]] = []
    target_interval_rows: list[list[float]] = []

    match_counters = {
        "amyloid": 0,
        "foxlab_bsi": 0,
        "picsl_ashs": 0,
        "fsx6": 0,
        "all_main_modalities": 0,
        "all_cortical_modalities": 0,
    }

    for pair in pairs:
        baseline_tau = observations.get(pair.get("baseline_loniuid", ""))
        target_tau = observations.get(pair.get("target_loniuid", ""))
        if baseline_tau is None or target_tau is None:
            continue
        baseline_tau_date = parse_date(pair.get("baseline_tau_date"))
        target_tau_date = parse_date(pair.get("target_tau_date"))
        tau_interval = parse_float(pair.get("target_time_years"))
        if baseline_tau_date is None or target_tau_date is None or not np.isfinite(tau_interval) or tau_interval <= 0.0:
            continue

        rid = normalize_rid(pair.get("RID", ""))
        cohort_row = cohort_by_rid.get(rid, {})
        demographics = demographics_by_rid.get(rid, {})
        plasma = nearest_dated_row(plasma_by_rid.get(rid, []), baseline_tau_date, max_days=max_date_distance_days)
        amyloid_pair = nearest_source_pair(amyloid_by_rid.get(rid, []), baseline_tau_date, target_tau_date, max_days=max_date_distance_days)
        fox_pair = nearest_source_pair(fox_by_rid.get(rid, []), baseline_tau_date, target_tau_date, max_days=max_date_distance_days)
        picsl_pair = nearest_source_pair(picsl_by_rid.get(rid, []), baseline_tau_date, target_tau_date, max_days=max_date_distance_days)
        fsx6_pair = nearest_source_pair(fsx6_by_rid.get(rid, []), baseline_tau_date, target_tau_date, max_days=max_date_distance_days)
        cognitive_baseline = nearest_cognitive_values(cognitive_sources, rid, baseline_tau_date, max_date_distance_days)
        cognitive_target = nearest_cognitive_values(cognitive_sources, rid, target_tau_date, max_date_distance_days)

        tau_baseline_values = [parse_float(baseline_tau.get(region_to_column[region])) for region in selected_regions]
        tau_target_values = [parse_float(target_tau.get(region_to_column[region])) for region in selected_regions]
        tau_baseline_meta = parse_float(baseline_tau.get("META_TEMPORAL_SUVR"))
        tau_target_meta = parse_float(target_tau.get("META_TEMPORAL_SUVR"))
        if not np.isfinite(tau_baseline_meta) or not np.isfinite(tau_target_meta):
            continue

        if amyloid_pair.usable:
            match_counters["amyloid"] += 1
        if fox_pair.usable:
            match_counters["foxlab_bsi"] += 1
        if picsl_pair.usable:
            match_counters["picsl_ashs"] += 1
        if fsx6_pair.usable:
            match_counters["fsx6"] += 1
        if amyloid_pair.usable and fox_pair.usable and picsl_pair.usable:
            match_counters["all_main_modalities"] += 1
        if amyloid_pair.usable and fox_pair.usable and picsl_pair.usable and fsx6_pair.usable:
            match_counters["all_cortical_modalities"] += 1

        age_years = age_at_date(demographics.get("PTDOBYY") or demographics.get("PTDOB"), baseline_tau_date)
        sex_female = sex_to_female(demographics.get("PTGENDER") or cohort_row.get("sex_code"))
        education = parse_float(demographics.get("PTEDUCAT") or cohort_row.get("education_years"))
        apoe4 = apoe4_dose(pair.get("apoe_genotype") or cohort_row.get("apoe_genotype"))

        feature_context = {
            "amyloid": amyloid_pair.baseline,
            "fox": fox_pair.baseline,
            "picsl": picsl_pair.baseline,
            "fsx6": fsx6_pair.baseline,
            "fsx6_columns": fsx6_columns,
        }
        feature_row = build_feature_row(
            feature_names,
            root_values={
                "age_years": age_years,
                "sex_female": sex_female,
                "education_years": education,
                "apoe4_dose": apoe4,
                "plasma_pt217": parse_float(plasma.get("pT217_F") if plasma else ""),
                "plasma_ab42_ab40": parse_float(plasma.get("AB42_AB40_F") if plasma else ""),
                "plasma_nfl": parse_float(plasma.get("NfL_Q") if plasma else ""),
                "plasma_gfap": parse_float(plasma.get("GFAP_Q") if plasma else ""),
                "tau_meta_temporal": tau_baseline_meta,
                "adas13": cognitive_baseline.get("adas13", float("nan")),
                "mmse": cognitive_baseline.get("mmse", float("nan")),
                "ravlt_immediate": cognitive_baseline.get("ravlt_immediate", float("nan")),
                "cdrsb": cognitive_baseline.get("cdrsb", float("nan")),
                **{f"tau_region:{region}": value for region, value in zip(selected_regions, tau_baseline_values, strict=True)},
            },
            context=feature_context,
        )

        target_baseline_row, target_observed_row, target_rate_row, target_interval_row = build_target_rows(
            target_names,
            selected_regions,
            tau_baseline_meta,
            tau_target_meta,
            tau_baseline_values,
            tau_target_values,
            tau_interval,
            amyloid_pair,
            fox_pair,
            picsl_pair,
            fsx6_pair,
            fsx6_columns,
            cognitive_baseline,
            cognitive_target,
        )

        metadata_rows.append(
            {
                "RID": rid,
                "PTID": pair.get("PTID", ""),
                "TRACER": pair.get("TRACER", ""),
                "target_role": pair.get("target_role", ""),
                "baseline_tau_date": baseline_tau_date.isoformat(),
                "target_tau_date": target_tau_date.isoformat(),
                "target_time_years": float(tau_interval),
                "dx_nearest_baseline": pair.get("dx_nearest_baseline", cohort_row.get("dx_nearest_baseline", "")),
                "amyloid_status": amyloid_status_to_float(amyloid_pair.baseline.get("AMYLOID_STATUS") if amyloid_pair.baseline else ""),
                "matched_amyloid": bool(amyloid_pair.usable),
                "matched_foxlab_bsi": bool(fox_pair.usable),
                "matched_picsl_ashs": bool(picsl_pair.usable),
                "matched_fsx6": bool(fsx6_pair.usable),
                "amyloid_interval_years": amyloid_pair.interval_years,
                "foxlab_interval_years": fox_pair.interval_years,
                "picsl_interval_years": picsl_pair.interval_years,
                "fsx6_interval_years": fsx6_pair.interval_years,
            }
        )
        feature_rows.append(feature_row)
        target_baseline_rows.append(target_baseline_row)
        target_observed_rows.append(target_observed_row)
        target_rate_rows.append(target_rate_row)
        target_interval_rows.append(target_interval_row)

    feature_matrix = np.asarray(feature_rows, dtype=float)
    target_baseline = np.asarray(target_baseline_rows, dtype=float)
    target_observed = np.asarray(target_observed_rows, dtype=float)
    target_rates = np.asarray(target_rate_rows, dtype=float)
    target_time_years = np.asarray(target_interval_rows, dtype=float)
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
            **match_counters,
        },
        "selected_tau_regions": selected_regions,
        "target_groups": target_group_counts(target_names),
        "feature_coverage": matrix_coverage(feature_matrix, feature_names),
        "target_coverage": matrix_coverage(target_rates, target_names),
        "target_time_years": target_time_year_summary(target_time_years, target_names),
        "notes": [
            "Rows are anchored on tau-PET forecast pairs.",
            "Non-tau rates use actual nearest-source scan intervals, not the tau-PET interval.",
            "Amyloid global rates should primarily use Centiloids; SUVR rate targets are exploratory when tracers differ.",
            "FOXLABBSI and PICSL/ASHS provide the main atrophy targets; UCSFFSX6 provides the cortical thickness/volume subset.",
            "FOXLAB BSI columns are not differenced as rate targets because they are already longitudinal boundary-shift measures.",
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
        target_time_years=target_time_years,
    )


def build_atn_target_names(selected_regions: list[str]) -> list[str]:
    names = [
        "amyloid_rate:centiloids",
        "amyloid_rate:summary_suvr",
    ]
    names.extend(f"amyloid_rate:{region}" for region in AMYLOID_REGION_COLUMNS)
    names.extend(["tau_rate:meta_temporal"])
    names.extend(f"tau_rate:{region}" for region in selected_regions)
    names.extend(f"atrophy_rate:{name}" for name in FOXLAB_COLUMNS)
    names.extend(f"ashs_rate:{name}" for name in PICSL_VOLUME_COLUMNS)
    names.extend(f"mri_thickness_rate:{region}" for region in selected_regions)
    names.extend(f"mri_volume_rate:{region}" for region in selected_regions)
    names.extend(["cognitive_rate:adas13", "cognitive_rate:mmse", "cognitive_rate:ravlt_immediate", "cognitive_rate:cdrsb"])
    return names


def build_atn_feature_names(selected_regions: list[str], fsx6_columns: dict[str, dict[str, str]]) -> list[str]:
    names = [
        "age_years",
        "sex_female",
        "education_years",
        "apoe4_dose",
        "plasma_pt217",
        "plasma_ab42_ab40",
        "plasma_nfl",
        "plasma_gfap",
        "amyloid_centiloids",
        "amyloid_summary_suvr",
        "amyloid_positive",
    ]
    names.extend(f"amyloid_region:{region}" for region in AMYLOID_REGION_COLUMNS)
    names.extend(["tau_meta_temporal"])
    names.extend(f"tau_region:{region}" for region in selected_regions)
    names.extend(
        [
            "mri_brain_volume",
            "mri_ventricle_volume",
            "mri_hippocampus_total_volume",
            "mri_hippocampus_left_volume",
            "mri_hippocampus_right_volume",
            "mri_hippocampus_vulnerability",
            "mri_temporal_cortical_thickness",
            "mri_temporal_cortical_volume",
        ]
    )
    names.extend(f"mri_ashs:{name}" for name in PICSL_VOLUME_COLUMNS)
    names.extend(f"mri_thickness:{region}" for region, columns in fsx6_columns.items() if columns.get("TA"))
    names.extend(f"mri_volume:{region}" for region, columns in fsx6_columns.items() if columns.get("CV"))
    names.extend(
        [
            "interaction:amyloid_centiloids_x_mri_hippocampus_vulnerability",
            "interaction:amyloid_centiloids_x_mri_temporal_thickness_vulnerability",
            "interaction:amyloid_centiloids_x_tau_meta_temporal",
            "interaction:plasma_pt217_x_mri_hippocampus_vulnerability",
            "adas13",
            "mmse",
            "ravlt_immediate",
            "cdrsb",
        ]
    )
    return names


def build_feature_row(feature_names: list[str], *, root_values: dict[str, float], context: dict[str, Any]) -> list[float]:
    amyloid = context["amyloid"]
    fox = context["fox"]
    picsl = context["picsl"]
    fsx6 = context["fsx6"]
    fsx6_columns = context["fsx6_columns"]
    values = dict(root_values)
    values.update(
        {
            "amyloid_centiloids": parse_float(amyloid.get("CENTILOIDS") if amyloid else ""),
            "amyloid_summary_suvr": parse_float(amyloid.get("SUMMARY_SUVR") if amyloid else ""),
            "amyloid_positive": amyloid_status_to_float(amyloid.get("AMYLOID_STATUS") if amyloid else ""),
        }
    )
    for region, column in AMYLOID_REGION_COLUMNS.items():
        values[f"amyloid_region:{region}"] = parse_float(amyloid.get(column) if amyloid else "")
    values.update(
        {
            "mri_brain_volume": parse_float(fox.get("BRAINVOL") if fox else ""),
            "mri_ventricle_volume": parse_float(fox.get("VENTVOL") if fox else ""),
            "mri_hippocampus_left_volume": parse_float(picsl.get("LEFT_HIPP_VOL") if picsl else ""),
            "mri_hippocampus_right_volume": parse_float(picsl.get("RIGHT_HIPP_VOL") if picsl else ""),
        }
    )
    values["mri_hippocampus_total_volume"] = finite_sum(
        [values["mri_hippocampus_left_volume"], values["mri_hippocampus_right_volume"]]
    )
    values["mri_hippocampus_vulnerability"] = -values["mri_hippocampus_total_volume"] if np.isfinite(values["mri_hippocampus_total_volume"]) else float("nan")
    for name, column in PICSL_VOLUME_COLUMNS.items():
        values[f"mri_ashs:{name}"] = parse_float(picsl.get(column) if picsl else "")
    for region, columns in fsx6_columns.items():
        if columns.get("TA"):
            values[f"mri_thickness:{region}"] = parse_float(fsx6.get(columns["TA"]) if fsx6 else "")
        if columns.get("CV"):
            values[f"mri_volume:{region}"] = parse_float(fsx6.get(columns["CV"]) if fsx6 else "")
    values["mri_temporal_cortical_thickness"] = finite_mean(
        values.get(f"mri_thickness:{region}", float("nan"))
        for region in fsx6_columns
        if any(token in region for token in ("entorhinal", "fusiform", "inferiortemporal", "middletemporal"))
    )
    values["mri_temporal_cortical_volume"] = finite_sum(
        values.get(f"mri_volume:{region}", float("nan"))
        for region in fsx6_columns
        if any(token in region for token in ("entorhinal", "fusiform", "inferiortemporal", "middletemporal"))
    )
    values["interaction:amyloid_centiloids_x_mri_hippocampus_vulnerability"] = product_or_nan(
        values.get("amyloid_centiloids"), values.get("mri_hippocampus_vulnerability")
    )
    temporal_thickness_vulnerability = -values["mri_temporal_cortical_thickness"] if np.isfinite(values["mri_temporal_cortical_thickness"]) else float("nan")
    values["interaction:amyloid_centiloids_x_mri_temporal_thickness_vulnerability"] = product_or_nan(
        values.get("amyloid_centiloids"), temporal_thickness_vulnerability
    )
    values["interaction:amyloid_centiloids_x_tau_meta_temporal"] = product_or_nan(
        values.get("amyloid_centiloids"), values.get("tau_meta_temporal")
    )
    values["interaction:plasma_pt217_x_mri_hippocampus_vulnerability"] = product_or_nan(
        values.get("plasma_pt217"), values.get("mri_hippocampus_vulnerability")
    )
    return [parse_float(values.get(name, float("nan"))) for name in feature_names]


def build_target_rows(
    target_names: list[str],
    selected_regions: list[str],
    tau_baseline_meta: float,
    tau_target_meta: float,
    tau_baseline_values: list[float],
    tau_target_values: list[float],
    tau_interval: float,
    amyloid_pair: DatedPair,
    fox_pair: DatedPair,
    picsl_pair: DatedPair,
    fsx6_pair: DatedPair,
    fsx6_columns: dict[str, dict[str, str]],
    cognitive_baseline: dict[str, float],
    cognitive_target: dict[str, float],
) -> tuple[list[float], list[float], list[float], list[float]]:
    baseline_values: dict[str, float] = {"tau_rate:meta_temporal": tau_baseline_meta}
    observed_values: dict[str, float] = {"tau_rate:meta_temporal": tau_target_meta}
    intervals: dict[str, float] = {"tau_rate:meta_temporal": tau_interval}
    for region, b_value, t_value in zip(selected_regions, tau_baseline_values, tau_target_values, strict=True):
        name = f"tau_rate:{region}"
        baseline_values[name] = b_value
        observed_values[name] = t_value
        intervals[name] = tau_interval

    add_pair_targets(baseline_values, observed_values, intervals, "amyloid_rate", amyloid_pair, {"centiloids": "CENTILOIDS", "summary_suvr": "SUMMARY_SUVR", **AMYLOID_REGION_COLUMNS})
    add_pair_targets(baseline_values, observed_values, intervals, "atrophy_rate", fox_pair, FOXLAB_COLUMNS)
    add_pair_targets(baseline_values, observed_values, intervals, "ashs_rate", picsl_pair, PICSL_VOLUME_COLUMNS)
    for region, columns in fsx6_columns.items():
        add_single_pair_target(baseline_values, observed_values, intervals, f"mri_thickness_rate:{region}", fsx6_pair, columns.get("TA", ""))
        add_single_pair_target(baseline_values, observed_values, intervals, f"mri_volume_rate:{region}", fsx6_pair, columns.get("CV", ""))

    for name in ("adas13", "mmse", "ravlt_immediate", "cdrsb"):
        target_name = f"cognitive_rate:{name}"
        baseline_values[target_name] = cognitive_baseline.get(name, float("nan"))
        observed_values[target_name] = cognitive_target.get(name, float("nan"))
        intervals[target_name] = tau_interval

    baseline_row = [baseline_values.get(name, float("nan")) for name in target_names]
    observed_row = [observed_values.get(name, float("nan")) for name in target_names]
    interval_row = [intervals.get(name, float("nan")) for name in target_names]
    rate_row = [
        rate_from_values(baseline, observed, interval)
        for baseline, observed, interval in zip(baseline_row, observed_row, interval_row, strict=True)
    ]
    return baseline_row, observed_row, rate_row, interval_row


def add_pair_targets(
    baseline_values: dict[str, float],
    observed_values: dict[str, float],
    intervals: dict[str, float],
    prefix: str,
    source_pair: DatedPair,
    columns: dict[str, str],
) -> None:
    for name, column in columns.items():
        add_single_pair_target(baseline_values, observed_values, intervals, f"{prefix}:{name}", source_pair, column)


def add_single_pair_target(
    baseline_values: dict[str, float],
    observed_values: dict[str, float],
    intervals: dict[str, float],
    target_name: str,
    source_pair: DatedPair,
    column: str,
) -> None:
    if not source_pair.usable or not column:
        baseline_values[target_name] = float("nan")
        observed_values[target_name] = float("nan")
        intervals[target_name] = float("nan")
        return
    baseline_values[target_name] = parse_float(source_pair.baseline.get(column))
    observed_values[target_name] = parse_float(source_pair.followup.get(column))
    intervals[target_name] = source_pair.interval_years


def nearest_source_pair(rows: list[dict[str, Any]], baseline_date: date, followup_date: date, *, max_days: int) -> DatedPair:
    baseline = nearest_dated_row(rows, baseline_date, max_days=max_days)
    followup = nearest_dated_row(rows, followup_date, max_days=max_days)
    baseline_days = abs((baseline["date"] - baseline_date).days) if baseline else float("nan")
    followup_days = abs((followup["date"] - followup_date).days) if followup else float("nan")
    interval = float("nan")
    if baseline is not None and followup is not None:
        interval = (followup["date"] - baseline["date"]).days / 365.25
        if interval <= 0.0:
            baseline = None
            followup = None
            interval = float("nan")
    return DatedPair(baseline, followup, float(baseline_days), float(followup_days), float(interval))


def fsx_region_columns(dictionary: dict[tuple[str, str, str], str], selected_regions: list[str]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for region in selected_regions:
        hemisphere, stem = split_region(region)
        output[region] = {
            "TA": dictionary.get((hemisphere, stem, "TA"), ""),
            "CV": dictionary.get((hemisphere, stem, "CV"), ""),
        }
    return output


def split_region(region: str) -> tuple[str, str]:
    prefix, _, stem = region.partition("_")
    hemisphere = "left" if prefix == "L" else "right"
    normalized = "".join(ch.lower() for ch in stem if ch.isalnum())
    return hemisphere, normalized


def foxlab_passes_qc(row: dict[str, str]) -> bool:
    qc_pass = str(row.get("QC_PASS", "")).strip()
    status = str(row.get("STATUS", "")).strip()
    return qc_pass in {"", "1"} and status in {"", "0", "1"}


def freesurfer_passes_qc(row: dict[str, str]) -> bool:
    overall = str(row.get("OVERALLQC", "")).strip().lower()
    status = str(row.get("STATUS", "")).strip().lower()
    return overall in {"pass", "partial", ""} and status in {"complete", "partial", ""}


def rate_from_values(baseline: float, observed: float, interval: float) -> float:
    if np.isfinite(baseline) and np.isfinite(observed) and np.isfinite(interval) and interval > 0.0:
        return float((observed - baseline) / interval)
    return float("nan")


def finite_sum(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.sum(finite)) if finite else float("nan")


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def product_or_nan(a: Any, b: Any) -> float:
    a_value = parse_float(a)
    b_value = parse_float(b)
    return float(a_value * b_value) if np.isfinite(a_value) and np.isfinite(b_value) else float("nan")


def target_group_counts(target_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in target_names:
        group = name.split(":", 1)[0]
        counts[group] = counts.get(group, 0) + 1
    return counts


def target_time_year_summary(intervals: np.ndarray, names: list[str]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    if intervals.size == 0:
        return output
    for idx, name in enumerate(names):
        values = intervals[:, idx]
        values = values[np.isfinite(values) & (values > 0.0)]
        output[name] = {
            "finite_rows": int(values.size),
            "median_years": float(np.median(values)) if values.size else float("nan"),
            "min_years": float(np.min(values)) if values.size else float("nan"),
            "max_years": float(np.max(values)) if values.size else float("nan"),
        }
    return output
