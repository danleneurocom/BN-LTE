#!/usr/bin/env python3
"""Run OASIS3 adapted baseline rows with the ADNI table metrics.

The downloaded OASIS3 archive is sometimes a streaming/malformed ZIP. This
script can reuse an already extracted tree or repair/extract the ZIP with
``zip -FF`` before constructing consecutive AV1451 tau-rate pairs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from bayesian_network_scm.constraints import default_variable_specs  # noqa: E402
from bayesian_network_scm.data import DEFAULT_SELECTED_REGIONS, MultimodalPairDataset  # noqa: E402
from bayesian_network_scm.reporting import make_subject_split  # noqa: E402
from machine_learning_baselines.run_ml_baselines import fit_ml_predictions, reconstruct_followup  # noqa: E402
from run_adni_baseline_table import (  # noqa: E402
    MODEL_ORDER,
    fit_dae_prognostic_rates,
    fit_deep_mtl_mlp_rates,
    fit_dyepad_dynamic_graph_rates,
    fit_gcn_xai_population_graph_rates,
    fit_hpbn_prototype_brainnet_rates,
    fit_jad_graphlasso_topology_rates,
    fit_karlsson_taupet_ml_rates,
    fit_ncomms2025_fusion_rates,
    fit_residual_deep_ensemble_rates,
    format_summary_rows,
    make_train_context,
    render_latex_table,
    render_markdown_table,
    score_baseline_table_rows,
    summarize_table_rows,
    write_csv_rows,
    write_json,
)
from machine_learning_baselines.run_ml_baselines import design_matrix  # noqa: E402
from run_paper_validation_experiments import validate_dataset, validate_predictions, validate_split  # noqa: E402


RANDOM_SEED = 20260521
OASIS_REGION_COLUMNS = {
    "L_entorhinal": "PET_fSUVR_L_CTX_ENTORHINAL",
    "R_entorhinal": "PET_fSUVR_R_CTX_ENTORHINAL",
    "L_fusiform": "PET_fSUVR_L_CTX_FUSIFORM",
    "R_fusiform": "PET_fSUVR_R_CTX_FUSIFORM",
    "L_inferiortemporal": "PET_fSUVR_L_CTX_INFRTMP",
    "R_inferiortemporal": "PET_fSUVR_R_CTX_INFTMP",
    "L_middletemporal": "PET_fSUVR_L_CTX_MIDTMP",
    "R_middletemporal": "PET_fSUVR_R_CTX_MIDTMP",
    "L_inferiorparietal": "PET_fSUVR_L_CTX_INFRPRTL",
    "R_inferiorparietal": "PET_fSUVR_R_CTX_INFPRTL",
}
META_TEMPORAL_REGIONS = (
    "L_entorhinal",
    "R_entorhinal",
    "L_fusiform",
    "R_fusiform",
    "L_inferiortemporal",
    "R_inferiortemporal",
    "L_middletemporal",
    "R_middletemporal",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oasis-root", type=Path, default=Path("/tmp/oasis3_extracted/oasis3"))
    parser.add_argument("--zip-path", type=Path, default=Path("~/Downloads/oasis3.zip").expanduser())
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "oasis3_baseline_table")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--deep-epochs", type=int, default=80)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--max-date-distance-days", type=int, default=1095)
    args = parser.parse_args()

    report = run_oasis3_baseline_table(
        oasis_root=args.oasis_root,
        zip_path=args.zip_path,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        repeats=args.repeats,
        deep_epochs=args.deep_epochs,
        ensemble_size=args.ensemble_size,
        max_date_distance_days=args.max_date_distance_days,
    )
    print("OASIS3 baseline table complete.")
    print(f"Formatted table: {report['tables']['formatted_markdown']}")
    print(f"Summary CSV: {report['tables']['summary_csv']}")
    print(f"Report: {report['report_path']}")
    return 0


def run_oasis3_baseline_table(
    *,
    oasis_root: str | Path,
    zip_path: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    random_seed: int,
    repeats: int,
    deep_epochs: int,
    ensemble_size: int,
    max_date_distance_days: int,
) -> dict[str, Any]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    root = ensure_oasis_root(Path(oasis_root).expanduser(), Path(zip_path).expanduser(), Path(work_dir).expanduser())
    dataset = build_oasis3_pair_dataset(root, max_date_distance_days=max_date_distance_days)
    selected_regions = list(dataset.report["selected_tau_regions"])
    target_names = [f"tau_rate:{region}" for region in selected_regions]
    target_indices = [dataset.target_index(name) for name in target_names]
    validate_dataset(dataset, target_indices)

    split_rows: list[dict[str, Any]] = []
    fit_reports: dict[str, Any] = {}
    split_seeds = [int(random_seed) + 1009 * idx for idx in range(int(repeats))]
    for repeat_index, seed in enumerate(split_seeds):
        print(f"OASIS3 split {repeat_index + 1}/{len(split_seeds)} seed={seed}")
        split = make_subject_split(dataset.metadata_rows, random_seed=seed)
        validate_split(split)
        context = make_train_context(dataset, split.train_indices, selected_regions, target_names, target_indices)

        predictions, reports = fit_oasis_prediction_rows(
            context=context,
            split=split,
            target_indices=target_indices,
            deep_epochs=deep_epochs,
            ensemble_size=ensemble_size,
            random_seed=seed,
        )
        ordered_predictions = {model: predictions[model] for model in MODEL_ORDER if model in predictions}
        validate_predictions(ordered_predictions, dataset, target_indices)
        split_rows.extend(
            score_baseline_table_rows(
                ordered_predictions,
                dataset,
                split,
                selected_regions,
                target_indices,
                repeat_index=repeat_index,
                seed=seed,
            )
        )
        fit_reports[f"split_{repeat_index}"] = reports

    summary_rows = summarize_table_rows(split_rows)
    formatted_rows = format_summary_rows(summary_rows)
    table_paths = {
        "split_metrics_csv": out / "oasis3_baseline_table_split_metrics.csv",
        "summary_csv": out / "oasis3_baseline_table_summary.csv",
        "formatted_csv": out / "oasis3_baseline_table_formatted.csv",
        "formatted_markdown": out / "oasis3_baseline_table_formatted.md",
        "latex": out / "oasis3_baseline_table.tex",
    }
    write_csv_rows(table_paths["split_metrics_csv"], split_rows)
    write_csv_rows(table_paths["summary_csv"], summary_rows)
    write_csv_rows(table_paths["formatted_csv"], formatted_rows)
    table_paths["formatted_markdown"].write_text(render_markdown_table(formatted_rows), encoding="utf-8")
    table_paths["latex"].write_text(render_latex_table(formatted_rows), encoding="utf-8")

    report = {
        "purpose": "OASIS3 adapted baseline rows scored with the same held-out group-map metrics as the ADNI table.",
        "configuration": {
            "random_seed": int(random_seed),
            "repeats": int(repeats),
            "split_seeds": split_seeds,
            "deep_epochs": int(deep_epochs),
            "ensemble_size": int(ensemble_size),
            "max_date_distance_days": int(max_date_distance_days),
        },
        "data": dataset.report,
        "metric_notes": [
            "MAE and B-MAE are SUVR errors multiplied by 100, matching the ADNI table scorer.",
            "OASIS3 pairs are consecutive QC-passed AV1451/AV1451L scans.",
            "No plasma biomarker table is present in the supplied OASIS3 archive, so plasma feature columns are missing and imputed inside each train-only preprocessor.",
            "OASIS3 has no ADAS13 field in the supplied archive; RAVLT-like memory uses psychometrics srttotal when available.",
            "Rows are endpoint adapters, retrained on OASIS3 tau-rate targets; they are not the original published endpoints for classifier/topology papers.",
        ],
        "models": fit_reports,
        "tables": {key: str(path) for key, path in table_paths.items()},
    }
    report_path = out / "oasis3_baseline_table_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def fit_oasis_prediction_rows(
    *,
    context: Any,
    split: Any,
    target_indices: list[int],
    deep_epochs: int,
    ensemble_size: int,
    random_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    dataset = context.dataset
    x = design_matrix(dataset)
    y = dataset.target_rates[:, target_indices]
    train = np.asarray(split.train_indices, dtype=int)

    predictions, reports = fit_ml_predictions(context, random_seed=random_seed)

    rates, report = fit_deep_mtl_mlp_rates(x, y, train, random_seed=random_seed + 11, max_iter=deep_epochs)
    predictions["DeepMTL-MLP"] = reconstruct_followup(context, rates)
    reports["DeepMTL-MLP"] = report

    rates, report = fit_dae_prognostic_rates(context, x, y, train, random_seed=random_seed + 17, max_iter=deep_epochs)
    predictions["DAE-Prognostic"] = reconstruct_followup(context, rates)
    reports["DAE-Prognostic"] = report

    rates, report = fit_residual_deep_ensemble_rates(
        x,
        y,
        train,
        random_seed=random_seed + 29,
        max_iter=deep_epochs,
        ensemble_size=ensemble_size,
    )
    predictions["ResidualDeepEnsemble"] = reconstruct_followup(context, rates)
    reports["ResidualDeepEnsemble"] = report

    rates, report = fit_karlsson_taupet_ml_rates(context, random_seed=random_seed + 67, max_iter=deep_epochs)
    predictions["Karlsson Tau-PET ML"] = reconstruct_followup(context, rates)
    reports["Karlsson Tau-PET ML"] = report

    rates, report = fit_ncomms2025_fusion_rates(context, random_seed=random_seed + 71, max_iter=deep_epochs)
    predictions["NComms2025 Fusion MLP"] = reconstruct_followup(context, rates)
    reports["NComms2025 Fusion MLP"] = report

    rates, report = fit_dyepad_dynamic_graph_rates(context, random_seed=random_seed + 79)
    predictions["DyEPAD Dynamic Graph"] = reconstruct_followup(context, rates)
    reports["DyEPAD Dynamic Graph"] = report

    rates, report = fit_gcn_xai_population_graph_rates(context, random_seed=random_seed + 83)
    predictions["GCN-XAI Population Graph"] = reconstruct_followup(context, rates)
    reports["GCN-XAI Population Graph"] = report

    rates, report = fit_jad_graphlasso_topology_rates(context, random_seed=random_seed + 89)
    predictions["JAD GraphLASSO Tau Topology"] = reconstruct_followup(context, rates)
    reports["JAD GraphLASSO Tau Topology"] = report

    rates, report = fit_hpbn_prototype_brainnet_rates(context, random_seed=random_seed + 97)
    predictions["HPBN Prototype Brain-Net"] = reconstruct_followup(context, rates)
    reports["HPBN Prototype Brain-Net"] = report

    return predictions, reports


def ensure_oasis_root(oasis_root: Path, zip_path: Path, work_dir: Path) -> Path:
    if oasis_root.exists() and find_csv(oasis_root, "OASIS3_AV1451_PUP.csv"):
        return oasis_root.resolve()
    extracted = work_dir / "oasis3_extracted" / "oasis3"
    if extracted.exists() and find_csv(extracted, "OASIS3_AV1451_PUP.csv"):
        return extracted.resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"OASIS3 root not found and ZIP is missing: {zip_path}")

    repair_dir = work_dir / "oasis3_repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    fixed_zip = repair_dir / "oasis3_fixed.zip"
    if not fixed_zip.exists():
        subprocess.run(
            ["zip", "-FF", str(zip_path), "--out", str(fixed_zip)],
            input=("y\n" * 5000),
            text=True,
            check=True,
        )
    extract_dir = work_dir / "oasis3_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", str(fixed_zip), "-d", str(extract_dir)], check=True)
    if not extracted.exists():
        raise FileNotFoundError(f"Repaired OASIS3 extraction did not create {extracted}")
    return extracted.resolve()


def build_oasis3_pair_dataset(oasis_root: Path, *, max_date_distance_days: int) -> MultimodalPairDataset:
    selected_regions = [region for region in DEFAULT_SELECTED_REGIONS if region in OASIS_REGION_COLUMNS]
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
    feature_names.extend(f"tau_region:{region}" for region in selected_regions)
    target_names = ["tau_rate:meta_temporal"]
    target_names.extend(f"tau_rate:{region}" for region in selected_regions)
    target_names.extend(["cognitive_rate:adas13", "cognitive_rate:mmse", "cognitive_rate:ravlt_immediate", "cognitive_rate:cdrsb"])

    tau = load_oasis_tau_rows(oasis_root, selected_regions)
    demographics = load_demographics(oasis_root)
    amyloid_by_subject = load_amyloid_records(oasis_root)
    mri_by_subject = load_mri_records(oasis_root)
    cdr_by_subject = load_cdr_records(oasis_root)
    psych_by_subject = load_psychometric_records(oasis_root)

    metadata_rows: list[dict[str, Any]] = []
    feature_rows: list[list[float]] = []
    target_baseline_rows: list[list[float]] = []
    target_observed_rows: list[list[float]] = []
    target_rate_rows: list[list[float]] = []

    for subject, subject_tau in tau.groupby("subject", sort=True):
        subject_tau = subject_tau.sort_values("day")
        if len(subject_tau) < 2:
            continue
        records = [row for _, row in subject_tau.iterrows()]
        for pair_index, (baseline, target) in enumerate(zip(records[:-1], records[1:])):
            dt_years = (float(target["day"]) - float(baseline["day"])) / 365.25
            if not np.isfinite(dt_years) or dt_years <= 0.25:
                continue

            baseline_tau_values = [as_float(baseline[OASIS_REGION_COLUMNS[region]]) for region in selected_regions]
            target_tau_values = [as_float(target[OASIS_REGION_COLUMNS[region]]) for region in selected_regions]
            if not all(np.isfinite(value) for value in baseline_tau_values + target_tau_values):
                continue
            baseline_meta = meta_temporal_value(baseline, selected_regions)
            target_meta = meta_temporal_value(target, selected_regions)
            if not np.isfinite(baseline_meta) or not np.isfinite(target_meta):
                continue

            baseline_day = float(baseline["day"])
            target_day = float(target["day"])
            demo = demographics.get(subject, {})
            amyloid = nearest_by_day(amyloid_by_subject.get(subject, []), baseline_day, max_date_distance_days)
            mri = nearest_by_day(mri_by_subject.get(subject, []), baseline_day, max_date_distance_days)
            cdr_base = nearest_by_day(cdr_by_subject.get(subject, []), baseline_day, max_date_distance_days)
            cdr_target = nearest_by_day(cdr_by_subject.get(subject, []), target_day, max_date_distance_days)
            psych_base = nearest_by_day(psych_by_subject.get(subject, []), baseline_day, max_date_distance_days)
            psych_target = nearest_by_day(psych_by_subject.get(subject, []), target_day, max_date_distance_days)

            age = value_or_nan(cdr_base, "age_years")
            if not np.isfinite(age):
                age = as_float(demo.get("AgeatEntry")) + baseline_day / 365.25
            mmse_base = value_or_nan(cdr_base, "mmse")
            cdrsb_base = value_or_nan(cdr_base, "cdrsb")
            ravlt_base = value_or_nan(psych_base, "ravlt_immediate")
            mmse_target = value_or_nan(cdr_target, "mmse")
            cdrsb_target = value_or_nan(cdr_target, "cdrsb")
            ravlt_target = value_or_nan(psych_target, "ravlt_immediate")

            feature_row = [
                age,
                oasis_sex_female(demo.get("GENDER")),
                as_float(demo.get("EDUC")),
                oasis_apoe4_dose(demo.get("APOE")),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                value_or_nan(amyloid, "summary_suvr"),
                value_or_nan(amyloid, "centiloids"),
                amyloid_positive(value_or_nan(amyloid, "centiloids")),
                baseline_meta,
                value_or_nan(mri, "hippocampus_volume"),
                value_or_nan(mri, "amygdala_volume"),
                value_or_nan(mri, "temporal_cortical_volume"),
                value_or_nan(mri, "temporal_cortical_thickness"),
                float("nan"),
                mmse_base,
                ravlt_base,
                cdrsb_base,
            ]
            feature_row.extend(baseline_tau_values)

            target_baseline_row = [baseline_meta] + baseline_tau_values + [float("nan"), mmse_base, ravlt_base, cdrsb_base]
            target_observed_row = [target_meta] + target_tau_values + [float("nan"), mmse_target, ravlt_target, cdrsb_target]
            target_rate_row = [(target_observed_row[idx] - target_baseline_row[idx]) / dt_years for idx in range(len(target_baseline_row))]

            metadata_rows.append(
                {
                    "RID": subject,
                    "PTID": subject,
                    "TRACER": "AV-1451",
                    "baseline_tau_day": baseline_day,
                    "target_tau_day": target_day,
                    "target_time_years": float(dt_years),
                    "baseline_tau_id": str(baseline["id"]),
                    "target_tau_id": str(target["id"]),
                    "pair_index": int(pair_index),
                    "amyloid_status": "positive" if value_or_nan(amyloid, "centiloids") >= 20.0 else "negative",
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

    report = {
        "source": {
            "oasis_root": str(oasis_root),
            "max_date_distance_days": int(max_date_distance_days),
        },
        "rows": {
            "usable_pairs": len(metadata_rows),
            "unique_subjects": len({row["RID"] for row in metadata_rows}),
            "qc_passed_tau_scans": int(len(tau)),
            "tau_subjects": int(tau["subject"].nunique()),
        },
        "selected_tau_regions": selected_regions,
        "feature_coverage": matrix_coverage(feature_matrix, feature_names),
        "target_coverage": matrix_coverage(target_rates, target_names),
        "notes": [
            "Meta-temporal tau is approximated as the mean of bilateral entorhinal, fusiform, inferior temporal, and middle temporal AV1451 fSUVR.",
            "Amyloid features come from nearest QC-passed amyloid PUP/Centiloid records at baseline.",
            "MRI summaries use nearest QC-passed FreeSurfer records at baseline.",
            "ADAS13 and plasma biomarkers are not present in the supplied OASIS3 archive.",
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
        variable_specs=default_variable_specs(feature_names, target_names),
        report=report,
    )


def load_oasis_tau_rows(oasis_root: Path, selected_regions: list[str]) -> pd.DataFrame:
    frames = []
    for filename in ("OASIS3_AV1451_PUP.csv", "OASIS3_AV1451L_PUP.csv"):
        path = find_csv(oasis_root, filename)
        if path is None:
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = filename
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("Could not find OASIS3 AV1451 PUP CSV files.")
    tau = pd.concat(frames, ignore_index=True, sort=False)
    tau["subject"] = tau["id"].map(extract_subject)
    tau["day"] = tau["id"].map(extract_day)
    for column in [OASIS_REGION_COLUMNS[region] for region in selected_regions]:
        tau[column] = pd.to_numeric(tau[column], errors="coerce")
    qc = (
        tau["FreeSurfer QC Status"].astype(str).str.contains("Passed", na=False)
        & tau["PET TC QC Status"].astype(str).str.contains("Passed", na=False)
    )
    needed = ["subject", "day", *[OASIS_REGION_COLUMNS[region] for region in selected_regions]]
    tau = tau[qc].dropna(subset=needed).copy()
    tau = tau.sort_values(["subject", "day", "source_file"]).drop_duplicates(["subject", "day"], keep="first")
    return tau


def load_demographics(oasis_root: Path) -> dict[str, dict[str, Any]]:
    path = require_csv(oasis_root, "OASIS3_demographics.csv")
    rows = pd.read_csv(path).to_dict(orient="records")
    return {str(row.get("OASISID")): row for row in rows if str(row.get("OASISID", "")).startswith("OAS")}


def load_amyloid_records(oasis_root: Path) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    pup_path = find_csv(oasis_root, "OASIS3_PUP.csv")
    if pup_path is not None:
        pup = pd.read_csv(pup_path)
        pup["subject"] = pup["id"].map(extract_subject)
        pup["day"] = pup["id"].map(extract_day)
        qc = (
            pup["tracer"].astype(str).isin(["AV45", "PIB"])
            & pup["FreeSurfer QC Status"].astype(str).str.contains("Passed", na=False)
            & pup["PET TC QC Status"].astype(str).str.contains("Passed", na=False)
        )
        for _, row in pup[qc].iterrows():
            subject = str(row.get("subject", ""))
            day = as_float(row.get("day"))
            if not subject or not np.isfinite(day):
                continue
            record = {
                "day": day,
                "summary_suvr": as_float(row.get("PET_fSUVR_TOT_CORTMEAN")),
                "centiloids": first_finite(
                    row.get("Centil_fSUVR_TOT_CORTMEAN"),
                    row.get("Centil_fBP_TOT_CORTMEAN"),
                    row.get("Centil_fSUVR_rsf_TOT_CORTMEAN"),
                    row.get("Centil_fBP_rsf_TOT_CORTMEAN"),
                ),
            }
            grouped.setdefault(subject, []).append(record)

    cent_path = find_csv(oasis_root, "OASIS3_amyloid_centiloid.csv")
    if cent_path is not None:
        cent = pd.read_csv(cent_path)
        for _, row in cent.iterrows():
            subject = str(row.get("subject_id", ""))
            day = extract_day(row.get("oasis_session_id"))
            if not subject or not np.isfinite(day):
                continue
            grouped.setdefault(subject, []).append(
                {
                    "day": day,
                    "summary_suvr": float("nan"),
                    "centiloids": first_finite(
                        row.get("Centiloid_fSUVR_TOT_CORTMEAN"),
                        row.get("Centiloid_fBP_TOT_CORTMEAN"),
                        row.get("Centiloid_fSUVR_rsf_TOT_CORTMEAN"),
                        row.get("Centiloid_fBP_rsf_TOT_CORTMEAN"),
                    ),
                }
            )
    return sort_grouped(grouped)


def load_mri_records(oasis_root: Path) -> dict[str, list[dict[str, float]]]:
    path = require_csv(oasis_root, "OASIS3_Freesurfer_output.csv")
    fs = pd.read_csv(path)
    fs["day"] = fs["MR_session"].map(extract_day)
    qc = fs["FS QC Status"].astype(str).str.contains("Passed", na=False)
    grouped: dict[str, list[dict[str, float]]] = {}
    temporal_regions = ("entorhinal", "fusiform", "inferiortemporal", "middletemporal")
    for _, row in fs[qc].iterrows():
        subject = str(row.get("Subject", ""))
        day = as_float(row.get("day"))
        if not subject or not np.isfinite(day):
            continue
        volume_cols = [f"{hemi}_{region}_volume" for hemi in ("lh", "rh") for region in temporal_regions]
        thickness_cols = [f"{hemi}_{region}_thickness" for hemi in ("lh", "rh") for region in temporal_regions]
        grouped.setdefault(subject, []).append(
            {
                "day": day,
                "hippocampus_volume": first_finite(
                    row.get("TOTAL_HIPPOCAMPUS_VOLUME"),
                    as_float(row.get("Left-Hippocampus_volume")) + as_float(row.get("Right-Hippocampus_volume")),
                ),
                "amygdala_volume": as_float(row.get("Left-Amygdala_volume")) + as_float(row.get("Right-Amygdala_volume")),
                "temporal_cortical_volume": finite_sum(row.get(col) for col in volume_cols),
                "temporal_cortical_thickness": finite_mean(row.get(col) for col in thickness_cols),
            }
        )
    return sort_grouped(grouped)


def load_cdr_records(oasis_root: Path) -> dict[str, list[dict[str, float]]]:
    path = require_csv(oasis_root, "OASIS3_UDSb4_cdr.csv")
    cdr = pd.read_csv(path)
    grouped: dict[str, list[dict[str, float]]] = {}
    for _, row in cdr.iterrows():
        subject = str(row.get("OASISID", ""))
        day = as_float(row.get("days_to_visit"))
        if not subject or not np.isfinite(day):
            continue
        grouped.setdefault(subject, []).append(
            {
                "day": day,
                "age_years": as_float(row.get("age at visit")),
                "mmse": as_float(row.get("MMSE")),
                "cdrsb": as_float(row.get("CDRSUM")),
            }
        )
    return sort_grouped(grouped)


def load_psychometric_records(oasis_root: Path) -> dict[str, list[dict[str, float]]]:
    path = require_csv(oasis_root, "OASIS3_UDSc1_cognitive_assessments.csv")
    psych = pd.read_csv(path)
    grouped: dict[str, list[dict[str, float]]] = {}
    for _, row in psych.iterrows():
        subject = str(row.get("OASISID", ""))
        day = as_float(row.get("days_to_visit"))
        if not subject or not np.isfinite(day):
            continue
        grouped.setdefault(subject, []).append(
            {
                "day": day,
                "ravlt_immediate": first_finite(row.get("srttotal"), row.get("srtfree")),
            }
        )
    return sort_grouped(grouped)


def find_csv(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None


def require_csv(root: Path, filename: str) -> Path:
    path = find_csv(root, filename)
    if path is None:
        raise FileNotFoundError(f"Could not find {filename} below {root}")
    return path


def extract_subject(value: Any) -> str:
    match = re.search(r"(OAS\d+)", str(value or ""))
    return match.group(1) if match else ""


def extract_day(value: Any) -> float:
    match = re.search(r"_d(\d+)", str(value or ""))
    return float(match.group(1)) if match else float("nan")


def meta_temporal_value(row: Any, selected_regions: list[str]) -> float:
    values = [as_float(row[OASIS_REGION_COLUMNS[region]]) for region in META_TEMPORAL_REGIONS if region in selected_regions]
    return finite_mean(values)


def nearest_by_day(records: list[dict[str, float]], day: float, max_days: int) -> dict[str, float] | None:
    if not records or not np.isfinite(day):
        return None
    best = min(records, key=lambda row: abs(as_float(row.get("day")) - day))
    return best if abs(as_float(best.get("day")) - day) <= float(max_days) else None


def sort_grouped(grouped: dict[str, list[dict[str, float]]]) -> dict[str, list[dict[str, float]]]:
    for rows in grouped.values():
        rows.sort(key=lambda row: as_float(row.get("day")))
    return grouped


def as_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def value_or_nan(row: dict[str, Any] | None, key: str) -> float:
    return as_float(row.get(key)) if row else float("nan")


def first_finite(*values: Any) -> float:
    for value in values:
        parsed = as_float(value)
        if np.isfinite(parsed):
            return parsed
    return float("nan")


def finite_sum(values: Any) -> float:
    arr = np.asarray([as_float(value) for value in values], dtype=float)
    return float(np.sum(arr[np.isfinite(arr)])) if np.any(np.isfinite(arr)) else float("nan")


def finite_mean(values: Any) -> float:
    arr = np.asarray([as_float(value) for value in values], dtype=float)
    return float(np.mean(arr[np.isfinite(arr)])) if np.any(np.isfinite(arr)) else float("nan")


def oasis_sex_female(value: Any) -> float:
    parsed = str(value or "").strip().lower()
    if parsed in {"2", "female", "f"}:
        return 1.0
    if parsed in {"1", "male", "m"}:
        return 0.0
    return float("nan")


def oasis_apoe4_dose(value: Any) -> float:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return float("nan")
    digits = re.findall(r"[234]", text)
    return float(sum(digit == "4" for digit in digits)) if digits else float("nan")


def amyloid_positive(centiloids: float) -> float:
    return float(centiloids >= 20.0) if np.isfinite(centiloids) else float("nan")


def matrix_coverage(matrix: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    arr = np.asarray(matrix, dtype=float)
    rows = []
    for idx, name in enumerate(names):
        values = arr[:, idx] if arr.size else np.asarray([], dtype=float)
        rows.append(
            {
                "name": name,
                "finite_fraction": float(np.mean(np.isfinite(values))) if values.size else 0.0,
                "finite_count": int(np.sum(np.isfinite(values))) if values.size else 0,
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
