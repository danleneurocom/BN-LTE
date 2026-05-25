#!/usr/bin/env python3
"""Create neuroscience-focused BN-LTE explainability figures and a notebook.

The figures in this notebook are intentionally disease-mechanism oriented rather
than generic feature-attribution plots.  They summarize how the fitted BN-LTE
model stages subjects, where regional tau becomes abnormal, which causal driver
families dominate by disease stage, and how local subject predictions change
under component-level counterfactual removal.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

from bayesian_network_scm.data import MultimodalPairDataset, build_multimodal_pair_dataset  # noqa: E402
from bayesian_network_scm.dynamic_scm import DynamicSCMFit, build_design_matrix, fit_dynamic_scm  # noqa: E402
from bayesian_network_scm.pseudotime import PseudotimeModel, fit_pseudotime  # noqa: E402
from bayesian_network_scm.reporting import SubjectSplit, make_subject_split  # noqa: E402
from run_paper_validation_experiments import (  # noqa: E402
    MODEL_COLORS,
    REGION_SHORT_NAMES,
    assign_stages,
    is_finite,
    plot_surface_grid,
    safe_correlation,
    save_figure,
    validate_dataset,
    validate_split,
)


OUT = HERE / "outputs" / "explainability"
FIG = OUT / "figures"
NOTEBOOK = HERE / "bn_lte_explainability_visualizations.ipynb"
RANDOM_SEED = 20260521
MAX_PARENTS = 6

TEXT = "#111827"
MUTED = "#4B5563"
GRID = "#E5E7EB"
BNLTE_COLOR = MODEL_COLORS["BayesianNetwork-SCM"]
GROUP_COLORS = {
    "amyloid": "#0072B2",
    "plasma_tau": "#D55E00",
    "plasma_neuroimmune": "#CC79A7",
    "genetic_demographic": "#E69F00",
    "mri_neurodegeneration": "#009E73",
    "clinical": "#7C3AED",
    "regional_tau": "#6B7280",
    "self_history": "#111827",
    "baseline_trajectory": "#7C3AED",
    "other": "#9CA3AF",
}

STAGE_CENTERS = {"early": 0.17, "mid": 0.50, "late": 0.83}
STAGE_ORDER = ("early", "mid", "late")
MECHANISM_GROUP_ORDER = (
    "amyloid",
    "plasma_tau",
    "plasma_neuroimmune",
    "mri_neurodegeneration",
    "genetic_demographic",
    "clinical",
    "regional_tau",
    "self_history",
    "baseline_trajectory",
    "other",
)
REGION_FAMILY_ORDER = ("entorhinal", "fusiform", "inferiortemporal", "middletemporal", "inferiorparietal")
REGION_FAMILY_LABELS = {
    "entorhinal": "Entorhinal",
    "fusiform": "Fusiform",
    "inferiortemporal": "Inferior temporal",
    "middletemporal": "Middle temporal",
    "inferiorparietal": "Inferior parietal",
}
BRAAK_SYSTEM_LABELS = {
    "entorhinal": "Braak I/II\ntransentorhinal",
    "ventral_temporal": "Braak III/IV\nventral temporal",
    "association": "Braak V/VI\nassociation",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for stale in list(FIG.glob("fig*.png")) + list(FIG.glob("fig*.svg")):
        stale.unlink()

    print("Step 1/5: fit current default BN-LTE explainability model")
    artifacts = fit_default_bnlte()
    print("Step 2/5: compute explanation tables")
    tables = compute_explanation_tables(artifacts)
    table_paths = write_tables(tables)
    print("Step 3/5: render explainability figures")
    figure_paths = render_figures(artifacts, tables)
    print("Step 4/5: write report")
    insight_summary = summarize_explainability_findings(tables)
    report = {
        "purpose": "Neuroscience-focused BN-LTE explainability visualizations for paper analysis.",
        "fit_scope": "current_default_train_only",
        "model_configuration": {
            "pseudotime_mode": "tau_free",
            "min_train_coverage": 0.50,
            "max_parents_per_target": MAX_PARENTS,
            "n_knots": 4,
            "spline_degree": 3,
            "ridge_profile": "balanced",
            "cv_folds": 5,
            "random_seed": RANDOM_SEED,
        },
        "data": {
            "pairs": int(artifacts["dataset"].pair_count),
            "train_pairs": int(artifacts["split"].train_indices.size),
            "validation_pairs": int(artifacts["split"].validation_indices.size),
            "test_pairs": int(artifacts["split"].test_indices.size),
            "regions": artifacts["regions"],
            "z_report": artifacts["pseudotime"].report(artifacts["dataset"].feature_matrix, artifacts["dataset"].metadata_rows),
        },
        "insight_summary": insight_summary,
        "tables": {key: str(value) for key, value in table_paths.items()},
        "figures": {key: str(value) for key, value in figure_paths.items()},
        "guardrails": [
            "These are explanations of the fitted observational BN-LTE/SCM, not randomized intervention evidence.",
            "Surface driver maps show regional summaries of learned edge magnitudes, not voxelwise PET effects.",
            "Subject waterfalls decompose the linear standardized design for a fitted target rate; signs are model-implied contributions to annualized tau change.",
        ],
    }
    report_path = OUT / "explainability_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    print("Step 5/5: create notebook")
    write_notebook(report, table_paths, figure_paths)
    print(NOTEBOOK)
    print(f"figures={len(figure_paths)}")
    return 0


def fit_default_bnlte() -> dict[str, Any]:
    dataset = build_multimodal_pair_dataset(PROJECT_ROOT)
    regions = list(dataset.report["selected_tau_regions"])
    target_names = [f"tau_rate:{region}" for region in regions]
    target_indices = [dataset.target_index(name) for name in target_names]
    validate_dataset(dataset, target_indices)
    split = make_subject_split(dataset.metadata_rows, random_seed=RANDOM_SEED)
    validate_split(split)
    pseudotime = fit_pseudotime(
        dataset.feature_matrix,
        dataset.feature_names,
        split.train_indices,
        mode="tau_free",
        min_train_coverage=0.50,
    )
    fit = fit_dynamic_scm(
        dataset,
        pseudotime,
        split.train_indices,
        target_names=target_names,
        max_parents_per_target=MAX_PARENTS,
        n_knots=4,
        spline_degree=3,
        ridge_alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0),
        cv_folds=5,
    )
    rates = fit.predict_rates(dataset)[:, target_indices]
    prediction = dataset.target_baseline[:, target_indices] + dataset.time_years[:, None] * rates
    z_values = pseudotime.transform(dataset.feature_matrix)
    return {
        "dataset": dataset,
        "regions": regions,
        "target_names": target_names,
        "target_indices": target_indices,
        "split": split,
        "pseudotime": pseudotime,
        "fit": fit,
        "rates": rates,
        "prediction": prediction,
        "z_values": z_values,
    }


def compute_explanation_tables(artifacts: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    dataset: MultimodalPairDataset = artifacts["dataset"]
    split: SubjectSplit = artifacts["split"]
    pseudotime: PseudotimeModel = artifacts["pseudotime"]
    fit: DynamicSCMFit = artifacts["fit"]
    regions: list[str] = artifacts["regions"]
    target_indices: list[int] = artifacts["target_indices"]
    prediction: np.ndarray = artifacts["prediction"]
    z_values: np.ndarray = artifacts["z_values"]

    archetypes = select_archetype_subjects(dataset, split, target_indices, prediction, z_values)
    subject_z_rows = subject_z_contribution_rows(dataset, pseudotime, archetypes)
    edge_rows = edge_attribution_rows(fit, regions)
    subject_rate_rows = subject_prediction_decomposition_rows(dataset, fit, regions, target_indices, prediction, archetypes, z_values)
    z_distribution_rows = z_distribution_table(dataset, z_values)
    z_group_rows = z_group_contribution_rows(dataset, pseudotime, z_values)
    trajectory_rows = regional_trajectory_rows(dataset, regions, target_indices, prediction, z_values)
    onset_rows = regional_onset_rows(dataset, regions, target_indices, prediction, z_values)
    stage_effect_rows = stage_mechanism_effect_rows(fit, regions)
    effect_curve_rows = mechanism_effect_curve_rows(fit, regions)
    gating_rows = gating_index_rows(stage_effect_rows, regions)
    subject_component_rows = subject_region_component_rows(dataset, fit, regions, target_indices, prediction, archetypes, z_values)
    return {
        "explainability_subjects": archetypes,
        "subject_pseudotime_contributions": subject_z_rows,
        "edge_group_attribution": edge_rows,
        "subject_prediction_decomposition": subject_rate_rows,
        "z_distribution": z_distribution_rows,
        "z_group_contributions": z_group_rows,
        "regional_trajectories": trajectory_rows,
        "regional_onset": onset_rows,
        "stage_mechanism_effects": stage_effect_rows,
        "mechanism_effect_curves": effect_curve_rows,
        "gating_indices": gating_rows,
        "subject_region_components": subject_component_rows,
    }


def select_archetype_subjects(
    dataset: MultimodalPairDataset,
    split: SubjectSplit,
    target_indices: list[int],
    prediction: np.ndarray,
    z_values: np.ndarray,
) -> list[dict[str, Any]]:
    test = np.asarray(split.test_indices, dtype=int)
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    empirical_rate = (observed - baseline) / dataset.time_years[:, None]
    predicted_delta = prediction - baseline
    empirical_delta = observed - baseline
    mean_empirical_rate = np.nanmean(empirical_rate, axis=1)
    mae = np.asarray([finite_mean_abs(prediction[idx] - observed[idx]) for idx in range(dataset.pair_count)], dtype=float)
    persistence_mae = np.asarray([finite_mean_abs(baseline[idx] - observed[idx]) for idx in range(dataset.pair_count)], dtype=float)
    candidates = [
        ("fast_progressor", int(test[np.nanargmax(mean_empirical_rate[test])])),
        ("slow_progressor", int(test[np.nanargmin(mean_empirical_rate[test])])),
        ("late_stage", int(test[np.nanargmax(z_values[test])])),
        ("bnlte_best_case", int(test[np.nanargmin(mae[test])])),
        ("bnlte_gain_over_persistence", int(test[np.nanargmax((persistence_mae - mae)[test])])),
    ]
    seen = set()
    rows = []
    for archetype, idx in candidates:
        if idx in seen:
            continue
        seen.add(idx)
        row = dataset.metadata_rows[idx]
        rows.append(
            {
                "archetype": archetype,
                "row_index": int(idx),
                "RID": row.get("RID", ""),
                "dx_nearest_baseline": row.get("dx_nearest_baseline", ""),
                "amyloid_status": row.get("amyloid_status", ""),
                "z": float(z_values[idx]),
                "stage": str(assign_stages(z_values)[idx]),
                "mean_empirical_tau_rate": float(mean_empirical_rate[idx]),
                "bnlte_mae_s1": float(mae[idx]),
                "persistence_mae_s1": float(persistence_mae[idx]),
                "bnlte_gain_vs_persistence": float(persistence_mae[idx] - mae[idx]),
                "delta_spearman": safe_correlation(empirical_delta[idx], predicted_delta[idx], rank=True),
            }
        )
    return rows


def subject_z_contribution_rows(
    dataset: MultimodalPairDataset,
    pseudotime: PseudotimeModel,
    archetypes: list[dict[str, Any]],
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    contributions = pseudotime.contributions(dataset.feature_matrix)
    rows = []
    for subject in archetypes:
        row_index = int(subject["row_index"])
        values = contributions[row_index]
        order = np.argsort(-np.abs(values), kind="mergesort")[:top_k]
        for rank, feature_idx in enumerate(order, start=1):
            feature = pseudotime.selected_feature_names[int(feature_idx)]
            rows.append(
                {
                    "archetype": subject["archetype"],
                    "row_index": row_index,
                    "RID": subject["RID"],
                    "rank": int(rank),
                    "feature": feature,
                    "layer": parent_group(feature),
                    "contribution": float(values[int(feature_idx)]),
                    "abs_contribution": abs(float(values[int(feature_idx)])),
                }
            )
    return rows


def edge_attribution_rows(fit: DynamicSCMFit, regions: list[str]) -> list[dict[str, Any]]:
    basis = fit.spline_basis.transform(fit.z_grid)
    rows = []
    for target_fit in fit.target_fits:
        target = target_fit.target_name
        if not target.startswith("tau_rate:"):
            continue
        region = target.replace("tau_rate:", "")
        if region not in regions:
            continue
        self_curve = target_fit.self_effect_curve(basis)
        rows.append(edge_row("self_history", target, region, self_curve, fit.z_grid))
        for parent in target_fit.parent_names:
            rows.append(edge_row(parent, target, region, target_fit.parent_effect_curve(parent, basis), fit.z_grid))
    return sorted(rows, key=lambda row: (-float(row["mean_abs_effect"]), row["target"], row["parent"]))


def edge_row(parent: str, target: str, region: str, curve: np.ndarray, z_grid: np.ndarray) -> dict[str, Any]:
    curve = np.asarray(curve, dtype=float)
    abs_curve = np.abs(curve)
    max_idx = int(np.nanargmax(abs_curve)) if abs_curve.size else 0
    return {
        "parent": parent,
        "parent_group": parent_group(parent),
        "target": target,
        "region": region,
        "mean_effect": float(np.nanmean(curve)),
        "mean_abs_effect": float(np.nanmean(abs_curve)),
        "max_abs_effect": float(np.nanmax(abs_curve)),
        "effect_z0": float(curve[0]),
        "effect_z033": float(curve[min(int(round(0.33 * (curve.size - 1))), curve.size - 1)]),
        "effect_z067": float(curve[min(int(round(0.67 * (curve.size - 1))), curve.size - 1)]),
        "effect_z1": float(curve[-1]),
        "z_at_max_abs": float(z_grid[max_idx]) if curve.size else float("nan"),
        "sign_at_max_abs": float(np.sign(curve[max_idx])) if curve.size else float("nan"),
    }


def subject_prediction_decomposition_rows(
    dataset: MultimodalPairDataset,
    fit: DynamicSCMFit,
    regions: list[str],
    target_indices: list[int],
    prediction: np.ndarray,
    archetypes: list[dict[str, Any]],
    z_values: np.ndarray,
) -> list[dict[str, Any]]:
    target_lookup = {target_fit.target_name: target_fit for target_fit in fit.target_fits}
    rows = []
    for subject in archetypes:
        row_index = int(subject["row_index"])
        baseline = dataset.target_baseline[row_index, target_indices]
        predicted_delta = prediction[row_index] - baseline
        local_target_idx = int(np.nanargmax(predicted_delta))
        region = regions[local_target_idx]
        target_name = f"tau_rate:{region}"
        target_fit = target_lookup[target_name]
        design = build_design_matrix(
            feature_matrix=dataset.feature_matrix,
            target_baseline=dataset.target_baseline[:, target_fit.target_index],
            z=z_values,
            spline_basis=fit.spline_basis,
            parent_names=target_fit.parent_names,
            feature_names=dataset.feature_names,
        )
        raw = design.values[row_index]
        filled = np.where(np.isfinite(raw), raw, target_fit.ridge.fill_values)
        scaled = np.nan_to_num(
            (filled - target_fit.ridge.center) / target_fit.ridge.scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        column_contrib = scaled * target_fit.ridge.coefficients
        grouped: defaultdict[str, float] = defaultdict(float)
        for name, value in zip(design.feature_names, column_contrib, strict=True):
            grouped[design_feature_group(name)] += float(value)
        grouped["intercept"] += float(target_fit.ridge.intercept)
        predicted_rate = float(target_fit.ridge.intercept + np.sum(column_contrib))
        for component, value in sorted(grouped.items(), key=lambda item: -abs(item[1])):
            rows.append(
                {
                    "archetype": subject["archetype"],
                    "row_index": row_index,
                    "RID": subject["RID"],
                    "target_region": region,
                    "target": target_name,
                    "component": component,
                    "component_group": parent_group(component),
                    "rate_contribution": float(value),
                    "abs_rate_contribution": abs(float(value)),
                    "predicted_rate": predicted_rate,
                    "baseline_tau": float(dataset.target_baseline[row_index, target_fit.target_index]),
                    "predicted_followup_tau": float(dataset.target_baseline[row_index, target_fit.target_index] + dataset.time_years[row_index] * predicted_rate),
                    "target_time_years": float(dataset.time_years[row_index]),
                }
            )
    return rows


def z_distribution_table(dataset: MultimodalPairDataset, z_values: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    stages = assign_stages(z_values)
    for idx, (z, stage) in enumerate(zip(z_values, stages, strict=True)):
        row = dataset.metadata_rows[idx]
        rows.append(
            {
                "row_index": int(idx),
                "RID": row.get("RID", ""),
                "dx_nearest_baseline": row.get("dx_nearest_baseline", ""),
                "amyloid_status": row.get("amyloid_status", ""),
                "z": float(z),
                "stage": str(stage),
            }
        )
    return rows


def z_group_contribution_rows(dataset: MultimodalPairDataset, pseudotime: PseudotimeModel, z_values: np.ndarray) -> list[dict[str, Any]]:
    contributions = pseudotime.contributions(dataset.feature_matrix)
    stages = assign_stages(z_values)
    feature_groups = [parent_group(name) for name in pseudotime.selected_feature_names]
    rows = []
    for stage in STAGE_ORDER:
        idx = np.asarray([i for i, label in enumerate(stages) if label == stage], dtype=int)
        if idx.size == 0:
            continue
        for group_name in MECHANISM_GROUP_ORDER:
            cols = np.asarray([i for i, label in enumerate(feature_groups) if label == group_name], dtype=int)
            if cols.size == 0:
                continue
            stage_values = np.sum(contributions[idx[:, None], cols[None, :]], axis=1)
            rows.append(
                {
                    "stage": stage,
                    "parent_group": group_name,
                    "mean_contribution": float(np.mean(stage_values)),
                    "mean_abs_contribution": float(np.mean(np.abs(stage_values))),
                    "n": int(idx.size),
                }
            )
    return rows


def regional_trajectory_rows(
    dataset: MultimodalPairDataset,
    regions: list[str],
    target_indices: list[int],
    prediction: np.ndarray,
    z_values: np.ndarray,
) -> list[dict[str, Any]]:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    empirical_rate = (observed - baseline) / dataset.time_years[:, None]
    predicted_rate = (prediction - baseline) / dataset.time_years[:, None]
    bins = np.linspace(0.0, 1.0, 9)
    rows = []
    for bin_idx in range(len(bins) - 1):
        lo, hi = float(bins[bin_idx]), float(bins[bin_idx + 1])
        if bin_idx == len(bins) - 2:
            mask = (z_values >= lo) & (z_values <= hi)
        else:
            mask = (z_values >= lo) & (z_values < hi)
        stage = str(assign_stages(np.asarray([(lo + hi) / 2.0]))[0])
        for local_idx, region in enumerate(regions):
            for measure_name, matrix in (
                ("baseline_tau", baseline),
                ("empirical_followup_tau", observed),
                ("bnlte_followup_tau", prediction),
                ("empirical_tau_rate", empirical_rate),
                ("bnlte_tau_rate", predicted_rate),
            ):
                values = np.asarray(matrix[:, local_idx], dtype=float)[mask]
                finite = values[np.isfinite(values)]
                rows.append(
                    {
                        "z_bin": int(bin_idx),
                        "z_low": lo,
                        "z_high": hi,
                        "z_mid": float((lo + hi) / 2.0),
                        "stage": stage,
                        "region": region,
                        "region_family": region_family(region),
                        "braak_system": braak_system(region),
                        "measure": measure_name,
                        "median": float(np.median(finite)) if finite.size else float("nan"),
                        "mean": float(np.mean(finite)) if finite.size else float("nan"),
                        "q25": float(np.quantile(finite, 0.25)) if finite.size else float("nan"),
                        "q75": float(np.quantile(finite, 0.75)) if finite.size else float("nan"),
                        "n": int(finite.size),
                    }
                )
    return rows


def regional_onset_rows(
    dataset: MultimodalPairDataset,
    regions: list[str],
    target_indices: list[int],
    prediction: np.ndarray,
    z_values: np.ndarray,
) -> list[dict[str, Any]]:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    low_mask = low_risk_mask(dataset, z_values)
    trajectory = pd.DataFrame(regional_trajectory_rows(dataset, regions, target_indices, prediction, z_values))
    rows = []
    for local_idx, region in enumerate(regions):
        low_values = baseline[low_mask, local_idx]
        low_values = low_values[np.isfinite(low_values)]
        fallback = baseline[:, local_idx]
        fallback = fallback[np.isfinite(fallback)]
        if low_values.size >= 8:
            threshold = float(np.quantile(low_values, 0.90))
        elif fallback.size:
            threshold = float(np.quantile(fallback, 0.35))
        else:
            threshold = float("nan")
        empirical_onset = onset_from_trajectory(trajectory, region, "empirical_followup_tau", threshold)
        predicted_onset = onset_from_trajectory(trajectory, region, "bnlte_followup_tau", threshold)
        baseline_onset = onset_from_trajectory(trajectory, region, "baseline_tau", threshold)
        rows.append(
            {
                "region": region,
                "region_label": REGION_SHORT_NAMES.get(region, region),
                "hemisphere": "L" if region.startswith("L_") else "R",
                "region_family": region_family(region),
                "braak_system": braak_system(region),
                "braak_rank": braak_rank(region),
                "threshold": threshold,
                "baseline_onset_z": baseline_onset,
                "empirical_followup_onset_z": empirical_onset,
                "bnlte_followup_onset_z": predicted_onset,
                "onset_abs_error": abs(predicted_onset - empirical_onset) if np.isfinite(predicted_onset) and np.isfinite(empirical_onset) else float("nan"),
                "onset_signed_error": predicted_onset - empirical_onset if np.isfinite(predicted_onset) and np.isfinite(empirical_onset) else float("nan"),
            }
        )
    return rows


def stage_mechanism_effect_rows(fit: DynamicSCMFit, regions: list[str]) -> list[dict[str, Any]]:
    rows = []
    for target_fit in fit.target_fits:
        target = target_fit.target_name
        if not target.startswith("tau_rate:"):
            continue
        region = target.replace("tau_rate:", "")
        if region not in regions:
            continue
        for parent_name in ["self_history", *target_fit.parent_names]:
            for stage, z in STAGE_CENTERS.items():
                basis = fit.spline_basis.transform(np.asarray([z], dtype=float))
                curve = target_fit.self_effect_curve(basis) if parent_name == "self_history" else target_fit.parent_effect_curve(parent_name, basis)
                effect = float(curve[0]) if curve.size else 0.0
                rows.append(
                    {
                        "stage": stage,
                        "z": float(z),
                        "parent": parent_name,
                        "parent_group": parent_group(parent_name),
                        "target": target,
                        "region": region,
                        "region_family": region_family(region),
                        "braak_system": braak_system(region),
                        "signed_effect": effect,
                        "abs_effect": abs(effect),
                    }
                )
    return rows


def mechanism_effect_curve_rows(fit: DynamicSCMFit, regions: list[str]) -> list[dict[str, Any]]:
    rows = []
    basis = fit.spline_basis.transform(fit.z_grid)
    for target_fit in fit.target_fits:
        target = target_fit.target_name
        if not target.startswith("tau_rate:"):
            continue
        region = target.replace("tau_rate:", "")
        if region not in regions:
            continue
        edge_curves = [("self_history", target_fit.self_effect_curve(basis))]
        edge_curves.extend((parent, target_fit.parent_effect_curve(parent, basis)) for parent in target_fit.parent_names)
        for parent_name, curve in edge_curves:
            for z, effect in zip(fit.z_grid, curve, strict=True):
                rows.append(
                    {
                        "z": float(z),
                        "stage": str(assign_stages(np.asarray([z]))[0]),
                        "parent": parent_name,
                        "parent_group": parent_group(parent_name),
                        "target": target,
                        "region": region,
                        "region_family": region_family(region),
                        "braak_system": braak_system(region),
                        "signed_effect": float(effect),
                        "abs_effect": abs(float(effect)),
                    }
                )
    return rows


def gating_index_rows(stage_effect_rows_: list[dict[str, Any]], regions: list[str]) -> list[dict[str, Any]]:
    df = pd.DataFrame(stage_effect_rows_)
    rows = []
    for region in regions:
        sub = df[df["region"] == region]
        def abs_sum(stage: str, group_name: str) -> float:
            values = sub[(sub["stage"] == stage) & (sub["parent_group"] == group_name)]["abs_effect"].to_numpy(float)
            return float(np.sum(values)) if values.size else 0.0

        early_amyloid = abs_sum("early", "amyloid")
        mid_amyloid = abs_sum("mid", "amyloid")
        late_amyloid = abs_sum("late", "amyloid")
        early_plasma_tau = abs_sum("early", "plasma_tau")
        late_plasma_tau = abs_sum("late", "plasma_tau")
        early_self = abs_sum("early", "self_history")
        late_self = abs_sum("late", "self_history")
        rows.append(
            {
                "region": region,
                "region_label": REGION_SHORT_NAMES.get(region, region),
                "region_family": region_family(region),
                "braak_system": braak_system(region),
                "early_amyloid_gate": early_amyloid,
                "mid_amyloid_gate": mid_amyloid,
                "late_amyloid_gate": late_amyloid,
                "early_plasma_tau_coupling": early_plasma_tau,
                "late_plasma_tau_coupling": late_plasma_tau,
                "early_self_history": early_self,
                "late_self_history": late_self,
                "amyloid_gate_shift": mid_amyloid - early_amyloid,
                "plasma_tau_decoupling": early_plasma_tau - late_plasma_tau,
                "tau_autonomy_shift": late_self - early_self,
                "late_autonomy_index": late_self / max(early_amyloid + early_plasma_tau + late_amyloid + late_plasma_tau, 1.0e-9),
            }
        )
    return rows


def subject_region_component_rows(
    dataset: MultimodalPairDataset,
    fit: DynamicSCMFit,
    regions: list[str],
    target_indices: list[int],
    prediction: np.ndarray,
    archetypes: list[dict[str, Any]],
    z_values: np.ndarray,
) -> list[dict[str, Any]]:
    target_lookup = {target_fit.target_name: target_fit for target_fit in fit.target_fits}
    rows = []
    for subject in archetypes:
        row_index = int(subject["row_index"])
        for local_idx, (region, target_index) in enumerate(zip(regions, target_indices, strict=True)):
            target_name = f"tau_rate:{region}"
            target_fit = target_lookup[target_name]
            grouped_rate = decompose_target_rate(dataset, fit, target_fit, row_index, z_values)
            predicted_delta = float(prediction[row_index, local_idx] - dataset.target_baseline[row_index, target_index])
            empirical_delta = float(dataset.target_observed[row_index, target_index] - dataset.target_baseline[row_index, target_index])
            for group_name, rate_value in grouped_rate.items():
                rows.append(
                    {
                        "archetype": subject["archetype"],
                        "row_index": row_index,
                        "RID": subject["RID"],
                        "z": subject["z"],
                        "stage": subject["stage"],
                        "region": region,
                        "region_label": REGION_SHORT_NAMES.get(region, region),
                        "region_family": region_family(region),
                        "component_group": parent_group(group_name),
                        "rate_contribution": float(rate_value),
                        "delta_contribution": float(rate_value * dataset.time_years[row_index]),
                        "predicted_delta": predicted_delta,
                        "empirical_delta": empirical_delta,
                        "baseline_tau": float(dataset.target_baseline[row_index, target_index]),
                        "predicted_followup_tau": float(prediction[row_index, local_idx]),
                        "empirical_followup_tau": float(dataset.target_observed[row_index, target_index]),
                    }
                )
    return rows


def decompose_target_rate(
    dataset: MultimodalPairDataset,
    fit: DynamicSCMFit,
    target_fit: Any,
    row_index: int,
    z_values: np.ndarray,
) -> dict[str, float]:
    design = build_design_matrix(
        feature_matrix=dataset.feature_matrix,
        target_baseline=dataset.target_baseline[:, target_fit.target_index],
        z=z_values,
        spline_basis=fit.spline_basis,
        parent_names=target_fit.parent_names,
        feature_names=dataset.feature_names,
    )
    raw = design.values[int(row_index)]
    filled = np.where(np.isfinite(raw), raw, target_fit.ridge.fill_values)
    scaled = np.nan_to_num((filled - target_fit.ridge.center) / target_fit.ridge.scale, nan=0.0, posinf=0.0, neginf=0.0)
    column_contrib = scaled * target_fit.ridge.coefficients
    grouped: defaultdict[str, float] = defaultdict(float)
    for name, value in zip(design.feature_names, column_contrib, strict=True):
        grouped[parent_group(design_feature_group(name))] += float(value)
    grouped["baseline_trajectory"] += float(target_fit.ridge.intercept)
    return dict(grouped)


def low_risk_mask(dataset: MultimodalPairDataset, z_values: np.ndarray) -> np.ndarray:
    dx = np.asarray([str(row.get("dx_nearest_baseline", "")).upper() for row in dataset.metadata_rows], dtype=object)
    amy = np.asarray([str(row.get("amyloid_status", "")).upper() for row in dataset.metadata_rows], dtype=object)
    mask = (z_values <= np.nanquantile(z_values, 0.33)) & (np.char.find(dx.astype(str), "CN") >= 0)
    amy_negative = np.char.find(amy.astype(str), "NEG") >= 0
    if int(np.sum(mask & amy_negative)) >= 8:
        mask = mask & amy_negative
    if int(np.sum(mask)) < 8:
        mask = z_values <= np.nanquantile(z_values, 0.25)
    return np.asarray(mask, dtype=bool)


def onset_from_trajectory(trajectory: pd.DataFrame, region: str, measure: str, threshold: float) -> float:
    if not np.isfinite(threshold):
        return float("nan")
    sub = trajectory[(trajectory["region"] == region) & (trajectory["measure"] == measure)].sort_values("z_mid")
    for _, row in sub.iterrows():
        if int(row.get("n", 0)) >= 3 and float(row["median"]) >= threshold:
            return float(row["z_mid"])
    return 1.0


def write_tables(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Path]:
    paths = {}
    for name, rows in tables.items():
        path = OUT / f"{name}.csv"
        write_csv_rows(path, rows)
        paths[name] = path
    return paths


def summarize_explainability_findings(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    onset = pd.DataFrame(tables["regional_onset"])
    valid = onset[np.isfinite(onset["empirical_followup_onset_z"]) & np.isfinite(onset["bnlte_followup_onset_z"])]
    gating = pd.DataFrame(tables["gating_indices"])
    stage = pd.DataFrame(tables["stage_mechanism_effects"])
    stage_group = (
        stage.groupby(["stage", "parent_group"], as_index=False)["abs_effect"].sum().sort_values(["stage", "abs_effect"], ascending=[True, False])
        if not stage.empty
        else pd.DataFrame()
    )
    dominant = {}
    for stage_name in STAGE_ORDER:
        sub = stage_group[stage_group["stage"] == stage_name] if not stage_group.empty else pd.DataFrame()
        dominant[stage_name] = str(sub.iloc[0]["parent_group"]) if not sub.empty else ""
    return {
        "regional_onset_pearson": safe_correlation(valid["empirical_followup_onset_z"].to_numpy(float), valid["bnlte_followup_onset_z"].to_numpy(float), rank=False) if not valid.empty else float("nan"),
        "regional_onset_spearman": safe_correlation(valid["empirical_followup_onset_z"].to_numpy(float), valid["bnlte_followup_onset_z"].to_numpy(float), rank=True) if not valid.empty else float("nan"),
        "mean_onset_abs_error": float(np.nanmean(valid["onset_abs_error"].to_numpy(float))) if not valid.empty else float("nan"),
        "dominant_driver_by_stage": dominant,
        "highest_late_autonomy_region": str(gating.sort_values("late_autonomy_index", ascending=False).iloc[0]["region"]) if not gating.empty else "",
        "highest_early_amyloid_gate_region": str(gating.sort_values("early_amyloid_gate", ascending=False).iloc[0]["region"]) if not gating.empty else "",
    }


def render_figures(artifacts: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> dict[str, Path]:
    figures = {
        "disease_time_neurobiology": FIG / "fig1_disease_time_neurobiology.png",
        "regional_onset_brain_maps": FIG / "fig2_regional_onset_brain_maps.png",
        "onset_agreement": FIG / "fig3_onset_agreement.png",
        "stage_mechanism_matrices": FIG / "fig4_stage_mechanism_matrices.png",
        "dynamic_gating_curves": FIG / "fig5_dynamic_gating_curves.png",
        "mechanism_brain_maps": FIG / "fig6_mechanism_brain_maps.png",
        "subject_causal_fingerprints": FIG / "fig7_subject_causal_fingerprints.png",
        "subject_counterfactual_brain_maps": FIG / "fig8_subject_counterfactual_brain_maps.png",
    }
    plot_disease_time_neurobiology(figures["disease_time_neurobiology"], artifacts, tables)
    plot_regional_onset_brain_maps(figures["regional_onset_brain_maps"], tables)
    plot_onset_agreement(figures["onset_agreement"], tables)
    plot_stage_mechanism_matrices(figures["stage_mechanism_matrices"], artifacts, tables)
    plot_dynamic_gating_curves(figures["dynamic_gating_curves"], tables)
    plot_mechanism_brain_maps(figures["mechanism_brain_maps"], tables)
    plot_subject_causal_fingerprints(figures["subject_causal_fingerprints"], tables)
    plot_subject_counterfactual_brain_maps(figures["subject_counterfactual_brain_maps"], tables)
    return figures


def plot_disease_time_neurobiology(path: Path, artifacts: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    z_df = pd.DataFrame(tables["z_distribution"])
    traj = pd.DataFrame(tables["regional_trajectories"])
    z_groups = pd.DataFrame(tables["z_group_contributions"])
    z_report = artifacts["pseudotime"].report(artifacts["dataset"].feature_matrix, artifacts["dataset"].metadata_rows)

    with plt.rc_context(publication_style()):
        fig = plt.figure(figsize=(14.4, 7.4), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], width_ratios=[1.05, 1.35])
        ax0 = fig.add_subplot(grid[0, 0])
        z_df["amyloid_label"] = z_df["amyloid_status"].map(clean_amyloid_status)
        z_df["stratum"] = z_df["dx_nearest_baseline"].fillna("unknown").astype(str) + "\n" + z_df["amyloid_label"]
        preferred = ["CN\nA-", "CN\nA+", "MCI\nA-", "MCI\nA+", "AD\nA+"]
        strata = [label for label in preferred if label in set(z_df["stratum"])]
        strata.extend([label for label in sorted(z_df["stratum"].unique()) if label not in strata])
        rng = np.random.default_rng(13)
        for x, label in enumerate(strata):
            values = z_df.loc[z_df["stratum"] == label, "z"].to_numpy(float)
            if values.size == 0:
                continue
            jitter = rng.normal(0.0, 0.045, values.size)
            ax0.scatter(np.full(values.size, x) + jitter, values, s=13, alpha=0.40, color=BNLTE_COLOR, edgecolor="none")
            parts = ax0.violinplot(values, positions=[x], widths=0.62, showmeans=False, showmedians=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor("#F4A261")
                body.set_edgecolor("none")
                body.set_alpha(0.25)
            if "cmedians" in parts:
                parts["cmedians"].set_color(TEXT)
                parts["cmedians"].set_linewidth(1.2)
        ax0.set_xticks(np.arange(len(strata)), strata, rotation=30, ha="right")
        ax0.set_ylabel("Latent disease time Z")
        ax0.set_title("A. Z separates clinical/amyloid strata")
        ax0.grid(axis="y", alpha=0.22)
        ax0.text(0.02, 0.97, f"burden r={float(z_report['burden_correlation']):.3f}", transform=ax0.transAxes, va="top", ha="left", fontsize=9, color=MUTED)

        ax1 = fig.add_subplot(grid[0, 1])
        grouped = z_groups.pivot(index="parent_group", columns="stage", values="mean_contribution").reindex(index=[g for g in MECHANISM_GROUP_ORDER if g in set(z_groups["parent_group"])], columns=list(STAGE_ORDER)).fillna(0.0)
        bottom_pos = np.zeros(len(STAGE_ORDER))
        bottom_neg = np.zeros(len(STAGE_ORDER))
        x = np.arange(len(STAGE_ORDER))
        for group_name in grouped.index:
            values = grouped.loc[group_name].to_numpy(float)
            pos = np.where(values > 0, values, 0.0)
            neg = np.where(values < 0, values, 0.0)
            ax1.bar(x, pos, bottom=bottom_pos, width=0.62, color=GROUP_COLORS.get(group_name, "#999"), label=clean_group(group_name), alpha=0.88)
            ax1.bar(x, neg, bottom=bottom_neg, width=0.62, color=GROUP_COLORS.get(group_name, "#999"), alpha=0.88)
            bottom_pos += pos
            bottom_neg += neg
        ax1.axhline(0.0, color=TEXT, linewidth=0.8)
        ax1.set_xticks(x, [stage.title() for stage in STAGE_ORDER])
        ax1.set_ylabel("Mean grouped contribution to Z")
        ax1.set_title("B. Which biological layers move subjects along Z?")
        ax1.legend(frameon=False, fontsize=7.6, ncols=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax1.grid(axis="y", alpha=0.20)

        ax2 = fig.add_subplot(grid[1, :])
        sub = traj[traj["measure"] == "empirical_followup_tau"].copy()
        family = sub.groupby(["z_mid", "region_family"], as_index=False)["median"].mean()
        for family_name in REGION_FAMILY_ORDER:
            fam = family[family["region_family"] == family_name].sort_values("z_mid")
            if fam.empty:
                continue
            ax2.plot(
                fam["z_mid"],
                fam["median"],
                marker="o",
                linewidth=2.1,
                label=REGION_FAMILY_LABELS.get(family_name, family_name),
            )
        for x0, x1, color, label in ((0, 1 / 3, "#DBEAFE", "early"), (1 / 3, 2 / 3, "#FEF3C7", "mid"), (2 / 3, 1, "#FEE2E2", "late")):
            ax2.axvspan(x0, x1, color=color, alpha=0.28, linewidth=0)
            ax2.text((x0 + x1) / 2, 0.98, label, transform=ax2.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color=MUTED)
        ax2.set_xlabel("Latent disease time Z")
        ax2.set_ylabel("Median follow-up tau SUVR")
        ax2.set_title("C. Regional tau burden progresses from medial temporal to association cortex")
        ax2.legend(frameon=False, ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        ax2.grid(alpha=0.22)
        fig.suptitle("Neurobiological interpretation of latent disease time", x=0.01, ha="left", fontsize=15, fontweight="bold")
        save_figure(fig, path)


def plot_regional_onset_brain_maps(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    onset = pd.DataFrame(tables["regional_onset"])
    panels = []
    for label, col in (
        ("S0 baseline", "baseline_onset_z"),
        ("Empirical S1", "empirical_followup_onset_z"),
        ("BN-LTE S1", "bnlte_followup_onset_z"),
    ):
        panels.append((label, dict(zip(onset["region"], onset[col], strict=True))))
    plot_surface_grid(
        path,
        "Where does tau become abnormal along latent disease time?",
        panels,
        cmap="viridis_r",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Onset pseudotime Z; brighter = earlier",
        note="Onset is the first Z-bin where median regional tau exceeds a region-specific threshold estimated from low-risk CN/low-Z baseline scans. This turns model explainability into a spatial staging test.",
    )


def plot_onset_agreement(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    onset = pd.DataFrame(tables["regional_onset"]).sort_values(["braak_rank", "hemisphere", "region"])
    valid = onset[np.isfinite(onset["empirical_followup_onset_z"]) & np.isfinite(onset["bnlte_followup_onset_z"])].copy()
    pearson = safe_correlation(valid["empirical_followup_onset_z"].to_numpy(float), valid["bnlte_followup_onset_z"].to_numpy(float), rank=False)
    spearman = safe_correlation(valid["empirical_followup_onset_z"].to_numpy(float), valid["bnlte_followup_onset_z"].to_numpy(float), rank=True)
    with plt.rc_context(publication_style()):
        fig = plt.figure(figsize=(13.8, 5.8), constrained_layout=True)
        grid = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.55])
        ax0 = fig.add_subplot(grid[0, 0])
        colors = [GROUP_COLORS["amyloid"] if braak_system(region) == "entorhinal" else GROUP_COLORS["plasma_tau"] if braak_system(region) == "ventral_temporal" else GROUP_COLORS["plasma_neuroimmune"] for region in valid["region"]]
        ax0.scatter(valid["empirical_followup_onset_z"], valid["bnlte_followup_onset_z"], s=70, color=colors, alpha=0.88, edgecolor="white", linewidth=0.8)
        for _, row in valid.iterrows():
            ax0.text(row["empirical_followup_onset_z"] + 0.012, row["bnlte_followup_onset_z"] + 0.012, row["region_label"], fontsize=7.5)
        ax0.plot([0, 1], [0, 1], color=TEXT, linewidth=1.0, linestyle="--")
        ax0.set_xlim(0, 1.03)
        ax0.set_ylim(0, 1.03)
        ax0.set_xlabel("Empirical onset Z")
        ax0.set_ylabel("BN-LTE predicted onset Z")
        ax0.set_title("A. Spatial staging agreement")
        ax0.text(0.04, 0.96, f"Pearson r={pearson:.2f}\nSpearman rho={spearman:.2f}", transform=ax0.transAxes, ha="left", va="top", fontsize=9, color=MUTED)
        ax0.grid(alpha=0.20)

        ax1 = fig.add_subplot(grid[0, 1])
        y = np.arange(onset.shape[0])
        for i, (_, row) in enumerate(onset.iterrows()):
            emp = float(row["empirical_followup_onset_z"])
            pred = float(row["bnlte_followup_onset_z"])
            ax1.plot([emp, pred], [i, i], color="#CBD5E1", linewidth=2.1, zorder=1)
            ax1.scatter(emp, i, marker="o", s=60, color="#111827", label="Empirical" if i == 0 else "", zorder=2)
            ax1.scatter(pred, i, marker="D", s=48, color=BNLTE_COLOR, label="BN-LTE" if i == 0 else "", zorder=3)
        ax1.set_yticks(y, [REGION_SHORT_NAMES.get(region, region) for region in onset["region"]])
        ax1.invert_yaxis()
        ax1.set_xlim(0, 1.03)
        ax1.set_xlabel("Onset pseudotime Z")
        ax1.set_title("B. Region-wise onset displacement")
        ax1.legend(frameon=False, loc="lower right")
        ax1.grid(axis="x", alpha=0.20)
        fig.suptitle("Does BN-LTE recover the empirical tau-spread order?", x=0.01, ha="left", fontsize=15, fontweight="bold")
        save_figure(fig, path)


def plot_stage_mechanism_matrices(path: Path, artifacts: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    df = pd.DataFrame(tables["stage_mechanism_effects"])
    group_order = [g for g in MECHANISM_GROUP_ORDER if g in set(df["parent_group"])]
    region_order = artifacts["regions"]
    matrices = []
    for stage in STAGE_ORDER:
        sub = df[df["stage"] == stage].groupby(["region", "parent_group"], as_index=False)["signed_effect"].sum()
        mat = sub.pivot(index="region", columns="parent_group", values="signed_effect").reindex(index=region_order, columns=group_order).fillna(0.0)
        matrices.append(mat)
    bound = max(max(abs(float(np.nanmin(mat.to_numpy()))), abs(float(np.nanmax(mat.to_numpy())))) for mat in matrices)
    bound = max(bound, 1.0e-6)
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.8), constrained_layout=True, sharey=True)
        image = None
        for ax, stage, mat in zip(axes, STAGE_ORDER, matrices, strict=True):
            image = ax.imshow(mat.to_numpy(float), cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound), aspect="auto")
            ax.set_title(f"{stage.title()} Z={STAGE_CENTERS[stage]:.2f}")
            ax.set_xticks(np.arange(len(group_order)), [clean_group(g) for g in group_order], rotation=38, ha="right")
            ax.set_yticks(np.arange(len(region_order)), [REGION_SHORT_NAMES.get(region, region) for region in region_order])
            ax.grid(False)
            for row in range(mat.shape[0]):
                for col in range(mat.shape[1]):
                    val = float(mat.iloc[row, col])
                    if abs(val) >= 0.04 * bound:
                        ax.text(col, row, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(val) > 0.55 * bound else TEXT)
        if image is not None:
            fig.colorbar(image, ax=axes, shrink=0.78, label="Signed standardized edge effect")
        fig.suptitle("Dynamic causal architecture across disease stage", x=0.01, ha="left", fontsize=15, fontweight="bold")
        save_figure(fig, path)


def plot_dynamic_gating_curves(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    curves = pd.DataFrame(tables["mechanism_effect_curves"])
    grouped = curves.groupby(["z", "region_family", "parent_group"], as_index=False)["signed_effect"].mean()
    groups_to_show = ["amyloid", "plasma_tau", "self_history", "mri_neurodegeneration"]
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(1, len(REGION_FAMILY_ORDER), figsize=(16.2, 4.8), constrained_layout=True, sharey=True)
        for ax, family in zip(axes, REGION_FAMILY_ORDER, strict=True):
            fam = grouped[grouped["region_family"] == family]
            for group_name in groups_to_show:
                sub = fam[fam["parent_group"] == group_name].sort_values("z")
                if sub.empty or float(np.nanmax(np.abs(sub["signed_effect"].to_numpy(float)))) <= 1.0e-9:
                    continue
                ax.plot(sub["z"], sub["signed_effect"], color=GROUP_COLORS.get(group_name, "#777"), linewidth=2.1, label=clean_group(group_name))
            ax.axhline(0.0, color=TEXT, linewidth=0.8)
            ax.axvline(1 / 3, color="#9CA3AF", linewidth=0.8, linestyle=":")
            ax.axvline(2 / 3, color="#9CA3AF", linewidth=0.8, linestyle=":")
            ax.set_title(REGION_FAMILY_LABELS.get(family, family))
            ax.set_xlabel("Z")
            ax.grid(alpha=0.18)
        axes[0].set_ylabel("Mean signed edge effect")
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, ncols=len(handles), loc="lower center", bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Amyloid gating, plasma-tau coupling, and tau-autonomy over pseudotime", x=0.01, ha="left", fontsize=15, fontweight="bold")
        save_figure(fig, path)


def plot_mechanism_brain_maps(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    gating = pd.DataFrame(tables["gating_indices"])
    panels = [
        ("Amyloid gate", dict(zip(gating["region"], gating["early_amyloid_gate"], strict=True))),
        ("Plasma tau", dict(zip(gating["region"], gating["early_plasma_tau_coupling"], strict=True))),
        ("Self-history", dict(zip(gating["region"], gating["late_self_history"], strict=True))),
        ("Autonomy index", dict(zip(gating["region"], gating["late_autonomy_index"], strict=True))),
    ]
    vmax = max(abs(float(value)) for _, values in panels for value in values.values())
    plot_surface_grid(
        path,
        "Spatial mechanism maps from the dynamic SCM",
        panels,
        cmap="inferno",
        vmin=0.0,
        vmax=max(vmax, 1.0e-6),
        colorbar_label="Integrated standardized effect",
        note="Rows are not voxelwise statistics. They map region-level dynamic SCM mechanism indices onto cortical surfaces to show where each driver family contributes most strongly.",
    )


def plot_subject_causal_fingerprints(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    components = pd.DataFrame(tables["subject_region_components"])
    subjects = pd.DataFrame(tables["explainability_subjects"])
    group_order = [g for g in MECHANISM_GROUP_ORDER if g in set(components["component_group"])]
    pivot = components.groupby(["archetype", "component_group"], as_index=False)["delta_contribution"].sum().pivot(index="archetype", columns="component_group", values="delta_contribution")
    archetype_order = list(subjects["archetype"])
    pivot = pivot.reindex(index=archetype_order, columns=group_order).fillna(0.0)
    bound = max(abs(float(np.nanmin(pivot.to_numpy()))), abs(float(np.nanmax(pivot.to_numpy()))), 1.0e-6)
    with plt.rc_context(publication_style()):
        fig = plt.figure(figsize=(13.8, 6.2), constrained_layout=True)
        grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
        ax0 = fig.add_subplot(grid[0, 0])
        image = ax0.imshow(pivot.to_numpy(float), cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound), aspect="auto")
        ax0.set_yticks(np.arange(pivot.shape[0]), [label.replace("_", " ").title() for label in pivot.index])
        ax0.set_xticks(np.arange(pivot.shape[1]), [clean_group(label) for label in pivot.columns], rotation=35, ha="right")
        ax0.set_title("A. Subject-level mechanism fingerprint")
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = float(pivot.iloc[row, col])
                if abs(value) > 0.04 * bound:
                    ax0.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(value) > 0.55 * bound else TEXT)
        fig.colorbar(image, ax=ax0, shrink=0.78, label="Summed contribution to predicted cortical tau delta")

        ax1 = fig.add_subplot(grid[0, 1])
        region_family = components.groupby(["archetype", "region_family"], as_index=False)[["predicted_delta", "empirical_delta"]].mean()
        x = np.arange(len(REGION_FAMILY_ORDER))
        width = 0.34
        selected = "bnlte_gain_over_persistence" if "bnlte_gain_over_persistence" in archetype_order else (archetype_order[0] if archetype_order else "")
        if selected:
            sub = region_family[region_family["archetype"] == selected].set_index("region_family").reindex(REGION_FAMILY_ORDER)
            ax1.bar(x - width / 2, sub["empirical_delta"], width=width, color="#111827", alpha=0.82, label="Empirical")
            ax1.bar(x + width / 2, sub["predicted_delta"], width=width, color=BNLTE_COLOR, alpha=0.82, label="BN-LTE")
        ax1.axhline(0.0, color=TEXT, linewidth=0.8)
        ax1.set_xticks(x, [REGION_FAMILY_LABELS.get(f, f) for f in REGION_FAMILY_ORDER], rotation=30, ha="right")
        ax1.set_ylabel("Mean follow-up tau delta")
        ax1.set_title(f"B. Regional delta check: {selected.replace('_', ' ').title()}")
        ax1.legend(frameon=False)
        ax1.grid(axis="y", alpha=0.20)
        fig.suptitle("Local causal fingerprints for representative subjects", x=0.01, ha="left", fontsize=15, fontweight="bold")
        save_figure(fig, path)


def plot_subject_counterfactual_brain_maps(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    components = pd.DataFrame(tables["subject_region_components"])
    subjects = pd.DataFrame(tables["explainability_subjects"])
    archetype = "bnlte_gain_over_persistence" if "bnlte_gain_over_persistence" in set(subjects["archetype"]) else str(subjects.iloc[0]["archetype"])
    sub = components[components["archetype"] == archetype]
    full = sub.groupby("region", as_index=False)["predicted_delta"].mean()
    panels = [("Full prediction", dict(zip(full["region"], full["predicted_delta"], strict=True)))]
    for group_name in ("amyloid", "plasma_tau", "self_history", "mri_neurodegeneration"):
        group = sub[sub["component_group"] == group_name].groupby("region", as_index=False)["delta_contribution"].sum()
        if group.empty or float(np.nanmax(np.abs(group["delta_contribution"].to_numpy(float)))) <= 1.0e-9:
            continue
        panels.append((f"Remove {clean_group(group_name)}", dict(zip(group["region"], group["delta_contribution"], strict=True))))
    vmax = max(abs(float(value)) for _, values in panels for value in values.values())
    plot_surface_grid(
        path,
        f"Subject-level counterfactual mechanism maps: {archetype.replace('_', ' ').title()}",
        panels,
        cmap="RdBu_r",
        vmin=-max(vmax, 1.0e-6),
        vmax=max(vmax, 1.0e-6),
        colorbar_label="Contribution to follow-up tau delta",
        note="For the representative subject where BN-LTE improves most over persistence, each row maps the signed component contribution. Positive values increase predicted tau accumulation; negative values suppress the predicted increase.",
    )


def plot_pseudotime_signature(path: Path, artifacts: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    pseudotime: PseudotimeModel = artifacts["pseudotime"]
    z_df = pd.DataFrame(tables["z_distribution"])
    loadings = pd.DataFrame(pseudotime.loading_rows()).head(12)
    contributions = pseudotime.contributions(artifacts["dataset"].feature_matrix)
    stages = assign_stages(artifacts["z_values"])
    top_features = list(loadings["feature"].head(9))
    stage_rows = []
    for stage in ("early", "mid", "late"):
        idx = np.asarray([i for i, label in enumerate(stages) if label == stage], dtype=int)
        for feature in top_features:
            feature_idx = pseudotime.selected_feature_names.index(feature)
            stage_rows.append({"stage": stage, "feature": feature, "mean_contribution": float(np.mean(contributions[idx, feature_idx])) if idx.size else float("nan")})
    heat = pd.DataFrame(stage_rows).pivot(index="feature", columns="stage", values="mean_contribution").reindex(top_features)

    with plt.rc_context(publication_style()):
        fig = plt.figure(figsize=(13.2, 5.5), constrained_layout=True)
        grid = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.05])
        ax0 = fig.add_subplot(grid[0, 0])
        labels = sorted(z_df["dx_nearest_baseline"].fillna("unknown").unique())
        rng = np.random.default_rng(7)
        for x, label in enumerate(labels):
            values = z_df.loc[z_df["dx_nearest_baseline"] == label, "z"].to_numpy(float)
            jitter = rng.normal(0.0, 0.045, size=values.size)
            ax0.scatter(np.full(values.size, x) + jitter, values, s=14, alpha=0.42, color=BNLTE_COLOR, edgecolor="none")
            if values.size:
                ax0.boxplot(values, positions=[x], widths=0.34, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "edgecolor": TEXT}, medianprops={"color": TEXT})
        ax0.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
        ax0.set_ylabel("Latent disease time Z")
        ax0.set_title("A. Z orders subjects")
        ax0.grid(axis="y", alpha=0.22)

        ax1 = fig.add_subplot(grid[0, 1])
        y = np.arange(loadings.shape[0])
        colors = [GROUP_COLORS.get(parent_group(name), "#777") for name in loadings["feature"]]
        ax1.barh(y, loadings["loading"], color=colors, alpha=0.88)
        ax1.axvline(0.0, color=TEXT, linewidth=0.8)
        ax1.set_yticks(y, [clean_feature(name) for name in loadings["feature"]])
        ax1.invert_yaxis()
        ax1.set_xlabel("Signed loading")
        ax1.set_title("B. Z feature loadings")
        ax1.grid(axis="x", alpha=0.22)

        ax2 = fig.add_subplot(grid[0, 2])
        values = heat.to_numpy(float)
        bound = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.1)
        image = ax2.imshow(values, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound), aspect="auto")
        ax2.set_yticks(np.arange(len(heat.index)), [clean_feature(name) for name in heat.index])
        ax2.set_xticks(np.arange(len(heat.columns)), [str(col).title() for col in heat.columns])
        ax2.set_title("C. Contribution shifts by stage")
        fig.colorbar(image, ax=ax2, shrink=0.72, label="Mean contribution to Z")
        fig.suptitle("Explainable latent disease time", x=0.01, ha="left", fontsize=14, fontweight="bold")
        save_figure(fig, path)


def plot_subject_z_waterfalls(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    df = pd.DataFrame(tables["subject_pseudotime_contributions"])
    subjects = pd.DataFrame(tables["explainability_subjects"])
    archetypes = list(subjects["archetype"])[:5]
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(len(archetypes), 1, figsize=(9.8, 1.9 * len(archetypes)), constrained_layout=True)
        if len(archetypes) == 1:
            axes = [axes]
        for ax, archetype in zip(axes, archetypes, strict=False):
            sub = df[df["archetype"] == archetype].sort_values("contribution")
            colors = ["#0072B2" if value < 0 else BNLTE_COLOR for value in sub["contribution"]]
            ax.barh(np.arange(sub.shape[0]), sub["contribution"], color=colors, alpha=0.88)
            ax.set_yticks(np.arange(sub.shape[0]), [clean_feature(name) for name in sub["feature"]])
            meta = subjects[subjects["archetype"] == archetype].iloc[0]
            ax.set_title(f"{archetype.replace('_', ' ').title()} | RID {meta['RID']} | Z={float(meta['z']):.2f} | {meta['stage']}", loc="left", fontsize=10)
            ax.axvline(0.0, color=TEXT, linewidth=0.8)
            ax.grid(axis="x", alpha=0.20)
            ax.set_xlabel("Feature contribution to Z")
        fig.suptitle("Subject-level pseudotime explanations", x=0.01, ha="left", fontsize=14, fontweight="bold")
        save_figure(fig, path)


def plot_regional_driver_attribution(path: Path, artifacts: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    df = pd.DataFrame(tables["edge_group_attribution"])
    group = df.groupby(["parent_group", "region"], as_index=False)["mean_abs_effect"].sum()
    region_order = artifacts["regions"]
    group_order = [g for g in ["amyloid", "plasma_tau", "plasma_neuroimmune", "genetic_demographic", "mri_neurodegeneration", "regional_tau", "self_history", "other"] if g in set(group["parent_group"])]
    heat = group.pivot(index="parent_group", columns="region", values="mean_abs_effect").reindex(index=group_order, columns=region_order).fillna(0.0)
    dominance = heat.div(heat.sum(axis=0).replace(0.0, np.nan), axis=1).fillna(0.0)
    with plt.rc_context(publication_style()):
        fig = plt.figure(figsize=(13.2, 5.7), constrained_layout=True)
        grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.95])
        ax0 = fig.add_subplot(grid[0, 0])
        image = ax0.imshow(heat.to_numpy(float), cmap="magma", aspect="auto")
        ax0.set_yticks(np.arange(len(heat.index)), [clean_group(name) for name in heat.index])
        ax0.set_xticks(np.arange(len(heat.columns)), [REGION_SHORT_NAMES.get(name, name) for name in heat.columns], rotation=35, ha="right")
        ax0.set_title("A. Integrated absolute edge effect")
        fig.colorbar(image, ax=ax0, shrink=0.75, label="Mean |standardized effect|")
        for row in range(heat.shape[0]):
            for col in range(heat.shape[1]):
                val = heat.iloc[row, col]
                if val > 0:
                    ax0.text(col, row, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > heat.to_numpy().max() * 0.45 else TEXT)

        ax1 = fig.add_subplot(grid[0, 1])
        left = np.zeros(dominance.shape[1])
        x = np.arange(dominance.shape[1])
        for group_name in dominance.index:
            vals = dominance.loc[group_name].to_numpy(float)
            ax1.bar(x, vals, bottom=left, label=clean_group(group_name), color=GROUP_COLORS.get(group_name, "#999"), width=0.72)
            left += vals
        ax1.set_xticks(x, [REGION_SHORT_NAMES.get(name, name) for name in dominance.columns], rotation=35, ha="right")
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("Fraction of modeled driver strength")
        ax1.set_title("B. Driver composition by region")
        ax1.legend(frameon=False, fontsize=8, ncols=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.suptitle("What drives each regional tau-rate prediction?", x=0.01, ha="left", fontsize=14, fontweight="bold")
        save_figure(fig, path)


def plot_dynamic_edge_storyboard(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    df = pd.DataFrame(tables["edge_group_attribution"])
    top = df[df["parent"] != "self_history"].sort_values("mean_abs_effect", ascending=False).head(8)
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.0), constrained_layout=True)
        for ax, (_, row) in zip(axes.flat, top.iterrows(), strict=False):
            z = np.asarray([0.0, 0.33, 0.67, 1.0], dtype=float)
            y = np.asarray([row["effect_z0"], row["effect_z033"], row["effect_z067"], row["effect_z1"]], dtype=float)
            ax.axvspan(0.0, 1 / 3, color="#DBEAFE", alpha=0.5)
            ax.axvspan(1 / 3, 2 / 3, color="#FEF3C7", alpha=0.5)
            ax.axvspan(2 / 3, 1.0, color="#FEE2E2", alpha=0.5)
            ax.plot(z, y, color=GROUP_COLORS.get(row["parent_group"], BNLTE_COLOR), marker="o", linewidth=2.0)
            ax.axhline(0.0, color=TEXT, linewidth=0.8)
            ax.set_title(f"{clean_feature(row['parent'])}\n-> {REGION_SHORT_NAMES.get(row['region'], row['region'])}", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.grid(alpha=0.18)
        for ax in axes.flat[top.shape[0] :]:
            ax.set_axis_off()
        fig.suptitle("Stage-varying causal edge trajectories", x=0.01, ha="left", fontsize=14, fontweight="bold")
        save_figure(fig, path)


def plot_subject_prediction_decomposition(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    df = pd.DataFrame(tables["subject_prediction_decomposition"])
    subjects = list(dict.fromkeys(df["archetype"]))[:5]
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(len(subjects), 1, figsize=(10.8, 2.0 * len(subjects)), constrained_layout=True)
        if len(subjects) == 1:
            axes = [axes]
        for ax, archetype in zip(axes, subjects, strict=False):
            sub = df[df["archetype"] == archetype].copy()
            sub = sub.sort_values("abs_rate_contribution", ascending=False).head(8).sort_values("rate_contribution")
            colors = [GROUP_COLORS.get(parent_group(component), "#777") for component in sub["component"]]
            ax.barh(np.arange(sub.shape[0]), sub["rate_contribution"], color=colors, alpha=0.88)
            ax.axvline(0.0, color=TEXT, linewidth=0.8)
            ax.set_yticks(np.arange(sub.shape[0]), [clean_component(name) for name in sub["component"]])
            first = sub.iloc[0]
            ax.set_title(f"{archetype.replace('_', ' ').title()} | RID {first['RID']} | target {REGION_SHORT_NAMES.get(first['target_region'], first['target_region'])}", loc="left", fontsize=10)
            ax.set_xlabel("Contribution to predicted annual tau-rate")
            ax.grid(axis="x", alpha=0.20)
        fig.suptitle("Local prediction decomposition for representative subjects", x=0.01, ha="left", fontsize=14, fontweight="bold")
        save_figure(fig, path)


def plot_brain_driver_maps(path: Path, artifacts: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> None:
    df = pd.DataFrame(tables["edge_group_attribution"])
    regions = artifacts["regions"]
    group = df.groupby(["parent_group", "region"], as_index=False)["mean_abs_effect"].sum()
    panel_groups = [
        ("Amyloid/PET drivers", "amyloid"),
        ("Plasma tau drivers", "plasma_tau"),
        ("MRI/neurodegeneration drivers", "mri_neurodegeneration"),
        ("Genetic/demographic drivers", "genetic_demographic"),
        ("Regional tau baseline drivers", "regional_tau"),
        ("Self-history drivers", "self_history"),
    ]
    panels = []
    for label, group_name in panel_groups:
        sub = group[group["parent_group"] == group_name]
        values = {region: 0.0 for region in regions}
        for _, row in sub.iterrows():
            values[str(row["region"])] = float(row["mean_abs_effect"])
        if any(value > 0 for value in values.values()):
            panels.append((label, values))
    if not panels:
        return
    vmax = max(value for _, values in panels for value in values.values())
    plot_surface_grid(
        path,
        "BN-LTE causal driver attribution on cortical surface",
        panels,
        cmap="inferno",
        vmin=0.0,
        vmax=max(vmax, 1.0e-6),
        colorbar_label="Integrated |edge effect|",
        note="Each row maps one parent-driver family onto the modeled tau target regions. Values are regional sums of mean absolute standardized dynamic edge effects.",
    )


def parent_group(name: str) -> str:
    text = str(name).lower()
    if text in {"self_history", "self-history"}:
        return "self_history"
    if "trajectory" in text or text == "intercept":
        return "baseline_trajectory"
    if any(token in text for token in ("amyloid", "centiloid", "ab42_ab40")):
        return "amyloid"
    if any(token in text for token in ("pt217", "ptau", "plasma_tau")):
        return "plasma_tau"
    if any(token in text for token in ("nfl", "gfap")):
        return "plasma_neuroimmune"
    if any(token in text for token in ("apoe", "age", "sex", "education")):
        return "genetic_demographic"
    if any(token in text for token in ("adas", "mmse", "ravlt", "cdrsb", "cognitive")):
        return "clinical"
    if any(token in text for token in ("mri", "hippocampus", "amygdala", "volume", "thickness")):
        return "mri_neurodegeneration"
    if any(token in text for token in ("tau_region", "tau_meta", "regional_tau")):
        return "regional_tau"
    return "other"


def region_family(region: str) -> str:
    text = str(region)
    if "_" in text:
        return text.split("_", 1)[1]
    return text


def braak_system(region: str) -> str:
    family = region_family(region)
    if family == "entorhinal":
        return "entorhinal"
    if family in {"fusiform", "inferiortemporal", "middletemporal"}:
        return "ventral_temporal"
    if family == "inferiorparietal":
        return "association"
    return "other"


def braak_rank(region: str) -> int:
    order = {"entorhinal": 1, "fusiform": 2, "inferiortemporal": 3, "middletemporal": 4, "inferiorparietal": 5}
    return order.get(region_family(region), 99)


def design_feature_group(name: str) -> str:
    text = str(name)
    if text.startswith("trajectory_spline"):
        return "baseline_trajectory"
    if text.startswith("self_history"):
        return "self_history"
    if text.startswith("edge:"):
        return text.split(":spline_", 1)[0].replace("edge:", "")
    return text


def clean_feature(name: str) -> str:
    text = str(name)
    text = text.replace("tau_region:", "")
    text = text.replace("interaction:", "")
    text = text.replace("mri_", "MRI ")
    text = text.replace("plasma_", "plasma ")
    text = text.replace("amyloid_", "amyloid ")
    text = text.replace("_", " ")
    return text[:34]


def clean_group(name: str) -> str:
    label = str(name).replace("_", " ").title()
    return label.replace("Mri", "MRI").replace("Bnlte", "BN-LTE")


def clean_amyloid_status(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"1", "1.0", "POSITIVE", "POS", "A+"}:
        return "A+"
    if text in {"0", "0.0", "NEGATIVE", "NEG", "A-"}:
        return "A-"
    if not text or text == "NAN":
        return "A?"
    return text.replace("AMYLOID_", "")


def clean_component(name: str) -> str:
    if str(name) == "baseline_trajectory":
        return "baseline trajectory"
    return clean_feature(name)


def finite_mean_abs(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(np.abs(arr))) if arr.size else float("nan")


def publication_style() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }


def write_notebook(report: dict[str, Any], table_paths: dict[str, Path], figure_paths: dict[str, Path]) -> None:
    cells = [
        md(
            """# FINAL BN-LTE Explainability Visualizations

This notebook revises BN-LTE explainability around computational-neuroscience questions rather than generic feature importance.

The panels ask:

- Does latent disease time recover clinically meaningful AD staging?
- Where does regional tau first become abnormal, and does BN-LTE recover that spatial order?
- Which causal driver families dominate early, mid, and late disease?
- Does the model show an amyloid/plasma-tau coupling phase followed by regional tau self-history/autonomy?
- What mechanisms explain a representative subject's predicted progression on the cortical surface?

The figures are generated from the current default BN-LTE setting:

`tau_free` pseudotime, 6 parents per target, cubic splines with 4 knots, balanced ridge grid, and 5-fold grouped CV.
"""
        ),
        md("## Setup"),
        code(
            """from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown

BASE_CANDIDATES = [Path("outputs/explainability"), Path("BayesianNetwork-SCM/outputs/explainability")]
BASE = next((path for path in BASE_CANDIDATES if path.exists()), None)
if BASE is None:
    raise FileNotFoundError("Could not find outputs/explainability")
FIG = BASE / "figures"
REPORT = json.loads((BASE / "explainability_report.json").read_text())
TABLES = {name: pd.read_csv(path) for name, path in REPORT["tables"].items()}
display(pd.DataFrame([REPORT["model_configuration"]]).T.rename(columns={0: "value"}))
display(pd.DataFrame([REPORT["insight_summary"]]).T.rename(columns={0: "value"}))
"""
        ),
    ]
    ordered_figures = [
        ("Figure 1. Disease-time neurobiology", "disease_time_neurobiology"),
        ("Figure 2. Regional onset brain maps", "regional_onset_brain_maps"),
        ("Figure 3. Empirical-vs-model onset agreement", "onset_agreement"),
        ("Figure 4. Stage-resolved causal mechanism matrices", "stage_mechanism_matrices"),
        ("Figure 5. Dynamic gating and autonomy curves", "dynamic_gating_curves"),
        ("Figure 6. Brain-surface mechanism maps", "mechanism_brain_maps"),
        ("Figure 7. Subject causal fingerprints", "subject_causal_fingerprints"),
        ("Figure 8. Subject counterfactual brain maps", "subject_counterfactual_brain_maps"),
    ]
    for title, key in ordered_figures:
        if key not in figure_paths:
            continue
        rel = relative_notebook_path(figure_paths[key])
        cells.extend([md(f"## {title}"), md(f"![{title}]({rel})")])
    cells.extend(
        [
            md("## Full Tables"),
            code(
                """for name, df in TABLES.items():
    display(Markdown(f"### {name}"))
    display(df)
"""
            ),
            md("## Guardrails"),
            md("\n".join(f"- {item}" for item in report["guardrails"])),
        ]
    )
    NOTEBOOK.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "pygments_lexer": "ipython3"}}, "nbformat": 4, "nbformat_minor": 5}, indent=2), encoding="utf-8")


def relative_notebook_path(path: Path) -> str:
    return str(path.resolve().relative_to(HERE.resolve()))


def md(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
