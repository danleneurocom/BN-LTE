#!/usr/bin/env python3
"""Hypothesis experiments and visual diagnostics for Dynamic BN-SCM.

This runner stays inside the leakage-controlled Dynamic BN-SCM formulation:
baseline multimodal variables at t0 predict annualized future tau-rate targets.
It writes machine-readable tables plus dependency-free SVG figures so the
notebook can be opened in the current lightweight project environment.
"""

from __future__ import annotations

import csv
import html
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_network_scm.constraints import default_variable_specs  # noqa: E402
from bayesian_network_scm.data import MultimodalPairDataset, build_multimodal_pair_dataset  # noqa: E402
from bayesian_network_scm.dynamic_scm import fit_dynamic_scm  # noqa: E402
from bayesian_network_scm.pseudotime import fit_pseudotime  # noqa: E402
from bayesian_network_scm.reporting import make_subject_split  # noqa: E402


SELECTED_TARGET_PREFIX = "tau_rate:"
MODEL_COLOR = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"
PINK = "#CC79A7"
GOLD = "#E69F00"
GRAY = "#6B7280"


_DK_COORDS_LEFT: dict[str, tuple[float, float]] = {
    "L_frontalpole": (0.05, 0.58),
    "L_superiorfrontal": (0.22, 0.88),
    "L_rostralmiddlefrontal": (0.18, 0.68),
    "L_caudalmiddlefrontal": (0.34, 0.74),
    "L_medialorbitofrontal": (0.10, 0.30),
    "L_lateralorbitofrontal": (0.16, 0.35),
    "L_parsorbitalis": (0.18, 0.43),
    "L_parstriangularis": (0.25, 0.52),
    "L_parsopercularis": (0.30, 0.58),
    "L_precentral": (0.42, 0.80),
    "L_paracentral": (0.48, 0.92),
    "L_postcentral": (0.50, 0.78),
    "L_superiorparietal": (0.65, 0.80),
    "L_inferiorparietal": (0.68, 0.55),
    "L_supramarginal": (0.55, 0.62),
    "L_precuneus": (0.74, 0.78),
    "L_lateraloccipital": (0.92, 0.50),
    "L_cuneus": (0.93, 0.68),
    "L_pericalcarine": (0.96, 0.45),
    "L_lingual": (0.93, 0.30),
    "L_superiortemporal": (0.45, 0.38),
    "L_middletemporal": (0.55, 0.30),
    "L_inferiortemporal": (0.62, 0.20),
    "L_bankssts": (0.55, 0.45),
    "L_transversetemporal": (0.42, 0.45),
    "L_fusiform": (0.70, 0.13),
    "L_temporalpole": (0.20, 0.22),
    "L_entorhinal": (0.32, 0.10),
    "L_parahippocampal": (0.50, 0.10),
    "L_caudalanteriorcingulate": (0.30, 0.65),
    "L_rostralanteriorcingulate": (0.18, 0.55),
    "L_posteriorcingulate": (0.55, 0.70),
    "L_isthmuscingulate": (0.72, 0.60),
    "L_insula": (0.38, 0.48),
}
_DK_COORDS_RIGHT = {("R" + key[1:]): (1.0 - x, y) for key, (x, y) in _DK_COORDS_LEFT.items()}
_DK_COORDS = {**_DK_COORDS_LEFT, **_DK_COORDS_RIGHT}
_MEDIAL_PARTS = {
    "medialorbitofrontal",
    "entorhinal",
    "parahippocampal",
    "precuneus",
    "cuneus",
    "pericalcarine",
    "lingual",
    "caudalanteriorcingulate",
    "rostralanteriorcingulate",
    "posteriorcingulate",
    "isthmuscingulate",
    "fusiform",
}


def run_hypothesis_experiments(
    *,
    project_root: str | Path = PROJECT_ROOT,
    output_dir: str | Path = THIS_DIR / "outputs" / "hypothesis_experiments",
    random_seed: int = 20260519,
    max_parents_per_target: int = 6,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_path = resolve_path(output_dir, root)
    figure_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("[BN-SCM hypotheses] Building multimodal pair dataset...")
    dataset = build_multimodal_pair_dataset(root)
    split = make_subject_split(dataset.metadata_rows, random_seed=random_seed)
    selected_regions = list(dataset.report["selected_tau_regions"])
    target_names = [f"{SELECTED_TARGET_PREFIX}{region}" for region in selected_regions]

    print("[BN-SCM hypotheses] Fitting base tau-free Dynamic BN-SCM...")
    base = fit_scenario(
        dataset,
        split,
        target_names=target_names,
        pseudotime_mode="tau_free",
        max_parents_per_target=max_parents_per_target,
    )

    print("[BN-SCM hypotheses] Testing H1 pT217 decoupling...")
    h1_rows, h1_summary = evaluate_h1_decoupling(base)

    print("[BN-SCM hypotheses] Checking H2 transcriptomic data gate...")
    h2_summary = evaluate_h2_data_gate(root, base)

    print("[BN-SCM hypotheses] Testing H3 PART-like vs AD-continuum routes...")
    h3_rows, h3_summary = evaluate_h3_part_ad(dataset, split, base, selected_regions)

    print("[BN-SCM hypotheses] Running pseudotime sensitivity...")
    pseudotime_rows = evaluate_pseudotime_sensitivity(
        dataset,
        split,
        target_names,
        max_parents_per_target=max_parents_per_target,
    )

    print("[BN-SCM hypotheses] Running parent ablations...")
    ablation_rows = evaluate_parent_ablations(
        dataset,
        split,
        target_names,
        max_parents_per_target=max_parents_per_target,
    )

    print("[BN-SCM hypotheses] Computing pseudotime-bin diagnostics and brain-map panels...")
    z_bin_rows = evaluate_z_bins(base, split)
    subject_rows = select_representative_subjects(dataset, split, base)
    region_labels = load_region_labels(root)

    figures = {
        "hypothesis_status": str(figure_dir / "fig1_hypothesis_status.svg"),
        "h1_pt217_decoupling": str(figure_dir / "fig2_h1_pt217_decoupling.svg"),
        "ablation_summary": str(figure_dir / "fig3_ablation_summary.svg"),
        "brain_group_forecast": str(figure_dir / "fig4_brain_group_forecast.svg"),
        "h3_part_ad_route": str(figure_dir / "fig5_h3_part_ad_route.svg"),
        "brain_subject_cn_stable": str(figure_dir / "fig6_subject_cn_stable.svg"),
        "brain_subject_mci_progressing": str(figure_dir / "fig7_subject_mci_progressing.svg"),
        "brain_subject_ad_high_baseline": str(figure_dir / "fig8_subject_ad_high_baseline.svg"),
        "z_bin_errors": str(figure_dir / "fig9_z_bin_errors.svg"),
    }
    hypothesis_status_rows = hypothesis_status(h1_summary, h2_summary, h3_summary)
    write_hypothesis_status_svg(Path(figures["hypothesis_status"]), hypothesis_status_rows)
    write_h1_svg(Path(figures["h1_pt217_decoupling"]), h1_rows)
    write_ablation_svg(Path(figures["ablation_summary"]), pseudotime_rows, ablation_rows)
    write_group_brain_svg(Path(figures["brain_group_forecast"]), region_labels, selected_regions, base, split)
    write_h3_brain_svg(Path(figures["h3_part_ad_route"]), region_labels, selected_regions, h3_summary)
    for row, key in zip(
        subject_rows,
        ["brain_subject_cn_stable", "brain_subject_mci_progressing", "brain_subject_ad_high_baseline"],
        strict=True,
    ):
        write_subject_brain_svg(Path(figures[key]), region_labels, selected_regions, base, row)
    write_z_bin_svg(Path(figures["z_bin_errors"]), z_bin_rows)

    csv_write(output_path / "h1_pt217_decoupling.csv", h1_rows)
    csv_write(output_path / "h3_part_ad_route.csv", h3_rows)
    csv_write(output_path / "pseudotime_sensitivity.csv", pseudotime_rows)
    csv_write(output_path / "parent_ablation.csv", ablation_rows)
    csv_write(output_path / "z_bin_diagnostics.csv", z_bin_rows)
    csv_write(output_path / "representative_subjects.csv", subject_rows)

    report = {
        "purpose": (
            "Hypothesis-driven Dynamic BN-SCM analysis using baseline-to-follow-up ADNI tau pairs, "
            "train-only pseudotime, and selected temporolimbic/inferior-parietal tau targets."
        ),
        "data": {
            "pairs": dataset.pair_count,
            "subjects": len({row["RID"] for row in dataset.metadata_rows}),
            "selected_regions": selected_regions,
            "feature_count": len(dataset.feature_names),
            "target_count": len(target_names),
        },
        "split": split.report(),
        "base_model": {
            "pseudotime_mode": "tau_free",
            "max_parents_per_target": int(max_parents_per_target),
            "test_rate_mae_median": summarize_metric(base["pair_rows"], "test", "rate_mae")["median"],
            "test_delta_spearman_median": summarize_metric(base["pair_rows"], "test", "delta_spearman")["median"],
            "pseudotime_report": base["pseudotime"].report(dataset.feature_matrix, dataset.metadata_rows),
        },
        "hypotheses": {
            "H1_pT217_tau_decoupling": h1_summary,
            "H2_transcriptomic_resilience_gating": h2_summary,
            "H3_PART_AD_continuum": h3_summary,
        },
        "hypothesis_status": hypothesis_status_rows,
        "recommended_experiments": recommended_experiments(),
        "figures": figures,
        "tables": {
            "h1": str(output_path / "h1_pt217_decoupling.csv"),
            "h3": str(output_path / "h3_part_ad_route.csv"),
            "pseudotime_sensitivity": str(output_path / "pseudotime_sensitivity.csv"),
            "parent_ablation": str(output_path / "parent_ablation.csv"),
            "z_bins": str(output_path / "z_bin_diagnostics.csv"),
            "subjects": str(output_path / "representative_subjects.csv"),
        },
        "limitations": [
            "This is still a ridge/bootstrap Dynamic BN-SCM prototype, not a full posterior graph MCMC.",
            "H2 cannot be tested without AHBA or another regional gene-expression matrix aligned to DK/aparc.",
            "Brain maps show selected BN-SCM target regions; unmodelled DK regions are greyed out.",
            "A/T grouping uses a train-derived tau threshold because no project-level tau-positivity threshold is configured.",
        ],
    }
    json_write(output_path / "hypothesis_experiment_report.json", report)
    print(f"[BN-SCM hypotheses] Wrote report: {output_path / 'hypothesis_experiment_report.json'}")
    return report


def fit_scenario(
    dataset: MultimodalPairDataset,
    split: Any,
    *,
    target_names: list[str],
    pseudotime_mode: str,
    max_parents_per_target: int,
    exclude_features: tuple[str, ...] = (),
) -> dict[str, Any]:
    working = drop_features(dataset, exclude_features) if exclude_features else dataset
    pseudotime = fit_pseudotime(working.feature_matrix, working.feature_names, split.train_indices, mode=pseudotime_mode)
    fit = fit_dynamic_scm(
        working,
        pseudotime,
        split.train_indices,
        target_names=target_names,
        max_parents_per_target=max_parents_per_target,
    )
    target_indices = [working.target_index(name) for name in target_names]
    pred_rates_full = fit.predict_rates(working)
    pred_rates = pred_rates_full[:, target_indices]
    baseline = working.target_baseline[:, target_indices]
    observed = working.target_observed[:, target_indices]
    predicted = baseline + working.time_years[:, None] * pred_rates
    pair_rows = compute_pair_rows(
        model="Dynamic BN-SCM",
        baseline=baseline,
        observed=observed,
        predicted=predicted,
        time_years=working.time_years,
        metadata_rows=working.metadata_rows,
        z=pseudotime.transform(working.feature_matrix),
        split=split,
    )
    return {
        "dataset": working,
        "pseudotime": pseudotime,
        "fit": fit,
        "target_names": target_names,
        "target_indices": target_indices,
        "baseline": baseline,
        "observed": observed,
        "predicted": predicted,
        "predicted_rates": pred_rates,
        "observed_rates": (observed - baseline) / working.time_years[:, None],
        "z": pseudotime.transform(working.feature_matrix),
        "pair_rows": pair_rows,
        "exclude_features": exclude_features,
        "pseudotime_mode": pseudotime_mode,
    }


def drop_features(dataset: MultimodalPairDataset, exclude_features: tuple[str, ...]) -> MultimodalPairDataset:
    exclude = set(exclude_features)
    keep = [idx for idx, name in enumerate(dataset.feature_names) if name not in exclude]
    feature_names = [dataset.feature_names[idx] for idx in keep]
    return replace(
        dataset,
        feature_names=feature_names,
        feature_matrix=dataset.feature_matrix[:, keep],
        variable_specs=default_variable_specs(feature_names, dataset.target_names),
    )


def compute_pair_rows(
    *,
    model: str,
    baseline: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    time_years: np.ndarray,
    metadata_rows: list[dict[str, Any]],
    z: np.ndarray,
    split: Any,
) -> list[dict[str, Any]]:
    labels = split_labels(split, observed.shape[0])
    rows = []
    for idx in range(observed.shape[0]):
        dt = float(time_years[idx])
        obs_rate = (observed[idx] - baseline[idx]) / dt
        pred_rate = (predicted[idx] - baseline[idx]) / dt
        rows.append(
            {
                "model": model,
                "split": labels[idx],
                "row_index": idx,
                "RID": metadata_rows[idx].get("RID", ""),
                "PTID": metadata_rows[idx].get("PTID", ""),
                "diagnosis": metadata_rows[idx].get("dx_nearest_baseline", ""),
                "target_time_years": dt,
                "z": float(z[idx]),
                "mae_suvr": mean_abs(predicted[idx] - observed[idx]),
                "rate_mae": mean_abs(pred_rate - obs_rate),
                "rate_rmse": rmse(pred_rate - obs_rate),
                "subject_spearman": safe_corr(observed[idx], predicted[idx], rank=True),
                "delta_spearman": safe_corr(observed[idx] - baseline[idx], predicted[idx] - baseline[idx], rank=True),
            }
        )
    return rows


def split_labels(split: Any, n: int) -> list[str]:
    labels = ["unknown"] * n
    for name, indices in (
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
    ):
        for idx in indices:
            labels[int(idx)] = name
    return labels


def evaluate_h1_decoupling(base: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fit = base["fit"]
    basis = fit.spline_basis.transform(fit.z_grid)
    rows = []
    for target_fit in fit.target_fits:
        if not target_fit.target_name.startswith("tau_rate:"):
            continue
        pt217 = target_fit.parent_effect_curve("plasma_pt217", basis)
        self_effect = target_fit.self_effect_curve(basis)
        early = fit.z_grid <= 0.30
        mid = (fit.z_grid > 0.30) & (fit.z_grid < 0.70)
        late = fit.z_grid >= 0.70
        early_abs = float(np.mean(np.abs(pt217[early])))
        mid_abs = float(np.mean(np.abs(pt217[mid])))
        late_abs = float(np.mean(np.abs(pt217[late])))
        self_early_abs = float(np.mean(np.abs(self_effect[early])))
        self_late_abs = float(np.mean(np.abs(self_effect[late])))
        max_abs = float(np.max(np.abs(pt217))) if pt217.size else 0.0
        decoupling_z = first_z_below(fit.z_grid, np.abs(pt217), threshold=0.005, after=0.30)
        rows.append(
            {
                "target": target_fit.target_name,
                "pt217_parent_selected": "plasma_pt217" in target_fit.parent_names,
                "pt217_early_abs_mean": early_abs,
                "pt217_mid_abs_mean": mid_abs,
                "pt217_late_abs_mean": late_abs,
                "pt217_max_abs": max_abs,
                "pt217_late_less_than_early": late_abs < early_abs,
                "pt217_late_near_zero": late_abs < 0.005,
                "self_early_abs_mean": self_early_abs,
                "self_late_abs_mean": self_late_abs,
                "self_late_greater_than_early": self_late_abs > self_early_abs,
                "z_decouple_threshold_0_005": decoupling_z,
                "strict_h1_pattern": (early_abs >= 0.005 and late_abs < 0.005 and late_abs < early_abs and self_late_abs > self_early_abs),
            }
        )
    summary = {
        "claim": "Soluble pT217 directly influences future regional tau early, then decouples late while self-history rises.",
        "test_type": "descriptive varying-effect curve from ridge Dynamic BN-SCM; not posterior PIP.",
        "target_count": len(rows),
        "pt217_selected_fraction": fraction(row["pt217_parent_selected"] for row in rows),
        "late_less_than_early_fraction": fraction(row["pt217_late_less_than_early"] for row in rows),
        "late_near_zero_fraction": fraction(row["pt217_late_near_zero"] for row in rows),
        "strict_h1_pattern_fraction": fraction(row["strict_h1_pattern"] for row in rows),
        "mean_pt217_early_abs": safe_mean([row["pt217_early_abs_mean"] for row in rows]),
        "mean_pt217_late_abs": safe_mean([row["pt217_late_abs_mean"] for row in rows]),
        "interpretation": "Not supported as a strong decoupling result unless strict_h1_pattern_fraction is high; current evidence is exploratory.",
    }
    return rows, summary


def evaluate_h2_data_gate(root: Path, base: dict[str, Any]) -> dict[str, Any]:
    candidate_files = []
    for directory in [root / "experiments", root / "BayesianNetwork-SCM", root / "data"]:
        if not directory.exists():
            continue
        for pattern in ("*AHBA*", "*Allen*", "*transcript*", "*expression*", "*gene*", "*resilien*"):
            candidate_files.extend(str(path.relative_to(root)) for path in directory.rglob(pattern) if path.is_file())
            if len(candidate_files) > 50:
                break
    fit = base["fit"]
    basis = fit.spline_basis.transform(fit.z_grid)
    amyloid_max = []
    amyloid_targets = 0
    for target_fit in fit.target_fits:
        effect = target_fit.parent_effect_curve("amyloid_summary_suvr", basis)
        max_abs = float(np.max(np.abs(effect))) if effect.size else 0.0
        amyloid_max.append(max_abs)
        if "amyloid_summary_suvr" in target_fit.parent_names:
            amyloid_targets += 1
    has_gene_data = len(candidate_files) > 0
    return {
        "claim": "Regional selective-resilience gene expression gates the amyloid-to-tau cascade.",
        "status": "not_testable_current_data" if not has_gene_data else "candidate_gene_files_found",
        "candidate_gene_files": sorted(candidate_files)[:50],
        "required_data": [
            "AHBA or other regional expression matrix aligned to DK/aparc labels",
            "gene module scores for SR-NA or named genes such as HSP90, SV2A, GAP43",
            "interaction design: amyloid_region_or_summary * regional_expression",
        ],
        "precondition_amyloid_parent_selected_fraction": float(amyloid_targets / max(len(fit.target_fits), 1)),
        "precondition_mean_amyloid_max_abs_effect": safe_mean(amyloid_max),
        "interpretation": (
            "The amyloid parent is evaluable, but transcriptomic gating itself is not testable "
            "until region-level expression features are added."
        ),
    }


def evaluate_h3_part_ad(
    dataset: MultimodalPairDataset,
    split: Any,
    base: dict[str, Any],
    selected_regions: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    amy = dataset.feature_matrix[:, dataset.feature_index("amyloid_positive")]
    tau = dataset.feature_matrix[:, dataset.feature_index("tau_meta_temporal")]
    train_cn = [
        int(idx)
        for idx in split.train_indices
        if dataset.metadata_rows[int(idx)].get("dx_nearest_baseline", "") == "CN" and np.isfinite(tau[int(idx)])
    ]
    tau_threshold = float(np.quantile(tau[train_cn], 0.75)) if train_cn else float(np.nanquantile(tau, 0.75))
    groups = []
    for idx in range(dataset.pair_count):
        if not np.isfinite(amy[idx]) or not np.isfinite(tau[idx]):
            groups.append("unclassified")
            continue
        groups.append(("A+" if amy[idx] >= 0.5 else "A-") + ("T+" if tau[idx] >= tau_threshold else "T-"))
    group_array = np.asarray(groups, dtype=object)
    rows = []
    summary_by_split = {}
    for split_name, indices in [
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
        ("all", np.arange(dataset.pair_count, dtype=int)),
    ]:
        part_idx = np.asarray([idx for idx in indices if group_array[int(idx)] == "A-T+"], dtype=int)
        ad_idx = np.asarray([idx for idx in indices if group_array[int(idx)] == "A+T+"], dtype=int)
        result = part_ad_route_metrics(base, part_idx, ad_idx, selected_regions)
        result["split"] = split_name
        rows.append(result)
        summary_by_split[split_name] = result
    test = summary_by_split["test"]
    summary = {
        "claim": "PART-like A-T+ and AD-continuum A+T+ share a spatial tau route, with AD showing kinetic acceleration.",
        "a_status_source": "amyloid_positive feature from nearest baseline amyloid PET",
        "t_status_source": "tau_meta_temporal >= train CN 75th percentile",
        "tau_positive_threshold": tau_threshold,
        "group_counts_all": {label: int(np.sum(group_array == label)) for label in sorted(set(group_array))},
        "test_result": test,
        "route_similarity_supported": bool(test.get("observed_route_spearman", float("nan")) >= 0.70),
        "kinetic_acceleration_supported": bool(test.get("observed_kinetic_ratio_ad_over_part", float("nan")) > 1.20),
        "interpretation": (
            "Supports a shared-route claim only if A-T+ and A+T+ spatial-rate vectors are highly correlated; "
            "supports acceleration only if A+T+ rate magnitude is materially larger."
        ),
    }
    return rows, summary


def part_ad_route_metrics(base: dict[str, Any], part_idx: np.ndarray, ad_idx: np.ndarray, selected_regions: list[str]) -> dict[str, Any]:
    obs_rate = base["observed_rates"]
    pred_rate = base["predicted_rates"]
    empty = {
        "part_like_n": int(part_idx.size),
        "ad_continuum_n": int(ad_idx.size),
        "observed_route_spearman": float("nan"),
        "predicted_route_spearman": float("nan"),
        "observed_kinetic_ratio_ad_over_part": float("nan"),
        "predicted_kinetic_ratio_ad_over_part": float("nan"),
        "part_observed_rate_mean_by_region": {},
        "ad_observed_rate_mean_by_region": {},
        "part_predicted_rate_mean_by_region": {},
        "ad_predicted_rate_mean_by_region": {},
    }
    if part_idx.size < 3 or ad_idx.size < 3:
        return empty
    part_obs = np.nanmean(obs_rate[part_idx], axis=0)
    ad_obs = np.nanmean(obs_rate[ad_idx], axis=0)
    part_pred = np.nanmean(pred_rate[part_idx], axis=0)
    ad_pred = np.nanmean(pred_rate[ad_idx], axis=0)
    return {
        "part_like_n": int(part_idx.size),
        "ad_continuum_n": int(ad_idx.size),
        "observed_route_spearman": safe_corr(part_obs, ad_obs, rank=True),
        "predicted_route_spearman": safe_corr(part_pred, ad_pred, rank=True),
        "observed_route_pearson": safe_corr(part_obs, ad_obs, rank=False),
        "predicted_route_pearson": safe_corr(part_pred, ad_pred, rank=False),
        "observed_kinetic_ratio_ad_over_part": vector_magnitude(ad_obs) / max(vector_magnitude(part_obs), 1.0e-12),
        "predicted_kinetic_ratio_ad_over_part": vector_magnitude(ad_pred) / max(vector_magnitude(part_pred), 1.0e-12),
        "part_observed_rate_mean_by_region": dict(zip(selected_regions, map(float, part_obs), strict=True)),
        "ad_observed_rate_mean_by_region": dict(zip(selected_regions, map(float, ad_obs), strict=True)),
        "part_predicted_rate_mean_by_region": dict(zip(selected_regions, map(float, part_pred), strict=True)),
        "ad_predicted_rate_mean_by_region": dict(zip(selected_regions, map(float, ad_pred), strict=True)),
    }


def evaluate_pseudotime_sensitivity(
    dataset: MultimodalPairDataset,
    split: Any,
    target_names: list[str],
    *,
    max_parents_per_target: int,
) -> list[dict[str, Any]]:
    rows = []
    for mode in ["tau_free", "global", "clinical_free", "pt217_free"]:
        scenario = fit_scenario(
            dataset,
            split,
            target_names=target_names,
            pseudotime_mode=mode,
            max_parents_per_target=max_parents_per_target,
        )
        rows.append(
            {
                "scenario": mode,
                "type": "pseudotime_mode",
                "excluded_features": "",
                "test_rate_mae_median": summarize_metric(scenario["pair_rows"], "test", "rate_mae")["median"],
                "test_rate_rmse_median": summarize_metric(scenario["pair_rows"], "test", "rate_rmse")["median"],
                "test_delta_spearman_median": summarize_metric(scenario["pair_rows"], "test", "delta_spearman")["median"],
                "validation_rate_mae_median": summarize_metric(scenario["pair_rows"], "validation", "rate_mae")["median"],
                "selected_pseudotime_features": len(scenario["pseudotime"].selected_feature_names),
            }
        )
    return rows


def evaluate_parent_ablations(
    dataset: MultimodalPairDataset,
    split: Any,
    target_names: list[str],
    *,
    max_parents_per_target: int,
) -> list[dict[str, Any]]:
    ablations = [
        ("full", ()),
        ("no_pt217", ("plasma_pt217",)),
        ("no_amyloid_pet", ("amyloid_summary_suvr", "amyloid_centiloids", "amyloid_positive")),
        ("no_plasma_abeta_ratio", ("plasma_ab42_ab40",)),
        ("no_apoe4", ("apoe4_dose",)),
    ]
    rows = []
    for name, excluded in ablations:
        scenario = fit_scenario(
            dataset,
            split,
            target_names=target_names,
            pseudotime_mode="tau_free",
            max_parents_per_target=max_parents_per_target,
            exclude_features=excluded,
        )
        rows.append(
            {
                "scenario": name,
                "type": "parent_ablation",
                "excluded_features": ";".join(excluded),
                "test_rate_mae_median": summarize_metric(scenario["pair_rows"], "test", "rate_mae")["median"],
                "test_rate_rmse_median": summarize_metric(scenario["pair_rows"], "test", "rate_rmse")["median"],
                "test_delta_spearman_median": summarize_metric(scenario["pair_rows"], "test", "delta_spearman")["median"],
                "validation_rate_mae_median": summarize_metric(scenario["pair_rows"], "validation", "rate_mae")["median"],
                "selected_pseudotime_features": len(scenario["pseudotime"].selected_feature_names),
            }
        )
    base_mae = next(row["test_rate_mae_median"] for row in rows if row["scenario"] == "full")
    for row in rows:
        row["test_rate_mae_delta_vs_full"] = float(row["test_rate_mae_median"] - base_mae)
    return rows


def evaluate_z_bins(base: dict[str, Any], split: Any) -> list[dict[str, Any]]:
    z = base["z"]
    rows = []
    bins = [("early", 0.0, 0.33), ("middle", 0.33, 0.66), ("late", 0.66, 1.000001)]
    for split_name, split_indices in [
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
    ]:
        for label, lo, hi in bins:
            selected = np.asarray([idx for idx in split_indices if lo <= z[int(idx)] < hi], dtype=int)
            if selected.size == 0:
                rows.append(empty_z_bin_row(split_name, label, lo, hi))
                continue
            rate_error = base["predicted_rates"][selected] - base["observed_rates"][selected]
            rows.append(
                {
                    "split": split_name,
                    "z_bin": label,
                    "z_lower": lo,
                    "z_upper": hi,
                    "n_pairs": int(selected.size),
                    "rate_mae": mean_abs(rate_error),
                    "rate_rmse": rmse(rate_error),
                    "delta_spearman_median": safe_median(
                        [
                            safe_corr(
                                base["observed"][idx] - base["baseline"][idx],
                                base["predicted"][idx] - base["baseline"][idx],
                                rank=True,
                            )
                            for idx in selected
                        ]
                    ),
                }
            )
    return rows


def empty_z_bin_row(split_name: str, label: str, lo: float, hi: float) -> dict[str, Any]:
    return {
        "split": split_name,
        "z_bin": label,
        "z_lower": lo,
        "z_upper": hi,
        "n_pairs": 0,
        "rate_mae": float("nan"),
        "rate_rmse": float("nan"),
        "delta_spearman_median": float("nan"),
    }


def select_representative_subjects(dataset: MultimodalPairDataset, split: Any, base: dict[str, Any]) -> list[dict[str, Any]]:
    test_indices = np.asarray(split.test_indices, dtype=int)
    obs_delta = base["observed"] - base["baseline"]
    base_mean = np.nanmean(base["baseline"], axis=1)
    delta_mean = np.nanmean(obs_delta, axis=1)
    delta_abs = np.nanmean(np.abs(obs_delta), axis=1)

    def pick(label: str, diagnosis: str, score: np.ndarray, largest: bool) -> dict[str, Any]:
        candidates = [
            int(idx)
            for idx in test_indices
            if str(dataset.metadata_rows[int(idx)].get("dx_nearest_baseline", "")) == diagnosis
        ]
        if not candidates:
            candidates = [int(idx) for idx in test_indices]
        values = np.asarray([score[idx] for idx in candidates], dtype=float)
        order = int(np.nanargmax(values) if largest else np.nanargmin(values))
        idx = candidates[order]
        return {
            "label": label,
            "row_index": idx,
            "RID": dataset.metadata_rows[idx].get("RID", ""),
            "PTID": dataset.metadata_rows[idx].get("PTID", ""),
            "diagnosis": dataset.metadata_rows[idx].get("dx_nearest_baseline", ""),
            "z": float(base["z"][idx]),
            "baseline_mean_tau": float(base_mean[idx]),
            "observed_mean_delta": float(delta_mean[idx]),
            "observed_abs_delta": float(delta_abs[idx]),
        }

    return [
        pick("CN stable", "CN", delta_abs, largest=False),
        pick("MCI progressing", "MCI", delta_mean, largest=True),
        pick("AD high baseline", "AD", base_mean, largest=True),
    ]


def hypothesis_status(h1: dict[str, Any], h2: dict[str, Any], h3: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "hypothesis": "H1 pT217 -> tau decoupling",
            "status": "weak / not confirmed" if h1["strict_h1_pattern_fraction"] < 0.5 else "supported",
            "evidence": f"strict pattern fraction={h1['strict_h1_pattern_fraction']:.2f}; late<early={h1['late_less_than_early_fraction']:.2f}",
        },
        {
            "hypothesis": "H2 transcriptomic resilience gating",
            "status": "not testable",
            "evidence": "no AHBA/regional gene-expression matrix found in project outputs",
        },
        {
            "hypothesis": "H3 PART vs AD shared route",
            "status": "supported" if h3["route_similarity_supported"] else "not confirmed",
            "evidence": f"test route rho={h3['test_result'].get('observed_route_spearman', float('nan')):.2f}; kinetic ratio={h3['test_result'].get('observed_kinetic_ratio_ad_over_part', float('nan')):.2f}",
        },
    ]


def recommended_experiments() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "experiment": "Full posterior edge sampling for the current leakage-controlled design",
            "reason": "The present edge curves are ridge estimates; H1 and causal PIP claims require posterior uncertainty.",
        },
        {
            "priority": "2",
            "experiment": "AHBA DK/aparc expression join for H2",
            "reason": "Transcriptomic gating cannot be evaluated until regional SR/SV gene-module scores are available.",
        },
        {
            "priority": "3",
            "experiment": "A/T threshold sensitivity for PART-like H3",
            "reason": "The current T+ cutoff is train-derived; robustness over accepted tau-positivity thresholds is needed.",
        },
        {
            "priority": "4",
            "experiment": "Longitudinal calibration curves by pseudotime bin",
            "reason": "Shows whether BN-SCM only helps early disease or also handles late autonomous tau dynamics.",
        },
        {
            "priority": "5",
            "experiment": "Target expansion to all 68 regional tau rates",
            "reason": "The brain-map diagnostics are selected-region only; whole-cortex forecasting is needed for spatial route claims.",
        },
    ]


# ---------------------------------------------------------------------------
# SVG figures
# ---------------------------------------------------------------------------


def write_hypothesis_status_svg(path: Path, rows: list[dict[str, str]]) -> None:
    width = 1120
    height = 330
    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, "BN-SCM Hypothesis Test Status", size=23, weight="700"))
    parts.append(svg_text(36, 64, "Statuses are based on the current leakage-controlled ridge Dynamic BN-SCM prototype.", size=13, fill="#4B5563"))
    y0 = 104
    for idx, row in enumerate(rows):
        y = y0 + idx * 66
        color = {"supported": GREEN, "not testable": GRAY}.get(row["status"], GOLD)
        parts.append(svg_rect(36, y - 28, 1048, 52, fill="#FFFFFF", stroke="#E5E7EB", radius=6))
        parts.append(svg_rect(36, y - 28, 8, 52, fill=color, radius=4))
        parts.append(svg_text(60, y - 7, row["hypothesis"], size=15, weight="700"))
        parts.append(svg_text(372, y - 7, row["status"], size=14, fill=color, weight="700"))
        parts.append(svg_text(520, y - 7, row["evidence"], size=12, fill="#374151"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_h1_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width = 1180
    height = 570
    plot_x = 78
    plot_y = 104
    plot_w = 710
    plot_h = 330
    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, "H1: pT217-to-Tau Decoupling Test", size=23, weight="700"))
    parts.append(svg_text(36, 64, "Each line connects early, middle, and late absolute pT217 effect per target. Lower late effect is the decoupling signature.", size=13, fill="#4B5563"))
    max_value = max([float(row["pt217_max_abs"]) for row in rows] + [0.012])
    parts.append(svg_rect(plot_x, plot_y, plot_w, plot_h, fill="#FFFFFF", stroke="#E5E7EB", radius=6))
    for tick in np.linspace(0.0, max_value, 5):
        y = plot_y + plot_h - (tick / max_value) * (plot_h - 30) - 15
        parts.append(svg_line(plot_x, y, plot_x + plot_w, y, "#E5E7EB", 1))
        parts.append(svg_text(plot_x - 10, y + 4, f"{tick:.3f}", size=10, anchor="end", fill="#6B7280"))
    x_positions = [plot_x + 120, plot_x + 350, plot_x + 580]
    labels = ["early Z<=0.3", "mid", "late Z>=0.7"]
    for x, label in zip(x_positions, labels, strict=True):
        parts.append(svg_text(x, plot_y + plot_h + 28, label, size=12, anchor="middle", fill="#374151"))
    colors = [MODEL_COLOR, BLUE, GREEN, PINK, GOLD, "#56B4E9", "#8B5CF6", "#A16207", "#475569", "#EF4444"]
    for idx, row in enumerate(rows):
        vals = [row["pt217_early_abs_mean"], row["pt217_mid_abs_mean"], row["pt217_late_abs_mean"]]
        points = []
        for x, value in zip(x_positions, vals, strict=True):
            y = plot_y + plot_h - (float(value) / max_value) * (plot_h - 30) - 15
            points.append((x, y))
        color = colors[idx % len(colors)]
        parts.append(svg_polyline(points, color, width=2.0, opacity=0.82))
        for x, y in points:
            parts.append(svg_circle(x, y, 3.8, fill=color, opacity=0.9))
        legend_y = 110 + idx * 31
        parts.append(svg_line(830, legend_y - 4, 856, legend_y - 4, color, 2.5))
        target = str(row["target"]).replace("tau_rate:", "").replace("_", " ")
        flag = "strict" if row["strict_h1_pattern"] else "no"
        parts.append(svg_text(866, legend_y, f"{target}: {flag}", size=11, fill="#111827"))
    parts.append(svg_text(plot_x + plot_w / 2, height - 28, "Pseudotime window", size=13, anchor="middle", fill="#374151"))
    parts.append(svg_text(32, plot_y + 20, "abs effect", size=11, fill="#374151"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_ablation_svg(path: Path, pseudotime_rows: list[dict[str, Any]], ablation_rows: list[dict[str, Any]]) -> None:
    width = 1180
    height = 640
    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, "Model Sensitivity: Pseudotime Modes and Parent Ablations", size=23, weight="700"))
    parts.append(svg_text(36, 64, "Lower test rate MAE is better. Parent ablation deltas are relative to the full tau-free model.", size=13, fill="#4B5563"))
    parts.extend(draw_horizontal_bars(54, 110, 500, 430, "Pseudotime mode test rate MAE", pseudotime_rows, "scenario", "test_rate_mae_median", lower_better=True))
    parts.extend(draw_horizontal_bars(630, 110, 500, 430, "Parent ablation delta vs full", ablation_rows, "scenario", "test_rate_mae_delta_vs_full", lower_better=True, zero_line=True))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def draw_horizontal_bars(
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
    *,
    lower_better: bool,
    zero_line: bool = False,
) -> list[str]:
    parts = [svg_rect(x0, y0, width, height, fill="#FFFFFF", stroke="#E5E7EB", radius=6), svg_text(x0 + 18, y0 + 30, title, size=15, weight="700")]
    values = np.asarray([float(row[value_key]) for row in rows if np.isfinite(float(row[value_key]))], dtype=float)
    if values.size == 0:
        return parts
    vmin = min(0.0, float(np.min(values))) if zero_line else 0.0
    vmax = max(float(np.max(values)), 1.0e-6)
    if zero_line:
        span = max(vmax - vmin, 1.0e-6)
        vmin -= 0.05 * span
        vmax += 0.05 * span
    bar_x = x0 + 175
    bar_w = width - 250
    start_y = y0 + 68
    row_h = 52
    zero_x = bar_x + (0.0 - vmin) / max(vmax - vmin, 1.0e-12) * bar_w
    if zero_line:
        parts.append(svg_line(zero_x, start_y - 18, zero_x, start_y + row_h * len(rows) - 18, "#9CA3AF", 1))
    for idx, row in enumerate(rows):
        y = start_y + idx * row_h
        label = str(row[label_key])
        value = float(row[value_key])
        x_value = bar_x + (value - vmin) / max(vmax - vmin, 1.0e-12) * bar_w
        left = min(zero_x, x_value) if zero_line else bar_x
        bw = max(abs(x_value - zero_x), 1.0) if zero_line else max(x_value - bar_x, 1.0)
        color = MODEL_COLOR if idx == 0 else BLUE
        if zero_line and value > 0:
            color = GOLD
        parts.append(svg_text(x0 + 18, y + 17, label, size=12, fill="#111827", weight="700" if idx == 0 else "400"))
        parts.append(svg_rect(left, y, bw, 24, fill=color, radius=4, opacity=0.9))
        parts.append(svg_text(bar_x + bar_w + 12, y + 17, format_float(value), size=12, fill="#111827"))
    return parts


def write_group_brain_svg(path: Path, region_labels: list[str], selected_regions: list[str], base: dict[str, Any], split: Any) -> None:
    idx = np.asarray(split.test_indices, dtype=int)
    baseline = np.nanmean(base["baseline"][idx], axis=0)
    observed = np.nanmean(base["observed"][idx], axis=0)
    predicted = np.nanmean(base["predicted"][idx], axis=0)
    obs_delta = observed - baseline
    pred_delta = predicted - baseline
    error = predicted - observed
    suvr_vals = np.concatenate([baseline, observed, predicted])
    delta_bound = max(float(np.nanmax(np.abs(np.concatenate([obs_delta, pred_delta, error])))), 0.02)
    panels = [
        {"title": "Initial baseline tau", "values": selected_to_full(region_labels, selected_regions, baseline), "mode": "sequential", "vmin": float(np.nanmin(suvr_vals)), "vmax": float(np.nanmax(suvr_vals))},
        {"title": "Empirical follow-up tau", "values": selected_to_full(region_labels, selected_regions, observed), "mode": "sequential", "vmin": float(np.nanmin(suvr_vals)), "vmax": float(np.nanmax(suvr_vals))},
        {"title": "BN-SCM predicted follow-up", "values": selected_to_full(region_labels, selected_regions, predicted), "mode": "sequential", "vmin": float(np.nanmin(suvr_vals)), "vmax": float(np.nanmax(suvr_vals))},
        {"title": "Empirical change", "values": selected_to_full(region_labels, selected_regions, obs_delta), "mode": "diverging", "vmin": -delta_bound, "vmax": delta_bound},
        {"title": "Predicted change", "values": selected_to_full(region_labels, selected_regions, pred_delta), "mode": "diverging", "vmin": -delta_bound, "vmax": delta_bound},
        {"title": "Prediction error", "values": selected_to_full(region_labels, selected_regions, error), "mode": "diverging", "vmin": -delta_bound, "vmax": delta_bound},
    ]
    write_brain_panel_grid(path, panels, region_labels, "Group-Average Test Brain Maps: Baseline vs Empirical vs BN-SCM")


def write_h3_brain_svg(path: Path, region_labels: list[str], selected_regions: list[str], h3_summary: dict[str, Any]) -> None:
    test = h3_summary["test_result"]
    panels = []
    for title, key in [
        ("A-T+ observed rate", "part_observed_rate_mean_by_region"),
        ("A+T+ observed rate", "ad_observed_rate_mean_by_region"),
        ("A-T+ predicted rate", "part_predicted_rate_mean_by_region"),
        ("A+T+ predicted rate", "ad_predicted_rate_mean_by_region"),
    ]:
        values = np.asarray([test.get(key, {}).get(region, float("nan")) for region in selected_regions], dtype=float)
        panels.append({"title": title, "values": selected_to_full(region_labels, selected_regions, values), "mode": "diverging"})
    finite_panel_values = [panel["values"][np.isfinite(panel["values"])] for panel in panels if np.isfinite(panel["values"]).any()]
    all_values = np.concatenate(finite_panel_values) if finite_panel_values else np.asarray([], dtype=float)
    bound = max(float(np.nanmax(np.abs(all_values))) if all_values.size else 0.02, 0.02)
    for panel in panels:
        panel["vmin"] = -bound
        panel["vmax"] = bound
    title = "H3 PART-like vs AD-Continuum Spatial Route"
    write_brain_panel_grid(path, panels, region_labels, title, ncols=2)


def write_subject_brain_svg(path: Path, region_labels: list[str], selected_regions: list[str], base: dict[str, Any], subject_row: dict[str, Any]) -> None:
    idx = int(subject_row["row_index"])
    baseline = base["baseline"][idx]
    observed = base["observed"][idx]
    predicted = base["predicted"][idx]
    obs_delta = observed - baseline
    pred_delta = predicted - baseline
    error = predicted - observed
    suvr_vals = np.concatenate([baseline, observed, predicted])
    delta_bound = max(float(np.nanmax(np.abs(np.concatenate([obs_delta, pred_delta, error])))), 0.02)
    panels = [
        {"title": "Initial tau", "values": selected_to_full(region_labels, selected_regions, baseline), "mode": "sequential", "vmin": float(np.nanmin(suvr_vals)), "vmax": float(np.nanmax(suvr_vals))},
        {"title": "Empirical follow-up", "values": selected_to_full(region_labels, selected_regions, observed), "mode": "sequential", "vmin": float(np.nanmin(suvr_vals)), "vmax": float(np.nanmax(suvr_vals))},
        {"title": "BN-SCM predicted", "values": selected_to_full(region_labels, selected_regions, predicted), "mode": "sequential", "vmin": float(np.nanmin(suvr_vals)), "vmax": float(np.nanmax(suvr_vals))},
        {"title": "Empirical change", "values": selected_to_full(region_labels, selected_regions, obs_delta), "mode": "diverging", "vmin": -delta_bound, "vmax": delta_bound},
        {"title": "Predicted change", "values": selected_to_full(region_labels, selected_regions, pred_delta), "mode": "diverging", "vmin": -delta_bound, "vmax": delta_bound},
        {"title": "Error", "values": selected_to_full(region_labels, selected_regions, error), "mode": "diverging", "vmin": -delta_bound, "vmax": delta_bound},
    ]
    title = f"{subject_row['label']}: RID {subject_row['RID']}, dx={subject_row['diagnosis']}, Z={subject_row['z']:.2f}"
    write_brain_panel_grid(path, panels, region_labels, title)


def write_brain_panel_grid(path: Path, panels: list[dict[str, Any]], region_labels: list[str], title: str, *, ncols: int = 3) -> None:
    panel_w = 360
    panel_h = 230
    margin = 36
    header = 82
    nrows = int(math.ceil(len(panels) / ncols))
    width = margin * 2 + ncols * panel_w
    height = header + nrows * panel_h + 36
    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, title, size=23, weight="700"))
    parts.append(svg_text(36, 64, "Schematic DK/aparc brain map. Colored nodes are the BN-SCM target regions; grey nodes were not modeled in this run.", size=12, fill="#4B5563"))
    for idx, panel in enumerate(panels):
        x = margin + (idx % ncols) * panel_w
        y = header + (idx // ncols) * panel_h
        parts.extend(draw_brain_panel(x, y, panel_w - 18, panel_h - 18, panel, region_labels))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def draw_brain_panel(x0: float, y0: float, width: float, height: float, panel: dict[str, Any], region_labels: list[str]) -> list[str]:
    values = np.asarray(panel["values"], dtype=float)
    vmin = float(panel.get("vmin", np.nanmin(values[np.isfinite(values)]) if np.isfinite(values).any() else 0.0))
    vmax = float(panel.get("vmax", np.nanmax(values[np.isfinite(values)]) if np.isfinite(values).any() else 1.0))
    mode = str(panel.get("mode", "sequential"))
    parts = [svg_rect(x0, y0, width, height, fill="#FFFFFF", stroke="#E5E7EB", radius=6), svg_text(x0 + 12, y0 + 24, panel.get("title", ""), size=13, weight="700")]
    brain_y = y0 + 42
    hemi_w = (width - 42) / 2
    hemi_h = height - 74
    for side_idx, side in enumerate(["L", "R"]):
        hx = x0 + 14 + side_idx * (hemi_w + 14)
        cy = brain_y + hemi_h / 2
        parts.append(f'<ellipse cx="{hx + hemi_w / 2:.2f}" cy="{cy:.2f}" rx="{hemi_w / 2:.2f}" ry="{hemi_h / 2.15:.2f}" fill="#F3F4F6" stroke="#CBD5E1" stroke-width="1"/>')
        for region in region_labels:
            if not region.startswith(side + "_"):
                continue
            coord = _DK_COORDS.get(region)
            if coord is None:
                continue
            rx, ry = coord
            px = hx + rx * hemi_w
            py = brain_y + (1.02 - ry) / 1.08 * hemi_h
            value = values[region_labels.index(region)]
            color = map_color(value, vmin, vmax, mode)
            stripped = region[2:]
            edge = "#111827" if stripped not in _MEDIAL_PARTS else "#9CA3AF"
            radius = 5.5 if np.isfinite(value) else 3.0
            parts.append(svg_circle(px, py, radius, fill=color, stroke=edge, opacity=0.95 if np.isfinite(value) else 0.45))
    # mini legend
    lx = x0 + 14
    ly = y0 + height - 20
    for i in range(84):
        value = vmin + (vmax - vmin) * i / 83
        parts.append(svg_rect(lx + i * 2, ly, 2, 8, fill=map_color(value, vmin, vmax, mode)))
    parts.append(svg_text(lx, ly + 20, format_float(vmin), size=9, fill="#6B7280"))
    parts.append(svg_text(lx + 168, ly + 20, format_float(vmax), size=9, fill="#6B7280", anchor="end"))
    return parts


def write_z_bin_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    test_rows = [row for row in rows if row["split"] == "test"]
    width = 820
    height = 420
    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, "Held-Out Error by Pseudotime Bin", size=23, weight="700"))
    parts.append(svg_text(36, 64, "Rate error is lowest where the model has enough comparable training support and strongest where trajectories are heterogeneous.", size=12, fill="#4B5563"))
    parts.extend(draw_horizontal_bars(70, 100, 680, 260, "Test rate MAE by Z bin", test_rows, "z_bin", "rate_mae", lower_better=True))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def load_region_labels(root: Path) -> list[str]:
    path = root / "experiments" / "group_average_enigma" / "adni_to_enigma_aparc_mapping.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["enigma_index"]))
    return [row["enigma_label"] for row in rows]


def selected_to_full(region_labels: list[str], selected_regions: list[str], values: np.ndarray) -> np.ndarray:
    out = np.full(len(region_labels), float("nan"), dtype=float)
    for region, value in zip(selected_regions, values, strict=True):
        out[region_labels.index(region)] = float(value)
    return out


def summarize_metric(rows: list[dict[str, Any]], split_name: str, metric: str) -> dict[str, float | int]:
    values = np.asarray([float(row[metric]) for row in rows if row["split"] == split_name and np.isfinite(float(row[metric]))], dtype=float)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "median": float(np.median(values)) if values.size else float("nan"),
    }


def first_z_below(z_grid: np.ndarray, values: np.ndarray, *, threshold: float, after: float) -> float:
    for z, value in zip(z_grid, values, strict=True):
        if z >= after and value < threshold:
            return float(z)
    return float("nan")


def mean_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    return float(np.mean(np.abs(arr[mask]))) if np.any(mask) else float("nan")


def rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    return float(np.sqrt(np.mean(arr[mask] ** 2))) if np.any(mask) else float("nan")


def safe_corr(a: np.ndarray, b: np.ndarray, *, rank: bool) -> float:
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


def safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def fraction(values: Any) -> float:
    items = list(values)
    return float(np.mean([bool(value) for value in items])) if items else float("nan")


def vector_magnitude(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def map_color(value: float, vmin: float, vmax: float, mode: str) -> str:
    if not np.isfinite(value):
        return "#CBD5E1"
    t = min(1.0, max(0.0, (float(value) - vmin) / max(vmax - vmin, 1.0e-12)))
    if mode == "diverging":
        if t < 0.5:
            return mix_hex("#2166AC", "#F7F7F7", t / 0.5)
        return mix_hex("#F7F7F7", "#B2182B", (t - 0.5) / 0.5)
    if t < 0.5:
        return mix_hex("#E8F2EE", "#E9C46A", t / 0.5)
    return mix_hex("#E9C46A", "#D55E00", (t - 0.5) / 0.5)


def mix_hex(a: str, b: str, t: float) -> str:
    ca = np.asarray([int(a[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    cb = np.asarray([int(b[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    c = ca + (cb - ca) * min(1.0, max(0.0, t))
    return "#" + "".join(f"{int(round(x)):02X}" for x in c)


def svg_header(width: float, height: float) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}</style>',
    ]


def svg_text(
    x: float,
    y: float,
    text: Any,
    *,
    size: int = 12,
    fill: str = "#111827",
    weight: str = "400",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(text))}</text>'
    )


def svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str | None = None,
    radius: float = 0.0,
    opacity: float = 1.0,
) -> str:
    stroke_attr = f' stroke="{stroke}"' if stroke else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'rx="{radius:.2f}" fill="{fill}"{stroke_attr} opacity="{opacity:.3f}"/>'
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float, *, opacity: float = 1.0) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width:.2f}" opacity="{opacity:.3f}"/>'


def svg_circle(x: float, y: float, radius: float, *, fill: str, stroke: str | None = None, opacity: float = 1.0) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="0.6"' if stroke else ""
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}"{stroke_attr} opacity="{opacity:.3f}"/>'


def svg_polyline(points: list[tuple[float, float]], stroke: str, *, width: float = 2.0, opacity: float = 1.0) -> str:
    text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{text}" fill="none" stroke="{stroke}" stroke-width="{width:.2f}" opacity="{opacity:.3f}"/>'


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    if abs(number) >= 1.0:
        return f"{number:.2f}"
    return f"{number:.3f}"


def main() -> int:
    run_hypothesis_experiments()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
