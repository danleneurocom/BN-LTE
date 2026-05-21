#!/usr/bin/env python3
"""Paper-grade validation and visualization suite for BN-LTE/BN-SCM.

The script intentionally keeps the core model code unchanged.  It orchestrates
repeated subject-level validation, leakage/ablation controls, stage-stratified
analysis, bootstrap edge stability, counterfactual response curves, and
multi-view nilearn surface figures from the existing ADNI-first BN-SCM
implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from bayesian_network_scm.data import MultimodalPairDataset, build_multimodal_pair_dataset  # noqa: E402
from bayesian_network_scm.dynamic_scm import DynamicSCMFit, fit_dynamic_scm  # noqa: E402
from bayesian_network_scm.pseudotime import PseudotimeModel, fit_pseudotime  # noqa: E402
from bayesian_network_scm.reporting import SubjectSplit, make_subject_split  # noqa: E402
from spread_toolbox.forecasting import MinMaxStateScaler, load_forecast_dataset, load_labeled_matrix  # noqa: E402
from spread_toolbox.io_adni import load_yaml_config  # noqa: E402
from spread_toolbox.models.esm import EpidemicSpreadingModel  # noqa: E402
from spread_toolbox.models.ndm import NetworkDiffusionModel  # noqa: E402
from spread_toolbox.models.sir import GraphSIRModel  # noqa: E402


MODEL_ORDER = ["BayesianNetwork-SCM", "NDM", "ESM", "SIR", "S0 persistence"]
PAPER_MODEL_ORDER = ["BayesianNetwork-SCM", "NDM", "ESM", "SIR", "S0 persistence"]
MODEL_COLORS = {
    "BayesianNetwork-SCM": "#D55E00",
    "NDM": "#0072B2",
    "ESM": "#009E73",
    "SIR": "#CC79A7",
    "S0 persistence": "#6B7280",
}
STAGE_COLORS = {"early": "#56B4E9", "mid": "#E69F00", "late": "#D55E00"}
REGION_SHORT_NAMES = {
    "L_entorhinal": "L-Ent",
    "R_entorhinal": "R-Ent",
    "L_fusiform": "L-Fus",
    "R_fusiform": "R-Fus",
    "L_inferiortemporal": "L-IT",
    "R_inferiortemporal": "R-IT",
    "L_middletemporal": "L-MT",
    "R_middletemporal": "R-MT",
    "L_inferiorparietal": "L-IP",
    "R_inferiorparietal": "R-IP",
}
DK_TO_DESTRIEUX = {
    "entorhinal": ["G_oc-temp_med-Parahip", "S_collat_transv_ant"],
    "fusiform": ["G_oc-temp_lat-fusifor", "S_oc-temp_lat"],
    "inferiortemporal": ["G_temporal_inf", "S_temporal_inf"],
    "middletemporal": ["G_temporal_middle"],
    "inferiorparietal": ["G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
}
SURFACE_VIEWS = [
    ("left", "lateral", "L lateral"),
    ("right", "lateral", "R lateral"),
    ("left", "medial", "L medial"),
    ("right", "medial", "R medial"),
    ("left", "ventral", "L ventral"),
    ("right", "ventral", "R ventral"),
]


class FixedZPseudotime:
    """Pseudotime negative-control wrapper with a fixed row-wise Z vector."""

    def __init__(self, base: PseudotimeModel, fixed_z: np.ndarray, mode: str):
        self.base = base
        self.fixed_z = np.asarray(fixed_z, dtype=float)
        self.mode = mode
        self.feature_names = base.feature_names
        self.selected_feature_names = base.selected_feature_names
        self.selected_indices = base.selected_indices
        self.explained_variance_ratio = base.explained_variance_ratio
        self.burden_correlation = base.burden_correlation

    def transform(self, feature_matrix: np.ndarray, *, clip: bool = True) -> np.ndarray:
        if int(feature_matrix.shape[0]) != int(self.fixed_z.size):
            raise ValueError("FixedZPseudotime can only transform the original row order.")
        z = self.fixed_z.copy()
        return np.clip(z, 0.0, 1.0) if clip else z

    def report(self, feature_matrix: np.ndarray, metadata_rows: list[dict[str, object]] | None = None) -> dict[str, object]:
        z = self.transform(feature_matrix)
        return {
            "mode": self.mode,
            "selected_feature_count": len(self.selected_feature_names),
            "selected_features": list(self.selected_feature_names),
            "explained_variance_ratio": float(self.explained_variance_ratio),
            "burden_correlation": float(self.burden_correlation),
            "z_min": float(np.min(z)),
            "z_median": float(np.median(z)),
            "z_max": float(np.max(z)),
            "negative_control": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "paper_validation")
    parser.add_argument("--random-seed", type=int, default=20260521)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--bootstrap-iterations", type=int, default=60)
    parser.add_argument("--max-parents", type=int, default=6)
    parser.add_argument("--skip-nilearn", action="store_true")
    args = parser.parse_args()

    report = run_paper_validation_suite(
        project_root=args.project_root,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        repeats=args.repeats,
        bootstrap_iterations=args.bootstrap_iterations,
        max_parents=args.max_parents,
        skip_nilearn=args.skip_nilearn,
    )
    print(render_console_summary(report))
    return 0


def run_paper_validation_suite(
    *,
    project_root: str | Path,
    output_dir: str | Path,
    random_seed: int,
    repeats: int,
    bootstrap_iterations: int,
    max_parents: int,
    skip_nilearn: bool,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    out = resolve_path(output_dir, root)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/7: loading and checking datasets")
    dataset = build_multimodal_pair_dataset(root)
    selected_regions = list(dataset.report["selected_tau_regions"])
    selected_target_names = [f"tau_rate:{region}" for region in selected_regions]
    selected_target_indices = [dataset.target_index(name) for name in selected_target_names]
    validate_dataset(dataset, selected_target_indices)
    graph = load_graph_resources(root, dataset, selected_regions)

    print("Step 2/7: fitting primary split models")
    primary_split = make_subject_split(dataset.metadata_rows, random_seed=random_seed)
    validate_split(primary_split)
    primary = fit_all_prediction_models(
        dataset=dataset,
        graph=graph,
        split=primary_split,
        selected_regions=selected_regions,
        selected_target_names=selected_target_names,
        selected_target_indices=selected_target_indices,
        max_parents=max_parents,
    )
    validate_predictions(primary["predictions"], dataset, selected_target_indices)
    primary_pair_rows = metric_rows_for_predictions(
        primary["predictions"],
        dataset,
        selected_target_indices,
        primary_split,
        selected_regions,
        experiment="primary",
        seed=random_seed,
    )
    primary_summary = summarize_metric_rows(primary_pair_rows, group_fields=["model", "split"])

    print("Step 3/7: repeated subject-level validation")
    repeated_rows = []
    split_seeds = [random_seed + 1009 * idx for idx in range(int(repeats))]
    for repeat_idx, seed in enumerate(split_seeds):
        print(f"  repeated split {repeat_idx + 1}/{len(split_seeds)} seed={seed}")
        split = make_subject_split(dataset.metadata_rows, random_seed=seed)
        validate_split(split)
        fitted = fit_all_prediction_models(
            dataset=dataset,
            graph=graph,
            split=split,
            selected_regions=selected_regions,
            selected_target_names=selected_target_names,
            selected_target_indices=selected_target_indices,
            max_parents=max_parents,
        )
        validate_predictions(fitted["predictions"], dataset, selected_target_indices)
        repeated_rows.extend(
            metric_rows_for_predictions(
                fitted["predictions"],
                dataset,
                selected_target_indices,
                split,
                selected_regions,
                experiment="repeated_split",
                seed=seed,
                repeat_index=repeat_idx,
            )
        )
    repeated_summary = summarize_metric_rows(repeated_rows, group_fields=["model", "split", "metric"])

    print("Step 4/7: leakage controls and feature ablations")
    ablation_rows, ablation_summary = run_ablation_suite(
        dataset=dataset,
        split=primary_split,
        selected_target_names=selected_target_names,
        selected_target_indices=selected_target_indices,
        selected_regions=selected_regions,
        max_parents=max_parents,
        random_seed=random_seed + 19,
    )

    print("Step 5/7: stage-stratified validation and counterfactual analysis")
    stage_rows = stage_stratified_metrics(
        predictions=primary["predictions"],
        dataset=dataset,
        split=primary_split,
        selected_target_indices=selected_target_indices,
        selected_regions=selected_regions,
        z_values=primary["z_values"],
    )
    counterfactual_rows = counterfactual_response_rows(
        dataset=dataset,
        fit=primary["bn_fit"],
        split=primary_split,
        selected_target_indices=selected_target_indices,
        selected_regions=selected_regions,
        z_values=primary["z_values"],
    )

    print("Step 6/7: bootstrap edge stability and graph summaries")
    stability_rows = bootstrap_stage_edge_stability(
        dataset=dataset,
        base_pseudotime=primary["pseudotime"],
        split=primary_split,
        target_names=selected_target_names,
        iterations=bootstrap_iterations,
        random_seed=random_seed + 29,
        max_parents=max_parents,
    )
    edge_grid_rows = edge_curve_grid_rows(primary["bn_fit"], top_k=30)
    graph_rows = dynamic_graph_stage_rows(primary["bn_fit"], top_k_per_stage=12)

    print("Step 7/7: writing tables, paper figures, and report")
    write_csv_rows(out / "primary_pair_metrics.csv", primary_pair_rows)
    write_csv_rows(out / "primary_summary.csv", primary_summary)
    write_csv_rows(out / "repeated_split_metrics.csv", repeated_rows)
    write_csv_rows(out / "repeated_split_summary.csv", repeated_summary)
    write_csv_rows(out / "ablation_metrics.csv", ablation_rows)
    write_csv_rows(out / "ablation_summary.csv", ablation_summary)
    write_csv_rows(out / "stage_stratified_metrics.csv", stage_rows)
    write_csv_rows(out / "counterfactual_effects.csv", counterfactual_rows)
    write_csv_rows(out / "bootstrap_stage_edge_stability.csv", stability_rows)
    write_csv_rows(out / "edge_curve_grid.csv", edge_grid_rows)
    write_csv_rows(out / "dynamic_graph_stage_edges.csv", graph_rows)

    figures = make_all_figures(
        fig_dir=fig_dir,
        repeated_rows=repeated_rows,
        repeated_summary=repeated_summary,
        ablation_summary=ablation_summary,
        stage_rows=stage_rows,
        counterfactual_rows=counterfactual_rows,
        edge_grid_rows=edge_grid_rows,
        graph_rows=graph_rows,
        primary=primary,
        dataset=dataset,
        split=primary_split,
        selected_regions=selected_regions,
        selected_target_indices=selected_target_indices,
        skip_nilearn=skip_nilearn,
    )

    report = {
        "purpose": "Paper-grade validation and visualization suite for ADNI-first BN-LTE/BN-SCM.",
        "configuration": {
            "random_seed": int(random_seed),
            "repeats": int(repeats),
            "bootstrap_iterations": int(bootstrap_iterations),
            "max_parents": int(max_parents),
            "skip_nilearn": bool(skip_nilearn),
        },
        "data": {
            "pairs": dataset.pair_count,
            "subjects": len({row["RID"] for row in dataset.metadata_rows}),
            "selected_regions": selected_regions,
            "feature_count": len(dataset.feature_names),
        },
        "primary_split": primary_split.report(),
        "pseudotime": primary["pseudotime"].report(dataset.feature_matrix, dataset.metadata_rows),
        "fit_reports": primary["fit_reports"],
        "primary_test_summary": nested_summary(primary_summary, split_name="test"),
        "repeated_test_summary": nested_summary(repeated_summary, split_name="test"),
        "ablation_summary": ablation_summary,
        "stage_summary": summarize_stage_rows(stage_rows),
        "counterfactual_summary": summarize_counterfactual_rows(counterfactual_rows),
        "top_stable_edges": sorted(
            stability_rows,
            key=lambda row: (-float(row["inclusion_probability"]), -float(row["mean_abs_effect"])),
        )[:20],
        "figures": figures,
        "tables": {
            "primary_pair_metrics": str(out / "primary_pair_metrics.csv"),
            "primary_summary": str(out / "primary_summary.csv"),
            "repeated_split_metrics": str(out / "repeated_split_metrics.csv"),
            "repeated_split_summary": str(out / "repeated_split_summary.csv"),
            "ablation_metrics": str(out / "ablation_metrics.csv"),
            "ablation_summary": str(out / "ablation_summary.csv"),
            "stage_stratified_metrics": str(out / "stage_stratified_metrics.csv"),
            "counterfactual_effects": str(out / "counterfactual_effects.csv"),
            "bootstrap_stage_edge_stability": str(out / "bootstrap_stage_edge_stability.csv"),
            "edge_curve_grid": str(out / "edge_curve_grid.csv"),
            "dynamic_graph_stage_edges": str(out / "dynamic_graph_stage_edges.csv"),
        },
        "interpretation_guardrails": [
            "All splits are by participant identifier, not by row.",
            "BN-LTE causal direction comes from baseline-to-future-rate temporal ordering plus biological constraints.",
            "Counterfactual curves are model-implied intervention responses, not randomized clinical trial evidence.",
            "Bootstrap stability quantifies finite-sample robustness of ridge-estimated edge curves, not a full Bayesian graph posterior.",
        ],
    }
    write_json(out / "paper_validation_report.json", report)
    (out / "paper_validation_report.md").write_text(render_markdown_report(report), encoding="utf-8")
    return report


def validate_dataset(dataset: MultimodalPairDataset, target_indices: list[int]) -> None:
    if dataset.pair_count < 50:
        raise ValueError(f"Too few pairs for paper validation: {dataset.pair_count}")
    if len({row["RID"] for row in dataset.metadata_rows}) < 30:
        raise ValueError("Too few unique participants for subject-level validation.")
    if not target_indices:
        raise ValueError("No selected target indices.")
    target_rates = dataset.target_rates[:, target_indices]
    coverage = np.mean(np.isfinite(target_rates), axis=0)
    if np.any(coverage < 0.70):
        bad = [dataset.target_names[target_indices[idx]] for idx, cov in enumerate(coverage) if cov < 0.70]
        raise ValueError(f"Selected tau targets have insufficient rate coverage: {bad}")


def validate_split(split: SubjectSplit) -> None:
    sets = [set(split.train_rids), set(split.validation_rids), set(split.test_rids)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Subject split leakage detected.")
    if min(split.train_indices.size, split.validation_indices.size, split.test_indices.size) <= 0:
        raise ValueError("One split has no rows.")


def validate_predictions(predictions: dict[str, np.ndarray], dataset: MultimodalPairDataset, target_indices: list[int]) -> None:
    expected = (dataset.pair_count, len(target_indices))
    for model, values in predictions.items():
        if values.shape != expected:
            raise ValueError(f"{model} prediction shape {values.shape} != {expected}")
        finite_fraction = float(np.mean(np.isfinite(values)))
        if finite_fraction < 0.95:
            raise ValueError(f"{model} predictions have low finite fraction: {finite_fraction:.3f}")


def load_graph_resources(root: Path, dataset: MultimodalPairDataset, selected_regions: list[str]) -> dict[str, Any]:
    config_path = root / "experiments" / "group_average_enigma" / "config.yaml"
    if not config_path.exists():
        config_path = root / "experiments" / "group_average_enigma" / "config.example.yaml"
    config = load_yaml_config(config_path)
    forecast_dataset = load_forecast_dataset(config, root)
    assert_aligned_pairs(forecast_dataset.pairs, dataset.metadata_rows)
    output_root = root / config["paths"]["output_dir"]
    outputs = config.get("outputs", {})
    labels, adjacency = load_labeled_matrix(output_root / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv"))
    lap_labels, laplacian = load_labeled_matrix(output_root / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"))
    if labels != forecast_dataset.region_labels or lap_labels != forecast_dataset.region_labels:
        raise ValueError("ENIGMA matrix labels do not match forecast region labels.")
    selected_region_indices = [forecast_dataset.region_labels.index(region) for region in selected_regions]
    return {
        "config": config,
        "forecast_dataset": forecast_dataset,
        "adjacency": adjacency,
        "laplacian": laplacian,
        "selected_region_indices": selected_region_indices,
    }


def fit_all_prediction_models(
    *,
    dataset: MultimodalPairDataset,
    graph: dict[str, Any],
    split: SubjectSplit,
    selected_regions: list[str],
    selected_target_names: list[str],
    selected_target_indices: list[int],
    max_parents: int,
) -> dict[str, Any]:
    baseline_selected = dataset.target_baseline[:, selected_target_indices]
    time_years = dataset.time_years
    pseudotime = fit_pseudotime(dataset.feature_matrix, dataset.feature_names, split.train_indices, mode="tau_free")
    bn_fit = fit_dynamic_scm(
        dataset,
        pseudotime,
        split.train_indices,
        target_names=selected_target_names,
        max_parents_per_target=max_parents,
    )
    bn_rates = bn_fit.predict_rates(dataset)[:, selected_target_indices]
    predictions = {
        "BayesianNetwork-SCM": baseline_selected + time_years[:, None] * bn_rates,
        "S0 persistence": baseline_selected.copy(),
    }
    fit_reports: dict[str, dict[str, Any]] = {
        "BayesianNetwork-SCM": {
            "target_count": len(selected_target_names),
            "pseudotime_mode": pseudotime.mode,
            "max_parents": int(max_parents),
        },
        "S0 persistence": {"fit_scope": "no learned parameters"},
    }

    forecast_dataset = graph["forecast_dataset"]
    config = graph["config"]
    selected_region_indices = graph["selected_region_indices"]
    ndm = NetworkDiffusionModel(graph["laplacian"])
    ndm_fit = ndm.fit_global_rho(
        forecast_dataset.baseline[split.train_indices],
        forecast_dataset.observed[split.train_indices],
        forecast_dataset.time_years[split.train_indices],
        bounds=parameter_bounds(config, "rho", (0.0, 10.0)),
    )
    predictions["NDM"] = ndm.predict(forecast_dataset.baseline, forecast_dataset.time_years, ndm_fit.rho)[:, selected_region_indices]
    fit_reports["NDM"] = {
        "rho": float(ndm_fit.rho),
        "train_mse": float(ndm_fit.train_mse),
        "optimizer_success": bool(ndm_fit.optimizer_success),
    }

    scaler = MinMaxStateScaler.fit(
        forecast_dataset.baseline[split.train_indices],
        forecast_dataset.observed[split.train_indices],
    )
    baseline_scaled = scaler.transform(forecast_dataset.baseline)
    observed_scaled = scaler.transform(forecast_dataset.observed)
    esm = EpidemicSpreadingModel(graph["adjacency"], steps_per_year=int(config.get("modeling", {}).get("esm_steps_per_year", 12)))
    esm_fit = esm.fit_global_beta(
        baseline_scaled[split.train_indices],
        observed_scaled[split.train_indices],
        forecast_dataset.time_years[split.train_indices],
        bounds=parameter_bounds(config, "beta", (0.0, 10.0)),
    )
    predictions["ESM"] = scaler.inverse_transform(esm.predict(baseline_scaled, forecast_dataset.time_years, esm_fit.beta))[:, selected_region_indices]
    fit_reports["ESM"] = {
        "beta": float(esm_fit.beta),
        "train_mse_scaled": float(esm_fit.train_mse),
        "optimizer_success": bool(esm_fit.optimizer_success),
    }

    sir = GraphSIRModel(graph["adjacency"], steps_per_year=int(config.get("modeling", {}).get("sir_steps_per_year", 12)))
    sir_fit = sir.fit_global_parameters(
        baseline_scaled[split.train_indices],
        observed_scaled[split.train_indices],
        forecast_dataset.time_years[split.train_indices],
        beta_bounds=parameter_bounds(config, "beta", (0.0, 10.0)),
        gamma_bounds=parameter_bounds(config, "gamma", (0.0, 10.0)),
        maxiter=int(config.get("modeling", {}).get("sir_optimizer_maxiter", 80)),
    )
    predictions["SIR"] = scaler.inverse_transform(sir.predict(baseline_scaled, forecast_dataset.time_years, beta=sir_fit.beta, gamma=sir_fit.gamma))[
        :, selected_region_indices
    ]
    fit_reports["SIR"] = {
        "beta": float(sir_fit.beta),
        "gamma": float(sir_fit.gamma),
        "train_mse_scaled": float(sir_fit.train_mse),
        "optimizer_success": bool(sir_fit.optimizer_success),
    }

    return {
        "predictions": predictions,
        "pseudotime": pseudotime,
        "bn_fit": bn_fit,
        "z_values": pseudotime.transform(dataset.feature_matrix),
        "fit_reports": fit_reports,
        "selected_regions": selected_regions,
    }


def run_ablation_suite(
    *,
    dataset: MultimodalPairDataset,
    split: SubjectSplit,
    selected_target_names: list[str],
    selected_target_indices: list[int],
    selected_regions: list[str],
    max_parents: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenarios = [
        {"name": "full", "kind": "reference"},
        {"name": "static_edges_no_z", "kind": "pseudotime_control", "n_knots": 1, "degree": 0},
        {"name": "shuffled_z", "kind": "pseudotime_control", "shuffle_z": True},
        {"name": "global_pseudotime_tau_allowed", "kind": "pseudotime_control", "pseudotime_mode": "global"},
        {"name": "clinical_free_pseudotime", "kind": "pseudotime_control", "pseudotime_mode": "clinical_free"},
        {"name": "no_pt217", "kind": "feature_ablation", "exclude_features": ["plasma_pt217"]},
        {
            "name": "no_amyloid_pet",
            "kind": "feature_ablation",
            "exclude_features": ["amyloid_summary_suvr", "amyloid_centiloids", "amyloid_positive"],
        },
        {"name": "no_plasma_abeta_ratio", "kind": "feature_ablation", "exclude_features": ["plasma_ab42_ab40"]},
        {"name": "no_apoe4", "kind": "feature_ablation", "exclude_features": ["apoe4_dose"]},
        {"name": "no_fluid_markers", "kind": "feature_ablation", "exclude_features": ["plasma_pt217", "plasma_ab42_ab40", "plasma_nfl", "plasma_gfap"]},
        {"name": "no_tau_parent_features", "kind": "feature_ablation", "exclude_prefixes": ["tau_region:"], "exclude_features": ["tau_meta_temporal"]},
        {"name": "no_target_self_history", "kind": "structural_ablation", "zero_target_baseline": True},
    ]

    rows = []
    baseline_selected = dataset.target_baseline[:, selected_target_indices]
    for scenario in scenarios:
        working = dataset
        excluded = list(scenario.get("exclude_features", []))
        for prefix in scenario.get("exclude_prefixes", []):
            excluded.extend([name for name in dataset.feature_names if name.startswith(str(prefix))])
        if excluded:
            working = mask_dataset_features(working, excluded)
        if scenario.get("zero_target_baseline"):
            working = zero_selected_target_baseline(working, selected_target_indices)
        mode = str(scenario.get("pseudotime_mode", "tau_free"))
        pseudotime = fit_pseudotime(working.feature_matrix, working.feature_names, split.train_indices, mode=mode)
        if scenario.get("shuffle_z"):
            rng = np.random.default_rng(int(random_seed))
            z = pseudotime.transform(working.feature_matrix)
            shuffled = z.copy()
            rng.shuffle(shuffled)
            pseudotime = FixedZPseudotime(pseudotime, shuffled, mode="shuffled_tau_free")
        fit = fit_dynamic_scm(
            working,
            pseudotime,
            split.train_indices,
            target_names=selected_target_names,
            max_parents_per_target=max_parents,
            n_knots=int(scenario.get("n_knots", 4)),
            spline_degree=int(scenario.get("degree", 3)),
        )
        predicted_rates = fit.predict_rates(working)[:, selected_target_indices]
        predicted_observed = baseline_selected + dataset.time_years[:, None] * predicted_rates
        scenario_rows = metric_rows_for_predictions(
            {"BayesianNetwork-SCM": predicted_observed},
            dataset,
            selected_target_indices,
            split,
            selected_regions,
            experiment=str(scenario["name"]),
            seed=random_seed,
        )
        for row in scenario_rows:
            row["scenario"] = scenario["name"]
            row["scenario_kind"] = scenario["kind"]
            row["excluded_features"] = ";".join(excluded)
            row["pseudotime_mode"] = getattr(pseudotime, "mode", mode)
            row["spline_degree"] = int(scenario.get("degree", 3))
            row["n_knots"] = int(scenario.get("n_knots", 4))
        rows.extend(scenario_rows)
    summary = summarize_metric_rows(rows, group_fields=["scenario", "scenario_kind", "split", "metric"])
    full_lookup = {
        (row["split"], row["metric"]): float(row["median"])
        for row in summary
        if row["scenario"] == "full" and is_finite(row["median"])
    }
    for row in summary:
        baseline = full_lookup.get((row["split"], row["metric"]), float("nan"))
        row["delta_vs_full_median"] = float(row["median"]) - baseline if is_finite(row["median"]) and is_finite(baseline) else float("nan")
    return rows, summary


def mask_dataset_features(dataset: MultimodalPairDataset, feature_names: Iterable[str]) -> MultimodalPairDataset:
    masked = dataset.feature_matrix.copy()
    name_to_index = {name: idx for idx, name in enumerate(dataset.feature_names)}
    for name in feature_names:
        if name in name_to_index:
            masked[:, name_to_index[name]] = np.nan
    return replace(dataset, feature_matrix=masked)


def zero_selected_target_baseline(dataset: MultimodalPairDataset, target_indices: list[int]) -> MultimodalPairDataset:
    target_baseline = dataset.target_baseline.copy()
    target_baseline[:, target_indices] = 0.0
    return replace(dataset, target_baseline=target_baseline)


def metric_rows_for_predictions(
    predictions: dict[str, np.ndarray],
    dataset: MultimodalPairDataset,
    target_indices: list[int],
    split: SubjectSplit,
    regions: list[str],
    *,
    experiment: str,
    seed: int,
    repeat_index: int | None = None,
) -> list[dict[str, Any]]:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    time_years = dataset.time_years
    split_labels = split_labels_by_index(split, dataset.pair_count)
    rows = []
    for model, predicted in predictions.items():
        for idx in range(dataset.pair_count):
            base = baseline[idx]
            y = observed[idx]
            pred = predicted[idx]
            dt = float(time_years[idx])
            obs_rate = (y - base) / dt
            pred_rate = (pred - base) / dt
            rows.append(
                {
                    "experiment": experiment,
                    "seed": int(seed),
                    "repeat_index": "" if repeat_index is None else int(repeat_index),
                    "model": model,
                    "split": split_labels[idx],
                    "RID": dataset.metadata_rows[idx]["RID"],
                    "dx_nearest_baseline": dataset.metadata_rows[idx].get("dx_nearest_baseline", ""),
                    "amyloid_status": dataset.metadata_rows[idx].get("amyloid_status", ""),
                    "target_time_years": dt,
                    "mae_suvr": finite_mean_abs(pred - y),
                    "rmse_suvr": finite_rmse(pred - y),
                    "rate_mae": finite_mean_abs(pred_rate - obs_rate),
                    "rate_rmse": finite_rmse(pred_rate - obs_rate),
                    "subject_spearman": safe_correlation(y, pred, rank=True),
                    "delta_spearman": safe_correlation(y - base, pred - base, rank=True),
                    "delta_pearson": safe_correlation(y - base, pred - base, rank=False),
                }
            )
    return rows


def stage_stratified_metrics(
    *,
    predictions: dict[str, np.ndarray],
    dataset: MultimodalPairDataset,
    split: SubjectSplit,
    selected_target_indices: list[int],
    selected_regions: list[str],
    z_values: np.ndarray,
) -> list[dict[str, Any]]:
    test_indices = np.asarray(split.test_indices, dtype=int)
    stage_labels = assign_stages(z_values)
    rows = []
    for model, predicted in predictions.items():
        for stage in ("early", "mid", "late"):
            indices = np.asarray([idx for idx in test_indices if stage_labels[idx] == stage], dtype=int)
            if indices.size == 0:
                continue
            stage_split = subset_split(indices)
            stage_rows = metric_rows_for_predictions(
                {model: predicted},
                dataset,
                selected_target_indices,
                stage_split,
                selected_regions,
                experiment=f"stage_{stage}",
                seed=0,
            )
            values = [row for row in stage_rows if row["split"] == "test"]
            summary = summarize_metric_rows(values, group_fields=["model", "metric"])
            for row in summary:
                row["stage"] = stage
                row["n_pairs"] = int(indices.size)
                row["z_min"] = float(np.min(z_values[indices]))
                row["z_median"] = float(np.median(z_values[indices]))
                row["z_max"] = float(np.max(z_values[indices]))
                rows.append(row)
    return rows


def subset_split(test_indices: np.ndarray) -> SubjectSplit:
    return SubjectSplit(
        train_indices=np.asarray([], dtype=int),
        validation_indices=np.asarray([], dtype=int),
        test_indices=np.asarray(test_indices, dtype=int),
        train_rids=[],
        validation_rids=[],
        test_rids=[],
    )


def assign_stages(z_values: np.ndarray) -> np.ndarray:
    z = np.asarray(z_values, dtype=float)
    labels = np.full(z.shape, "mid", dtype=object)
    labels[z <= 1.0 / 3.0] = "early"
    labels[z >= 2.0 / 3.0] = "late"
    return labels


def counterfactual_response_rows(
    *,
    dataset: MultimodalPairDataset,
    fit: DynamicSCMFit,
    split: SubjectSplit,
    selected_target_indices: list[int],
    selected_regions: list[str],
    z_values: np.ndarray,
) -> list[dict[str, Any]]:
    train = np.asarray(split.train_indices, dtype=int)
    test = np.asarray(split.test_indices, dtype=int)
    factual_rates = fit.predict_rates(dataset)[:, selected_target_indices]
    scenarios = []
    feature_sets = [
        ("amyloid_pet_low", {"amyloid_summary_suvr": 0.25, "amyloid_centiloids": 0.25, "amyloid_positive": "zero"}),
        ("plasma_abeta_ratio_high", {"plasma_ab42_ab40": 0.75}),
        ("pt217_low", {"plasma_pt217": 0.25}),
        ("apoe4_zero", {"apoe4_dose": "zero"}),
    ]
    for name, rules in feature_sets:
        scenarios.append((name, intervention_dataset_features(dataset, train, rules)))
    scenarios.append(("local_tau_low", intervention_local_tau_baseline(dataset, train, selected_target_indices, quantile=0.25)))

    stage_labels = assign_stages(z_values)
    rows = []
    for scenario_name, intervened in scenarios:
        rates = fit.predict_rates(intervened)[:, selected_target_indices]
        delta = rates - factual_rates
        for stage in ("early", "mid", "late"):
            indices = np.asarray([idx for idx in test if stage_labels[idx] == stage], dtype=int)
            if indices.size == 0:
                continue
            for region_idx, region in enumerate(selected_regions):
                values = delta[indices, region_idx]
                rows.append(
                    {
                        "scenario": scenario_name,
                        "stage": stage,
                        "region": region,
                        "n_pairs": int(indices.size),
                        "mean_delta_rate": finite_mean(values),
                        "median_delta_rate": finite_median(values),
                        "mean_abs_delta_rate": finite_mean_abs(values),
                        "fraction_rate_reduced": finite_fraction(values < 0.0),
                    }
                )
    return rows


def intervention_dataset_features(dataset: MultimodalPairDataset, train_indices: np.ndarray, rules: dict[str, Any]) -> MultimodalPairDataset:
    matrix = dataset.feature_matrix.copy()
    name_to_index = {name: idx for idx, name in enumerate(dataset.feature_names)}
    for name, rule in rules.items():
        if name not in name_to_index:
            continue
        idx = name_to_index[name]
        column = dataset.feature_matrix[train_indices, idx]
        if rule == "zero":
            value = 0.0
        else:
            value = float(np.nanquantile(column, float(rule)))
        matrix[:, idx] = value
    return replace(dataset, feature_matrix=matrix)


def intervention_local_tau_baseline(dataset: MultimodalPairDataset, train_indices: np.ndarray, target_indices: list[int], *, quantile: float) -> MultimodalPairDataset:
    target_baseline = dataset.target_baseline.copy()
    for target_idx in target_indices:
        value = float(np.nanquantile(dataset.target_baseline[train_indices, target_idx], float(quantile)))
        target_baseline[:, target_idx] = value
    return replace(dataset, target_baseline=target_baseline)


def bootstrap_stage_edge_stability(
    *,
    dataset: MultimodalPairDataset,
    base_pseudotime: PseudotimeModel,
    split: SubjectSplit,
    target_names: list[str],
    iterations: int,
    random_seed: int,
    max_parents: int,
) -> list[dict[str, Any]]:
    if int(iterations) <= 0:
        return []
    train_indices = np.asarray(split.train_indices, dtype=int)
    groups = np.asarray([row["RID"] for row in dataset.metadata_rows], dtype=object)
    unique_groups = np.unique(groups[train_indices])
    indices_by_group = {group: train_indices[groups[train_indices] == group] for group in unique_groups}
    rng = np.random.default_rng(int(random_seed))
    stage_z = {"early": 0.15, "mid": 0.50, "late": 0.85}
    accum: dict[tuple[str, str, str], list[float]] = {}
    included: dict[tuple[str, str, str], int] = {}
    seen_edges: set[tuple[str, str]] = set()

    for iteration in range(int(iterations)):
        if (iteration + 1) % max(1, int(iterations) // 5) == 0:
            print(f"  bootstrap edge stability {iteration + 1}/{iterations}")
        sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        sampled_indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        fit = fit_dynamic_scm(
            dataset,
            base_pseudotime,
            sampled_indices,
            target_names=target_names,
            max_parents_per_target=max_parents,
            cv_folds=3,
        )
        basis = fit.spline_basis.transform(np.asarray(list(stage_z.values()), dtype=float))
        edge_effects: dict[tuple[str, str], np.ndarray] = {}
        for target_fit in fit.target_fits:
            for parent in target_fit.parent_names:
                effect = target_fit.parent_effect_curve(parent, basis)
                edge_effects[(parent, target_fit.target_name)] = effect
                seen_edges.add((parent, target_fit.target_name))
        for edge in list(seen_edges):
            effect = edge_effects.get(edge)
            for stage_idx, stage in enumerate(stage_z):
                key = (edge[0], edge[1], stage)
                value = 0.0 if effect is None else float(effect[stage_idx])
                accum.setdefault(key, []).append(value)
                if abs(value) >= 0.01:
                    included[key] = included.get(key, 0) + 1

    rows = []
    for key, values in accum.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "parent": key[0],
                "target": key[1],
                "stage": key[2],
                "bootstrap_iterations": int(iterations),
                "inclusion_probability": float(included.get(key, 0) / int(iterations)),
                "mean_effect": float(np.mean(arr)),
                "mean_abs_effect": float(np.mean(np.abs(arr))),
                "sd_effect": float(np.std(arr)),
                "positive_sign_probability": float(np.mean(arr > 0.0)),
                "negative_sign_probability": float(np.mean(arr < 0.0)),
            }
        )
    rows.sort(key=lambda row: (-float(row["inclusion_probability"]), -float(row["mean_abs_effect"]), row["parent"], row["target"], row["stage"]))
    return rows


def edge_curve_grid_rows(fit: DynamicSCMFit, *, top_k: int) -> list[dict[str, Any]]:
    basis = fit.spline_basis.transform(fit.z_grid)
    summaries = []
    for target_fit in fit.target_fits:
        for parent in target_fit.parent_names:
            effect = target_fit.parent_effect_curve(parent, basis)
            summaries.append((float(np.max(np.abs(effect))) if effect.size else 0.0, parent, target_fit.target_name, effect))
    summaries.sort(key=lambda row: -row[0])
    rows = []
    for _, parent, target, effect in summaries[: int(top_k)]:
        for z, value in zip(fit.z_grid, effect, strict=True):
            rows.append({"parent": parent, "target": target, "z": float(z), "effect": float(value), "abs_effect": abs(float(value))})
    return rows


def dynamic_graph_stage_rows(fit: DynamicSCMFit, *, top_k_per_stage: int) -> list[dict[str, Any]]:
    stage_z = {"early": 0.15, "mid": 0.50, "late": 0.85}
    basis = fit.spline_basis.transform(np.asarray(list(stage_z.values()), dtype=float))
    rows = []
    for stage_idx, stage in enumerate(stage_z):
        stage_rows = []
        for target_fit in fit.target_fits:
            for parent in target_fit.parent_names:
                effect = target_fit.parent_effect_curve(parent, basis)[stage_idx]
                stage_rows.append(
                    {
                        "stage": stage,
                        "z": stage_z[stage],
                        "parent": parent,
                        "target": target_fit.target_name,
                        "effect": float(effect),
                        "abs_effect": abs(float(effect)),
                    }
                )
        stage_rows.sort(key=lambda row: -float(row["abs_effect"]))
        rows.extend(stage_rows[: int(top_k_per_stage)])
    return rows


def make_all_figures(
    *,
    fig_dir: Path,
    repeated_rows: list[dict[str, Any]],
    repeated_summary: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    edge_grid_rows: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    primary: dict[str, Any],
    dataset: MultimodalPairDataset,
    split: SubjectSplit,
    selected_regions: list[str],
    selected_target_indices: list[int],
    skip_nilearn: bool,
) -> dict[str, str]:
    figures = {
        "repeated_validation": str(fig_dir / "fig1_repeated_validation.png"),
        "ablation_controls": str(fig_dir / "fig2_ablation_controls.png"),
        "stage_stratified": str(fig_dir / "fig3_stage_stratified_performance.png"),
        "edge_heatmap": str(fig_dir / "fig4_dynamic_edge_heatmap.png"),
        "dynamic_graph": str(fig_dir / "fig5_dynamic_graph_early_mid_late.png"),
        "counterfactual_windows": str(fig_dir / "fig6_counterfactual_windows.png"),
    }
    plot_repeated_validation(Path(figures["repeated_validation"]), repeated_rows)
    plot_ablation_controls(Path(figures["ablation_controls"]), ablation_summary)
    plot_stage_stratified(Path(figures["stage_stratified"]), stage_rows)
    plot_edge_heatmap(Path(figures["edge_heatmap"]), edge_grid_rows)
    plot_dynamic_graph(Path(figures["dynamic_graph"]), graph_rows)
    plot_counterfactual_windows(Path(figures["counterfactual_windows"]), counterfactual_rows)

    if not skip_nilearn:
        try:
            figures["brain_multiview_absolute"] = str(fig_dir / "fig7_brain_multiview_tau_burden.png")
            figures["brain_multiview_error"] = str(fig_dir / "fig8_brain_multiview_prediction_error.png")
            figures["brain_stage_cascade"] = str(fig_dir / "fig9_brain_stage_cascade_bn_lte.png")
            plot_brain_multiview_absolute(
                Path(figures["brain_multiview_absolute"]),
                primary["predictions"],
                dataset,
                split,
                selected_regions,
                selected_target_indices,
            )
            plot_brain_multiview_error(
                Path(figures["brain_multiview_error"]),
                primary["predictions"],
                dataset,
                split,
                selected_regions,
                selected_target_indices,
            )
            plot_brain_stage_cascade(
                Path(figures["brain_stage_cascade"]),
                primary["predictions"]["BayesianNetwork-SCM"],
                dataset,
                split,
                primary["z_values"],
                selected_regions,
                selected_target_indices,
            )
        except Exception as exc:
            figures["brain_multiview_error_message"] = f"{type(exc).__name__}: {exc}"
    return figures


def plot_repeated_validation(path: Path, rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("rate_mae", "Rate MAE (SUVR/year)", False),
        ("mae_suvr", "Follow-up MAE (SUVR)", False),
        ("subject_spearman", "Spatial Spearman", True),
        ("delta_spearman", "Delta Spearman", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    fig.suptitle("Repeated subject-level validation", fontsize=16, fontweight="bold")
    for ax, (metric, title, higher_better) in zip(axes.flat, metrics, strict=True):
        data = []
        labels = []
        for model in PAPER_MODEL_ORDER:
            values = [
                float(row[metric])
                for row in rows
                if row["split"] == "test" and row["model"] == model and is_finite(row[metric])
            ]
            data.append(values)
            labels.append(short_model(model))
        try:
            box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        except TypeError:
            box = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
        for patch, model in zip(box["boxes"], PAPER_MODEL_ORDER, strict=True):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.65)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
        if not higher_better:
            best_idx = int(np.nanargmin([np.nanmedian(values) if values else np.nan for values in data]))
        else:
            best_idx = int(np.nanargmax([np.nanmedian(values) if values else np.nan for values in data]))
        ax.text(0.02, 0.95, f"Best median: {labels[best_idx]}", transform=ax.transAxes, va="top", fontsize=9, color="#374151")
    save_figure(fig, path)


def plot_ablation_controls(path: Path, summary: list[dict[str, Any]]) -> None:
    rows = [
        row
        for row in summary
        if row["split"] == "test" and row["metric"] == "rate_mae" and row["scenario"] != "full" and is_finite(row["delta_vs_full_median"])
    ]
    rows.sort(key=lambda row: float(row["delta_vs_full_median"]), reverse=True)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * len(rows) + 1.6)), constrained_layout=True)
    values = [float(row["delta_vs_full_median"]) for row in rows]
    labels = [str(row["scenario"]).replace("_", " ") for row in rows]
    colors = ["#D55E00" if value > 0 else "#009E73" for value in values]
    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, alpha=0.82)
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Delta median test rate MAE vs full BN-LTE")
    ax.set_title("Leakage controls and modality ablations", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, path)


def plot_stage_stratified(path: Path, rows: list[dict[str, Any]]) -> None:
    metric_rows = [row for row in rows if row["metric"] == "rate_mae" and is_finite(row["median"])]
    stages = ["early", "mid", "late"]
    x = np.arange(len(stages))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for model_idx, model in enumerate(PAPER_MODEL_ORDER):
        values = []
        for stage in stages:
            match = [row for row in metric_rows if row["model"] == model and row["stage"] == stage]
            values.append(float(match[0]["median"]) if match else np.nan)
        ax.bar(x + (model_idx - 2) * width, values, width=width, label=short_model(model), color=MODEL_COLORS[model], alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels([stage.title() for stage in stages])
    ax.set_ylabel("Median test rate MAE (SUVR/year)")
    ax.set_title("Performance by latent disease stage", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3, fontsize=9)
    save_figure(fig, path)


def plot_edge_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    edges = []
    z_values = sorted({float(row["z"]) for row in rows})
    for row in rows:
        edge = f"{short_parent(row['parent'])} -> {short_target(row['target'])}"
        if edge not in edges:
            edges.append(edge)
    edges = edges[:24]
    matrix = np.full((len(edges), len(z_values)), np.nan, dtype=float)
    z_to_col = {z: idx for idx, z in enumerate(z_values)}
    edge_to_row = {edge: idx for idx, edge in enumerate(edges)}
    for row in rows:
        edge = f"{short_parent(row['parent'])} -> {short_target(row['target'])}"
        if edge in edge_to_row:
            matrix[edge_to_row[edge], z_to_col[float(row["z"])]] = float(row["effect"])
    bound = max(float(np.nanmax(np.abs(matrix))) if np.any(np.isfinite(matrix)) else 0.01, 0.01)
    fig, ax = plt.subplots(figsize=(12.5, max(6, 0.32 * len(edges) + 2)), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set_yticks(np.arange(len(edges)))
    ax.set_yticklabels(edges, fontsize=8)
    tick_cols = np.linspace(0, len(z_values) - 1, 6, dtype=int)
    ax.set_xticks(tick_cols)
    ax.set_xticklabels([f"{z_values[idx]:.2f}" for idx in tick_cols])
    ax.set_xlabel("Latent disease time Z")
    ax.set_title("Stage-varying direct effect curves", fontsize=15, fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Standardized edge effect")
    save_figure(fig, path)


def plot_dynamic_graph(path: Path, rows: list[dict[str, Any]]) -> None:
    stages = ["early", "mid", "late"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    fig.suptitle("Dynamic BN-LTE graph: strongest stage-specific edges", fontsize=16, fontweight="bold")
    for ax, stage in zip(axes, stages, strict=True):
        stage_rows = [row for row in rows if row["stage"] == stage]
        draw_stage_graph(ax, stage, stage_rows)
    save_figure(fig, path)


def draw_stage_graph(ax: Any, stage: str, rows: list[dict[str, Any]]) -> None:
    ax.set_title(stage.title(), fontweight="bold")
    ax.set_axis_off()
    nodes = sorted({row["parent"] for row in rows} | {row["target"] for row in rows})
    columns = {
        "root": 0.08,
        "fluid": 0.28,
        "pathology": 0.50,
        "neurodegeneration": 0.70,
        "target": 0.90,
    }
    grouped: dict[str, list[str]] = {key: [] for key in columns}
    for node in nodes:
        grouped[node_group(node)].append(node)
    positions = {}
    for group, group_nodes in grouped.items():
        if not group_nodes:
            continue
        ys = np.linspace(0.85, 0.15, len(group_nodes))
        for node, y in zip(group_nodes, ys, strict=True):
            positions[node] = (columns[group], float(y))
    max_abs = max([float(row["abs_effect"]) for row in rows] + [0.01])
    for row in rows:
        parent = row["parent"]
        target = row["target"]
        if parent not in positions or target not in positions:
            continue
        x1, y1 = positions[parent]
        x2, y2 = positions[target]
        color = "#D55E00" if float(row["effect"]) > 0 else "#0072B2"
        width = 0.8 + 4.0 * float(row["abs_effect"]) / max_abs
        arrow = FancyArrowPatch(
            (x1 + 0.035, y1),
            (x2 - 0.035, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            color=color,
            alpha=0.72,
            connectionstyle="arc3,rad=0.08",
        )
        ax.add_patch(arrow)
    for node, (x, y) in positions.items():
        group = node_group(node)
        ax.scatter([x], [y], s=520, color=node_color(group), edgecolor="white", linewidth=1.3, zorder=3)
        ax.text(x, y, short_node(node), ha="center", va="center", fontsize=7.2, color="#111827", zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_counterfactual_windows(path: Path, rows: list[dict[str, Any]]) -> None:
    scenarios = ["amyloid_pet_low", "plasma_abeta_ratio_high", "pt217_low", "apoe4_zero", "local_tau_low"]
    stages = ["early", "mid", "late"]
    fig, ax = plt.subplots(figsize=(11.5, 6), constrained_layout=True)
    for scenario in scenarios:
        values = []
        for stage in stages:
            stage_values = [float(row["mean_delta_rate"]) for row in rows if row["scenario"] == scenario and row["stage"] == stage and is_finite(row["mean_delta_rate"])]
            values.append(float(np.mean(stage_values)) if stage_values else np.nan)
        ax.plot(stages, values, marker="o", linewidth=2.0, label=scenario.replace("_", " "))
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_ylabel("Mean intervention-induced rate change (SUVR/year)")
    ax.set_title("Model-implied therapeutic response windows", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    save_figure(fig, path)


def plot_brain_multiview_absolute(
    path: Path,
    predictions: dict[str, np.ndarray],
    dataset: MultimodalPairDataset,
    split: SubjectSplit,
    selected_regions: list[str],
    target_indices: list[int],
) -> None:
    test = np.asarray(split.test_indices, dtype=int)
    baseline = region_mean_dict(dataset.target_baseline[test][:, target_indices], selected_regions)
    observed = region_mean_dict(dataset.target_observed[test][:, target_indices], selected_regions)
    panels = [("Baseline S0", baseline), ("Empirical follow-up", observed)]
    for model in ("BayesianNetwork-SCM", "NDM", "ESM", "SIR", "S0 persistence"):
        panels.append((short_model(model), region_mean_dict(predictions[model][test], selected_regions)))
    finite = [value for _, row in panels for value in row.values() if is_finite(value)]
    vmin, vmax = padded_bounds(finite, minimum_pad=0.03)
    note = "Held-out test-set mean tau burden. Rows compare empirical baseline/follow-up against model-predicted follow-up. Views include lateral, medial, and ventral cortex."
    plot_surface_grid(path, "Multi-view regional tau burden", panels, cmap="magma", vmin=vmin, vmax=vmax, colorbar_label="Tau SUVR", note=note)


def plot_brain_multiview_error(
    path: Path,
    predictions: dict[str, np.ndarray],
    dataset: MultimodalPairDataset,
    split: SubjectSplit,
    selected_regions: list[str],
    target_indices: list[int],
) -> None:
    test = np.asarray(split.test_indices, dtype=int)
    observed = region_mean_dict(dataset.target_observed[test][:, target_indices], selected_regions)
    panels = []
    for model in ("BayesianNetwork-SCM", "NDM", "ESM", "SIR", "S0 persistence"):
        pred = region_mean_dict(predictions[model][test], selected_regions)
        panels.append((short_model(model), {region: pred[region] - observed[region] for region in selected_regions}))
    bound = max([abs(value) for _, row in panels for value in row.values() if is_finite(value)] + [0.05])
    note = "Prediction error is model-predicted follow-up minus empirical follow-up; warm colors overpredict and cool colors underpredict."
    plot_surface_grid(path, "Multi-view regional tau prediction error", panels, cmap="RdBu_r", vmin=-bound, vmax=bound, colorbar_label="Predicted - empirical SUVR", note=note)


def plot_brain_stage_cascade(
    path: Path,
    bn_prediction: np.ndarray,
    dataset: MultimodalPairDataset,
    split: SubjectSplit,
    z_values: np.ndarray,
    selected_regions: list[str],
    target_indices: list[int],
) -> None:
    test = np.asarray(split.test_indices, dtype=int)
    stage_labels = assign_stages(z_values)
    panels = []
    for stage in ("early", "mid", "late"):
        indices = np.asarray([idx for idx in test if stage_labels[idx] == stage], dtype=int)
        if indices.size == 0:
            continue
        panels.append((f"{stage.title()} S0", region_mean_dict(dataset.target_baseline[indices][:, target_indices], selected_regions)))
        panels.append((f"{stage.title()} empirical", region_mean_dict(dataset.target_observed[indices][:, target_indices], selected_regions)))
        panels.append((f"{stage.title()} BN-LTE", region_mean_dict(bn_prediction[indices], selected_regions)))
    finite = [value for _, row in panels for value in row.values() if is_finite(value)]
    vmin, vmax = padded_bounds(finite, minimum_pad=0.03)
    note = "Stage-binned held-out subjects show baseline, empirical follow-up, and BN-LTE predicted follow-up across latent disease time."
    plot_surface_grid(path, "BN-LTE stage cascade on cortical surface", panels, cmap="magma", vmin=vmin, vmax=vmax, colorbar_label="Tau SUVR", note=note)


def plot_surface_grid(
    path: Path,
    title: str,
    panels: list[tuple[str, dict[str, float]]],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    note: str,
) -> None:
    from nilearn import datasets, plotting

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        atlas = datasets.fetch_atlas_surf_destrieux()
    fig = plt.figure(figsize=(19.5, 2.15 * len(panels) + 1.6), facecolor="white")
    grid_bottom = 0.055
    grid_top = 0.84
    grid = fig.add_gridspec(
        len(panels),
        len(SURFACE_VIEWS),
        left=0.055,
        right=0.90,
        bottom=grid_bottom,
        top=grid_top,
        wspace=0.00,
        hspace=0.04,
    )
    for row_idx, (row_label, values) in enumerate(panels):
        maps = build_surface_maps(values, atlas)
        for col_idx, (hemi, view, col_label) in enumerate(SURFACE_VIEWS):
            ax = fig.add_subplot(grid[row_idx, col_idx], projection="3d")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plotting.plot_surf_stat_map(
                    fsaverage[f"infl_{hemi}"],
                    maps[hemi],
                    hemi=hemi,
                    view=view,
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    axes=ax,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar=False,
                    title=col_label if row_idx == 0 else "",
                )
        row_step = (grid_top - grid_bottom) / len(panels)
        row_center = grid_top - (row_idx + 0.5) * row_step
        fig.text(0.012, row_center, row_label, ha="left", va="center", fontsize=10.5, weight="bold")
    fig.suptitle(title, x=0.055, y=0.985, ha="left", fontsize=16, weight="bold")
    fig.text(0.055, 0.937, textwrap.fill(note, width=160), ha="left", va="top", fontsize=9.5, color="#4B5563")
    norm = Normalize(vmin=vmin, vmax=vmax)
    cax = fig.add_axes([0.925, 0.18, 0.016, 0.60])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(colorbar_label, fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    save_figure(fig, path, dpi=170, write_svg=False)


def build_surface_maps(values: dict[str, float], atlas: Any) -> dict[str, np.ndarray]:
    label_to_index = {label: idx for idx, label in enumerate(atlas["labels"])}
    maps = {
        "left": np.full(np.asarray(atlas["map_left"]).shape, np.nan, dtype=float),
        "right": np.full(np.asarray(atlas["map_right"]).shape, np.nan, dtype=float),
    }
    atlas_maps = {"left": np.asarray(atlas["map_left"]), "right": np.asarray(atlas["map_right"])}
    for region, value in values.items():
        if not is_finite(value):
            continue
        hemi = "left" if region.startswith("L_") else "right"
        for label in DK_TO_DESTRIEUX.get(region.split("_", 1)[1], []):
            label_idx = label_to_index.get(label)
            if label_idx is not None:
                maps[hemi][atlas_maps[hemi] == label_idx] = float(value)
    return maps


def assert_aligned_pairs(forecast_pairs: list[dict[str, str]], metadata_rows: list[dict[str, Any]]) -> None:
    if len(forecast_pairs) != len(metadata_rows):
        raise ValueError(f"Pair count mismatch: forecast={len(forecast_pairs)}, bn_scm={len(metadata_rows)}")
    for idx, (forecast_row, bn_row) in enumerate(zip(forecast_pairs, metadata_rows, strict=True)):
        if (
            str(forecast_row.get("RID", "")) != str(bn_row.get("RID", ""))
            or str(forecast_row.get("baseline_tau_date", "")) != str(bn_row.get("baseline_tau_date", ""))
            or str(forecast_row.get("target_tau_date", "")) != str(bn_row.get("target_tau_date", ""))
        ):
            raise ValueError(f"Forecast and BN-SCM pair ordering diverges at row {idx}.")


def parameter_bounds(config: dict[str, Any], name: str, default: tuple[float, float]) -> tuple[float, float]:
    values = config.get("modeling", {}).get("parameter_bounds", {}).get(name, default)
    return float(values[0]), float(values[1])


def split_labels_by_index(split: SubjectSplit, row_count: int) -> list[str]:
    labels = ["unknown"] * row_count
    for name, indices in (("train", split.train_indices), ("validation", split.validation_indices), ("test", split.test_indices)):
        for idx in indices:
            labels[int(idx)] = name
    return labels


def summarize_metric_rows(rows: list[dict[str, Any]], *, group_fields: list[str]) -> list[dict[str, Any]]:
    metrics = ["mae_suvr", "rmse_suvr", "rate_mae", "rate_rmse", "subject_spearman", "delta_spearman", "delta_pearson"]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        base = {field: value for field, value in zip(group_fields, key, strict=True)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group_rows if is_finite(row.get(metric))], dtype=float)
            if values.size == 0:
                continue
            output.append(
                {
                    **base,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    return output


def nested_summary(rows: list[dict[str, Any]], *, split_name: str) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("split") != split_name:
            continue
        model = str(row.get("model", row.get("scenario", "")))
        metric = str(row.get("metric", ""))
        value = row.get("median", row.get(metric))
        if metric and is_finite(value):
            output.setdefault(model, {})[metric] = float(value)
    return output


def summarize_stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["metric"] in {"rate_mae", "delta_spearman"}]


def summarize_counterfactual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scenario in sorted({row["scenario"] for row in rows}):
        for stage in ("early", "mid", "late"):
            values = [float(row["mean_delta_rate"]) for row in rows if row["scenario"] == scenario and row["stage"] == stage and is_finite(row["mean_delta_rate"])]
            output.append(
                {
                    "scenario": scenario,
                    "stage": stage,
                    "mean_delta_rate_over_regions": float(np.mean(values)) if values else float("nan"),
                    "n_regions": len(values),
                }
            )
    return output


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    return float(np.mean(values[mask])) if np.any(mask) else float("nan")


def finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    return float(np.median(values[mask])) if np.any(mask) else float("nan")


def finite_mean_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    return float(np.mean(np.abs(values[mask]))) if np.any(mask) else float("nan")


def finite_rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    return float(np.sqrt(np.mean(values[mask] ** 2))) if np.any(mask) else float("nan")


def finite_fraction(mask_values: np.ndarray) -> float:
    values = np.asarray(mask_values)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


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


def region_mean_dict(values: np.ndarray, regions: list[str]) -> dict[str, float]:
    return {region: finite_mean(values[:, idx]) for idx, region in enumerate(regions)}


def padded_bounds(values: Iterable[float], *, minimum_pad: float) -> tuple[float, float]:
    finite = [float(value) for value in values if is_finite(value)]
    if not finite:
        return 0.0, 1.0
    lo = min(finite)
    hi = max(finite)
    pad = max((hi - lo) * 0.15, minimum_pad)
    return lo - pad, hi + pad


def node_group(name: str) -> str:
    text = name.lower()
    if text.startswith("tau_rate:"):
        return "target"
    if text in {"age_years", "sex_female", "education_years", "apoe4_dose"}:
        return "root"
    if any(token in text for token in ("plasma", "pt217", "nfl", "gfap", "ab42")):
        return "fluid"
    if any(token in text for token in ("amyloid", "tau")):
        return "pathology"
    if any(token in text for token in ("mri", "volume", "thickness", "atrophy")):
        return "neurodegeneration"
    return "pathology"


def node_color(group: str) -> str:
    return {
        "root": "#F0E442",
        "fluid": "#56B4E9",
        "pathology": "#D55E00",
        "neurodegeneration": "#009E73",
        "target": "#CC79A7",
    }.get(group, "#E5E7EB")


def short_model(model: str) -> str:
    return {
        "BayesianNetwork-SCM": "BN-LTE",
        "S0 persistence": "Persistence",
    }.get(model, model)


def short_parent(name: str) -> str:
    mapping = {
        "plasma_ab42_ab40": "plasma AB42/40",
        "amyloid_summary_suvr": "amyloid PET",
        "amyloid_centiloids": "centiloids",
        "amyloid_positive": "amyloid +",
        "plasma_pt217": "p-tau217",
        "apoe4_dose": "APOE4",
        "age_years": "age",
        "tau_meta_temporal": "meta-temporal tau",
    }
    return mapping.get(str(name), str(name).replace("tau_region:", "tau ").replace("_", " "))


def short_target(name: str) -> str:
    return str(name).replace("tau_rate:", "").replace("_", " ")


def short_node(name: str) -> str:
    text = short_parent(name)
    text = text.replace("tau rate:", "").replace("tau_rate:", "")
    if len(text) > 16:
        text = text[:14] + ".."
    return text


def is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
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


def resolve_path(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def save_figure(fig: Any, path: Path, *, dpi: int = 180, write_svg: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    if write_svg:
        svg_path = path.with_suffix(".svg")
        fig.savefig(svg_path)
    plt.close(fig)


def render_markdown_report(report: dict[str, Any]) -> str:
    primary = report["primary_test_summary"]
    repeated = report["repeated_test_summary"]
    bnlte_primary = primary.get("BayesianNetwork-SCM", {})
    bnlte_repeated = repeated.get("BayesianNetwork-SCM", {})
    lines = [
        "# BN-LTE Paper Validation Report",
        "",
        "## Scope",
        "",
        "This report summarizes repeated validation, negative controls, stage-stratified performance, bootstrap edge stability, counterfactual response curves, and brain-surface visualizations for the ADNI-first BN-LTE/BN-SCM pipeline.",
        "",
        "## Data",
        "",
        f"- Pairs: {report['data']['pairs']}",
        f"- Subjects: {report['data']['subjects']}",
        f"- Selected tau regions: {', '.join(report['data']['selected_regions'])}",
        f"- Repeated splits: {report['configuration']['repeats']}",
        f"- Bootstrap edge iterations: {report['configuration']['bootstrap_iterations']}",
        "",
        "## Primary Held-Out Test Result",
        "",
        f"- BN-LTE median rate MAE: {format_float(bnlte_primary.get('rate_mae'))} SUVR/year",
        f"- BN-LTE median follow-up MAE: {format_float(bnlte_primary.get('mae_suvr'))} SUVR",
        f"- BN-LTE median spatial Spearman: {format_float(bnlte_primary.get('subject_spearman'))}",
        f"- BN-LTE median delta Spearman: {format_float(bnlte_primary.get('delta_spearman'))}",
        "",
        "## Repeated-Split Summary",
        "",
        f"- BN-LTE repeated median rate MAE: {format_float(bnlte_repeated.get('rate_mae'))}",
        f"- BN-LTE repeated median delta Spearman: {format_float(bnlte_repeated.get('delta_spearman'))}",
        "",
        "## Strongest Stable Edges",
        "",
    ]
    for row in report.get("top_stable_edges", [])[:10]:
        lines.append(
            f"- {short_parent(row['parent'])} -> {short_target(row['target'])} ({row['stage']}): "
            f"PIP-like stability={format_float(row['inclusion_probability'])}, mean_abs={format_float(row['mean_abs_effect'])}"
        )
    lines.extend(["", "## Figures", ""])
    for name, path in report["figures"].items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Counterfactual curves are model-implied SCM responses, not randomized intervention evidence.",
            "- Bootstrap stability is a finite-sample robustness analysis, not full posterior graph MCMC.",
            "- The strongest paper claim should combine predictive superiority, shuffled-Z/static-edge controls, and stable stage-varying edge patterns.",
            "",
        ]
    )
    return "\n".join(lines)


def render_console_summary(report: dict[str, Any]) -> str:
    primary = report["primary_test_summary"].get("BayesianNetwork-SCM", {})
    return "\n".join(
        [
            "BN-LTE paper validation complete.",
            f"Output report: {report['tables']['primary_pair_metrics']}",
            f"Figures: {len(report['figures'])}",
            f"Primary test median rate MAE: {format_float(primary.get('rate_mae'))}",
            f"Primary test median delta Spearman: {format_float(primary.get('delta_spearman'))}",
        ]
    )


def format_float(value: Any) -> str:
    if not is_finite(value):
        return "NA"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
