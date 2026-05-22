#!/usr/bin/env python3
"""Extended paper experiments for BN-LTE progression topology and brain maps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from bayesian_network_scm.data import MultimodalPairDataset, build_multimodal_pair_dataset  # noqa: E402
from bayesian_network_scm.dynamic_scm import fit_dynamic_scm  # noqa: E402
from bayesian_network_scm.reporting import make_subject_split  # noqa: E402
from run_paper_validation_experiments import (  # noqa: E402
    MODEL_COLORS,
    MODEL_ORDER,
    REGION_SHORT_NAMES,
    STAGE_COLORS,
    assign_stages,
    edge_curve_grid_rows,
    fit_all_prediction_models,
    finite_mean,
    finite_mean_abs,
    finite_median,
    finite_rmse,
    is_finite,
    load_graph_resources,
    plot_surface_grid,
    safe_correlation,
    save_figure,
    short_model,
    short_parent,
    short_target,
    validate_dataset,
    validate_predictions,
    validate_split,
)


EXTENDED_MODELS = ["BayesianNetwork-SCM", "ESM", "SIR", "NDM", "S0 persistence"]
BRAAK_GROUPS = {
    "entorhinal": ["L_entorhinal", "R_entorhinal"],
    "ventral_temporal": ["L_fusiform", "R_fusiform", "L_inferiortemporal", "R_inferiortemporal"],
    "lateral_temporal": ["L_middletemporal", "R_middletemporal"],
    "inferior_parietal": ["L_inferiorparietal", "R_inferiorparietal"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "paper_extended")
    parser.add_argument("--random-seed", type=int, default=20260521)
    parser.add_argument("--max-parents", type=int, default=6)
    parser.add_argument("--edge-bootstrap-iterations", type=int, default=40)
    parser.add_argument("--skip-brain", action="store_true")
    args = parser.parse_args()
    report = run_extended_experiments(
        project_root=args.project_root,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        max_parents=args.max_parents,
        edge_bootstrap_iterations=args.edge_bootstrap_iterations,
        skip_brain=args.skip_brain,
    )
    print("Extended BN-LTE paper experiments complete.")
    print(f"Report: {report['report_path']}")
    print(f"Figures: {len(report['figures'])}")
    return 0


def run_extended_experiments(
    *,
    project_root: str | Path,
    output_dir: str | Path,
    random_seed: int,
    max_parents: int,
    edge_bootstrap_iterations: int,
    skip_brain: bool,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    out = resolve_path(output_dir, root)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1/7: load data, split, and fit shared models")
    dataset = build_multimodal_pair_dataset(root)
    selected_regions = list(dataset.report["selected_tau_regions"])
    target_names = [f"tau_rate:{region}" for region in selected_regions]
    target_indices = [dataset.target_index(name) for name in target_names]
    validate_dataset(dataset, target_indices)
    split = make_subject_split(dataset.metadata_rows, random_seed=random_seed)
    validate_split(split)
    graph = load_graph_resources(root, dataset, selected_regions)
    fitted = fit_all_prediction_models(
        dataset=dataset,
        graph=graph,
        split=split,
        selected_regions=selected_regions,
        selected_target_names=target_names,
        selected_target_indices=target_indices,
        max_parents=max_parents,
    )
    validate_predictions(fitted["predictions"], dataset, target_indices)

    print("Step 2/7: progression-topology and group-map metrics")
    topology_rows = progression_topology_rows(fitted["predictions"], dataset, split, selected_regions, target_indices)
    topology_summary = summarize_topology_rows(topology_rows)
    group_map_rows = group_map_progression_metrics(fitted["predictions"], dataset, split, selected_regions, target_indices, fitted["z_values"])

    print("Step 3/7: Braak-like anatomical ordering and fast-progressor classification")
    braak_rows, braak_summary = braak_ordering_rows(fitted["predictions"], dataset, split, selected_regions, target_indices)
    classifier_rows = fast_progressor_classification(fitted["predictions"], dataset, split, selected_regions, target_indices)

    print("Step 4/7: pseudotime explainability")
    loading_rows = fitted["pseudotime"].loading_rows()
    contribution_rows = pseudotime_contribution_rows(fitted["pseudotime"], dataset, fitted["z_values"])

    print("Step 5/7: bootstrap edge confidence bands")
    edge_band_rows, edge_band_summary = bootstrap_edge_confidence_bands(
        dataset=dataset,
        split=split,
        fitted=fitted,
        target_names=target_names,
        iterations=edge_bootstrap_iterations,
        random_seed=random_seed + 919,
        max_parents=max_parents,
    )

    print("Step 6/7: subject archetypes and brain-map inputs")
    archetype_rows = subject_archetype_rows(fitted["predictions"], dataset, split, selected_regions, target_indices, fitted["z_values"])

    print("Step 7/7: write tables, figures, and report")
    tables = {
        "progression_topology_pair_metrics": out / "progression_topology_pair_metrics.csv",
        "progression_topology_summary": out / "progression_topology_summary.csv",
        "group_map_progression_metrics": out / "group_map_progression_metrics.csv",
        "braak_stage_deltas": out / "braak_stage_deltas.csv",
        "braak_ordering_summary": out / "braak_ordering_summary.csv",
        "fast_progressor_classification": out / "fast_progressor_classification.csv",
        "pseudotime_loadings": out / "pseudotime_loadings.csv",
        "pseudotime_group_contributions": out / "pseudotime_group_contributions.csv",
        "edge_confidence_bands": out / "edge_confidence_bands.csv",
        "edge_confidence_summary": out / "edge_confidence_summary.csv",
        "subject_archetypes": out / "subject_archetypes.csv",
    }
    write_csv_rows(tables["progression_topology_pair_metrics"], topology_rows)
    write_csv_rows(tables["progression_topology_summary"], topology_summary)
    write_csv_rows(tables["group_map_progression_metrics"], group_map_rows)
    write_csv_rows(tables["braak_stage_deltas"], braak_rows)
    write_csv_rows(tables["braak_ordering_summary"], braak_summary)
    write_csv_rows(tables["fast_progressor_classification"], classifier_rows)
    write_csv_rows(tables["pseudotime_loadings"], loading_rows)
    write_csv_rows(tables["pseudotime_group_contributions"], contribution_rows)
    write_csv_rows(tables["edge_confidence_bands"], edge_band_rows)
    write_csv_rows(tables["edge_confidence_summary"], edge_band_summary)
    write_csv_rows(tables["subject_archetypes"], archetype_rows)

    figures = make_extended_figures(
        fig_dir=fig_dir,
        topology_summary=topology_summary,
        topology_rows=topology_rows,
        group_map_rows=group_map_rows,
        braak_rows=braak_rows,
        braak_summary=braak_summary,
        classifier_rows=classifier_rows,
        loading_rows=loading_rows,
        contribution_rows=contribution_rows,
        edge_band_rows=edge_band_rows,
        fitted=fitted,
        dataset=dataset,
        split=split,
        selected_regions=selected_regions,
        target_indices=target_indices,
        archetype_rows=archetype_rows,
        skip_brain=skip_brain,
    )

    report = {
        "purpose": "Extended BN-LTE experiments emphasizing progression topology, anatomical ordering, fast-progressor classification, edge uncertainty, and additional brain visualizations.",
        "configuration": {
            "random_seed": int(random_seed),
            "max_parents": int(max_parents),
            "edge_bootstrap_iterations": int(edge_bootstrap_iterations),
            "skip_brain": bool(skip_brain),
        },
        "data": {
            "pairs": dataset.pair_count,
            "subjects": len({row["RID"] for row in dataset.metadata_rows}),
            "selected_regions": selected_regions,
        },
        "topology_summary": topology_summary,
        "group_map_progression_metrics": group_map_rows,
        "braak_ordering_summary": braak_summary,
        "fast_progressor_classification": classifier_rows,
        "edge_confidence_summary": edge_band_summary,
        "figures": {key: str(value) for key, value in figures.items()},
        "tables": {key: str(value) for key, value in tables.items()},
        "guardrails": [
            "Progression-topology metrics complement but do not replace MAE/RMSE.",
            "Fast-progressor thresholds are learned from the training split only.",
            "Braak-like grouping is approximate because only 10 selected tau regions are modeled.",
            "Edge confidence bands are bootstrap stability summaries of the ridge-estimated BN-LTE, not full MCMC credible intervals.",
        ],
    }
    report_path = out / "extended_paper_experiments_report.json"
    analysis_path = out / "extended_paper_experiments_analysis.md"
    write_json(report_path, report)
    analysis_path.write_text(render_extended_analysis(report), encoding="utf-8")
    report["report_path"] = str(report_path)
    report["analysis_path"] = str(analysis_path)
    return report


def progression_topology_rows(
    predictions: dict[str, np.ndarray],
    dataset: MultimodalPairDataset,
    split: Any,
    regions: list[str],
    target_indices: list[int],
) -> list[dict[str, Any]]:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    dt = dataset.time_years
    labels = split_labels(split, dataset.pair_count)
    rows = []
    for model in EXTENDED_MODELS:
        pred = predictions[model]
        for idx in range(dataset.pair_count):
            empirical_delta = observed[idx] - baseline[idx]
            predicted_delta = pred[idx] - baseline[idx]
            empirical_rate = empirical_delta / float(dt[idx])
            predicted_rate = predicted_delta / float(dt[idx])
            rows.append(
                {
                    "model": model,
                    "split": labels[idx],
                    "RID": dataset.metadata_rows[idx]["RID"],
                    "dx_nearest_baseline": dataset.metadata_rows[idx].get("dx_nearest_baseline", ""),
                    "z_stage": "",
                    "delta_spearman": safe_correlation(empirical_delta, predicted_delta, rank=True),
                    "delta_pearson": safe_correlation(empirical_delta, predicted_delta, rank=False),
                    "delta_cosine": cosine_similarity(empirical_delta, predicted_delta),
                    "rate_cosine": cosine_similarity(empirical_rate, predicted_rate),
                    "direction_accuracy": direction_accuracy(empirical_delta, predicted_delta),
                    "top2_overlap": topk_overlap(empirical_delta, predicted_delta, 2),
                    "top3_overlap": topk_overlap(empirical_delta, predicted_delta, 3),
                    "top5_overlap": topk_overlap(empirical_delta, predicted_delta, 5),
                    "weighted_top3_capture": weighted_topk_capture(empirical_delta, predicted_delta, 3),
                    "delta_mae": finite_mean_abs(predicted_delta - empirical_delta),
                    "delta_rmse": finite_rmse(predicted_delta - empirical_delta),
                }
            )
    return rows


def summarize_topology_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "delta_spearman",
        "delta_pearson",
        "delta_cosine",
        "rate_cosine",
        "direction_accuracy",
        "top2_overlap",
        "top3_overlap",
        "top5_overlap",
        "weighted_top3_capture",
        "delta_mae",
        "delta_rmse",
    ]
    output = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["split"]), []).append(row)
    for (model, split), group_rows in sorted(groups.items()):
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group_rows if is_finite(row[metric])], dtype=float)
            if values.size == 0:
                continue
            output.append(
                {
                    "model": model,
                    "split": split,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            )
    return output


def group_map_progression_metrics(
    predictions: dict[str, np.ndarray],
    dataset: MultimodalPairDataset,
    split: Any,
    regions: list[str],
    target_indices: list[int],
    z_values: np.ndarray,
) -> list[dict[str, Any]]:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    stage_labels = assign_stages(z_values)
    rows = []
    stage_items = [("all_test", np.asarray(split.test_indices, dtype=int))]
    for stage in ("early", "mid", "late"):
        stage_items.append((stage, np.asarray([idx for idx in split.test_indices if stage_labels[idx] == stage], dtype=int)))
    for stage, indices in stage_items:
        if indices.size < 3:
            continue
        empirical_s1 = np.nanmean(observed[indices], axis=0)
        empirical_delta = np.nanmean(observed[indices] - baseline[indices], axis=0)
        for model in EXTENDED_MODELS:
            pred_s1 = np.nanmean(predictions[model][indices], axis=0)
            pred_delta = np.nanmean(predictions[model][indices] - baseline[indices], axis=0)
            rows.append(
                {
                    "model": model,
                    "stage": stage,
                    "n_pairs": int(indices.size),
                    "group_map_mae_s1": finite_mean_abs(pred_s1 - empirical_s1),
                    "group_map_rmse_s1": finite_rmse(pred_s1 - empirical_s1),
                    "s1_map_spearman": safe_correlation(empirical_s1, pred_s1, rank=True),
                    "delta_map_spearman": safe_correlation(empirical_delta, pred_delta, rank=True),
                    "delta_map_pearson": safe_correlation(empirical_delta, pred_delta, rank=False),
                    "delta_cosine": cosine_similarity(empirical_delta, pred_delta),
                    "direction_accuracy": direction_accuracy(empirical_delta, pred_delta),
                    "top3_overlap": topk_overlap(empirical_delta, pred_delta, 3),
                    "weighted_top3_capture": weighted_topk_capture(empirical_delta, pred_delta, 3),
                }
            )
    return rows


def braak_ordering_rows(
    predictions: dict[str, np.ndarray],
    dataset: MultimodalPairDataset,
    split: Any,
    regions: list[str],
    target_indices: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    region_to_idx = {region: idx for idx, region in enumerate(regions)}
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    test = np.asarray(split.test_indices, dtype=int)
    rows = []
    empirical_group_delta = {}
    for group, group_regions in BRAAK_GROUPS.items():
        idxs = [region_to_idx[region] for region in group_regions if region in region_to_idx]
        empirical_group_delta[group] = float(np.nanmean(observed[test][:, idxs] - baseline[test][:, idxs])) if idxs else float("nan")
    for model in ["Empirical", *EXTENDED_MODELS]:
        source = observed if model == "Empirical" else predictions[model]
        group_delta = {}
        for group, group_regions in BRAAK_GROUPS.items():
            idxs = [region_to_idx[region] for region in group_regions if region in region_to_idx]
            value = float(np.nanmean(source[test][:, idxs] - baseline[test][:, idxs])) if idxs else float("nan")
            group_delta[group] = value
            rows.append({"model": model, "group": group, "mean_delta_suvr": value, "n_pairs": int(test.size), "n_regions": len(idxs)})
    summary = []
    empirical_order = np.asarray([empirical_group_delta[group] for group in BRAAK_GROUPS], dtype=float)
    for model in EXTENDED_MODELS:
        model_order = np.asarray([
            next(row["mean_delta_suvr"] for row in rows if row["model"] == model and row["group"] == group)
            for group in BRAAK_GROUPS
        ])
        summary.append(
            {
                "model": model,
                "braak_group_spearman": safe_correlation(empirical_order, model_order, rank=True),
                "braak_group_pearson": safe_correlation(empirical_order, model_order, rank=False),
                "braak_group_mae": finite_mean_abs(model_order - empirical_order),
                "top_group_empirical": list(BRAAK_GROUPS)[int(np.nanargmax(empirical_order))],
                "top_group_predicted": list(BRAAK_GROUPS)[int(np.nanargmax(model_order))],
                "top_group_correct": bool(int(np.nanargmax(empirical_order)) == int(np.nanargmax(model_order))),
            }
        )
    return rows, summary


def fast_progressor_classification(
    predictions: dict[str, np.ndarray],
    dataset: MultimodalPairDataset,
    split: Any,
    regions: list[str],
    target_indices: list[int],
) -> list[dict[str, Any]]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_curve, roc_auc_score

    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    observed_rate = (observed - baseline) / dataset.time_years[:, None]
    empirical_score = np.nanmean(observed_rate, axis=1)
    threshold = float(np.nanquantile(empirical_score[np.asarray(split.train_indices, dtype=int)], 0.75))
    y_test = empirical_score[np.asarray(split.test_indices, dtype=int)] >= threshold
    rows = []
    for model in EXTENDED_MODELS:
        pred_rate = (predictions[model] - baseline) / dataset.time_years[:, None]
        score = np.nanmean(pred_rate, axis=1)[np.asarray(split.test_indices, dtype=int)]
        rows.append(
            {
                "model": model,
                "threshold_train_q75": threshold,
                "test_fast_progressor_fraction": float(np.mean(y_test)),
                "auroc": safe_auroc(y_test, score),
                "auprc": safe_auprc(y_test, score),
                "balanced_accuracy_at_train_threshold": safe_balanced_accuracy(y_test, score >= threshold),
                "top_decile_precision": top_fraction_precision(y_test, score, fraction=0.10),
                "top_quartile_precision": top_fraction_precision(y_test, score, fraction=0.25),
            }
        )
    return rows


def pseudotime_contribution_rows(pseudotime: Any, dataset: MultimodalPairDataset, z_values: np.ndarray) -> list[dict[str, Any]]:
    contributions = pseudotime.contributions(dataset.feature_matrix)
    rows = []
    diagnoses = [str(row.get("dx_nearest_baseline", "") or "unknown") for row in dataset.metadata_rows]
    stages = assign_stages(z_values)
    for group_name, labels in [("diagnosis", diagnoses), ("stage", stages)]:
        for label in sorted(set(labels)):
            indices = np.asarray([idx for idx, item in enumerate(labels) if item == label], dtype=int)
            if indices.size == 0:
                continue
            for feature_idx, feature in enumerate(pseudotime.selected_feature_names):
                values = contributions[indices, feature_idx]
                rows.append(
                    {
                        "group_type": group_name,
                        "group": label,
                        "feature": feature,
                        "n": int(indices.size),
                        "mean_contribution": finite_mean(values),
                        "median_contribution": finite_median(values),
                        "mean_abs_contribution": finite_mean_abs(values),
                    }
                )
    return rows


def bootstrap_edge_confidence_bands(
    *,
    dataset: MultimodalPairDataset,
    split: Any,
    fitted: dict[str, Any],
    target_names: list[str],
    iterations: int,
    random_seed: int,
    max_parents: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    top_edges = []
    for row in edge_curve_grid_rows(fitted["bn_fit"], top_k=12):
        edge = (row["parent"], row["target"])
        if edge not in top_edges:
            top_edges.append(edge)
    top_edges = top_edges[:8]
    z_grid = fitted["bn_fit"].z_grid
    train_indices = np.asarray(split.train_indices, dtype=int)
    groups = np.asarray([row["RID"] for row in dataset.metadata_rows], dtype=object)
    unique_groups = np.unique(groups[train_indices])
    indices_by_group = {group: train_indices[groups[train_indices] == group] for group in unique_groups}
    rng = np.random.default_rng(int(random_seed))
    curves: dict[tuple[str, str], list[np.ndarray]] = {edge: [] for edge in top_edges}
    print(f"  edge confidence bootstrap: {iterations} iterations, {len(top_edges)} edges")
    for iteration in range(int(iterations)):
        if (iteration + 1) % max(1, int(iterations) // 4) == 0:
            print(f"    edge CI bootstrap {iteration + 1}/{iterations}")
        sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        sampled_indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        fit = fit_dynamic_scm(
            dataset,
            fitted["pseudotime"],
            sampled_indices,
            target_names=target_names,
            max_parents_per_target=max_parents,
            cv_folds=3,
        )
        basis = fit.spline_basis.transform(z_grid)
        lookup = {}
        for target_fit in fit.target_fits:
            for parent in target_fit.parent_names:
                lookup[(parent, target_fit.target_name)] = target_fit.parent_effect_curve(parent, basis)
        for edge in top_edges:
            curves[edge].append(lookup.get(edge, np.zeros_like(z_grid)))
    rows = []
    summary = []
    for parent, target in top_edges:
        arr = np.vstack(curves[(parent, target)])
        med = np.quantile(arr, 0.50, axis=0)
        lo = np.quantile(arr, 0.025, axis=0)
        hi = np.quantile(arr, 0.975, axis=0)
        inclusion = np.mean(np.max(np.abs(arr), axis=1) >= 0.01)
        summary.append(
            {
                "parent": parent,
                "target": target,
                "bootstrap_iterations": int(iterations),
                "inclusion_probability": float(inclusion),
                "max_abs_median_effect": float(np.max(np.abs(med))),
                "z_at_max_abs_median": float(z_grid[int(np.argmax(np.abs(med)))]),
            }
        )
        for z, q025, q50, q975 in zip(z_grid, lo, med, hi, strict=True):
            rows.append(
                {
                    "parent": parent,
                    "target": target,
                    "z": float(z),
                    "effect_q025": float(q025),
                    "effect_q50": float(q50),
                    "effect_q975": float(q975),
                }
            )
    return rows, sorted(summary, key=lambda row: -float(row["inclusion_probability"]))


def subject_archetype_rows(
    predictions: dict[str, np.ndarray],
    dataset: MultimodalPairDataset,
    split: Any,
    regions: list[str],
    target_indices: list[int],
    z_values: np.ndarray,
) -> list[dict[str, Any]]:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    test = np.asarray(split.test_indices, dtype=int)
    empirical_delta = observed - baseline
    bnlte_error = np.asarray([finite_mean_abs(predictions["BayesianNetwork-SCM"][idx] - observed[idx]) for idx in range(dataset.pair_count)])
    persistence_error = np.asarray([finite_mean_abs(baseline[idx] - observed[idx]) for idx in range(dataset.pair_count)])
    mean_rate = np.nanmean(empirical_delta / dataset.time_years[:, None], axis=1)
    candidates = {
        "fast_progressor": int(test[np.nanargmax(mean_rate[test])]),
        "slow_progressor": int(test[np.nanargmin(mean_rate[test])]),
        "bnlte_best_case": int(test[np.nanargmin(bnlte_error[test])]),
        "bnlte_worst_case": int(test[np.nanargmax(bnlte_error[test])]),
        "bnlte_gain_over_persistence": int(test[np.nanargmax((persistence_error - bnlte_error)[test])]),
        "late_stage_case": int(test[np.nanargmax(z_values[test])]),
    }
    rows = []
    for label, idx in candidates.items():
        rows.append(
            {
                "archetype": label,
                "row_index": int(idx),
                "RID": dataset.metadata_rows[idx]["RID"],
                "dx_nearest_baseline": dataset.metadata_rows[idx].get("dx_nearest_baseline", ""),
                "z": float(z_values[idx]),
                "mean_empirical_rate": float(mean_rate[idx]),
                "bnlte_mae_s1": float(bnlte_error[idx]),
                "persistence_mae_s1": float(persistence_error[idx]),
                "bnlte_gain_vs_persistence": float(persistence_error[idx] - bnlte_error[idx]),
            }
        )
    return rows


def make_extended_figures(
    *,
    fig_dir: Path,
    topology_summary: list[dict[str, Any]],
    topology_rows: list[dict[str, Any]],
    group_map_rows: list[dict[str, Any]],
    braak_rows: list[dict[str, Any]],
    braak_summary: list[dict[str, Any]],
    classifier_rows: list[dict[str, Any]],
    loading_rows: list[dict[str, Any]],
    contribution_rows: list[dict[str, Any]],
    edge_band_rows: list[dict[str, Any]],
    fitted: dict[str, Any],
    dataset: MultimodalPairDataset,
    split: Any,
    selected_regions: list[str],
    target_indices: list[int],
    archetype_rows: list[dict[str, Any]],
    skip_brain: bool,
) -> dict[str, Path]:
    figures = {
        "progression_topology": fig_dir / "fig10_progression_topology_metrics.png",
        "group_map_progression": fig_dir / "fig11_group_map_progression_metrics.png",
        "topk_overlap": fig_dir / "fig12_topk_overlap_distribution.png",
        "braak_ordering": fig_dir / "fig13_braak_ordering.png",
        "fast_progressor": fig_dir / "fig14_fast_progressor_classification.png",
        "pseudotime_explainability": fig_dir / "fig15_pseudotime_explainability.png",
        "edge_confidence_bands": fig_dir / "fig16_edge_confidence_bands.png",
    }
    plot_progression_topology(figures["progression_topology"], topology_summary)
    plot_group_map_progression(figures["group_map_progression"], group_map_rows)
    plot_topk_overlap(figures["topk_overlap"], topology_rows)
    plot_braak_ordering(figures["braak_ordering"], braak_rows, braak_summary)
    plot_fast_progressor(figures["fast_progressor"], classifier_rows)
    plot_pseudotime_explainability(figures["pseudotime_explainability"], loading_rows, contribution_rows)
    plot_edge_confidence_bands(figures["edge_confidence_bands"], edge_band_rows)
    if not skip_brain:
        figures["brain_stage_delta_models"] = fig_dir / "fig17_brain_stage_delta_models.png"
        figures["brain_topk_overlap"] = fig_dir / "fig18_brain_topk_progression_overlap.png"
        figures["brain_subject_archetypes"] = fig_dir / "fig19_brain_subject_archetypes.png"
        figures["brain_bilateral_delta_butterfly"] = fig_dir / "fig20_brain_bilateral_delta_butterfly.png"
        figures["brain_region_delta_heatmap"] = fig_dir / "fig21_brain_region_delta_heatmap.png"
        figures["brain_causal_flow_schematic"] = fig_dir / "fig22_brain_causal_flow_schematic.png"
        figures["brain_stage_radial_fingerprint"] = fig_dir / "fig23_brain_stage_radial_fingerprint.png"
        figures["brain_progression_heatmap"] = fig_dir / "fig24_brain_surface_progression_heatmap.png"
        figures["brain_prediction_error_heatmap"] = fig_dir / "fig25_brain_surface_prediction_error_heatmap.png"
        figures["brain_bnlte_advantage_heatmap"] = fig_dir / "fig26_brain_surface_bnlte_advantage_heatmap.png"
        plot_brain_stage_delta_models(figures["brain_stage_delta_models"], fitted, dataset, split, selected_regions, target_indices)
        plot_brain_topk_overlap(figures["brain_topk_overlap"], fitted, dataset, split, selected_regions, target_indices)
        plot_brain_subject_archetypes(figures["brain_subject_archetypes"], fitted, dataset, selected_regions, target_indices, archetype_rows)
        plot_brain_bilateral_delta_butterfly(figures["brain_bilateral_delta_butterfly"], fitted, dataset, split, selected_regions, target_indices)
        plot_brain_region_delta_heatmap(figures["brain_region_delta_heatmap"], fitted, dataset, split, selected_regions, target_indices)
        plot_brain_causal_flow_schematic(figures["brain_causal_flow_schematic"], fitted, dataset, split, selected_regions, target_indices, edge_band_rows)
        plot_brain_stage_radial_fingerprint(figures["brain_stage_radial_fingerprint"], fitted, dataset, split, selected_regions, target_indices)
        plot_brain_progression_heatmap(figures["brain_progression_heatmap"], fitted, dataset, split, selected_regions, target_indices)
        plot_brain_prediction_error_heatmap(figures["brain_prediction_error_heatmap"], fitted, dataset, split, selected_regions, target_indices)
        plot_brain_bnlte_advantage_heatmap(figures["brain_bnlte_advantage_heatmap"], fitted, dataset, split, selected_regions, target_indices)
    return figures


def plot_progression_topology(path: Path, summary: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(summary)
    metrics = ["delta_spearman", "delta_cosine", "top3_overlap", "direction_accuracy"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Progression-topology metrics on held-out subjects", fontsize=16, fontweight="bold")
    for ax, metric in zip(axes.flat, metrics, strict=True):
        sub = df[(df["split"] == "test") & (df["metric"] == metric)]
        sub = sub.set_index("model").reindex(EXTENDED_MODELS).reset_index()
        colors = [MODEL_COLORS.get(model, "#777777") for model in sub["model"]]
        ax.bar([short_model(model) for model in sub["model"]], sub["median"], color=colors, alpha=0.82)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    save_figure(fig, path)


def plot_group_map_progression(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    metrics = ["group_map_mae_s1", "delta_map_spearman", "delta_cosine", "top3_overlap"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Group-average brain-map progression agreement", fontsize=16, fontweight="bold")
    for ax, metric in zip(axes.flat, metrics, strict=True):
        sub = df[df["stage"] == "all_test"].set_index("model").reindex(EXTENDED_MODELS).reset_index()
        ax.bar([short_model(model) for model in sub["model"]], sub[metric], color=[MODEL_COLORS.get(model, "#777") for model in sub["model"]], alpha=0.82)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    save_figure(fig, path)


def plot_topk_overlap(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    metric = "top3_overlap"
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    data = [df[(df["split"] == "test") & (df["model"] == model)][metric].dropna().to_numpy(float) for model in EXTENDED_MODELS]
    try:
        box = ax.boxplot(data, tick_labels=[short_model(model) for model in EXTENDED_MODELS], patch_artist=True, showfliers=False)
    except TypeError:
        box = ax.boxplot(data, labels=[short_model(model) for model in EXTENDED_MODELS], patch_artist=True, showfliers=False)
    for patch, model in zip(box["boxes"], EXTENDED_MODELS, strict=True):
        patch.set_facecolor(MODEL_COLORS.get(model, "#777777"))
        patch.set_alpha(0.72)
    ax.set_title("Top-3 progressing-region overlap per held-out subject", fontsize=15, fontweight="bold")
    ax.set_ylabel("Overlap with empirical top-3 regional tau increases")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def plot_braak_ordering(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    groups = list(BRAAK_GROUPS)
    for model in ["Empirical", "BayesianNetwork-SCM", "ESM", "SIR", "NDM"]:
        sub = df[df["model"] == model].set_index("group").reindex(groups)
        color = "#111827" if model == "Empirical" else MODEL_COLORS.get(model, "#777777")
        axes[0].plot(groups, sub["mean_delta_suvr"], marker="o", label=short_model(model), color=color)
    axes[0].set_title("Braak-like regional progression profile")
    axes[0].set_ylabel("Mean tau SUVR change")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    sdf = pd.DataFrame(summary).set_index("model").reindex(EXTENDED_MODELS).reset_index()
    axes[1].bar([short_model(model) for model in sdf["model"]], sdf["braak_group_spearman"], color=[MODEL_COLORS.get(model, "#777") for model in sdf["model"]], alpha=0.82)
    axes[1].set_title("Braak-group ordering agreement")
    axes[1].set_ylabel("Spearman vs empirical group deltas")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Anatomical progression ordering", fontsize=16, fontweight="bold")
    save_figure(fig, path)


def plot_fast_progressor(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows).set_index("model").reindex(EXTENDED_MODELS).reset_index()
    metrics = ["auroc", "auprc", "top_decile_precision", "top_quartile_precision"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Fast-progressor classification", fontsize=16, fontweight="bold")
    for ax, metric in zip(axes.flat, metrics, strict=True):
        ax.bar([short_model(model) for model in df["model"]], df[metric], color=[MODEL_COLORS.get(model, "#777") for model in df["model"]], alpha=0.82)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    save_figure(fig, path)


def plot_pseudotime_explainability(path: Path, loading_rows: list[dict[str, Any]], contribution_rows: list[dict[str, Any]]) -> None:
    loadings = pd.DataFrame(loading_rows).head(12)
    contrib = pd.DataFrame(contribution_rows)
    stage = contrib[contrib["group_type"] == "stage"]
    top_features = list(loadings["feature"].head(10))
    heat = stage[stage["feature"].isin(top_features)].pivot(index="feature", columns="group", values="mean_contribution").reindex(top_features)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    axes[0].barh(loadings["feature"][::-1], loadings["loading"][::-1], color="#D55E00", alpha=0.82)
    axes[0].set_title("Top pseudotime loadings")
    axes[0].grid(axis="x", alpha=0.25)
    image = axes[1].imshow(heat.fillna(0.0).to_numpy(float), aspect="auto", cmap="RdBu_r")
    axes[1].set_yticks(np.arange(len(heat.index)))
    axes[1].set_yticklabels(heat.index)
    axes[1].set_xticks(np.arange(len(heat.columns)))
    axes[1].set_xticklabels(heat.columns)
    axes[1].set_title("Mean feature contribution by Z stage")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("Explainability of latent disease time Z", fontsize=16, fontweight="bold")
    save_figure(fig, path)


def plot_edge_confidence_bands(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    edges = list(dict.fromkeys(zip(df["parent"], df["target"])))[:8]
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), constrained_layout=True)
    fig.suptitle("Bootstrap confidence bands for dynamic edge curves", fontsize=16, fontweight="bold")
    for ax, edge in zip(axes.flat, edges, strict=False):
        sub = df[(df["parent"] == edge[0]) & (df["target"] == edge[1])].sort_values("z")
        ax.fill_between(sub["z"], sub["effect_q025"], sub["effect_q975"], color="#56B4E9", alpha=0.25)
        ax.plot(sub["z"], sub["effect_q50"], color="#D55E00", linewidth=2.0)
        ax.axhline(0.0, color="#111827", linewidth=0.8)
        ax.set_title(f"{short_parent(edge[0])} -> {short_target(edge[1])}", fontsize=9)
        ax.grid(alpha=0.20)
    for ax in axes.flat[len(edges):]:
        ax.set_axis_off()
    save_figure(fig, path)


def plot_brain_stage_delta_models(path: Path, fitted: dict[str, Any], dataset: MultimodalPairDataset, split: Any, regions: list[str], target_indices: list[int]) -> None:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    z_stage = assign_stages(fitted["z_values"])
    panels = []
    for stage in ("early", "mid", "late"):
        indices = np.asarray([idx for idx in split.test_indices if z_stage[idx] == stage], dtype=int)
        if indices.size < 3:
            continue
        empirical_delta = np.nanmean(observed[indices] - baseline[indices], axis=0)
        panels.append((f"{stage.title()} empirical delta", dict(zip(regions, empirical_delta, strict=True))))
        for model in ("BayesianNetwork-SCM", "ESM", "SIR"):
            pred_delta = np.nanmean(fitted["predictions"][model][indices] - baseline[indices], axis=0)
            panels.append((f"{stage.title()} {short_model(model)} delta", dict(zip(regions, pred_delta, strict=True))))
    bound = max([abs(value) for _, row in panels for value in row.values() if is_finite(value)] + [0.02])
    plot_surface_grid(
        path,
        "Stage-specific tau progression deltas",
        panels,
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        colorbar_label="Tau SUVR change",
        note="Rows show empirical and model-predicted follow-up minus baseline tau maps by latent disease stage; views include lateral, medial, and ventral surfaces.",
    )


def plot_brain_topk_overlap(path: Path, fitted: dict[str, Any], dataset: MultimodalPairDataset, split: Any, regions: list[str], target_indices: list[int]) -> None:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    test = np.asarray(split.test_indices, dtype=int)
    empirical_delta = np.nanmean(observed[test] - baseline[test], axis=0)
    empirical_top = set(np.argsort(-empirical_delta)[:3])
    panels = []
    for model in ("BayesianNetwork-SCM", "ESM", "SIR", "NDM"):
        pred_delta = np.nanmean(fitted["predictions"][model][test] - baseline[test], axis=0)
        pred_top = set(np.argsort(-pred_delta)[:3])
        values = {}
        for idx, region in enumerate(regions):
            if idx in empirical_top and idx in pred_top:
                value = 1.5
            elif idx in empirical_top:
                value = 0.5
            elif idx in pred_top:
                value = 1.0
            else:
                value = 0.0
            values[region] = value
        panels.append((f"{short_model(model)} top-3 overlap", values))
    plot_surface_grid(
        path,
        "Top progressing-region overlap on brain surface",
        panels,
        cmap="viridis",
        vmin=0.0,
        vmax=1.5,
        colorbar_label="0 none, 0.5 empirical, 1.0 predicted, 1.5 overlap",
        note="Categorical map of empirical and predicted top-3 progressing regions. Overlap regions indicate model recovery of the empirical progression hot spots.",
    )


def plot_brain_subject_archetypes(path: Path, fitted: dict[str, Any], dataset: MultimodalPairDataset, regions: list[str], target_indices: list[int], archetype_rows: list[dict[str, Any]]) -> None:
    panels = []
    seen = set()
    for row in archetype_rows:
        idx = int(row["row_index"])
        if idx in seen:
            continue
        seen.add(idx)
        prefix = str(row["archetype"]).replace("_", " ")
        baseline = dataset.target_baseline[idx, target_indices]
        observed = dataset.target_observed[idx, target_indices]
        panels.append((f"{prefix}: baseline", dict(zip(regions, baseline, strict=True))))
        panels.append((f"{prefix}: empirical", dict(zip(regions, observed, strict=True))))
        panels.append((f"{prefix}: BN-LTE", dict(zip(regions, fitted["predictions"]["BayesianNetwork-SCM"][idx], strict=True))))
        if len(panels) >= 12:
            break
    finite = [value for _, row in panels for value in row.values() if is_finite(value)]
    vmin = min(finite) - 0.03
    vmax = max(finite) + 0.03
    plot_surface_grid(
        path,
        "Representative subject archetypes",
        panels,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        colorbar_label="Tau SUVR",
        note="Each archetype shows baseline, empirical follow-up, and BN-LTE predicted follow-up on multi-view surfaces.",
    )


def plot_brain_bilateral_delta_butterfly(path: Path, fitted: dict[str, Any], dataset: MultimodalPairDataset, split: Any, regions: list[str], target_indices: list[int]) -> None:
    deltas = test_group_delta_maps(fitted, dataset, split, regions, target_indices)
    models = ["Empirical", "BayesianNetwork-SCM", "ESM", "SIR", "NDM"]
    bases = ["entorhinal", "fusiform", "inferiortemporal", "middletemporal", "inferiorparietal"]
    labels = ["Entorhinal", "Fusiform", "Inferior temporal", "Middle temporal", "Inferior parietal"]
    finite_values = [value for model in models for value in deltas.get(model, np.array([])) if is_finite(value)]
    bound = max([abs(value) for value in finite_values] + [0.02])
    norm = Normalize(vmin=-bound, vmax=bound)
    cmap = plt.get_cmap("RdBu_r")
    fig, axes = plt.subplots(1, len(models), figsize=(18, 7), sharey=True, constrained_layout=True)
    y = np.arange(len(bases), dtype=float)
    for ax, model in zip(axes, models, strict=True):
        values = region_value_dict(regions, deltas[model])
        for row_idx, base in enumerate(bases):
            left = values.get(f"L_{base}", np.nan)
            right = values.get(f"R_{base}", np.nan)
            if is_finite(left):
                ax.barh(row_idx, -abs(float(left)), color=cmap(norm(float(left))), height=0.36, edgecolor="white", linewidth=0.7)
            if is_finite(right):
                ax.barh(row_idx, abs(float(right)), color=cmap(norm(float(right))), height=0.36, edgecolor="white", linewidth=0.7)
        ax.axvline(0.0, color="#111827", linewidth=1.0)
        ax.set_xlim(-bound * 1.16, bound * 1.16)
        ax.set_title(short_model(model), fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.18)
        ticks = np.array([-bound, -bound / 2.0, 0.0, bound / 2.0, bound])
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{abs(t):.2f}" if abs(t) > 1e-8 else "0" for t in ticks], fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    fig.suptitle("Bilateral tau progression butterfly", x=0.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.02,
        0.91,
        "Outward bar length is the magnitude of test-set mean tau change; color encodes signed SUVR change. Left hemisphere is plotted to the left and right hemisphere to the right.",
        ha="left",
        fontsize=9.5,
        color="#4B5563",
    )
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.018, pad=0.025)
    cbar.set_label("Signed tau SUVR change", fontsize=9)
    save_figure(fig, path)


def plot_brain_region_delta_heatmap(path: Path, fitted: dict[str, Any], dataset: MultimodalPairDataset, split: Any, regions: list[str], target_indices: list[int]) -> None:
    deltas = test_group_delta_maps(fitted, dataset, split, regions, target_indices)
    order = anatomical_region_order(regions)
    order_idx = [regions.index(region) for region in order]
    models = ["Empirical", "BayesianNetwork-SCM", "ESM", "SIR", "NDM"]
    delta_matrix = np.vstack([deltas[model][order_idx] for model in models])
    residual_models = ["BayesianNetwork-SCM", "ESM", "SIR", "NDM"]
    residual_matrix = np.vstack([deltas[model][order_idx] - deltas["Empirical"][order_idx] for model in residual_models])
    delta_bound = max([abs(value) for value in delta_matrix.ravel() if is_finite(value)] + [0.02])
    resid_bound = max([abs(value) for value in residual_matrix.ravel() if is_finite(value)] + [0.02])
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.8), constrained_layout=True)
    image0 = axes[0].imshow(delta_matrix, aspect="auto", cmap="RdBu_r", vmin=-delta_bound, vmax=delta_bound)
    image1 = axes[1].imshow(residual_matrix, aspect="auto", cmap="RdBu_r", vmin=-resid_bound, vmax=resid_bound)
    axes[0].set_title("Regional tau-progression fingerprint")
    axes[1].set_title("Model residual fingerprint vs empirical")
    axes[0].set_yticks(np.arange(len(models)))
    axes[0].set_yticklabels([short_model(model) for model in models])
    axes[1].set_yticks(np.arange(len(residual_models)))
    axes[1].set_yticklabels([short_model(model) for model in residual_models])
    labels = [REGION_SHORT_NAMES.get(region, region) for region in order]
    for ax in axes:
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        add_anatomical_group_lines(ax, order)
        ax.tick_params(axis="both", labelsize=9)
    axes[0].add_patch(Rectangle((-0.5, 1 - 0.5), len(order), 1, fill=False, edgecolor=MODEL_COLORS["BayesianNetwork-SCM"], linewidth=2.5))
    axes[1].add_patch(Rectangle((-0.5, 0 - 0.5), len(order), 1, fill=False, edgecolor=MODEL_COLORS["BayesianNetwork-SCM"], linewidth=2.5))
    fig.colorbar(image0, ax=axes[0], fraction=0.046, pad=0.03, label="Tau SUVR change")
    fig.colorbar(image1, ax=axes[1], fraction=0.046, pad=0.03, label="Predicted - empirical")
    fig.suptitle("Anatomical heatmap of progression and residuals", x=0.02, ha="left", fontsize=16, fontweight="bold")
    save_figure(fig, path)


def plot_brain_causal_flow_schematic(
    path: Path,
    fitted: dict[str, Any],
    dataset: MultimodalPairDataset,
    split: Any,
    regions: list[str],
    target_indices: list[int],
    edge_band_rows: list[dict[str, Any]],
) -> None:
    deltas = test_group_delta_maps(fitted, dataset, split, regions, target_indices)
    bnlte = region_value_dict(regions, deltas["BayesianNetwork-SCM"])
    edge_summary = edge_strength_summary(edge_band_rows)
    finite_values = [value for value in bnlte.values() if is_finite(value)]
    bound = max([abs(value) for value in finite_values] + [0.02])
    norm = Normalize(vmin=-bound, vmax=bound)
    cmap = plt.get_cmap("RdBu_r")
    coords = schematic_region_coordinates()
    fig, ax = plt.subplots(figsize=(12.8, 8.2), constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(-1.35, 1.55)
    ax.axis("off")
    for center, angle, hemi in [((-1.05, -0.06), -12, "Left lateral"), ((1.05, -0.06), 12, "Right lateral")]:
        ax.add_patch(Ellipse(center, 1.75, 2.35, angle=angle, facecolor="#F3F4F6", edgecolor="#9CA3AF", linewidth=1.4, zorder=0))
        ax.add_patch(Ellipse((center[0], center[1] - 0.07), 1.35, 1.85, angle=angle, facecolor="none", edgecolor="#D1D5DB", linewidth=0.9, zorder=0))
        ax.text(center[0], -1.23, hemi, ha="center", va="center", fontsize=10, color="#374151", fontweight="bold")
    source = (0.0, 1.22)
    ax.scatter([source[0]], [source[1]], s=950, marker="s", color="#111827", edgecolor="white", linewidth=1.4, zorder=5)
    ax.text(source[0], source[1], "Plasma\nAbeta42/40", ha="center", va="center", fontsize=9, color="white", fontweight="bold", zorder=6)
    max_edge = max([row["max_abs_effect"] for row in edge_summary.values()] + [0.01])
    for region in regions:
        target = f"tau_rate:{region}"
        if region not in coords or target not in edge_summary:
            continue
        effect = edge_summary[target]["effect_at_max"]
        width = 0.9 + 4.0 * edge_summary[target]["max_abs_effect"] / max_edge
        color = "#D55E00" if effect >= 0 else "#0072B2"
        arrow = FancyArrowPatch(
            source,
            coords[region],
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=width,
            color=color,
            alpha=0.25 + 0.50 * edge_summary[target]["max_abs_effect"] / max_edge,
            connectionstyle="arc3,rad=0.10",
            zorder=1,
        )
        ax.add_patch(arrow)
    for region in anatomical_region_order(regions):
        if region not in coords:
            continue
        value = bnlte.get(region, np.nan)
        size = 250 + 1800 * min(abs(float(value)) / bound, 1.0) if is_finite(value) else 250
        node_value = float(value) if is_finite(value) else 0.0
        ax.scatter([coords[region][0]], [coords[region][1]], s=size, color=cmap(norm(node_value)), edgecolor="#111827", linewidth=0.9, zorder=4)
        ax.text(coords[region][0], coords[region][1] - 0.17, REGION_SHORT_NAMES.get(region, region), ha="center", va="top", fontsize=8.2, zorder=6)
    ax.text(-2.26, 1.42, "Causal-flow brain schematic", ha="left", va="top", fontsize=16, fontweight="bold")
    ax.text(
        -2.26,
        1.24,
        "Node color and size show BN-LTE predicted test-set tau progression. Arrow thickness shows bootstrap-stable dynamic edge magnitude from the plasma Abeta root into regional tau-rate nodes.",
        ha="left",
        va="top",
        fontsize=9.2,
        color="#4B5563",
        wrap=True,
    )
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("BN-LTE tau SUVR change", fontsize=9)
    save_figure(fig, path)


def plot_brain_stage_radial_fingerprint(path: Path, fitted: dict[str, Any], dataset: MultimodalPairDataset, split: Any, regions: list[str], target_indices: list[int]) -> None:
    order = anatomical_region_order(regions)
    order_idx = [regions.index(region) for region in order]
    labels = [REGION_SHORT_NAMES.get(region, region) for region in order]
    stage_maps = stage_delta_maps(fitted, dataset, split, regions, target_indices)
    models = ["Empirical", "BayesianNetwork-SCM", "ESM", "SIR"]
    stages = ["early", "mid", "late"]
    all_values = []
    for stage in stages:
        for model in models:
            if stage in stage_maps and model in stage_maps[stage]:
                all_values.extend(np.maximum(stage_maps[stage][model][order_idx], 0.0))
    bound = max([float(value) for value in all_values if is_finite(value)] + [0.02])
    theta = np.linspace(0, 2 * np.pi, len(order), endpoint=False)
    closed_theta = np.r_[theta, theta[0]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), subplot_kw={"projection": "polar"}, constrained_layout=True)
    for ax, stage in zip(axes, stages, strict=True):
        ax.set_title(stage.title(), fontsize=12, fontweight="bold", pad=14)
        for model in models:
            if stage not in stage_maps or model not in stage_maps[stage]:
                continue
            values = np.maximum(stage_maps[stage][model][order_idx], 0.0)
            closed_values = np.r_[values, values[0]]
            color = "#111827" if model == "Empirical" else MODEL_COLORS.get(model, "#777777")
            linewidth = 2.4 if model in {"Empirical", "BayesianNetwork-SCM"} else 1.5
            ax.plot(closed_theta, closed_values, color=color, linewidth=linewidth, label=short_model(model))
            if model == "BayesianNetwork-SCM":
                ax.fill(closed_theta, closed_values, color=color, alpha=0.13)
        ax.set_ylim(0, bound * 1.08)
        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels([])
        ax.grid(alpha=0.28)
    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.34, 1.18), fontsize=8)
    fig.suptitle("Stage-wise regional progression fingerprints", x=0.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.02,
        0.91,
        "Polar axes follow the anatomical region order. Radius is positive tau SUVR increase from baseline to follow-up within each latent disease-time stage.",
        ha="left",
        fontsize=9.5,
        color="#4B5563",
    )
    save_figure(fig, path)


def plot_brain_progression_heatmap(path: Path, fitted: dict[str, Any], dataset: MultimodalPairDataset, split: Any, regions: list[str], target_indices: list[int]) -> None:
    deltas = test_group_delta_maps(fitted, dataset, split, regions, target_indices)
    panels = []
    for model in ["Empirical", "BayesianNetwork-SCM", "ESM", "SIR", "NDM"]:
        positive_delta = np.maximum(deltas[model], 0.0)
        panels.append((f"{short_model(model)} progression heatmap", region_value_dict(regions, positive_delta)))
    vmax = max([value for _, row in panels for value in row.values() if is_finite(value)] + [0.02])
    plot_surface_grid(
        path,
        "Brain-surface heatmap of tau progression intensity",
        panels,
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        colorbar_label="Positive tau SUVR increase",
        note="Heatmap is computed from held-out test subjects as mean positive follow-up minus baseline tau change. The empirical row is the observed progression target; model rows show predicted progression intensity.",
    )


def plot_brain_prediction_error_heatmap(path: Path, fitted: dict[str, Any], dataset: MultimodalPairDataset, split: Any, regions: list[str], target_indices: list[int]) -> None:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    test = np.asarray(split.test_indices, dtype=int)
    empirical_s1 = np.nanmean(observed[test], axis=0)
    panels = []
    for model in ["BayesianNetwork-SCM", "ESM", "SIR", "NDM", "S0 persistence"]:
        pred_s1 = np.nanmean(fitted["predictions"][model][test], axis=0)
        error = np.abs(pred_s1 - empirical_s1)
        panels.append((f"{short_model(model)} absolute error", region_value_dict(regions, error)))
    vmax = max([value for _, row in panels for value in row.values() if is_finite(value)] + [0.01])
    plot_surface_grid(
        path,
        "Brain-surface heatmap of prediction error",
        panels,
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        colorbar_label="Absolute SUVR error",
        note="Regional heatmap shows absolute error between the group-mean empirical follow-up tau map and each model's group-mean predicted follow-up map.",
    )


def plot_brain_bnlte_advantage_heatmap(path: Path, fitted: dict[str, Any], dataset: MultimodalPairDataset, split: Any, regions: list[str], target_indices: list[int]) -> None:
    observed = dataset.target_observed[:, target_indices]
    test = np.asarray(split.test_indices, dtype=int)
    empirical_s1 = np.nanmean(observed[test], axis=0)
    bnlte_s1 = np.nanmean(fitted["predictions"]["BayesianNetwork-SCM"][test], axis=0)
    bnlte_error = np.abs(bnlte_s1 - empirical_s1)
    panels = []
    for model in ["ESM", "SIR", "NDM", "S0 persistence"]:
        pred_s1 = np.nanmean(fitted["predictions"][model][test], axis=0)
        competitor_error = np.abs(pred_s1 - empirical_s1)
        advantage = competitor_error - bnlte_error
        panels.append((f"BN-LTE advantage vs {short_model(model)}", region_value_dict(regions, advantage)))
    bound = max([abs(value) for _, row in panels for value in row.values() if is_finite(value)] + [0.01])
    plot_surface_grid(
        path,
        "Brain-surface heatmap of BN-LTE regional advantage",
        panels,
        cmap="BrBG",
        vmin=-bound,
        vmax=bound,
        colorbar_label="Competitor error minus BN-LTE error",
        note="Positive regions indicate where BN-LTE has lower group-mean follow-up error than the comparator; negative regions indicate comparator advantage.",
    )


def test_group_delta_maps(fitted: dict[str, Any], dataset: MultimodalPairDataset, split: Any, regions: list[str], target_indices: list[int]) -> dict[str, np.ndarray]:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    test = np.asarray(split.test_indices, dtype=int)
    deltas = {"Empirical": np.nanmean(observed[test] - baseline[test], axis=0)}
    for model in ("BayesianNetwork-SCM", "ESM", "SIR", "NDM", "S0 persistence"):
        if model in fitted["predictions"]:
            deltas[model] = np.nanmean(fitted["predictions"][model][test] - baseline[test], axis=0)
    return deltas


def stage_delta_maps(fitted: dict[str, Any], dataset: MultimodalPairDataset, split: Any, regions: list[str], target_indices: list[int]) -> dict[str, dict[str, np.ndarray]]:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    stage_labels = assign_stages(fitted["z_values"])
    out: dict[str, dict[str, np.ndarray]] = {}
    for stage in ("early", "mid", "late"):
        indices = np.asarray([idx for idx in split.test_indices if stage_labels[idx] == stage], dtype=int)
        if indices.size < 3:
            continue
        out[stage] = {"Empirical": np.nanmean(observed[indices] - baseline[indices], axis=0)}
        for model in ("BayesianNetwork-SCM", "ESM", "SIR", "NDM"):
            out[stage][model] = np.nanmean(fitted["predictions"][model][indices] - baseline[indices], axis=0)
    return out


def anatomical_region_order(regions: list[str]) -> list[str]:
    ordered: list[str] = []
    for group_regions in BRAAK_GROUPS.values():
        for region in group_regions:
            if region in regions and region not in ordered:
                ordered.append(region)
    for region in regions:
        if region not in ordered:
            ordered.append(region)
    return ordered


def region_value_dict(regions: list[str], values: np.ndarray) -> dict[str, float]:
    return {region: float(values[idx]) for idx, region in enumerate(regions)}


def add_anatomical_group_lines(ax: Any, ordered_regions: list[str]) -> None:
    base_to_group = {region: group for group, group_regions in BRAAK_GROUPS.items() for region in group_regions}
    groups = [base_to_group.get(region, "other") for region in ordered_regions]
    for idx in range(1, len(groups)):
        if groups[idx] != groups[idx - 1]:
            ax.axvline(idx - 0.5, color="#111827", linewidth=0.8, alpha=0.35)


def edge_strength_summary(edge_band_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    df = pd.DataFrame(edge_band_rows)
    if df.empty:
        return out
    for target, sub in df.groupby("target"):
        effects = sub["effect_q50"].to_numpy(float)
        if effects.size == 0 or not np.any(np.isfinite(effects)):
            continue
        idx = int(np.nanargmax(np.abs(effects)))
        out[str(target)] = {
            "max_abs_effect": float(abs(effects[idx])),
            "effect_at_max": float(effects[idx]),
            "z_at_max": float(sub["z"].to_numpy(float)[idx]),
        }
    return out


def schematic_region_coordinates() -> dict[str, tuple[float, float]]:
    left = {
        "L_entorhinal": (-1.48, -0.63),
        "L_fusiform": (-1.16, -0.78),
        "L_inferiortemporal": (-1.58, -0.30),
        "L_middletemporal": (-1.46, 0.12),
        "L_inferiorparietal": (-1.05, 0.56),
    }
    right = {
        "R_entorhinal": (1.48, -0.63),
        "R_fusiform": (1.16, -0.78),
        "R_inferiortemporal": (1.58, -0.30),
        "R_middletemporal": (1.46, 0.12),
        "R_inferiorparietal": (1.05, 0.56),
    }
    return {**left, **right}


def split_labels(split: Any, row_count: int) -> list[str]:
    labels = ["unknown"] * row_count
    for name, indices in (("train", split.train_indices), ("validation", split.validation_indices), ("test", split.test_indices)):
        for idx in indices:
            labels[int(idx)] = name
    return labels


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1.0e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def direction_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1.0e-12)
    if int(np.sum(mask)) == 0:
        return float("nan")
    return float(np.mean(np.sign(x[mask]) == np.sign(y[mask])))


def topk_overlap(empirical_delta: np.ndarray, predicted_delta: np.ndarray, k: int) -> float:
    mask = np.isfinite(empirical_delta) & np.isfinite(predicted_delta)
    if int(np.sum(mask)) < int(k):
        return float("nan")
    emp_order = np.argsort(-empirical_delta[mask])[: int(k)]
    pred_order = np.argsort(-predicted_delta[mask])[: int(k)]
    return float(len(set(emp_order) & set(pred_order)) / int(k))


def weighted_topk_capture(empirical_delta: np.ndarray, predicted_delta: np.ndarray, k: int) -> float:
    mask = np.isfinite(empirical_delta) & np.isfinite(predicted_delta)
    if int(np.sum(mask)) < int(k):
        return float("nan")
    emp = empirical_delta[mask]
    emp_positive = np.maximum(emp, 0.0)
    denom = float(np.sum(np.sort(emp_positive)[-int(k):]))
    if denom <= 1.0e-12:
        return float("nan")
    pred_top = np.argsort(-predicted_delta[mask])[: int(k)]
    return float(np.sum(emp_positive[pred_top]) / denom)


def safe_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(bool(x) for x in y_true)) < 2 or np.std(score) <= 1.0e-12:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, score))
    except ValueError:
        return float("nan")


def safe_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(set(bool(x) for x in y_true)) < 2 or np.std(score) <= 1.0e-12:
        return float("nan")
    try:
        return float(average_precision_score(y_true, score))
    except ValueError:
        return float("nan")


def safe_balanced_accuracy(y_true: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    if len(set(bool(x) for x in y_true)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, pred))


def top_fraction_precision(y_true: np.ndarray, score: np.ndarray, *, fraction: float) -> float:
    if y_true.size == 0 or not np.any(np.isfinite(score)):
        return float("nan")
    k = max(1, int(round(y_true.size * float(fraction))))
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf))[:k]
    return float(np.mean(y_true[order]))


def resolve_path(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


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


def render_extended_analysis(report: dict[str, Any]) -> str:
    topology = pd.DataFrame(report["topology_summary"])
    group_map = pd.DataFrame(report["group_map_progression_metrics"])
    classifier = pd.DataFrame(report["fast_progressor_classification"])

    def frame_block(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        return "```\n" + df.to_string(index=False) + "\n```"

    lines = [
        "# Extended BN-LTE Paper Experiments",
        "",
        "## Scope",
        "",
        "This extension adds progression-topology metrics, group-map metrics, Braak-like anatomical ordering, fast-progressor classification, pseudotime explainability, bootstrap edge confidence bands, and additional multi-view brain visualizations.",
        "",
        "## Key Progression-Topology Metrics",
        "",
    ]
    if not topology.empty:
        sub = topology[(topology["split"] == "test") & (topology["metric"].isin(["delta_spearman", "delta_cosine", "top3_overlap", "direction_accuracy"]))]
        lines.append(frame_block(sub[["model", "metric", "median", "q25", "q75"]]))
    lines.extend(["", "## Group-Map Progression Metrics", ""])
    if not group_map.empty:
        lines.append(frame_block(group_map[group_map["stage"] == "all_test"]))
    lines.extend(["", "## Fast-Progressor Classification", ""])
    if not classifier.empty:
        lines.append(frame_block(classifier))
    lines.extend(["", "## Figures", ""])
    for name, path in report["figures"].items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(["", "## Guardrails", ""])
    for item in report["guardrails"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
