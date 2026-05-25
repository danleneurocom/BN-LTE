#!/usr/bin/env python3
"""Validation-locked hyperparameter search for the ADNI BN-LTE model.

The search uses the existing subject-level train/validation/test split.  Model
selection is performed only on the validation subjects; the held-out test set is
evaluated once for the selected configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from bayesian_network_scm.data import MultimodalPairDataset, build_multimodal_pair_dataset  # noqa: E402
from bayesian_network_scm.dynamic_scm import DynamicSCMFit, fit_dynamic_scm  # noqa: E402
from bayesian_network_scm.pseudotime import PseudotimeModel, fit_pseudotime  # noqa: E402
from bayesian_network_scm.reporting import SubjectSplit, make_subject_split  # noqa: E402


SELECTED_MODEL_NAME = "BN-LTE"
BRAAK_GROUPS = {
    "entorhinal": ["L_entorhinal", "R_entorhinal"],
    "ventral_temporal": ["L_fusiform", "R_fusiform", "L_inferiortemporal", "R_inferiortemporal"],
    "lateral_temporal": ["L_middletemporal", "R_middletemporal"],
    "inferior_parietal": ["L_inferiorparietal", "R_inferiorparietal"],
}
RIDGE_PROFILES = {
    "balanced": (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    "flexible": (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    "conservative": (10.0, 100.0, 1000.0, 10000.0, 100000.0),
}


@dataclass(frozen=True)
class SearchConfig:
    config_id: str
    pseudotime_mode: str
    min_train_coverage: float
    max_parents: int
    n_knots: int
    spline_degree: int
    ridge_profile: str
    cv_folds: int

    @property
    def ridge_alphas(self) -> tuple[float, ...]:
        return RIDGE_PROFILES[self.ridge_profile]


@dataclass
class FittedConfig:
    config: SearchConfig
    pseudotime: PseudotimeModel
    fit: DynamicSCMFit
    predictions: np.ndarray
    z_values: np.ndarray


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "bnlte_hyperparameter_search")
    parser.add_argument("--random-seed", type=int, default=20260521)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--limit-configs", type=int, default=0, help="Optional smoke-test limit; 0 means full default grid.")
    args = parser.parse_args()

    report = run_search(
        project_root=args.project_root,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        top_k=args.top_k,
        limit_configs=args.limit_configs,
    )
    best = report["best_validation_config"]
    test = report["selected_test_metrics"]
    causal = report["best_tau_free_validation_config"]
    causal_test = report["tau_free_test_metrics"]
    default = report["current_default_validation_config"]
    default_test = report["current_default_test_metrics"]
    print("BN-LTE hyperparameter search complete.")
    print(f"Best validation config: {best['config_id']} score={best['selection_score']:.4f}")
    print(
        "Held-out test: "
        f"group_mae={test['group_map_mae_s1']:.6f}, "
        f"delta_spearman={test['delta_map_spearman']:.4f}, "
        f"auroc={test['fast_progressor_auroc']:.4f}, "
        f"braak_spearman={test['braak_group_spearman']:.4f}"
    )
    print(f"Best tau-free config: {causal['config_id']} score={causal['selection_score']:.4f}")
    print(
        "Tau-free held-out test: "
        f"group_mae={causal_test['group_map_mae_s1']:.6f}, "
        f"delta_spearman={causal_test['delta_map_spearman']:.4f}, "
        f"auroc={causal_test['fast_progressor_auroc']:.4f}, "
        f"braak_spearman={causal_test['braak_group_spearman']:.4f}"
    )
    print(f"Current default config: {default['config_id']} score={default['selection_score']:.4f}")
    print(
        "Current default held-out test: "
        f"group_mae={default_test['group_map_mae_s1']:.6f}, "
        f"delta_spearman={default_test['delta_map_spearman']:.4f}, "
        f"auroc={default_test['fast_progressor_auroc']:.4f}, "
        f"braak_spearman={default_test['braak_group_spearman']:.4f}"
    )
    return 0


def run_search(
    *,
    project_root: str | Path,
    output_dir: str | Path,
    random_seed: int,
    top_k: int,
    limit_configs: int,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    out = resolve_path(output_dir, root)
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_multimodal_pair_dataset(root)
    selected_regions = list(dataset.report["selected_tau_regions"])
    selected_target_names = [f"tau_rate:{region}" for region in selected_regions]
    selected_target_indices = [dataset.target_index(name) for name in selected_target_names]
    split = make_subject_split(dataset.metadata_rows, random_seed=random_seed)

    configs = default_search_grid()
    if int(limit_configs) > 0:
        configs = configs[: int(limit_configs)]
    print(f"Search grid: {len(configs)} configurations")
    print(
        "Split: "
        f"train={split.train_indices.size}, "
        f"validation={split.validation_indices.size}, "
        f"test={split.test_indices.size}"
    )

    validation_rows: list[dict[str, Any]] = []
    fit_errors: list[dict[str, Any]] = []
    for idx, config in enumerate(configs):
        print(f"  {idx + 1:03d}/{len(configs):03d} {config.config_id}")
        try:
            fitted = fit_config(dataset, split.train_indices, selected_target_names, selected_target_indices, config)
            metrics = evaluate_prediction(
                dataset=dataset,
                selected_regions=selected_regions,
                selected_target_indices=selected_target_indices,
                train_indices=split.train_indices,
                eval_indices=split.validation_indices,
                prediction=fitted.predictions,
                z_values=fitted.z_values,
            )
            metrics.update(config_row(config))
            metrics["split"] = "validation"
            metrics["status"] = "ok"
            validation_rows.append(metrics)
        except Exception as exc:  # noqa: BLE001 - search should continue and report failed configs.
            fit_errors.append({**config_row(config), "status": "failed", "error": repr(exc)})

    if not validation_rows:
        raise RuntimeError("No BN-LTE hyperparameter configuration completed successfully.")

    scored_rows = attach_selection_scores(validation_rows)
    scored_rows = sorted(scored_rows, key=lambda row: (-float(row["selection_score"]), str(row["config_id"])))
    best_config = config_from_row(scored_rows[0])
    tau_free_rows = [row for row in scored_rows if row["pseudotime_mode"] == "tau_free"]
    best_tau_free_row = tau_free_rows[0] if tau_free_rows else scored_rows[0]
    best_tau_free_config = config_from_row(best_tau_free_row)
    default_config_id = "tau_free_cov0.50_p6_k4_d3_balanced_cv5"
    default_rows = [row for row in scored_rows if row["config_id"] == default_config_id]
    current_default_row = default_rows[0] if default_rows else scored_rows[0]
    current_default_config = config_from_row(current_default_row)
    top_rows = scored_rows[: max(1, int(top_k))]

    trainval_indices = np.sort(np.concatenate([split.train_indices, split.validation_indices]))
    selected_specs = [
        {
            "selection_name": "best_overall_validation",
            "selection_rule": "highest composite validation score across all searched pseudotime modes",
            "row": scored_rows[0],
            "config": best_config,
        },
        {
            "selection_name": "best_tau_free_validation",
            "selection_rule": "highest composite validation score among tau-free pseudotime configurations",
            "row": best_tau_free_row,
            "config": best_tau_free_config,
        },
        {
            "selection_name": "current_default_reference",
            "selection_rule": "pre-search BN-LTE default: tau-free pseudotime, 6 parents, 4 knots, cubic splines, balanced ridge grid, 5-fold CV",
            "row": current_default_row,
            "config": current_default_config,
        },
    ]
    selected_metrics_rows: list[dict[str, Any]] = []
    selected_reports: dict[str, dict[str, Any]] = {}
    edge_summary_by_selection: dict[str, list[dict[str, Any]]] = {}
    for spec in selected_specs:
        config = spec["config"]
        selection_prefix = {"selection_name": spec["selection_name"], "selection_rule": spec["selection_rule"]}
        train_fit = fit_config(dataset, split.train_indices, selected_target_names, selected_target_indices, config)
        validation_metrics = evaluate_prediction(
            dataset=dataset,
            selected_regions=selected_regions,
            selected_target_indices=selected_target_indices,
            train_indices=split.train_indices,
            eval_indices=split.validation_indices,
            prediction=train_fit.predictions,
            z_values=train_fit.z_values,
        )
        test_metrics_train_only = evaluate_prediction(
            dataset=dataset,
            selected_regions=selected_regions,
            selected_target_indices=selected_target_indices,
            train_indices=split.train_indices,
            eval_indices=split.test_indices,
            prediction=train_fit.predictions,
            z_values=train_fit.z_values,
        )
        trainval_fit = fit_config(dataset, trainval_indices, selected_target_names, selected_target_indices, config)
        test_metrics_refit = evaluate_prediction(
            dataset=dataset,
            selected_regions=selected_regions,
            selected_target_indices=selected_target_indices,
            train_indices=trainval_indices,
            eval_indices=split.test_indices,
            prediction=trainval_fit.predictions,
            z_values=trainval_fit.z_values,
        )
        edge_summary_by_selection[str(spec["selection_name"])] = summarize_edges(trainval_fit.fit)
        selected_metrics_rows.extend(
            [
                {**selection_prefix, "split": "validation", "fit_scope": "train_only", **config_row(config), **validation_metrics},
                {**selection_prefix, "split": "test", "fit_scope": "train_only", **config_row(config), **test_metrics_train_only},
                {**selection_prefix, "split": "test", "fit_scope": "train_plus_validation_refit", **config_row(config), **test_metrics_refit},
            ]
        )
        selected_reports[str(spec["selection_name"])] = {
            "selection_rule": str(spec["selection_rule"]),
            "validation_row": dict(spec["row"]),
            "validation_metrics": validation_metrics,
            "test_metrics_train_only": test_metrics_train_only,
            "test_metrics_train_plus_validation_refit": test_metrics_refit,
        }

    write_csv(out / "validation_search_results.csv", scored_rows)
    write_csv(out / "top_validation_configs.csv", top_rows)
    write_csv(out / "fit_errors.csv", fit_errors)
    write_csv(out / "selected_config_metrics.csv", selected_metrics_rows)
    edge_rows = []
    for selection_name, rows_for_selection in edge_summary_by_selection.items():
        edge_rows.extend({"selection_name": selection_name, **row} for row in rows_for_selection)
    write_csv(out / "selected_edge_summary.csv", edge_rows)

    figures = {
        "validation_scoreboard": fig_dir / "bnlte_hyperparameter_validation_scoreboard.png",
        "metric_tradeoff": fig_dir / "bnlte_hyperparameter_tradeoff.png",
        "selected_test_profile": fig_dir / "bnlte_selected_test_profile.png",
    }
    plot_validation_scoreboard(figures["validation_scoreboard"], top_rows)
    plot_metric_tradeoff(figures["metric_tradeoff"], scored_rows)
    overall = selected_reports["best_overall_validation"]
    plot_selected_test_profile(
        figures["selected_test_profile"],
        overall["validation_metrics"],
        overall["test_metrics_train_only"],
        overall["test_metrics_train_plus_validation_refit"],
    )

    report = {
        "purpose": "Validation-only BN-LTE hyperparameter search; held-out test evaluated once for selected configuration.",
        "configuration": {
            "random_seed": int(random_seed),
            "grid_size": len(configs),
            "successful_configs": len(scored_rows),
            "failed_configs": len(fit_errors),
            "selection_metrics": selection_metric_specs(),
            "split": split.report(),
            "selected_regions": selected_regions,
            "target_names": selected_target_names,
        },
        "best_validation_config": top_rows[0],
        "best_tau_free_validation_config": best_tau_free_row,
        "current_default_validation_config": current_default_row,
        "selected_configs": selected_reports,
        "selected_validation_metrics": selected_reports["best_overall_validation"]["validation_metrics"],
        "selected_test_metrics": selected_reports["best_overall_validation"]["test_metrics_train_only"],
        "selected_test_metrics_train_plus_validation_refit": selected_reports["best_overall_validation"]["test_metrics_train_plus_validation_refit"],
        "tau_free_validation_metrics": selected_reports["best_tau_free_validation"]["validation_metrics"],
        "tau_free_test_metrics": selected_reports["best_tau_free_validation"]["test_metrics_train_only"],
        "tau_free_test_metrics_train_plus_validation_refit": selected_reports["best_tau_free_validation"]["test_metrics_train_plus_validation_refit"],
        "current_default_validation_metrics": selected_reports["current_default_reference"]["validation_metrics"],
        "current_default_test_metrics": selected_reports["current_default_reference"]["test_metrics_train_only"],
        "current_default_test_metrics_train_plus_validation_refit": selected_reports["current_default_reference"]["test_metrics_train_plus_validation_refit"],
        "selected_edge_summary_top": edge_rows[:20],
        "tables": {
            "validation_search_results": str(out / "validation_search_results.csv"),
            "top_validation_configs": str(out / "top_validation_configs.csv"),
            "fit_errors": str(out / "fit_errors.csv"),
            "selected_config_metrics": str(out / "selected_config_metrics.csv"),
            "selected_edge_summary": str(out / "selected_edge_summary.csv"),
        },
        "figures": {key: str(value) for key, value in figures.items()},
        "guardrails": [
            "The validation split, not the test split, selects the configuration.",
            "The train_plus_validation_refit test result is a standard final-refit estimate, but it should be labeled separately from the train-only comparison.",
            "Configurations with tau-inclusive pseudotime may improve forecasting but should be described as forecasting-tuned, not as the cleanest causal pseudotime.",
        ],
    }
    (out / "bnlte_hyperparameter_search_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    (out / "bnlte_hyperparameter_search_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def default_search_grid() -> list[SearchConfig]:
    modes = ["tau_free", "pt217_free", "clinical_free"]
    coverages = [0.40, 0.50, 0.55]
    parent_counts = [3, 6, 10]
    spline_shapes = [(1, 0), (3, 1), (4, 2), (4, 3), (5, 3)]
    ridge_profiles = ["balanced", "flexible"]
    cv_folds_values = [3, 5]
    configs = []
    for mode in modes:
        for coverage in coverages:
            for parents in parent_counts:
                for n_knots, degree in spline_shapes:
                    for ridge_profile in ridge_profiles:
                        for cv_folds in cv_folds_values:
                            config_id = f"{mode}_cov{coverage:.2f}_p{parents}_k{n_knots}_d{degree}_{ridge_profile}_cv{cv_folds}"
                            configs.append(
                                SearchConfig(
                                    config_id=config_id,
                                    pseudotime_mode=mode,
                                    min_train_coverage=coverage,
                                    max_parents=parents,
                                    n_knots=n_knots,
                                    spline_degree=degree,
                                    ridge_profile=ridge_profile,
                                    cv_folds=cv_folds,
                                )
                            )
    return configs


def fit_config(
    dataset: MultimodalPairDataset,
    train_indices: np.ndarray,
    selected_target_names: list[str],
    selected_target_indices: list[int],
    config: SearchConfig,
) -> FittedConfig:
    pseudotime = fit_pseudotime(
        dataset.feature_matrix,
        dataset.feature_names,
        train_indices,
        mode=config.pseudotime_mode,
        min_train_coverage=config.min_train_coverage,
    )
    fit = fit_dynamic_scm(
        dataset,
        pseudotime,
        train_indices,
        target_names=selected_target_names,
        max_parents_per_target=config.max_parents,
        n_knots=config.n_knots,
        spline_degree=config.spline_degree,
        ridge_alphas=config.ridge_alphas,
        cv_folds=config.cv_folds,
    )
    rates = fit.predict_rates(dataset)[:, selected_target_indices]
    prediction = dataset.target_baseline[:, selected_target_indices] + dataset.time_years[:, None] * rates
    return FittedConfig(config=config, pseudotime=pseudotime, fit=fit, predictions=prediction, z_values=pseudotime.transform(dataset.feature_matrix))


def evaluate_prediction(
    *,
    dataset: MultimodalPairDataset,
    selected_regions: list[str],
    selected_target_indices: list[int],
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    prediction: np.ndarray,
    z_values: np.ndarray,
) -> dict[str, float | int]:
    baseline = dataset.target_baseline[:, selected_target_indices]
    observed = dataset.target_observed[:, selected_target_indices]
    dt = dataset.time_years
    idx = np.asarray(eval_indices, dtype=int)
    train = np.asarray(train_indices, dtype=int)
    empirical_s1 = np.nanmean(observed[idx], axis=0)
    predicted_s1 = np.nanmean(prediction[idx], axis=0)
    empirical_delta = np.nanmean(observed[idx] - baseline[idx], axis=0)
    predicted_delta = np.nanmean(prediction[idx] - baseline[idx], axis=0)

    pair_mae = []
    pair_rate_mae = []
    pair_delta_spearman = []
    pair_subject_spearman = []
    for row_idx in idx:
        y = observed[row_idx]
        pred = prediction[row_idx]
        base = baseline[row_idx]
        pair_mae.append(finite_mean_abs(pred - y))
        pair_rate_mae.append(finite_mean_abs(((pred - base) - (y - base)) / float(dt[row_idx])))
        pair_delta_spearman.append(safe_correlation(y - base, pred - base, rank=True))
        pair_subject_spearman.append(safe_correlation(y, pred, rank=True))

    observed_rate = (observed - baseline) / dt[:, None]
    predicted_rate = (prediction - baseline) / dt[:, None]
    empirical_score = np.nanmean(observed_rate, axis=1)
    predicted_score = np.nanmean(predicted_rate, axis=1)
    fast_threshold = float(np.nanquantile(empirical_score[train], 0.75))
    y_fast = empirical_score[idx] >= fast_threshold
    fast_score = predicted_score[idx]

    braak = braak_metrics(selected_regions, baseline, observed, prediction, idx)
    edge_proxy = float("nan")
    if z_values.size:
        edge_proxy = float(np.nanstd(z_values[idx]))
    return {
        "n_pairs": int(idx.size),
        "pair_median_mae_suvr": finite_median(pair_mae),
        "pair_median_rate_mae": finite_median(pair_rate_mae),
        "pair_median_delta_spearman": finite_median(pair_delta_spearman),
        "pair_median_subject_spearman": finite_median(pair_subject_spearman),
        "group_map_mae_s1": finite_mean_abs(predicted_s1 - empirical_s1),
        "group_map_rmse_s1": finite_rmse(predicted_s1 - empirical_s1),
        "s1_map_spearman": safe_correlation(empirical_s1, predicted_s1, rank=True),
        "delta_map_spearman": safe_correlation(empirical_delta, predicted_delta, rank=True),
        "delta_map_pearson": safe_correlation(empirical_delta, predicted_delta, rank=False),
        "delta_cosine": cosine_similarity(empirical_delta, predicted_delta),
        "direction_accuracy": direction_accuracy(empirical_delta, predicted_delta),
        "top3_overlap": topk_overlap(empirical_delta, predicted_delta, 3),
        "weighted_top3_capture": weighted_topk_capture(empirical_delta, predicted_delta, 3),
        "fast_progressor_threshold_train_q75": fast_threshold,
        "fast_progressor_fraction": float(np.mean(y_fast)) if y_fast.size else float("nan"),
        "fast_progressor_auroc": safe_auroc(y_fast, fast_score),
        "fast_progressor_auprc": safe_auprc(y_fast, fast_score),
        "fast_progressor_top_decile_precision": top_fraction_precision(y_fast, fast_score, 0.10),
        "braak_group_spearman": braak["braak_group_spearman"],
        "braak_group_mae": braak["braak_group_mae"],
        "z_eval_std": edge_proxy,
    }


def attach_selection_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = selection_metric_specs()
    scored = [dict(row) for row in rows]
    for spec in specs:
        key = spec["key"]
        values = np.asarray([float(row.get(key, float("nan"))) for row in scored], dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            for row in scored:
                row[f"score_component:{key}"] = float("nan")
            continue
        lo = float(np.nanmin(values[finite]))
        hi = float(np.nanmax(values[finite]))
        denom = hi - lo
        for row, value in zip(scored, values, strict=True):
            if not np.isfinite(value):
                component = 0.0
            elif denom <= 1.0e-12:
                component = 1.0
            elif spec["direction"] == "min":
                component = (hi - value) / denom
            else:
                component = (value - lo) / denom
            row[f"score_component:{key}"] = float(np.clip(component, 0.0, 1.0))
    for row in scored:
        numerator = 0.0
        denom = 0.0
        for spec in specs:
            component = float(row.get(f"score_component:{spec['key']}", 0.0))
            weight = float(spec["weight"])
            if np.isfinite(component):
                numerator += weight * component
                denom += weight
        row["selection_score"] = float(numerator / denom) if denom > 0.0 else float("nan")
    return scored


def selection_metric_specs() -> list[dict[str, str | float]]:
    return [
        {"key": "group_map_mae_s1", "direction": "min", "weight": 0.22},
        {"key": "delta_map_spearman", "direction": "max", "weight": 0.20},
        {"key": "delta_cosine", "direction": "max", "weight": 0.14},
        {"key": "weighted_top3_capture", "direction": "max", "weight": 0.12},
        {"key": "fast_progressor_auroc", "direction": "max", "weight": 0.14},
        {"key": "braak_group_spearman", "direction": "max", "weight": 0.10},
        {"key": "pair_median_delta_spearman", "direction": "max", "weight": 0.08},
    ]


def braak_metrics(
    selected_regions: list[str],
    baseline: np.ndarray,
    observed: np.ndarray,
    prediction: np.ndarray,
    eval_indices: np.ndarray,
) -> dict[str, float]:
    region_to_idx = {region: idx for idx, region in enumerate(selected_regions)}
    empirical = []
    predicted = []
    for group_regions in BRAAK_GROUPS.values():
        idxs = [region_to_idx[region] for region in group_regions if region in region_to_idx]
        if not idxs:
            empirical.append(float("nan"))
            predicted.append(float("nan"))
            continue
        empirical.append(float(np.nanmean(observed[eval_indices][:, idxs] - baseline[eval_indices][:, idxs])))
        predicted.append(float(np.nanmean(prediction[eval_indices][:, idxs] - baseline[eval_indices][:, idxs])))
    empirical_arr = np.asarray(empirical, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    return {
        "braak_group_spearman": safe_correlation(empirical_arr, predicted_arr, rank=True),
        "braak_group_mae": finite_mean_abs(predicted_arr - empirical_arr),
    }


def summarize_edges(fit: DynamicSCMFit) -> list[dict[str, Any]]:
    rows = fit.edge_effect_rows()
    output = []
    for row in rows:
        output.append(
            {
                "parent": row["parent"],
                "target": row["target"],
                "max_abs_effect": row["max_abs_effect"],
                "mean_abs_effect": row["mean_abs_effect"],
                "z_at_max_abs_effect": row["z_at_max_abs_effect"],
                "included_by_effect_threshold": row["included_by_effect_threshold"],
            }
        )
    return output


def config_row(config: SearchConfig) -> dict[str, Any]:
    row = asdict(config)
    row["ridge_alphas"] = ";".join(f"{value:g}" for value in config.ridge_alphas)
    return row


def config_from_row(row: dict[str, Any]) -> SearchConfig:
    return SearchConfig(
        config_id=str(row["config_id"]),
        pseudotime_mode=str(row["pseudotime_mode"]),
        min_train_coverage=float(row["min_train_coverage"]),
        max_parents=int(row["max_parents"]),
        n_knots=int(row["n_knots"]),
        spline_degree=int(row["spline_degree"]),
        ridge_profile=str(row["ridge_profile"]),
        cv_folds=int(row["cv_folds"]),
    )


def plot_validation_scoreboard(path: Path, rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("selection_score", "Score"),
        ("group_map_mae_s1", "Group MAE"),
        ("delta_map_spearman", "Delta rho"),
        ("fast_progressor_auroc", "AUROC"),
        ("braak_group_spearman", "Braak rho"),
    ]
    labels = [short_config(row) for row in rows]
    values = np.asarray([[float(row.get(key, float("nan"))) for key, _ in metrics] for row in rows], dtype=float)
    normalized = values.copy()
    for col, (key, _) in enumerate(metrics):
        col_values = values[:, col]
        finite = np.isfinite(col_values)
        if not np.any(finite):
            normalized[:, col] = np.nan
            continue
        lo = np.nanmin(col_values[finite])
        hi = np.nanmax(col_values[finite])
        if hi - lo <= 1.0e-12:
            normalized[:, col] = 1.0
        elif key == "group_map_mae_s1":
            normalized[:, col] = (hi - col_values) / (hi - lo)
        else:
            normalized[:, col] = (col_values - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(10.4, max(4.2, 0.34 * len(rows) + 1.7)), constrained_layout=True)
    image = ax.imshow(normalized, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metrics)), [label for _, label in metrics], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            text = "NA" if not np.isfinite(values[y, x]) else f"{values[y, x]:.3f}" if metrics[x][0] != "group_map_mae_s1" else f"{values[y, x]:.4f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color="white" if normalized[y, x] < 0.55 else "#111827")
    ax.set_title("Top validation-selected BN-LTE configurations", loc="left", fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, shrink=0.75)
    cbar.set_label("Column-normalized desirability")
    save_figure(fig, path)


def plot_metric_tradeoff(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    modes = sorted({str(row["pseudotime_mode"]) for row in rows})
    colors = dict(zip(modes, ["#D55E00", "#0072B2", "#009E73", "#CC79A7"], strict=False))
    for mode in modes:
        sub = [row for row in rows if row["pseudotime_mode"] == mode]
        ax.scatter(
            [float(row["group_map_mae_s1"]) for row in sub],
            [float(row["delta_map_spearman"]) for row in sub],
            s=[36.0 + 105.0 * float(row["selection_score"]) for row in sub],
            alpha=0.70,
            label=mode,
            color=colors.get(mode, "#666666"),
            edgecolor="white",
            linewidth=0.6,
        )
    best = max(rows, key=lambda row: float(row["selection_score"]))
    ax.scatter([float(best["group_map_mae_s1"])], [float(best["delta_map_spearman"])], marker="*", s=260, color="#111827", label="selected")
    ax.set_xlabel("Validation group-map MAE, lower is better")
    ax.set_ylabel("Validation delta-map Spearman, higher is better")
    ax.set_title("Forecast accuracy versus spatial progression ordering", loc="left", fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, path)


def plot_selected_test_profile(path: Path, validation: dict[str, Any], test_train: dict[str, Any], test_refit: dict[str, Any]) -> None:
    labels = ["validation\ntrain-only", "test\ntrain-only", "test\ntrain+val"]
    metrics = [
        ("group_map_mae_s1", "Group MAE", "min"),
        ("delta_map_spearman", "Delta rho", "max"),
        ("fast_progressor_auroc", "AUROC", "max"),
        ("braak_group_spearman", "Braak rho", "max"),
    ]
    rows = [validation, test_train, test_refit]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.8, 3.2), constrained_layout=True)
    for ax, (key, label, direction) in zip(axes, metrics, strict=True):
        values = [float(row.get(key, float("nan"))) for row in rows]
        color = "#D55E00" if direction == "max" else "#0072B2"
        ax.plot(labels, values, marker="o", color=color, linewidth=2.0)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Selected BN-LTE configuration: validation and frozen test profile", fontsize=12, fontweight="bold")
    save_figure(fig, path)


def render_markdown(report: dict[str, Any]) -> str:
    best = report["best_validation_config"]
    test = report["selected_test_metrics"]
    refit = report["selected_test_metrics_train_plus_validation_refit"]
    tau_free = report["best_tau_free_validation_config"]
    tau_test = report["tau_free_test_metrics"]
    tau_refit = report["tau_free_test_metrics_train_plus_validation_refit"]
    default = report["current_default_validation_config"]
    default_test = report["current_default_test_metrics"]
    default_refit = report["current_default_test_metrics_train_plus_validation_refit"]
    lines = [
        "# BN-LTE Hyperparameter Search",
        "",
        "## Selection Rule",
        "",
        "Configurations were selected using validation subjects only. The held-out test set was evaluated once after selection.",
        "",
        "## Best Validation Configuration",
        "",
        f"- `config_id`: `{best['config_id']}`",
        f"- pseudotime mode: `{best['pseudotime_mode']}`",
        f"- minimum feature coverage: `{best['min_train_coverage']}`",
        f"- max parents per target: `{best['max_parents']}`",
        f"- spline knots / degree: `{best['n_knots']}` / `{best['spline_degree']}`",
        f"- ridge profile: `{best['ridge_profile']}`",
        f"- validation selection score: `{best['selection_score']:.4f}`",
        "",
        "## Held-Out Test Performance",
        "",
        f"- train-only test group-map MAE: `{test['group_map_mae_s1']:.6f}`",
        f"- train-only test delta-map Spearman: `{test['delta_map_spearman']:.4f}`",
        f"- train-only test fast-progressor AUROC: `{test['fast_progressor_auroc']:.4f}`",
        f"- train-only test Braak Spearman: `{test['braak_group_spearman']:.4f}`",
        "",
        "## Train+Validation Final Refit Performance",
        "",
        f"- refit test group-map MAE: `{refit['group_map_mae_s1']:.6f}`",
        f"- refit test delta-map Spearman: `{refit['delta_map_spearman']:.4f}`",
        f"- refit test fast-progressor AUROC: `{refit['fast_progressor_auroc']:.4f}`",
        f"- refit test Braak Spearman: `{refit['braak_group_spearman']:.4f}`",
        "",
        "## Best Tau-Free Causal Configuration",
        "",
        f"- `config_id`: `{tau_free['config_id']}`",
        f"- validation selection score: `{tau_free['selection_score']:.4f}`",
        f"- train-only test group-map MAE: `{tau_test['group_map_mae_s1']:.6f}`",
        f"- train-only test delta-map Spearman: `{tau_test['delta_map_spearman']:.4f}`",
        f"- train-only test fast-progressor AUROC: `{tau_test['fast_progressor_auroc']:.4f}`",
        f"- train-only test Braak Spearman: `{tau_test['braak_group_spearman']:.4f}`",
        f"- refit test group-map MAE: `{tau_refit['group_map_mae_s1']:.6f}`",
        f"- refit test delta-map Spearman: `{tau_refit['delta_map_spearman']:.4f}`",
        f"- refit test fast-progressor AUROC: `{tau_refit['fast_progressor_auroc']:.4f}`",
        f"- refit test Braak Spearman: `{tau_refit['braak_group_spearman']:.4f}`",
        "",
        "## Current Default Reference",
        "",
        f"- `config_id`: `{default['config_id']}`",
        f"- validation selection score: `{default['selection_score']:.4f}`",
        f"- train-only test group-map MAE: `{default_test['group_map_mae_s1']:.6f}`",
        f"- train-only test delta-map Spearman: `{default_test['delta_map_spearman']:.4f}`",
        f"- train-only test fast-progressor AUROC: `{default_test['fast_progressor_auroc']:.4f}`",
        f"- train-only test Braak Spearman: `{default_test['braak_group_spearman']:.4f}`",
        f"- refit test group-map MAE: `{default_refit['group_map_mae_s1']:.6f}`",
        f"- refit test delta-map Spearman: `{default_refit['delta_map_spearman']:.4f}`",
        f"- refit test fast-progressor AUROC: `{default_refit['fast_progressor_auroc']:.4f}`",
        f"- refit test Braak Spearman: `{default_refit['braak_group_spearman']:.4f}`",
        "",
        "## Outputs",
        "",
    ]
    for key, value in report["tables"].items():
        lines.append(f"- {key}: `{value}`")
    for key, value in report["figures"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Guardrails", ""])
    lines.extend(f"- {item}" for item in report["guardrails"])
    return "\n".join(lines)


def finite_mean_abs(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(np.abs(arr))) if arr.size else float("nan")


def finite_rmse(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def finite_median(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def safe_correlation(a: Any, b: Any, *, rank: bool) -> float:
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


def cosine_similarity(a: Any, b: Any) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) == 0:
        return float("nan")
    x = x[mask]
    y = y[mask]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1.0e-12 else float("nan")


def direction_accuracy(a: Any, b: Any) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1.0e-12)
    if int(np.sum(mask)) == 0:
        return float("nan")
    return float(np.mean(np.sign(x[mask]) == np.sign(y[mask])))


def topk_overlap(a: Any, b: Any, k: int) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < int(k):
        return float("nan")
    x = x[mask]
    y = y[mask]
    kk = int(k)
    observed = set(np.argsort(-x, kind="mergesort")[:kk])
    predicted = set(np.argsort(-y, kind="mergesort")[:kk])
    return float(len(observed & predicted) / kk) if kk else float("nan")


def weighted_topk_capture(a: Any, b: Any, k: int) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < int(k):
        return float("nan")
    x = x[mask]
    y = y[mask]
    kk = int(k)
    empirical_positive = np.maximum(x, 0.0)
    denom = float(np.sum(np.sort(empirical_positive)[-kk:]))
    if denom <= 1.0e-12:
        return float("nan")
    predicted_top = np.argsort(-y, kind="mergesort")[:kk]
    return float(np.sum(empirical_positive[predicted_top]) / denom)


def safe_auroc(y_true: Any, score: Any) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        y = np.asarray(y_true, dtype=bool)
        s = np.asarray(score, dtype=float)
        mask = np.isfinite(s)
        if int(np.sum(mask)) < 3 or np.unique(y[mask]).size < 2:
            return float("nan")
        return float(roc_auc_score(y[mask], s[mask]))
    except Exception:
        return float("nan")


def safe_auprc(y_true: Any, score: Any) -> float:
    try:
        from sklearn.metrics import average_precision_score

        y = np.asarray(y_true, dtype=bool)
        s = np.asarray(score, dtype=float)
        mask = np.isfinite(s)
        if int(np.sum(mask)) < 3 or np.unique(y[mask]).size < 2:
            return float("nan")
        return float(average_precision_score(y[mask], s[mask]))
    except Exception:
        return float("nan")


def top_fraction_precision(y_true: Any, score: Any, fraction: float) -> float:
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    if int(np.sum(mask)) == 0:
        return float("nan")
    y = y[mask]
    s = s[mask]
    k = max(1, int(math.ceil(y.size * float(fraction))))
    order = np.argsort(-s, kind="mergesort")[:k]
    return float(np.mean(y[order]))


def short_config(row: dict[str, Any]) -> str:
    return f"{row['pseudotime_mode']} p{row['max_parents']} k{row['n_knots']} d{row['spline_degree']} {row['ridge_profile'][:4]}"


def resolve_path(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "config_id",
        "split",
        "fit_scope",
        "selection_score",
        "pseudotime_mode",
        "min_train_coverage",
        "max_parents",
        "n_knots",
        "spline_degree",
        "ridge_profile",
        "cv_folds",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=260, bbox_inches="tight")
    svg_path = path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


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
