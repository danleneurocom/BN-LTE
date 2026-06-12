#!/usr/bin/env python3
"""Run the ADNI baseline comparison table with shared splits and metrics.

This script uses the repo's generated ADNI forecast pairs, ENIGMA graph, and
BN-LTE metric conventions. Heavy neural baselines are short-epoch sklearn
proxies so they can be run in the local environment without torch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import GraphicalLasso
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from bayesian_network_scm.data import MultimodalPairDataset, build_multimodal_pair_dataset  # noqa: E402
from bayesian_network_scm.dynamic_scm import fit_dynamic_scm  # noqa: E402
from bayesian_network_scm.pseudotime import fit_pseudotime  # noqa: E402
from bayesian_network_scm.reporting import make_subject_split  # noqa: E402
from machine_learning_baselines.run_ml_baselines import (  # noqa: E402
    RobustPreprocessor,
    TrainContext,
    design_matrix,
    fit_ml_predictions,
    mlp_predict_einsum,
    reconstruct_followup,
    train_rate_rmse,
)
from run_extended_paper_experiments import (  # noqa: E402
    BRAAK_GROUPS,
    cosine_similarity,
    weighted_topk_capture,
)
from run_paper_validation_experiments import (  # noqa: E402
    finite_mean_abs,
    fit_all_prediction_models,
    load_graph_resources,
    parameter_bounds,
    safe_correlation,
    validate_dataset,
    validate_predictions,
    validate_split,
)
from spread_toolbox.models.ndm import NetworkDiffusionModel  # noqa: E402


RANDOM_SEED = 20260521
MODEL_ORDER = [
    "ML-Prognostic Index",
    "Karlsson Tau-PET ML",
    "AdaBoost Tau-Rate",
    "MLP-Lite",
    "DeepMTL-MLP",
    "DAE-Prognostic",
    "ResidualDeepEnsemble",
    "NComms2025 Fusion MLP",
    "DyEPAD Dynamic Graph",
    "GCN-XAI Population Graph",
    "JAD GraphLASSO Tau Topology",
    "HPBN Prototype Brain-Net",
    "NDM+",
    "ESM+",
    "SIR+",
    "Bayesian NDM",
    "Probabilistic Stage",
    "BN-LTE",
    "BN-LTE + PCA-Z",
    "BN-LTE + ATN-Z tau-free",
]
MODEL_TYPES = {
    "ML-Prognostic Index": "ML",
    "Karlsson Tau-PET ML": "ML",
    "AdaBoost Tau-Rate": "ML",
    "MLP-Lite": "Neural/MLP",
    "DeepMTL-MLP": "Deep learning",
    "DAE-Prognostic": "Deep learning",
    "ResidualDeepEnsemble": "Deep learning",
    "NComms2025 Fusion MLP": "Deep learning",
    "DyEPAD Dynamic Graph": "Graph learning",
    "GCN-XAI Population Graph": "Graph learning",
    "JAD GraphLASSO Tau Topology": "Graph topology",
    "HPBN Prototype Brain-Net": "Prototype/graph",
    "NDM+": "Biophysical",
    "ESM+": "Biophysical",
    "SIR+": "Biophysical",
    "Bayesian NDM": "Bayesian/biophysical",
    "Probabilistic Stage": "Bayesian/staging",
    "BN-LTE": "Structural",
    "BN-LTE + PCA-Z": "Structural/Z",
    "BN-LTE + ATN-Z tau-free": "Structural/Z",
}
METRIC_COLUMNS = ["mae", "rho_delta", "cosine", "top3", "braak_rho", "braak_mae"]
DISPLAY_COLUMNS = {
    "mae": "MAE",
    "rho_delta": "rho_delta",
    "cosine": "Cos.",
    "top3": "Top-3",
    "braak_rho": "Braak rho",
    "braak_mae": "B-MAE",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "outputs" / "adni_baseline_table")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--deep-epochs", type=int, default=80)
    parser.add_argument("--bootstrap-samples", type=int, default=5)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--max-parents", type=int, default=5)
    args = parser.parse_args()

    report = run_adni_baseline_table(
        project_root=args.project_root,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        repeats=args.repeats,
        deep_epochs=args.deep_epochs,
        bootstrap_samples=args.bootstrap_samples,
        ensemble_size=args.ensemble_size,
        max_parents=args.max_parents,
    )
    print("ADNI baseline table complete.")
    print(f"Formatted table: {report['tables']['formatted_markdown']}")
    print(f"Summary CSV: {report['tables']['summary_csv']}")
    print(f"Report: {report['report_path']}")
    return 0


def run_adni_baseline_table(
    *,
    project_root: str | Path,
    output_dir: str | Path,
    random_seed: int,
    repeats: int,
    deep_epochs: int,
    bootstrap_samples: int,
    ensemble_size: int,
    max_parents: int,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    out = resolve_path(output_dir, root)
    out.mkdir(parents=True, exist_ok=True)

    dataset = build_multimodal_pair_dataset(root)
    selected_regions = list(dataset.report["selected_tau_regions"])
    target_names = [f"tau_rate:{region}" for region in selected_regions]
    target_indices = [dataset.target_index(name) for name in target_names]
    validate_dataset(dataset, target_indices)
    graph = load_graph_resources(root, dataset, selected_regions)

    split_rows: list[dict[str, Any]] = []
    fit_reports: dict[str, Any] = {}
    split_seeds = [int(random_seed) + 1009 * idx for idx in range(int(repeats))]
    for repeat_index, seed in enumerate(split_seeds):
        print(f"Split {repeat_index + 1}/{len(split_seeds)} seed={seed}")
        split = make_subject_split(dataset.metadata_rows, random_seed=seed)
        validate_split(split)
        context = make_train_context(dataset, split.train_indices, selected_regions, target_names, target_indices)

        fitted = fit_all_prediction_models(
            dataset=dataset,
            graph=graph,
            split=split,
            selected_regions=selected_regions,
            selected_target_names=target_names,
            selected_target_indices=target_indices,
            max_parents=max_parents,
        )
        predictions = rename_mechanistic_predictions(fitted["predictions"])
        reports = rename_mechanistic_reports(fitted["fit_reports"])

        ml_predictions, ml_reports = fit_ml_predictions(context, random_seed=seed)
        predictions.update(ml_predictions)
        reports.update(ml_reports)

        recent_predictions, recent_reports = fit_recent_short_epoch_baselines(
            context=context,
            graph=graph,
            split=split,
            fitted=fitted,
            target_names=target_names,
            target_indices=target_indices,
            max_parents=max_parents,
            deep_epochs=deep_epochs,
            bootstrap_samples=bootstrap_samples,
            ensemble_size=ensemble_size,
            random_seed=seed,
        )
        predictions.update(recent_predictions)
        reports.update(recent_reports)

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
        "split_metrics_csv": out / "adni_baseline_table_split_metrics.csv",
        "summary_csv": out / "adni_baseline_table_summary.csv",
        "formatted_csv": out / "adni_baseline_table_formatted.csv",
        "formatted_markdown": out / "adni_baseline_table_formatted.md",
        "latex": out / "adni_baseline_table.tex",
    }
    write_csv_rows(table_paths["split_metrics_csv"], split_rows)
    write_csv_rows(table_paths["summary_csv"], summary_rows)
    write_csv_rows(table_paths["formatted_csv"], formatted_rows)
    table_paths["formatted_markdown"].write_text(render_markdown_table(formatted_rows), encoding="utf-8")
    table_paths["latex"].write_text(render_latex_table(formatted_rows), encoding="utf-8")

    report = {
        "purpose": "ADNI-only baseline comparison table with shared subject splits and metrics.",
        "configuration": {
            "random_seed": int(random_seed),
            "repeats": int(repeats),
            "split_seeds": split_seeds,
            "deep_epochs": int(deep_epochs),
            "bootstrap_samples": int(bootstrap_samples),
            "ensemble_size": int(ensemble_size),
            "max_parents": int(max_parents),
        },
        "data": {
            "pairs": int(dataset.pair_count),
            "features": int(len(dataset.feature_names)),
            "targets": int(len(target_indices)),
            "selected_regions": selected_regions,
        },
        "metric_notes": [
            "MAE and B-MAE are SUVR errors multiplied by 100 to match the paper-table scale.",
            "MAE, rho_delta, Cos., and Top-3 are computed on held-out group-average regional maps.",
            "rho_delta is the Spearman correlation between observed and predicted group-average regional tau deltas.",
            "Cos. is the cosine similarity of group-average regional tau deltas.",
            "Top-3 is the weighted capture of empirically largest three group-average regional tau increases.",
            "Braak rho and B-MAE use the four approximate Braak-like groups available from the selected 10 tau regions.",
            "The deep-learning rows are short-epoch sklearn neural proxies because torch is not installed in this environment.",
            "Rows named after recent papers are endpoint adapters: all are retrained on the same ADNI tau-rate task and scored with identical held-out metrics, but classifier/topology papers are not their original published endpoints.",
            "Karlsson uses PET-free clinical/plasma/MRI features with gradient boosting; ncomms2025 uses PET-free multimodal fusion with a residual MLP head; DyEPAD and GCN-XAI use train-only patient-similarity graph smoothers; JAD uses graphical-lasso tau topology; HPBN uses soft prototype matching because FCN/SCN matrices are not present in this ADNI tau-rate dataset.",
        ],
        "model_order": MODEL_ORDER,
        "model_types": MODEL_TYPES,
        "fit_reports": fit_reports,
        "tables": {key: str(path) for key, path in table_paths.items()},
    }
    report_path = out / "adni_baseline_table_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def make_train_context(
    dataset: MultimodalPairDataset,
    train_indices: np.ndarray,
    selected_regions: list[str],
    target_names: list[str],
    target_indices: list[int],
) -> TrainContext:
    train = np.asarray(train_indices, dtype=int)
    train_values = np.concatenate(
        [dataset.target_baseline[train][:, target_indices], dataset.target_observed[train][:, target_indices]],
        axis=0,
    )
    return TrainContext(
        dataset=dataset,
        selected_regions=selected_regions,
        target_names=target_names,
        target_indices=target_indices,
        train_indices=train,
        train_observed_min=np.nanmin(train_values, axis=0),
        train_observed_max=np.nanmax(train_values, axis=0),
    )


def rename_mechanistic_predictions(predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rename = {
        "BayesianNetwork-SCM": "BN-LTE",
        "NDM": "NDM+",
        "ESM": "ESM+",
        "SIR": "SIR+",
    }
    return {rename[key]: value for key, value in predictions.items() if key in rename}


def rename_mechanistic_reports(reports: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rename = {
        "BayesianNetwork-SCM": "BN-LTE",
        "NDM": "NDM+",
        "ESM": "ESM+",
        "SIR": "SIR+",
    }
    return {rename[key]: value for key, value in reports.items() if key in rename}


def fit_recent_short_epoch_baselines(
    *,
    context: TrainContext,
    graph: dict[str, Any],
    split: Any,
    fitted: dict[str, Any],
    target_names: list[str],
    target_indices: list[int],
    max_parents: int,
    deep_epochs: int,
    bootstrap_samples: int,
    ensemble_size: int,
    random_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    dataset = context.dataset
    x = design_matrix(dataset)
    y = dataset.target_rates[:, target_indices]
    train = np.asarray(split.train_indices, dtype=int)

    predictions: dict[str, np.ndarray] = {}
    reports: dict[str, dict[str, Any]] = {}

    rates, report = fit_deep_mtl_mlp_rates(x, y, train, random_seed=random_seed, max_iter=deep_epochs)
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

    pred, report = fit_bayesian_ndm_prediction(graph, split, target_indices, bootstrap_samples=bootstrap_samples, random_seed=random_seed + 43)
    predictions["Bayesian NDM"] = pred
    reports["Bayesian NDM"] = report

    rates, report = fit_probabilistic_stage_rates(context, fitted["z_values"], random_seed=random_seed + 59)
    predictions["Probabilistic Stage"] = reconstruct_followup(context, rates)
    reports["Probabilistic Stage"] = report

    pred, report = fit_bnlte_variant_prediction(
        dataset,
        split,
        target_names,
        target_indices,
        max_parents=max_parents,
        pseudotime_mode="global",
        pseudotime_label="pca_z_all_features",
    )
    predictions["BN-LTE + PCA-Z"] = pred
    reports["BN-LTE + PCA-Z"] = report

    pred, report = fit_bnlte_variant_prediction(
        dataset,
        split,
        target_names,
        target_indices,
        max_parents=max_parents,
        pseudotime_mode="atn_tau_free",
        pseudotime_label="atn_z_tau_free",
    )
    predictions["BN-LTE + ATN-Z tau-free"] = pred
    reports["BN-LTE + ATN-Z tau-free"] = report

    return predictions, reports


def fit_deep_mtl_mlp_rates(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    random_seed: int,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.all(np.isfinite(y[train]), axis=1)
    if int(np.sum(mask)) < 16:
        raise ValueError("Too few complete training rows for DeepMTL-MLP baseline.")
    preprocessor = RobustPreprocessor.fit(x[train], use_scaling=True)
    x_train = preprocessor.transform(x[train])
    x_all = preprocessor.transform(x)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1.0e-3,
        learning_rate_init=1.0e-3,
        early_stopping=True,
        validation_fraction=0.18,
        n_iter_no_change=patience(max_iter),
        max_iter=int(max_iter),
        random_state=int(random_seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train[mask], y[train][mask])
    rates = mlp_predict_einsum(model, x_all)
    return rates, {
        "family": "short_epoch_multi_task_mlp_rate_regressor",
        "hidden_layer_sizes": [64, 32],
        "max_iter": int(max_iter),
        "iterations": int(getattr(model, "n_iter_", -1)),
        "best_validation_score": float(getattr(model, "best_validation_score_", float("nan"))),
        "train_rate_rmse": train_rate_rmse(y, rates, train),
    }


def fit_dae_prognostic_rates(
    context: TrainContext,
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    random_seed: int,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(random_seed))
    preprocessor = RobustPreprocessor.fit(x[train], use_scaling=True)
    x_train = preprocessor.transform(x[train])
    x_all = preprocessor.transform(x)
    clean = np.repeat(x_train, repeats=3, axis=0)
    noise = rng.normal(loc=0.0, scale=0.10, size=clean.shape)
    dropout = rng.random(clean.shape) < 0.05
    noisy = clean + noise
    noisy[dropout] = 0.0
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(48, 16, 48),
        activation="relu",
        alpha=1.0e-3,
        learning_rate_init=2.0e-3,
        early_stopping=True,
        validation_fraction=0.18,
        n_iter_no_change=patience(max_iter),
        max_iter=int(max_iter),
        random_state=int(random_seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        autoencoder.fit(noisy, clean)
    encoded_all = mlp_hidden_activation(autoencoder, x_all, hidden_layers=2)
    baseline = context.dataset.target_baseline[:, context.target_indices]
    dt = context.dataset.time_years[:, None]
    ridge_features = np.column_stack([encoded_all, baseline, dt, np.log1p(np.clip(dt, 0.0, None))])
    rates = fit_ridge_multi_output_rates(ridge_features, y, train, alpha=1.0)
    reconstruction = autoencoder.predict(x_train)
    train_recon_rmse = float(np.sqrt(np.mean((reconstruction - x_train) ** 2)))
    return rates, {
        "family": "denoising_autoencoder_embedding_plus_ridge_rate_head",
        "hidden_layer_sizes": [48, 16, 48],
        "max_iter": int(max_iter),
        "iterations": int(getattr(autoencoder, "n_iter_", -1)),
        "train_reconstruction_rmse": train_recon_rmse,
        "train_rate_rmse": train_rate_rmse(y, rates, train),
    }


def fit_residual_deep_ensemble_rates(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    random_seed: int,
    max_iter: int,
    ensemble_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.all(np.isfinite(y[train]), axis=1)
    if int(np.sum(mask)) < 16:
        raise ValueError("Too few complete training rows for residual deep ensemble baseline.")
    preprocessor = RobustPreprocessor.fit(x[train], use_scaling=True)
    x_train = preprocessor.transform(x[train])
    x_all = preprocessor.transform(x)
    complete_rows = np.where(mask)[0]
    global_rate = np.nanmean(y[train][mask], axis=0)
    residual = y[train][mask] - global_rate[None, :]
    rng = np.random.default_rng(int(random_seed))
    all_rates = []
    iterations = []
    for member in range(int(ensemble_size)):
        boot = rng.choice(complete_rows, size=complete_rows.size, replace=True)
        model = MLPRegressor(
            hidden_layer_sizes=(48, 24),
            activation="relu",
            alpha=2.0e-3,
            learning_rate_init=1.5e-3,
            early_stopping=True,
            validation_fraction=0.18,
            n_iter_no_change=patience(max_iter),
            max_iter=int(max_iter),
            random_state=int(random_seed) + member,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train[boot], residual[boot])
        all_rates.append(global_rate[None, :] + mlp_predict_einsum(model, x_all))
        iterations.append(int(getattr(model, "n_iter_", -1)))
    rates = np.mean(np.stack(all_rates, axis=0), axis=0)
    return rates, {
        "family": "short_epoch_bootstrap_mlp_residual_ensemble",
        "hidden_layer_sizes": [48, 24],
        "ensemble_size": int(ensemble_size),
        "max_iter": int(max_iter),
        "iterations": iterations,
        "train_rate_rmse": train_rate_rmse(y, rates, train),
    }


def fit_karlsson_taupet_ml_rates(
    context: TrainContext,
    *,
    random_seed: int,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adapt Karlsson et al. tau-PET ML to tau-rate regression.

    The original model compares clinical, plasma, and MRI feature blocks for
    tau-PET load/laterality. This adapter keeps the PET-free low-cost blocks
    and uses histogram gradient boosting as a local CatBoost substitute.
    """

    dataset = context.dataset
    keep = feature_indices_by_family(
        dataset,
        include_clinical=True,
        include_plasma=True,
        include_mri=True,
        include_amyloid_pet=False,
        include_tau_pet=False,
    )
    x = matrix_with_time(dataset.feature_matrix[:, keep], dataset.time_years)
    y = dataset.target_rates[:, context.target_indices]
    train = np.asarray(context.train_indices, dtype=int)
    rates, target_reports = fit_hist_gradient_boosting_multioutput(
        x,
        y,
        train,
        random_seed=random_seed,
        max_iter=max_iter,
        learning_rate=0.045,
        max_leaf_nodes=15,
        l2_regularization=0.05,
    )
    return rates, {
        "family": "karlsson_2025_taupet_ml_adapter",
        "adapter_note": "Original endpoint was tau-PET burden/laterality; this row predicts annualized regional tau rates for shared-table scoring.",
        "feature_policy": "clinical + plasma + MRI + time, excluding amyloid PET and baseline tau PET features",
        "estimator": "HistGradientBoostingRegressor per region, CatBoost-style local substitute",
        "feature_count": int(len(keep) + 2),
        "target_reports": target_reports,
        "train_rate_rmse": train_rate_rmse(y, rates, train),
    }


def fit_ncomms2025_fusion_rates(
    context: TrainContext,
    *,
    random_seed: int,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adapt the ncomms2025 multimodal PET-status model to tau rates."""

    dataset = context.dataset
    keep = feature_indices_by_family(
        dataset,
        include_clinical=True,
        include_plasma=True,
        include_mri=True,
        include_amyloid_pet=False,
        include_tau_pet=False,
    )
    x = matrix_with_missing_indicators(matrix_with_time(dataset.feature_matrix[:, keep], dataset.time_years))
    y = dataset.target_rates[:, context.target_indices]
    train = np.asarray(context.train_indices, dtype=int)
    base_rates = fit_ridge_multi_output_rates(x, y, train, alpha=5.0)
    mask = np.all(np.isfinite(y[train]), axis=1)
    if int(np.sum(mask)) < 16:
        return base_rates, {
            "family": "ncomms2025_fusion_adapter_fallback_ridge",
            "feature_count": int(x.shape[1]),
        }
    preprocessor = RobustPreprocessor.fit(x[train], use_scaling=True)
    x_train = preprocessor.transform(x[train])
    x_all = preprocessor.transform(x)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1.0e-2,
        learning_rate_init=7.5e-4,
        early_stopping=True,
        validation_fraction=0.18,
        n_iter_no_change=patience(max_iter),
        max_iter=int(max_iter),
        random_state=int(random_seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train[mask], y[train][mask] - base_rates[train][mask])
    rates = base_rates + 0.50 * mlp_predict_einsum(model, x_all)
    return rates, {
        "family": "ncomms2025_multimodal_fusion_adapter",
        "adapter_note": "Original endpoint was amyloid/meta-tau/regional tau PET positivity; this row predicts annualized regional tau rates.",
        "feature_policy": "PET-free multimodal clinical, plasma, MRI, APOE, cognition, and time features with missingness indicators",
        "head": "ridge base plus short-epoch residual MLP",
        "ridge_alpha": 5.0,
        "residual_scale": 0.50,
        "hidden_layer_sizes": [64, 32],
        "max_iter": int(max_iter),
        "iterations": int(getattr(model, "n_iter_", -1)),
        "feature_count": int(x.shape[1]),
        "best_validation_score": float(getattr(model, "best_validation_score_", float("nan"))),
        "train_rate_rmse": train_rate_rmse(y, rates, train),
    }


def fit_dyepad_dynamic_graph_rates(
    context: TrainContext,
    *,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adapt DyEPAD's dynamic patient graph idea with a short non-torch graph smoother."""

    dataset = context.dataset
    y = dataset.target_rates[:, context.target_indices]
    train = np.asarray(context.train_indices, dtype=int)
    x_graph = matrix_with_time(design_matrix(dataset), dataset.time_years)
    neighbor_rates, neighbor_report = knn_rate_smoother(
        x_graph,
        y,
        train,
        random_seed=random_seed,
        n_neighbors=32,
        use_scaling=True,
    )
    ridge_rates = fit_ridge_multi_output_rates(x_graph, y, train, alpha=1.0)
    rates = 0.55 * neighbor_rates + 0.45 * ridge_rates
    return rates, {
        "family": "dyepad_dynamic_patient_graph_adapter",
        "adapter_note": "Original DyEPAD uses visit-wise GCN embeddings plus tensor/frequency modeling for AD progression; no torch/torch-geometric is available here, so this row uses train-only dynamic patient-similarity graph smoothing plus ridge heads.",
        "graph_features": "all baseline features + elapsed time",
        "neighbor_report": neighbor_report,
        "ridge_alpha": 1.0,
        "train_rate_rmse": train_rate_rmse(y, rates, train),
    }


def fit_gcn_xai_population_graph_rates(
    context: TrainContext,
    *,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adapt the GCN-XAI ADNI population graph to tau-rate regression."""

    dataset = context.dataset
    y = dataset.target_rates[:, context.target_indices]
    train = np.asarray(context.train_indices, dtype=int)
    node_keep = feature_indices_by_family(
        dataset,
        include_clinical=True,
        include_plasma=True,
        include_mri=True,
        include_amyloid_pet=True,
        include_tau_pet=False,
    )
    adjacency_keep = [idx for idx, name in enumerate(dataset.feature_names) if name in {"adas13", "mmse", "ravlt_immediate", "cdrsb"}]
    if not adjacency_keep:
        adjacency_keep = node_keep
    x_node = matrix_with_time(dataset.feature_matrix[:, node_keep], dataset.time_years)
    x_adj = matrix_with_time(dataset.feature_matrix[:, adjacency_keep], dataset.time_years)
    graph_rates, graph_report = knn_rate_smoother(
        x_adj,
        y,
        train,
        random_seed=random_seed,
        n_neighbors=28,
        use_scaling=True,
    )
    ridge_rates = fit_ridge_multi_output_rates(x_node, y, train, alpha=2.0)
    rates = 0.45 * graph_rates + 0.55 * ridge_rates
    return rates, {
        "family": "gcn_xai_population_graph_adapter",
        "adapter_note": "Original endpoint was NC/MCI/AD node classification; this row uses the same population-graph idea for annualized regional tau-rate regression.",
        "node_feature_count": int(x_node.shape[1]),
        "adjacency_feature_policy": "cognitive test similarity, matching the paper's population graph construction as closely as this dataset allows",
        "graph_report": graph_report,
        "ridge_alpha": 2.0,
        "train_rate_rmse": train_rate_rmse(y, rates, train),
    }


def fit_jad_graphlasso_topology_rates(
    context: TrainContext,
    *,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adapt JAD2024 graphical-lasso tau topology into a rate forecaster."""

    del random_seed
    dataset = context.dataset
    train = np.asarray(context.train_indices, dtype=int)
    y = dataset.target_rates[:, context.target_indices]
    baseline_tau = dataset.target_baseline[:, context.target_indices]
    preprocessor = RobustPreprocessor.fit(baseline_tau[train], use_scaling=True)
    tau_train = preprocessor.transform(baseline_tau[train])
    alpha = 0.15
    topology_source = "graphical_lasso_precision"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gl = GraphicalLasso(alpha=alpha, max_iter=300, tol=1.0e-3).fit(tau_train)
        adjacency = np.abs(gl.precision_)
    except Exception:
        topology_source = "absolute_correlation_fallback"
        adjacency = np.abs(np.corrcoef(tau_train, rowvar=False))
    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(adjacency, 0.0)
    row_sum = adjacency.sum(axis=1, keepdims=True)
    weights = np.divide(adjacency, row_sum, out=np.zeros_like(adjacency), where=row_sum > 0.0)

    amyloid = amyloid_stage_values(dataset, train)
    cutpoints = np.nanquantile(amyloid[train], [1.0 / 3.0, 2.0 / 3.0])
    stages = np.digitize(amyloid, cutpoints, right=False)
    global_rate = np.nanmean(y[train], axis=0)
    stage_rates = []
    stage_counts = []
    shrinkage = 10.0
    for stage_id in range(3):
        rows = train[stages[train] == stage_id]
        stage_counts.append(int(rows.size))
        if rows.size:
            mean_rate = np.nanmean(y[rows], axis=0)
            rate = (rows.size * mean_rate + shrinkage * global_rate) / (rows.size + shrinkage)
        else:
            rate = global_rate.copy()
        smoothed_rate = 0.65 * rate + 0.35 * np.einsum("ij,j->i", weights, rate)
        stage_rates.append(smoothed_rate)
    rates = np.vstack(stage_rates)[np.clip(stages, 0, 2)]
    return rates, {
        "family": "jad2024_graphical_lasso_tau_topology_adapter",
        "adapter_note": "Original endpoint was tau-topology/efficiency across amyloid burden, not subject-level prediction; this row stratifies by amyloid burden and smooths train rates over the learned tau-dependency graph.",
        "alpha": float(alpha),
        "topology_source": topology_source,
        "stage_cutpoints": [float(value) for value in cutpoints],
        "stage_counts": stage_counts,
        "nonzero_edges": int(np.count_nonzero(np.triu(adjacency > 1.0e-8, k=1))),
        "train_rate_rmse": train_rate_rmse(y, rates, train),
    }


def fit_hpbn_prototype_brainnet_rates(
    context: TrainContext,
    *,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adapt HPBN's hierarchical prototype matching to tau-map rate prediction."""

    dataset = context.dataset
    train = np.asarray(context.train_indices, dtype=int)
    y = dataset.target_rates[:, context.target_indices]
    baseline_tau = dataset.target_baseline[:, context.target_indices]
    x = matrix_with_time(baseline_tau, dataset.time_years)
    n_prototypes = min(18, max(4, int(np.sqrt(train.size))))
    rates, report = soft_prototype_rate_regression(
        x,
        y,
        train,
        n_prototypes=n_prototypes,
        random_seed=random_seed,
    )
    return rates, {
        "family": "hpbn_hierarchical_prototype_adapter",
        "adapter_note": "Attached HPBN code expects FCN/SCN matrices and performs prototype matching, not GCN message passing. This adapter uses baseline regional tau maps as the available brain-network state and predicts rates with soft prototype matching.",
        **report,
        "train_rate_rmse": train_rate_rmse(y, rates, train),
    }


def fit_hist_gradient_boosting_multioutput(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    random_seed: int,
    max_iter: int,
    learning_rate: float,
    max_leaf_nodes: int,
    l2_regularization: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    preprocessor = RobustPreprocessor.fit(x[train], use_scaling=False)
    x_train = preprocessor.transform(x[train])
    x_all = preprocessor.transform(x)
    output = np.zeros((x.shape[0], y.shape[1]), dtype=float)
    reports = []
    for target_idx in range(y.shape[1]):
        y_train = y[train, target_idx]
        mask = np.isfinite(y_train)
        if int(np.sum(mask)) < 8:
            output[:, target_idx] = np.nanmean(y_train)
            reports.append({"target": int(target_idx), "fallback": "train_mean"})
            continue
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(learning_rate),
            max_iter=int(max_iter),
            max_leaf_nodes=int(max_leaf_nodes),
            min_samples_leaf=12,
            l2_regularization=float(l2_regularization),
            early_stopping=True,
            validation_fraction=0.18,
            n_iter_no_change=patience(max_iter),
            random_state=int(random_seed) + target_idx,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train[mask], y_train[mask])
        output[:, target_idx] = model.predict(x_all)
        reports.append(
            {
                "target": int(target_idx),
                "iterations": int(getattr(model, "n_iter_", -1)),
                "train_rmse": float(np.sqrt(np.mean((model.predict(x_train[mask]) - y_train[mask]) ** 2))),
            }
        )
    return output, reports


def knn_rate_smoother(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    random_seed: int,
    n_neighbors: int,
    use_scaling: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    del random_seed
    preprocessor = RobustPreprocessor.fit(x[train], use_scaling=use_scaling)
    x_train = preprocessor.transform(x[train])
    x_all = preprocessor.transform(x)
    k = min(int(n_neighbors), int(train.size))
    diff = x_all[:, None, :] - x_train[None, :, :]
    dist = np.sqrt(np.mean(diff * diff, axis=2))
    neighbor_order = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    neighbor_dist = np.take_along_axis(dist, neighbor_order, axis=1)
    scale = float(np.nanmedian(neighbor_dist[neighbor_dist > 0.0]))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    weights = np.exp(-0.5 * (neighbor_dist / scale) ** 2)
    train_y = y[train]
    global_rate = np.nanmean(train_y, axis=0)
    output = np.zeros((x.shape[0], y.shape[1]), dtype=float)
    for row_idx in range(x.shape[0]):
        source = train_y[neighbor_order[row_idx]]
        w = weights[row_idx]
        for target_idx in range(y.shape[1]):
            values = source[:, target_idx]
            mask = np.isfinite(values)
            if np.any(mask):
                output[row_idx, target_idx] = float(np.sum(w[mask] * values[mask]) / np.sum(w[mask]))
            else:
                output[row_idx, target_idx] = float(global_rate[target_idx])
    return output, {
        "n_neighbors": int(k),
        "distance_scale": float(scale),
        "train_source": "training subjects only",
    }


def soft_prototype_rate_regression(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    n_prototypes: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    preprocessor = RobustPreprocessor.fit(x[train], use_scaling=True)
    x_train = preprocessor.transform(x[train])
    x_all = preprocessor.transform(x)
    k = min(int(n_prototypes), int(train.size))
    model = KMeans(n_clusters=k, n_init=20, random_state=int(random_seed))
    model.fit(x_train)
    train_weights = soft_assignments(x_train, model.cluster_centers_)
    all_weights = soft_assignments(x_all, model.cluster_centers_)
    train_y = y[train]
    global_rate = np.nanmean(train_y, axis=0)
    proto_rates = np.zeros((k, y.shape[1]), dtype=float)
    shrinkage = 4.0
    for proto_idx in range(k):
        w = train_weights[:, proto_idx]
        for target_idx in range(y.shape[1]):
            values = train_y[:, target_idx]
            mask = np.isfinite(values)
            denom = float(np.sum(w[mask]))
            if denom > 1.0e-12:
                mean_rate = float(np.sum(w[mask] * values[mask]) / denom)
                proto_rates[proto_idx, target_idx] = (denom * mean_rate + shrinkage * global_rate[target_idx]) / (denom + shrinkage)
            else:
                proto_rates[proto_idx, target_idx] = global_rate[target_idx]
    rates = all_weights @ proto_rates
    return rates, {
        "n_prototypes": int(k),
        "prototype_space": "baseline regional tau map + elapsed time",
        "soft_assignment": "rbf over k-means prototype distances",
        "shrinkage_subjects": float(shrinkage),
    }


def soft_assignments(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - centers[None, :, :]
    dist_sq = np.mean(diff * diff, axis=2)
    scale = float(np.nanmedian(dist_sq[dist_sq > 0.0]))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    logits = -dist_sq / scale
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    return np.divide(weights, weights_sum, out=np.full_like(weights, 1.0 / weights.shape[1]), where=weights_sum > 0.0)


def feature_indices_by_family(
    dataset: MultimodalPairDataset,
    *,
    include_clinical: bool,
    include_plasma: bool,
    include_mri: bool,
    include_amyloid_pet: bool,
    include_tau_pet: bool,
) -> list[int]:
    keep = []
    for idx, name in enumerate(dataset.feature_names):
        text = name.lower()
        is_tau_pet = text == "tau_meta_temporal" or text.startswith("tau_region:")
        is_amyloid_pet = text.startswith("amyloid_")
        is_plasma = text.startswith("plasma_")
        is_mri = text.startswith("mri_")
        is_clinical = text in {
            "age_years",
            "sex_female",
            "education_years",
            "apoe4_dose",
            "adas13",
            "mmse",
            "ravlt_immediate",
            "cdrsb",
        }
        if is_tau_pet and include_tau_pet:
            keep.append(idx)
        elif is_amyloid_pet and include_amyloid_pet:
            keep.append(idx)
        elif is_plasma and include_plasma:
            keep.append(idx)
        elif is_mri and include_mri:
            keep.append(idx)
        elif is_clinical and include_clinical:
            keep.append(idx)
    if not keep:
        raise ValueError("Feature policy selected no columns.")
    return keep


def matrix_with_time(x: np.ndarray, time_years: np.ndarray) -> np.ndarray:
    dt = np.asarray(time_years, dtype=float)[:, None]
    return np.column_stack([np.asarray(x, dtype=float), dt, np.log1p(np.clip(dt, 0.0, None))])


def matrix_with_missing_indicators(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    missing = (~np.isfinite(arr)).astype(float)
    return np.column_stack([arr, missing])


def amyloid_stage_values(dataset: MultimodalPairDataset, train: np.ndarray) -> np.ndarray:
    candidates = ["amyloid_centiloids", "amyloid_summary_suvr", "amyloid_positive"]
    for name in candidates:
        if name in dataset.feature_names:
            values = np.asarray(dataset.feature_matrix[:, dataset.feature_index(name)], dtype=float)
            if np.any(np.isfinite(values[train])):
                median = float(np.nanmedian(values[train]))
                return np.where(np.isfinite(values), values, median)
    baseline = dataset.target_baseline[:, 0]
    median = float(np.nanmedian(baseline[train]))
    return np.where(np.isfinite(baseline), baseline, median)


def fit_bayesian_ndm_prediction(
    graph: dict[str, Any],
    split: Any,
    target_indices: list[int],
    *,
    bootstrap_samples: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    forecast = graph["forecast_dataset"]
    selected_region_indices = graph["selected_region_indices"]
    ndm = NetworkDiffusionModel(graph["laplacian"])
    rng = np.random.default_rng(int(random_seed))
    train = np.asarray(split.train_indices, dtype=int)
    predictions = []
    rhos = []
    train_mses = []
    for _ in range(int(bootstrap_samples)):
        sample = rng.choice(train, size=train.size, replace=True)
        fit = ndm.fit_global_rho(
            forecast.baseline[sample],
            forecast.observed[sample],
            forecast.time_years[sample],
            bounds=parameter_bounds(graph["config"], "rho", (0.0, 10.0)),
        )
        predictions.append(ndm.predict(forecast.baseline, forecast.time_years, fit.rho)[:, selected_region_indices])
        rhos.append(float(fit.rho))
        train_mses.append(float(fit.train_mse))
    pred = np.mean(np.stack(predictions, axis=0), axis=0)
    return pred, {
        "family": "bootstrap_posterior_predictive_ndm",
        "bootstrap_samples": int(bootstrap_samples),
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "train_mse_mean": float(np.mean(train_mses)),
        "target_indices": [int(idx) for idx in target_indices],
    }


def fit_probabilistic_stage_rates(
    context: TrainContext,
    z_values: np.ndarray,
    *,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    del random_seed
    dataset = context.dataset
    train = np.asarray(context.train_indices, dtype=int)
    y = dataset.target_rates[:, context.target_indices]
    z = np.asarray(z_values, dtype=float)
    cuts = np.nanquantile(z[train], [1.0 / 3.0, 2.0 / 3.0])
    stage = np.digitize(z, cuts, right=False)
    global_rate = np.nanmean(y[train], axis=0)
    stage_rates = []
    stage_counts = []
    shrinkage = 8.0
    for stage_id in range(3):
        rows = train[stage[train] == stage_id]
        stage_counts.append(int(rows.size))
        if rows.size == 0:
            stage_rates.append(global_rate.copy())
            continue
        mean_rate = np.nanmean(y[rows], axis=0)
        count = float(rows.size)
        stage_rates.append((count * mean_rate + shrinkage * global_rate) / (count + shrinkage))
    stage_rates_arr = np.vstack(stage_rates)
    rates = stage_rates_arr[np.clip(stage, 0, 2)]
    return rates, {
        "family": "tertile_pseudotime_stage_rate_model",
        "stage_cutpoints": [float(cut) for cut in cuts],
        "stage_counts": stage_counts,
        "shrinkage_subjects": float(shrinkage),
        "train_rate_rmse": train_rate_rmse(y, rates, train),
    }


def fit_bnlte_variant_prediction(
    dataset: MultimodalPairDataset,
    split: Any,
    target_names: list[str],
    target_indices: list[int],
    *,
    max_parents: int,
    pseudotime_mode: str,
    pseudotime_label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if pseudotime_mode == "atn_tau_free":
        pseudotime, selected_feature_count = fit_masked_atn_pseudotime(dataset, split.train_indices)
    else:
        pseudotime = fit_pseudotime(dataset.feature_matrix, dataset.feature_names, split.train_indices, mode=pseudotime_mode)
        selected_feature_count = len(pseudotime.selected_feature_names)
    pseudotime.mode = pseudotime_label
    fit = fit_dynamic_scm(
        dataset,
        pseudotime,
        split.train_indices,
        target_names=target_names,
        max_parents_per_target=max_parents,
        cv_folds=3,
    )
    rates = fit.predict_rates(dataset)[:, target_indices]
    pred = dataset.target_baseline[:, target_indices] + dataset.time_years[:, None] * rates
    return pred, {
        "family": "bn_lte_variant",
        "pseudotime_mode": pseudotime_label,
        "selected_feature_count": int(selected_feature_count),
        "selected_features": list(pseudotime.selected_feature_names),
        "max_parents": int(max_parents),
    }


def fit_masked_atn_pseudotime(dataset: MultimodalPairDataset, train_indices: np.ndarray) -> tuple[Any, int]:
    keep = [idx for idx, name in enumerate(dataset.feature_names) if is_atn_tau_free_feature(name)]
    if len(keep) < 2:
        raise ValueError("ATN tau-free pseudotime selected fewer than two features.")
    masked = np.full_like(dataset.feature_matrix, np.nan, dtype=float)
    masked[:, keep] = dataset.feature_matrix[:, keep]
    model = fit_pseudotime(masked, dataset.feature_names, train_indices, mode="global")
    return model, len(model.selected_feature_names)


def is_atn_tau_free_feature(name: str) -> bool:
    text = name.lower()
    if "tau" in text or "pt217" in text or "ptau" in text:
        return False
    tokens = (
        "amyloid",
        "ab42",
        "ab40",
        "centiloid",
        "nfl",
        "gfap",
        "mri",
        "hippocampus",
        "amygdala",
        "temporal",
        "adas",
        "mmse",
        "ravlt",
        "cdrsb",
        "age",
        "sex",
        "education",
        "apoe",
    )
    return any(token in text for token in tokens)


def mlp_hidden_activation(model: MLPRegressor, x: np.ndarray, *, hidden_layers: int) -> np.ndarray:
    activation = np.asarray(x, dtype=float)
    for layer_idx in range(int(hidden_layers)):
        activation = np.einsum("ij,jk->ik", activation, model.coefs_[layer_idx]) + model.intercepts_[layer_idx][None, :]
        activation = np.maximum(activation, 0.0)
    return np.asarray(activation, dtype=float)


def fit_ridge_multi_output_rates(x: np.ndarray, y: np.ndarray, train: np.ndarray, *, alpha: float) -> np.ndarray:
    preprocessor = RobustPreprocessor.fit(x[train], use_scaling=True)
    x_train = preprocessor.transform(x[train])
    x_all = preprocessor.transform(x)
    mask = np.all(np.isfinite(y[train]), axis=1)
    if int(np.sum(mask)) < 8:
        fill = np.nanmean(y[train], axis=0)
        return np.tile(fill[None, :], (x.shape[0], 1))
    design = np.column_stack([np.ones(np.sum(mask)), x_train[mask]])
    penalty = np.eye(design.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    gram = design.T @ design
    rhs = design.T @ y[train][mask]
    beta = np.linalg.solve(gram + penalty, rhs)
    all_design = np.column_stack([np.ones(x_all.shape[0]), x_all])
    return all_design @ beta


def score_baseline_table_rows(
    predictions: dict[str, np.ndarray],
    dataset: MultimodalPairDataset,
    split: Any,
    regions: list[str],
    target_indices: list[int],
    *,
    repeat_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    test = np.asarray(split.test_indices, dtype=int)
    rows = []
    empirical_s1 = np.nanmean(observed[test], axis=0)
    empirical_delta = np.nanmean(observed[test] - baseline[test], axis=0)
    for model in MODEL_ORDER:
        if model not in predictions:
            continue
        pred = predictions[model]
        pred_s1 = np.nanmean(pred[test], axis=0)
        predicted_delta = np.nanmean(pred[test] - baseline[test], axis=0)
        braak = braak_summary_for_model(pred, observed, baseline, test, regions)
        rows.append(
            {
                "model": model,
                "type": MODEL_TYPES[model],
                "repeat_index": int(repeat_index),
                "seed": int(seed),
                "n_test_pairs": int(test.size),
                "mae": 100.0 * finite_mean_abs(pred_s1 - empirical_s1),
                "rho_delta": safe_correlation(empirical_delta, predicted_delta, rank=True),
                "cosine": cosine_similarity(empirical_delta, predicted_delta),
                "top3": weighted_topk_capture(empirical_delta, predicted_delta, 3),
                "braak_rho": braak["braak_rho"],
                "braak_mae": 100.0 * braak["braak_mae"],
            }
        )
    return rows


def braak_summary_for_model(
    pred: np.ndarray,
    observed: np.ndarray,
    baseline: np.ndarray,
    test: np.ndarray,
    regions: list[str],
) -> dict[str, float]:
    region_to_idx = {region: idx for idx, region in enumerate(regions)}
    empirical_order = []
    model_order = []
    for group_regions in BRAAK_GROUPS.values():
        idxs = [region_to_idx[region] for region in group_regions if region in region_to_idx]
        if idxs:
            empirical_order.append(float(np.nanmean(observed[test][:, idxs] - baseline[test][:, idxs])))
            model_order.append(float(np.nanmean(pred[test][:, idxs] - baseline[test][:, idxs])))
        else:
            empirical_order.append(float("nan"))
            model_order.append(float("nan"))
    empirical = np.asarray(empirical_order, dtype=float)
    modeled = np.asarray(model_order, dtype=float)
    return {
        "braak_rho": safe_correlation(empirical, modeled, rank=True),
        "braak_mae": finite_mean_abs(modeled - empirical),
    }


def summarize_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in MODEL_ORDER:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        summary: dict[str, Any] = {
            "model": model,
            "type": MODEL_TYPES[model],
            "n_repeats": int(len(model_rows)),
        }
        for metric in METRIC_COLUMNS:
            values = np.asarray([float(row[metric]) for row in model_rows if np.isfinite(float(row[metric]))], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values)) if values.size else float("nan")
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        output.append(summary)
    return output


def format_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    formatted = []
    for row in rows:
        item = {
            "Model": str(row["model"]),
            "Type": str(row["type"]),
        }
        for metric in METRIC_COLUMNS:
            item[DISPLAY_COLUMNS[metric]] = format_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"])
        formatted.append(item)
    return formatted


def render_markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def render_latex_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0])
    output = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        output.append(" & ".join(escape_latex(str(row.get(header, ""))) for header in headers) + " \\\\")
    output.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(output)


def format_mean_std(mean: Any, std: Any) -> str:
    mean_value = float(mean)
    std_value = float(std)
    if not np.isfinite(mean_value):
        return "NA"
    return f"{mean_value:.2f}+-{std_value:.2f}"


def patience(max_iter: int) -> int:
    return max(5, min(20, int(max_iter) // 4))


def finite_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    return float(np.mean(arr[mask])) if np.any(mask) else float("nan")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def escape_latex(text: str) -> str:
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def resolve_path(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
