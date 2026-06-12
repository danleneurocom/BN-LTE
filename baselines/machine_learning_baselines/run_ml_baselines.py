#!/usr/bin/env python3
"""Run lightweight ML baselines on the shared BN-LTE ADNI tau-forecast split.

The baselines are intentionally kept in a separate folder from the BN-LTE code.
They are direct forecasting comparators, not causal models:

1. Supervised 1D ridge prognostic index.
2. AdaBoost regional tau-rate regressors.
3. Small MLP residual/rate forecaster as a lite deep-learning baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor


THIS_DIR = Path(__file__).resolve().parent
BN_DIR = THIS_DIR.parent
PROJECT_ROOT = BN_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(BN_DIR))

from bayesian_network_scm.data import MultimodalPairDataset, build_multimodal_pair_dataset  # noqa: E402
from bayesian_network_scm.reporting import make_subject_split  # noqa: E402
from run_extended_paper_experiments import (  # noqa: E402
    BRAAK_GROUPS,
    cosine_similarity,
    direction_accuracy,
    safe_auprc,
    safe_auroc,
    safe_balanced_accuracy,
    top_fraction_precision,
    topk_overlap,
    weighted_topk_capture,
    write_csv_rows,
    write_json,
)
from run_paper_validation_experiments import (  # noqa: E402
    finite_mean_abs,
    finite_rmse,
    metric_rows_for_predictions,
    safe_correlation,
    summarize_metric_rows,
    validate_dataset,
    validate_predictions,
    validate_split,
)


ML_MODEL_ORDER = ["ML-Prognostic Index", "AdaBoost Tau-Rate", "MLP-Lite"]
RANDOM_SEED = 20260521


@dataclass
class TrainContext:
    dataset: MultimodalPairDataset
    selected_regions: list[str]
    target_names: list[str]
    target_indices: list[int]
    train_indices: np.ndarray
    train_observed_min: np.ndarray
    train_observed_max: np.ndarray


@dataclass
class RobustPreprocessor:
    """Train-only median imputation, winsorization, and optional scaling."""

    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    use_scaling: bool

    @classmethod
    def fit(cls, x_train: np.ndarray, *, use_scaling: bool = True, lower_q: float = 0.01, upper_q: float = 0.99) -> "RobustPreprocessor":
        x = np.asarray(x_train, dtype=float)
        x = np.where(np.isfinite(x), x, np.nan)
        median = np.nanmedian(x, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        x_imp = np.where(np.isfinite(x), x, median[None, :])
        lower = np.nanquantile(x_imp, lower_q, axis=0)
        upper = np.nanquantile(x_imp, upper_q, axis=0)
        lower = np.where(np.isfinite(lower), lower, median)
        upper = np.where(np.isfinite(upper), upper, median)
        upper = np.maximum(upper, lower)
        x_clip = np.clip(x_imp, lower[None, :], upper[None, :])
        mean = np.mean(x_clip, axis=0)
        scale = np.std(x_clip, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1.0e-8), scale, 1.0)
        return cls(median=median, lower=lower, upper=upper, mean=mean, scale=scale, use_scaling=use_scaling)

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        arr = np.where(np.isfinite(arr), arr, self.median[None, :])
        arr = np.clip(arr, self.lower[None, :], self.upper[None, :])
        if self.use_scaling:
            arr = (arr - self.mean[None, :]) / self.scale[None, :]
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=BN_DIR / "outputs" / "machine_learning_baselines")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    report = run_ml_baselines(project_root=args.project_root, output_dir=args.output_dir, random_seed=args.random_seed)
    print("Machine-learning baselines complete.")
    print(f"Report: {report['report_path']}")
    return 0


def run_ml_baselines(*, project_root: str | Path, output_dir: str | Path, random_seed: int) -> dict[str, Any]:
    root = Path(project_root).resolve()
    out = resolve_path(output_dir, root)
    out.mkdir(parents=True, exist_ok=True)

    dataset = build_multimodal_pair_dataset(root)
    selected_regions = list(dataset.report["selected_tau_regions"])
    target_names = [f"tau_rate:{region}" for region in selected_regions]
    target_indices = [dataset.target_index(name) for name in target_names]
    validate_dataset(dataset, target_indices)
    split = make_subject_split(dataset.metadata_rows, random_seed=random_seed)
    validate_split(split)

    train = np.asarray(split.train_indices, dtype=int)
    train_values = np.concatenate(
        [dataset.target_baseline[train][:, target_indices], dataset.target_observed[train][:, target_indices]],
        axis=0,
    )
    context = TrainContext(
        dataset=dataset,
        selected_regions=selected_regions,
        target_names=target_names,
        target_indices=target_indices,
        train_indices=train,
        train_observed_min=np.nanmin(train_values, axis=0),
        train_observed_max=np.nanmax(train_values, axis=0),
    )

    predictions, fit_reports = fit_ml_predictions(context, random_seed=random_seed)
    validate_predictions(predictions, dataset, target_indices)

    pair_rows = metric_rows_for_predictions(
        predictions,
        dataset,
        target_indices,
        split,
        selected_regions,
        experiment="ml_baselines",
        seed=random_seed,
    )
    repeated_summary = summarize_metric_rows(pair_rows, group_fields=["model", "split", "metric"])
    group_map_rows = group_map_progression_metrics(predictions, dataset, split, target_indices)
    braak_rows, braak_summary = braak_ordering_rows(predictions, dataset, split, selected_regions, target_indices)
    classifier_rows = fast_progressor_classification(predictions, dataset, split, target_indices)

    tables = {
        "pair_metrics": out / "ml_pair_metrics.csv",
        "pair_summary": out / "ml_pair_summary.csv",
        "group_map": out / "ml_group_map_progression_metrics.csv",
        "braak_stage_deltas": out / "ml_braak_stage_deltas.csv",
        "braak_summary": out / "ml_braak_ordering_summary.csv",
        "fast_progressor": out / "ml_fast_progressor_classification.csv",
    }
    write_csv_rows(tables["pair_metrics"], pair_rows)
    write_csv_rows(tables["pair_summary"], repeated_summary)
    write_csv_rows(tables["group_map"], group_map_rows)
    write_csv_rows(tables["braak_stage_deltas"], braak_rows)
    write_csv_rows(tables["braak_summary"], braak_summary)
    write_csv_rows(tables["fast_progressor"], classifier_rows)
    np.savez_compressed(
        out / "ml_predictions.npz",
        **{model_key(model): value for model, value in predictions.items()},
    )

    report = {
        "purpose": "Recent peer-review-inspired lightweight ML baselines for the BN-LTE paper comparison.",
        "configuration": {
            "random_seed": int(random_seed),
            "subject_split": split.report(),
            "selected_regions": selected_regions,
            "target_names": target_names,
        },
        "models": fit_reports,
        "tables": {name: str(path) for name, path in tables.items()},
        "notes": [
            "All models are trained only on the training subject split.",
            "Feature imputation and scaling are fit on the training split only.",
            "Targets are annualized regional tau rates; follow-up SUVR is reconstructed as baseline + dt * predicted_rate.",
            "Follow-up predictions are clipped to the training split regional tau SUVR range to avoid biologically impossible extrapolation.",
            "Torch and xgboost were not installed; the lite deep-learning baseline uses sklearn MLPRegressor and the boosting baseline uses sklearn AdaBoostRegressor.",
        ],
        "references_to_cite": [
            "Giorgio et al., Nature Communications 2022: multimodal prognostic index for future tau accumulation.",
            "Rathore et al., Alzheimer's & Dementia 2024: ML/radiomics prediction of regional tau accumulation.",
            "Jung et al., NeuroImage 2021 / Xu et al., Medical Image Analysis 2022: longitudinal deep learning for AD progression.",
        ],
    }
    report_path = out / "ml_baseline_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def fit_ml_predictions(context: TrainContext, *, random_seed: int) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    dataset = context.dataset
    x = design_matrix(dataset)
    y = dataset.target_rates[:, context.target_indices]
    train = context.train_indices

    predictions: dict[str, np.ndarray] = {}
    reports: dict[str, dict[str, Any]] = {}

    rates, report = fit_supervised_prognostic_index(x, y, train)
    predictions["ML-Prognostic Index"] = reconstruct_followup(context, rates)
    reports["ML-Prognostic Index"] = report

    rates, report = fit_adaboost_rate_models(x, y, train, random_seed=random_seed)
    predictions["AdaBoost Tau-Rate"] = reconstruct_followup(context, rates)
    reports["AdaBoost Tau-Rate"] = report

    rates, report = fit_mlp_lite_rate_model(x, y, train, random_seed=random_seed)
    predictions["MLP-Lite"] = reconstruct_followup(context, rates)
    reports["MLP-Lite"] = report

    return predictions, reports


def design_matrix(dataset: MultimodalPairDataset) -> np.ndarray:
    dt = np.asarray(dataset.time_years, dtype=float)[:, None]
    log_dt = np.log1p(np.clip(dt, 0.0, None))
    x = np.column_stack([dataset.feature_matrix, dt, log_dt])
    return np.where(np.isfinite(x), x, np.nan)


def fit_supervised_prognostic_index(x: np.ndarray, y: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a stable one-dimensional prognostic index and region-specific heads."""

    train_y = y[train]
    burden_rate = np.nanmean(train_y, axis=1)
    mask = np.isfinite(burden_rate)
    if int(np.sum(mask)) < 8:
        raise ValueError("Too few complete training rows for supervised prognostic index.")
    preprocessor = RobustPreprocessor.fit(x[train], use_scaling=True)
    x_train = preprocessor.transform(x[train])
    x_all = preprocessor.transform(x)
    index_alpha = 10.0
    coef, intercept = fit_ridge_closed_form(x_train[mask], burden_rate[mask], alpha=index_alpha)
    z_all = (np.einsum("ij,j->i", x_all, coef) + intercept).reshape(-1, 1)
    z_train = z_all[train]
    z_mean = float(np.nanmean(z_train[mask]))
    z_std = float(np.nanstd(z_train[mask]))
    if not np.isfinite(z_std) or z_std < 1.0e-12:
        z_std = 1.0
    z_all = (z_all - z_mean) / z_std
    z_train = z_all[train]

    output = np.zeros((x.shape[0], y.shape[1]), dtype=float)
    target_alpha = 0.1
    for target_idx in range(y.shape[1]):
        y_train = y[train, target_idx]
        target_mask = np.isfinite(y_train) & np.isfinite(z_train[:, 0])
        if int(np.sum(target_mask)) < 8:
            output[:, target_idx] = np.nanmean(y_train)
            continue
        head_coef, head_intercept = fit_ridge_closed_form(z_train[target_mask], y_train[target_mask], alpha=target_alpha)
        output[:, target_idx] = np.einsum("ij,j->i", z_all, head_coef) + head_intercept
    return output, {
        "family": "supervised_1d_ridge_prognostic_index",
        "index_target": "mean_regional_tau_rate",
        "index_alpha": float(index_alpha),
        "target_head_alpha": float(target_alpha),
        "train_rate_rmse": train_rate_rmse(y, output, train),
    }


def fit_ridge_closed_form(x: np.ndarray, y: np.ndarray, *, alpha: float) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(x.shape[0]), x])
    penalty = np.eye(design.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    gram = np.einsum("ni,nj->ij", design, design)
    rhs = np.einsum("ni,n->i", design, y)
    beta = np.linalg.solve(gram + penalty, rhs)
    return np.asarray(beta[1:], dtype=float), float(beta[0])


def fit_adaboost_rate_models(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    preprocessor = RobustPreprocessor.fit(x[train], use_scaling=False)
    x_train_imp = preprocessor.transform(x[train])
    x_all = preprocessor.transform(x)
    output = np.zeros((x.shape[0], y.shape[1]), dtype=float)
    train_rmses = []
    estimator_arg = adaboost_estimator_keyword()
    for target_idx in range(y.shape[1]):
        y_train = y[train, target_idx]
        mask = np.isfinite(y_train)
        if int(np.sum(mask)) < 8:
            output[:, target_idx] = np.nanmean(y_train)
            continue
        stump = DecisionTreeRegressor(max_depth=3, min_samples_leaf=12, random_state=random_seed + target_idx)
        kwargs = {
            estimator_arg: stump,
            "n_estimators": 120,
            "learning_rate": 0.04,
            "loss": "linear",
            "random_state": random_seed + 101 + target_idx,
        }
        model = AdaBoostRegressor(**kwargs)
        model.fit(x_train_imp[mask], y_train[mask])
        output[:, target_idx] = model.predict(x_all)
        train_pred = model.predict(x_train_imp[mask])
        train_rmses.append(float(np.sqrt(np.mean((train_pred - y_train[mask]) ** 2))))
    return output, {
        "family": "regional_adaboost_regressors",
        "n_estimators": 120,
        "base_tree_max_depth": 3,
        "target_count": int(y.shape[1]),
        "train_rate_rmse_mean": float(np.mean(train_rmses)) if train_rmses else float("nan"),
    }


def adaboost_estimator_keyword() -> str:
    try:
        AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=1))
        return "estimator"
    except TypeError:
        return "base_estimator"


def fit_mlp_lite_rate_model(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    *,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.all(np.isfinite(y[train]), axis=1)
    if int(np.sum(mask)) < 16:
        raise ValueError("Too few complete training rows for MLP-lite baseline.")
    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=5.0e-3,
        learning_rate_init=2.0e-3,
        early_stopping=True,
        validation_fraction=0.18,
        n_iter_no_change=30,
        max_iter=1200,
        random_state=random_seed,
    )
    preprocessor = RobustPreprocessor.fit(x[train], use_scaling=True)
    x_train = preprocessor.transform(x[train])
    x_all = preprocessor.transform(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train[mask], y[train][mask])
    rates = mlp_predict_einsum(model, x_all)
    return rates, {
        "family": "small_mlp_rate_regressor",
        "hidden_layer_sizes": [32, 16],
        "alpha": 5.0e-3,
        "iterations": int(getattr(model, "n_iter_", -1)),
        "best_validation_score": float(getattr(model, "best_validation_score_", float("nan"))),
        "train_rate_rmse": train_rate_rmse(y, rates, train),
    }


def mlp_predict_einsum(model: MLPRegressor, x: np.ndarray) -> np.ndarray:
    activation = np.asarray(x, dtype=float)
    for layer_idx, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_, strict=True)):
        activation = np.einsum("ij,jk->ik", activation, coef) + intercept[None, :]
        if layer_idx < len(model.coefs_) - 1:
            activation = np.maximum(activation, 0.0)
    return np.asarray(activation, dtype=float)


def reconstruct_followup(context: TrainContext, predicted_rates: np.ndarray) -> np.ndarray:
    baseline = context.dataset.target_baseline[:, context.target_indices]
    pred = baseline + context.dataset.time_years[:, None] * np.asarray(predicted_rates, dtype=float)
    pred = np.nan_to_num(pred, nan=np.nanmedian(baseline), posinf=np.nanmax(context.train_observed_max), neginf=np.nanmin(context.train_observed_min))
    return np.clip(pred, context.train_observed_min[None, :], context.train_observed_max[None, :])


def train_rate_rmse(y: np.ndarray, pred: np.ndarray, train: np.ndarray) -> float:
    mask = np.isfinite(y[train]) & np.isfinite(pred[train])
    if int(np.sum(mask)) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[train][mask] - y[train][mask]) ** 2)))


def group_map_progression_metrics(
    predictions: dict[str, np.ndarray],
    dataset: MultimodalPairDataset,
    split: Any,
    target_indices: list[int],
) -> list[dict[str, Any]]:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    rows = []
    stage_items = [("all_test", np.asarray(split.test_indices, dtype=int))]
    for stage, indices in stage_items:
        empirical_s1 = np.nanmean(observed[indices], axis=0)
        empirical_delta = np.nanmean(observed[indices] - baseline[indices], axis=0)
        for model in ML_MODEL_ORDER:
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
    for model in ["Empirical", *ML_MODEL_ORDER]:
        source = observed if model == "Empirical" else predictions[model]
        for group, group_regions in BRAAK_GROUPS.items():
            idxs = [region_to_idx[region] for region in group_regions if region in region_to_idx]
            value = float(np.nanmean(source[test][:, idxs] - baseline[test][:, idxs])) if idxs else float("nan")
            rows.append({"model": model, "group": group, "mean_delta_suvr": value, "n_pairs": int(test.size), "n_regions": len(idxs)})
    summary = []
    empirical_order = np.asarray([empirical_group_delta[group] for group in BRAAK_GROUPS], dtype=float)
    for model in ML_MODEL_ORDER:
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
    target_indices: list[int],
) -> list[dict[str, Any]]:
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    observed_rate = (observed - baseline) / dataset.time_years[:, None]
    empirical_score = np.nanmean(observed_rate, axis=1)
    threshold = float(np.nanquantile(empirical_score[np.asarray(split.train_indices, dtype=int)], 0.75))
    y_test = empirical_score[np.asarray(split.test_indices, dtype=int)] >= threshold
    rows = []
    for model in ML_MODEL_ORDER:
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


def model_key(model: str) -> str:
    return model.lower().replace(" ", "_").replace("-", "_")


def resolve_path(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
