#!/usr/bin/env python3
"""Fit Stage-1 local-FKPP plus Bayesian linear-basis closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.adni_features import build_closure_covariates  # noqa: E402
from spread_toolbox.forecasting import (  # noqa: E402
    ForecastDataset,
    SubjectSplit,
    compute_aggregate_metrics,
    compute_gaussian_likelihood_metrics,
    compute_pair_metrics,
    load_forecast_dataset,
    load_labeled_matrix,
    make_subject_split,
    read_csv_rows,
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.models.closure import (  # noqa: E402
    apply_closure_delta,
    build_closure_feature_library,
    fit_horseshoe_linear_closure,
)
from spread_toolbox.models.fkpp import LocalFKPPModel, fit_local_fkpp_components  # noqa: E402


MODEL_NAME = "local_fkpp_horseshoe_closure"


def default_config_path() -> Path:
    experiment_dir = PROJECT_ROOT / "experiments" / "group_average_enigma"
    local_config = experiment_dir / "config.yaml"
    if local_config.exists():
        return local_config
    return experiment_dir / "config.example.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to config YAML. Defaults to config.yaml if present, otherwise config.example.yaml.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run without writing outputs.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    laplacian = load_required_matrix(output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"), dataset)
    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=int(config.get("experiment", {}).get("random_seed", 20260507)),
    )

    result = run_closure_model(dataset, split, laplacian, modeling, config)
    pair_metrics = result["pair_metrics"]
    metrics_summary = compute_aggregate_metrics(pair_metrics)
    likelihood_metrics = compute_gaussian_likelihood_metrics(
        dataset.observed,
        result["predicted"],
        split,
        MODEL_NAME,
        n_parameters=int(result["fit_report"]["n_likelihood_parameters"]),
        min_sigma=float(modeling.get("likelihood_min_sigma", 1.0e-6)),
    )

    baseline_summary = load_optional_rows(
        output_dir / outputs.get("baseline_comparison_metrics_summary", "baseline_comparison_metrics_summary.csv")
    )
    baseline_likelihood = load_optional_rows(
        output_dir / outputs.get("baseline_comparison_likelihood_metrics", "baseline_comparison_likelihood_metrics.csv")
    )
    comparison_rows = test_comparison_rows(baseline_summary + metrics_summary)

    report = {
        "model": MODEL_NAME,
        "purpose": (
            "Stage-1 pivot from the PhD decision report: keep local-FKPP transport/backbone and add "
            "a sparse Bayesian feature-conditioned linear closure on residual tau change."
        ),
        "equation": (
            "S_hat = local_FKPP(S0, dt) + dt * sum_j beta_j g_j("
            "tau, amyloid, cortical_thickness, cortical_volume, APOE, plasma_ptau181)"
        ),
        "method_note": (
            "Implemented with PyMC global-local shrinkage because this project environment already uses PyMC; "
            "this is the same Stage-1 idea as the report's NumPyro horseshoe recommendation."
        ),
        "split": {
            "train_pairs": int(split.train_indices.size),
            "test_pairs": int(split.test_indices.size),
            "train_subjects": len(split.train_rids),
            "test_subjects": len(split.test_rids),
        },
        "fit_report": result["fit_report"],
        "r2_report": result["r2_report"],
        "test_comparison": comparison_rows,
        "likelihood_comparison": baseline_likelihood + likelihood_metrics,
    }

    print(json.dumps(report["split"], indent=2, sort_keys=True))
    print("\nStage-1 closure test comparison:")
    print_comparison_table(comparison_rows)
    print("\nR2 report:")
    print(json.dumps(report["r2_report"], indent=2, sort_keys=True))
    print("\nGaussian likelihood comparison:")
    print_likelihood_table(baseline_likelihood + likelihood_metrics)
    print("\nClosure terms:")
    print_terms(result["term_rows"])

    if not args.no_write:
        pair_metrics_path = output_dir / outputs.get("closure_pair_metrics", "closure_pair_metrics.csv")
        metrics_summary_path = output_dir / outputs.get("closure_metrics_summary", "closure_metrics_summary.csv")
        likelihood_metrics_path = output_dir / outputs.get(
            "closure_likelihood_metrics",
            "closure_likelihood_metrics.csv",
        )
        terms_path = output_dir / outputs.get("closure_terms", "closure_terms.csv")
        report_path = output_dir / outputs.get("closure_report", "closure_report.json")

        write_csv_rows(pair_metrics_path, pair_metrics)
        write_csv_rows(metrics_summary_path, metrics_summary)
        write_csv_rows(likelihood_metrics_path, likelihood_metrics)
        write_csv_rows(terms_path, result["term_rows"])
        write_json(report_path, report)

        print("\nWrote closure outputs:")
        print(f"pair_metrics: {pair_metrics_path}")
        print(f"metrics_summary: {metrics_summary_path}")
        print(f"likelihood_metrics: {likelihood_metrics_path}")
        print(f"terms: {terms_path}")
        print(f"report: {report_path}")

    return 0


def load_required_matrix(path: Path, dataset: ForecastDataset) -> np.ndarray:
    labels, matrix = load_labeled_matrix(path)
    if labels != dataset.region_labels:
        raise ValueError(f"Matrix labels do not match forecast dataset region labels: {path}")
    return matrix


def run_closure_model(
    dataset: ForecastDataset,
    split: SubjectSplit,
    laplacian: np.ndarray,
    modeling: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    random_seed = int(config.get("experiment", {}).get("random_seed", 20260507))
    component_quantile = float(modeling.get("closure_component_quantile", modeling.get("local_fkpp_component_quantile", 0.99)))
    components = fit_local_fkpp_components(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
        carrying_capacity_quantile=component_quantile,
        random_seed=random_seed,
    )

    parameter_bounds = modeling.get("parameter_bounds", {})
    rho_bounds = tuple(float(value) for value in parameter_bounds.get("rho", [0.0, 10.0]))
    alpha_bounds = tuple(float(value) for value in parameter_bounds.get("alpha", [0.0, 10.0]))
    local_model = LocalFKPPModel(
        laplacian,
        u0=components.u0,
        cc=components.cc,
        steps_per_year=int(modeling.get("closure_fkpp_steps_per_year", modeling.get("local_fkpp_steps_per_year", 12))),
        laplacian_normalization=str(
            modeling.get("closure_fkpp_laplacian_normalization", modeling.get("local_fkpp_laplacian_normalization", "spectral"))
        ),
    )
    clipped_baseline = components.clip(dataset.baseline)
    clipped_observed = components.clip(dataset.observed)
    backbone_fit = local_model.fit_global_parameters(
        clipped_baseline[split.train_indices],
        clipped_observed[split.train_indices],
        dataset.time_years[split.train_indices],
        rho_bounds=(rho_bounds[0], rho_bounds[1]),
        alpha_bounds=(alpha_bounds[0], alpha_bounds[1]),
        maxiter=int(modeling.get("closure_fkpp_optimizer_maxiter", modeling.get("local_fkpp_optimizer_maxiter", 80))),
    )
    backbone_predicted = local_model.predict(
        clipped_baseline,
        dataset.time_years,
        rho=backbone_fit.rho,
        alpha=backbone_fit.alpha,
    )

    pair_covariates, regional_covariates, covariate_report = build_closure_covariates(dataset, split, config, PROJECT_ROOT)
    feature_library = build_closure_feature_library(
        clipped_baseline,
        u0=components.u0,
        cc=components.cc,
        pair_covariates=pair_covariates,
        regional_covariates=regional_covariates,
    )
    safe_time = np.maximum(dataset.time_years, 1.0e-6)[:, None]
    residual_rate = (clipped_observed - backbone_predicted) / safe_time
    closure = fit_horseshoe_linear_closure(
        feature_library,
        residual_rate,
        row_indices=split.train_indices,
        draws=int(modeling.get("closure_draws", 300)),
        tune=int(modeling.get("closure_tune", 300)),
        chains=int(modeling.get("closure_chains", 2)),
        target_accept=float(modeling.get("closure_target_accept", 0.9)),
        random_seed=random_seed,
        inclusion_threshold=float(modeling.get("closure_inclusion_threshold", 1.0e-4)),
        max_train_rows=int(modeling.get("closure_max_train_rows", 10000)),
    )
    closure_rate = closure.predict_rate(feature_library.values)
    predicted = apply_closure_delta(
        backbone_predicted,
        dataset.time_years,
        closure_rate,
        u0=components.u0,
        cc=components.cc,
    )

    pair_metrics = compute_pair_metrics(dataset.pairs, dataset.baseline, dataset.observed, predicted, split, MODEL_NAME)
    term_rows = closure.term_rows()
    included_count = int(sum(row["selected"] for row in term_rows if row["term"] != "1"))
    n_likelihood_parameters = 4 + included_count
    r2_report = build_r2_report(dataset, split, backbone_predicted, predicted)

    return {
        "model": MODEL_NAME,
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "term_rows": term_rows,
        "r2_report": r2_report,
        "fit_report": {
            "model": MODEL_NAME,
            "backbone": "local_fkpp_population",
            "backbone_rho": backbone_fit.rho,
            "backbone_alpha": backbone_fit.alpha,
            "backbone_train_mse_clipped_suvr": backbone_fit.train_mse,
            "closure_train_mse_rate": closure.train_mse_rate,
            "closure_train_r2_rate": closure.train_r2_rate,
            "closure_feature_count": len(closure.feature_names),
            "closure_selected_features_pip_ge_0_95": included_count,
            "closure_inclusion_threshold": closure.inclusion_threshold,
            "closure_sampling": closure.sampling_report,
            "covariates": covariate_report,
            "n_likelihood_parameters": n_likelihood_parameters,
            "component_quantile": component_quantile,
            "laplacian_normalization": local_model.laplacian_normalization,
            "laplacian_scale": local_model.laplacian_scale,
            "u0_min": float(components.u0.min()),
            "u0_max": float(components.u0.max()),
            "cc_min": float(components.cc.min()),
            "cc_max": float(components.cc.max()),
        },
    }


def build_r2_report(
    dataset: ForecastDataset,
    split: SubjectSplit,
    backbone_predicted: np.ndarray,
    closure_predicted: np.ndarray,
) -> dict[str, Any]:
    test = split.test_indices
    train = split.train_indices
    one_year = np.where((dataset.time_years >= 0.5) & (dataset.time_years <= 1.5))[0]
    test_one_year = np.intersect1d(test, one_year)
    temporal_mask = temporal_region_mask(dataset.region_labels)
    return {
        "train_scalar_r2_backbone": scalar_r2(dataset.observed[train], backbone_predicted[train]),
        "train_scalar_r2_closure": scalar_r2(dataset.observed[train], closure_predicted[train]),
        "test_scalar_r2_backbone": scalar_r2(dataset.observed[test], backbone_predicted[test]),
        "test_scalar_r2_closure": scalar_r2(dataset.observed[test], closure_predicted[test]),
        "test_scalar_r2_gain": scalar_r2(dataset.observed[test], closure_predicted[test])
        - scalar_r2(dataset.observed[test], backbone_predicted[test]),
        "test_one_year_pairs": int(test_one_year.size),
        "test_one_year_scalar_r2_closure": scalar_r2(dataset.observed[test_one_year], closure_predicted[test_one_year])
        if test_one_year.size
        else float("nan"),
        "test_one_year_temporal_r2_closure": scalar_r2(
            dataset.observed[test_one_year][:, temporal_mask],
            closure_predicted[test_one_year][:, temporal_mask],
        )
        if test_one_year.size and np.any(temporal_mask)
        else float("nan"),
    }


def temporal_region_mask(region_labels: list[str]) -> np.ndarray:
    keywords = ("temporal", "entorhinal", "parahippocampal", "fusiform", "bankssts")
    return np.asarray([any(keyword in label.lower() for keyword in keywords) for label in region_labels], dtype=bool)


def scalar_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    if not np.any(mask):
        return float("nan")
    y = observed[mask]
    yhat = predicted[mask]
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / total)


def load_optional_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for row in read_csv_rows(path)]


def test_comparison_rows(metrics_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted_metrics = ["mae", "rmse", "subject_spearman", "delta_spearman", "top5_overlap", "top10_overlap"]
    return [
        row
        for row in metrics_summary
        if row["split"] == "test" and row["metric"] in wanted_metrics
    ]


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), {})[str(row["metric"])] = row
    print(
        f"{'model':<32} {'MAE med':>9} {'RMSE med':>9} {'subject Spearman med':>21} "
        f"{'delta Spearman med':>19} {'top5 med':>9} {'top10 med':>10}"
    )
    for model in ["ndm", "esm", "global_fkpp", "local_fkpp", MODEL_NAME]:
        metrics = by_model.get(model, {})
        if not metrics:
            continue
        print(
            f"{model:<32} "
            f"{metric_value(metrics, 'mae', 'median'):>9.4f} "
            f"{metric_value(metrics, 'rmse', 'median'):>9.4f} "
            f"{metric_value(metrics, 'subject_spearman', 'median'):>21.4f} "
            f"{metric_value(metrics, 'delta_spearman', 'median'):>19.4f} "
            f"{metric_value(metrics, 'top5_overlap', 'median'):>9.4f} "
            f"{metric_value(metrics, 'top10_overlap', 'median'):>10.4f}"
        )


def print_likelihood_table(rows: list[dict[str, Any]]) -> None:
    by_model_split: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_model_split.setdefault(str(row["model"]), {})[str(row["split"])] = row
    print(f"{'model':<32} {'train BIC':>12} {'test ELPD':>12} {'test ELPD/obs':>15} {'sigma':>9} {'k':>5}")
    for model in ["ndm", "esm", "global_fkpp", "local_fkpp", MODEL_NAME]:
        train = by_model_split.get(model, {}).get("train", {})
        test = by_model_split.get(model, {}).get("test", {})
        if not train or not test:
            continue
        print(
            f"{model:<32} "
            f"{row_float(train, 'bic'):>12.1f} "
            f"{row_float(test, 'elpd'):>12.1f} "
            f"{row_float(test, 'elpd_per_observation'):>15.4f} "
            f"{row_float(train, 'sigma_train'):>9.4f} "
            f"{int(row_float(train, 'n_parameters')):>5}"
        )


def print_terms(rows: list[dict[str, Any]], limit: int = 16) -> None:
    rows = sorted(rows, key=lambda row: abs(float(row["coefficient_mean"])), reverse=True)
    for row in rows[:limit]:
        print(
            f"{float(row['coefficient_mean']): .6f} "
            f"[PIP={float(row['inclusion_probability']):.3f}] * {row['term']}"
        )
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more terms")


def metric_value(metrics: dict[str, dict[str, Any]], metric: str, field: str) -> float:
    return row_float(metrics[metric], field)


def row_float(row: dict[str, Any], field: str) -> float:
    return float(row[field])


if __name__ == "__main__":
    raise SystemExit(main())
