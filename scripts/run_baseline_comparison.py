#!/usr/bin/env python3
"""Train and compare baseline forecasting models on one train/test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.forecasting import (  # noqa: E402
    ForecastDataset,
    MinMaxStateScaler,
    SubjectSplit,
    compute_aggregate_metrics,
    compute_gaussian_likelihood_metrics,
    compute_pair_metrics,
    load_forecast_dataset,
    load_labeled_matrix,
    make_subject_split,
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.models.esm import EpidemicSpreadingModel  # noqa: E402
from spread_toolbox.models.fkpp import GraphFKPPModel, LocalFKPPModel, fit_local_fkpp_components  # noqa: E402
from spread_toolbox.models.local_fkpp_bayes import (  # noqa: E402
    group_indices_by_rid,
    median_or,
    predict_by_subject,
    require_pymc,
    sample_subject_posterior,
    summarize_subject_posteriors,
)
from spread_toolbox.models.ndm import NetworkDiffusionModel  # noqa: E402


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
        help="Run the comparison without writing outputs.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    adjacency = load_required_matrix(output_dir / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv"), dataset)
    laplacian = load_required_matrix(output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"), dataset)
    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=int(config.get("experiment", {}).get("random_seed", 20260507)),
    )

    model_results = [
        run_ndm(dataset, split, laplacian, modeling),
        run_esm(dataset, split, adjacency, modeling),
        run_fkpp(dataset, split, laplacian, modeling),
        run_local_fkpp(dataset, split, laplacian, modeling, config),
    ]
    pair_metrics = [row for result in model_results for row in result["pair_metrics"]]
    metrics_summary = compute_aggregate_metrics(pair_metrics)
    likelihood_metrics = [
        row
        for result in model_results
        for row in compute_gaussian_likelihood_metrics(
            dataset.observed,
            result["predicted"],
            split,
            result["model"],
            n_parameters=int(result["fit_report"]["n_likelihood_parameters"]),
            min_sigma=float(modeling.get("likelihood_min_sigma", 1.0e-6)),
        )
    ]
    test_comparison = test_comparison_rows(metrics_summary)

    report = {
        "purpose": "Baseline comparison only: fit each baseline on train and compare all baselines on the same test set.",
        "likelihood_metric_note": (
            "BIC and ELPD use a Gaussian residual likelihood with sigma estimated on train residuals. "
            "This is comparable within this pipeline, but it is not a numerical reproduction of Chaggar's "
            "full posterior predictive BIC/ELPD calculation."
        ),
        "split": {
            "train_pairs": int(split.train_indices.size),
            "test_pairs": int(split.test_indices.size),
            "train_subjects": len(split.train_rids),
            "test_subjects": len(split.test_rids),
        },
        "model_fit_reports": [result["fit_report"] for result in model_results],
        "test_comparison": test_comparison,
        "likelihood_comparison": likelihood_metrics,
    }

    print(json.dumps(report["split"], indent=2, sort_keys=True))
    print("\nBaseline test comparison:")
    print_comparison_table(test_comparison)
    print("\nGaussian likelihood comparison:")
    print_likelihood_table(likelihood_metrics)

    if not args.no_write:
        pair_metrics_path = output_dir / outputs.get(
            "baseline_comparison_pair_metrics",
            "baseline_comparison_pair_metrics.csv",
        )
        metrics_summary_path = output_dir / outputs.get(
            "baseline_comparison_metrics_summary",
            "baseline_comparison_metrics_summary.csv",
        )
        likelihood_metrics_path = output_dir / outputs.get(
            "baseline_comparison_likelihood_metrics",
            "baseline_comparison_likelihood_metrics.csv",
        )
        report_path = output_dir / outputs.get("baseline_comparison_report", "baseline_comparison_report.json")

        write_csv_rows(pair_metrics_path, pair_metrics)
        write_csv_rows(metrics_summary_path, metrics_summary)
        write_csv_rows(likelihood_metrics_path, likelihood_metrics)
        write_json(report_path, report)

        print("\nWrote baseline-comparison outputs:")
        print(f"pair_metrics: {pair_metrics_path}")
        print(f"metrics_summary: {metrics_summary_path}")
        print(f"likelihood_metrics: {likelihood_metrics_path}")
        print(f"baseline_comparison_report: {report_path}")

    return 0


def load_required_matrix(path: Path, dataset: ForecastDataset) -> np.ndarray:
    labels, matrix = load_labeled_matrix(path)
    if labels != dataset.region_labels:
        raise ValueError(f"Matrix labels do not match forecast dataset region labels: {path}")
    return matrix


def run_ndm(
    dataset: ForecastDataset,
    split: SubjectSplit,
    laplacian: np.ndarray,
    modeling: dict[str, Any],
) -> dict[str, Any]:
    bounds = tuple(float(value) for value in modeling.get("parameter_bounds", {}).get("rho", [0.0, 10.0]))
    model = NetworkDiffusionModel(laplacian)
    fit = model.fit_global_rho(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
        dataset.time_years[split.train_indices],
        bounds=(bounds[0], bounds[1]),
    )
    predicted = model.predict(dataset.baseline, dataset.time_years, fit.rho)
    pair_metrics = compute_pair_metrics(dataset.pairs, dataset.baseline, dataset.observed, predicted, split, "ndm")
    return {
        "model": "ndm",
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "fit_report": {
            "model": "ndm",
            "equation": "dS/dt = -rho L S",
            "n_likelihood_parameters": 2,
            "rho": fit.rho,
            "rho_bounds": list(bounds),
            "train_mse": fit.train_mse,
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
        },
    }


def run_esm(
    dataset: ForecastDataset,
    split: SubjectSplit,
    adjacency: np.ndarray,
    modeling: dict[str, Any],
) -> dict[str, Any]:
    scaler = MinMaxStateScaler.fit(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
    )
    baseline_scaled = scaler.transform(dataset.baseline)
    observed_scaled = scaler.transform(dataset.observed)
    bounds = tuple(float(value) for value in modeling.get("parameter_bounds", {}).get("beta", [0.0, 10.0]))
    model = EpidemicSpreadingModel(adjacency, steps_per_year=int(modeling.get("esm_steps_per_year", 12)))
    fit = model.fit_global_beta(
        baseline_scaled[split.train_indices],
        observed_scaled[split.train_indices],
        dataset.time_years[split.train_indices],
        bounds=(bounds[0], bounds[1]),
    )
    predicted = scaler.inverse_transform(model.predict(baseline_scaled, dataset.time_years, fit.beta))
    pair_metrics = compute_pair_metrics(dataset.pairs, dataset.baseline, dataset.observed, predicted, split, "esm")
    return {
        "model": "esm",
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "fit_report": {
            "model": "esm",
            "equation": "dS/dt = beta * (1 - S) * W S",
            "n_likelihood_parameters": 2,
            "beta": fit.beta,
            "beta_bounds": list(bounds),
            "train_mse_scaled": fit.train_mse,
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
        },
    }


def run_fkpp(
    dataset: ForecastDataset,
    split: SubjectSplit,
    laplacian: np.ndarray,
    modeling: dict[str, Any],
) -> dict[str, Any]:
    scaler = MinMaxStateScaler.fit(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
    )
    baseline_scaled = scaler.transform(dataset.baseline)
    observed_scaled = scaler.transform(dataset.observed)
    parameter_bounds = modeling.get("parameter_bounds", {})
    rho_bounds = tuple(float(value) for value in parameter_bounds.get("rho", [0.0, 10.0]))
    alpha_bounds = tuple(float(value) for value in parameter_bounds.get("alpha", [0.0, 10.0]))
    model = GraphFKPPModel(
        laplacian,
        steps_per_year=int(modeling.get("fkpp_steps_per_year", 12)),
        laplacian_normalization=str(modeling.get("fkpp_laplacian_normalization", "spectral")),
    )
    fit = model.fit_global_parameters(
        baseline_scaled[split.train_indices],
        observed_scaled[split.train_indices],
        dataset.time_years[split.train_indices],
        rho_bounds=(rho_bounds[0], rho_bounds[1]),
        alpha_bounds=(alpha_bounds[0], alpha_bounds[1]),
        maxiter=int(modeling.get("fkpp_optimizer_maxiter", 80)),
    )
    predicted = scaler.inverse_transform(model.predict(baseline_scaled, dataset.time_years, rho=fit.rho, alpha=fit.alpha))
    pair_metrics = compute_pair_metrics(
        dataset.pairs,
        dataset.baseline,
        dataset.observed,
        predicted,
        split,
        "global_fkpp",
    )
    return {
        "model": "global_fkpp",
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "fit_report": {
            "model": "global_fkpp",
            "equation": "dS/dt = -rho L S + alpha S(1 - S)",
            "n_likelihood_parameters": 3,
            "laplacian_normalization": model.laplacian_normalization,
            "laplacian_scale": model.laplacian_scale,
            "rho": fit.rho,
            "alpha": fit.alpha,
            "rho_bounds": list(rho_bounds),
            "alpha_bounds": list(alpha_bounds),
            "train_mse_scaled": fit.train_mse,
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
            "optimizer_iterations": fit.optimizer_iterations,
            "optimizer_evaluations": fit.optimizer_evaluations,
        },
    }


def run_local_fkpp(
    dataset: ForecastDataset,
    split: SubjectSplit,
    laplacian: np.ndarray,
    modeling: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    require_pymc()

    random_seed = int(config.get("experiment", {}).get("random_seed", 20260507))
    component_quantile = float(modeling.get("local_fkpp_component_quantile", 0.99))
    components = fit_local_fkpp_components(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
        carrying_capacity_quantile=component_quantile,
        random_seed=random_seed,
    )

    parameter_bounds = modeling.get("parameter_bounds", {})
    rho_bounds = tuple(float(value) for value in parameter_bounds.get("rho", [0.0, 10.0]))
    alpha_bounds = tuple(float(value) for value in parameter_bounds.get("alpha", [0.0, 10.0]))
    model = LocalFKPPModel(
        laplacian,
        u0=components.u0,
        cc=components.cc,
        steps_per_year=int(modeling.get("local_fkpp_steps_per_year", 12)),
        laplacian_normalization=str(modeling.get("local_fkpp_laplacian_normalization", "spectral")),
    )
    clipped_baseline = components.clip(dataset.baseline)
    clipped_observed = components.clip(dataset.observed)

    population_fit = model.fit_global_parameters(
        clipped_baseline[split.train_indices],
        clipped_observed[split.train_indices],
        dataset.time_years[split.train_indices],
        rho_bounds=(rho_bounds[0], rho_bounds[1]),
        alpha_bounds=(alpha_bounds[0], alpha_bounds[1]),
        maxiter=int(modeling.get("local_fkpp_optimizer_maxiter", 80)),
    )

    population_prediction = model.predict(
        clipped_baseline[split.train_indices],
        dataset.time_years[split.train_indices],
        rho=population_fit.rho,
        alpha=population_fit.alpha,
    )
    residual_sigma = float(np.std(population_prediction - clipped_observed[split.train_indices]))
    residual_sigma = max(residual_sigma, float(modeling.get("local_fkpp_bayes_min_sigma", 1.0e-3)))

    draws = int(modeling.get("local_fkpp_bayes_draws", 200))
    tune = int(modeling.get("local_fkpp_bayes_tune", 200))
    chains = int(modeling.get("local_fkpp_bayes_chains", 1))
    max_train_subjects = int(modeling.get("local_fkpp_bayes_max_train_subjects", 25))
    subject_indices = group_indices_by_rid(dataset.pairs, split.train_indices)
    sampled_rids = sorted(subject_indices, key=lambda value: int(value) if value.isdigit() else value)
    if max_train_subjects > 0:
        sampled_rids = sampled_rids[:max_train_subjects]

    subject_rows = []
    subject_parameters: dict[str, tuple[float, float]] = {}
    for count, rid in enumerate(sampled_rids, start=1):
        indices = subject_indices[rid]
        posterior = sample_subject_posterior(
            model,
            clipped_baseline[indices],
            clipped_observed[indices],
            dataset.time_years[indices],
            prior_rho=population_fit.rho,
            prior_alpha=population_fit.alpha,
            prior_scale=float(modeling.get("local_fkpp_bayes_prior_scale", 1.0)),
            sigma=residual_sigma,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed + count,
        )
        subject_parameters[rid] = (posterior["rho_median"], posterior["alpha_median"])
        subject_rows.append({"RID": rid, "pairs": int(indices.size), **posterior})
        print(
            f"sampled local_fkpp {count}/{len(sampled_rids)} RID={rid} "
            f"rho={posterior['rho_median']:.4f} alpha={posterior['alpha_median']:.4f}"
        )

    fallback_rho = median_or(population_fit.rho, [row["rho_median"] for row in subject_rows])
    fallback_alpha = median_or(population_fit.alpha, [row["alpha_median"] for row in subject_rows])
    predicted = predict_by_subject(
        model,
        dataset.pairs,
        dataset.baseline,
        dataset.time_years,
        dataset.observed.shape,
        subject_parameters,
        fallback_rho=fallback_rho,
        fallback_alpha=fallback_alpha,
    )
    pair_metrics = compute_pair_metrics(
        dataset.pairs,
        dataset.baseline,
        dataset.observed,
        predicted,
        split,
        "local_fkpp",
    )
    n_likelihood_parameters = 3 + 2 * len(sampled_rids)
    return {
        "model": "local_fkpp",
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "fit_report": {
            "model": "local_fkpp",
            "equation": "dS/dt = -rho_s L (S - u0) + alpha_s (S - u0)((cc - u0) - (S - u0))",
            "n_likelihood_parameters": n_likelihood_parameters,
            "implementation_note": "Bayesian Chaggar-style local FKPP. Subject-specific posterior sampling is run on training subjects; unseen test subjects use train-derived fallback posterior medians to avoid target-scan leakage.",
            "component_estimation": "Two-component Gaussian mixture per region fit on training baseline and target tau only.",
            "component_quantile": component_quantile,
            "laplacian_normalization": model.laplacian_normalization,
            "laplacian_scale": model.laplacian_scale,
            "population_prior_rho": population_fit.rho,
            "population_prior_alpha": population_fit.alpha,
            "fallback_rho_for_unseen_subjects": fallback_rho,
            "fallback_alpha_for_unseen_subjects": fallback_alpha,
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "sampled_train_subjects": len(sampled_rids),
            "available_train_subjects": len(subject_indices),
            "residual_sigma_fixed": residual_sigma,
            "prior_scale": float(modeling.get("local_fkpp_bayes_prior_scale", 1.0)),
            "rho_bounds": list(rho_bounds),
            "alpha_bounds": list(alpha_bounds),
            "train_mse_clipped_suvr": population_fit.train_mse,
            **summarize_subject_posteriors(subject_rows, "rho_median", prefix="subject_rho"),
            **summarize_subject_posteriors(subject_rows, "alpha_median", prefix="subject_alpha"),
            "u0_min": float(components.u0.min()),
            "u0_max": float(components.u0.max()),
            "cc_min": float(components.cc.min()),
            "cc_max": float(components.cc.max()),
        },
    }


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
    header = (
        "model",
        "MAE med",
        "RMSE med",
        "subject Spearman med",
        "delta Spearman med",
        "top5 med",
        "top10 med",
    )
    print(
        f"{header[0]:<14} {header[1]:>9} {header[2]:>9} {header[3]:>21} "
        f"{header[4]:>19} {header[5]:>9} {header[6]:>10}"
    )
    for model in ["ndm", "esm", "global_fkpp", "local_fkpp"]:
        metrics = by_model.get(model, {})
        print(
            f"{model:<14} "
            f"{median(metrics, 'mae'):>9.4f} "
            f"{median(metrics, 'rmse'):>9.4f} "
            f"{median(metrics, 'subject_spearman'):>21.4f} "
            f"{median(metrics, 'delta_spearman'):>19.4f} "
            f"{median(metrics, 'top5_overlap'):>9.4f} "
            f"{median(metrics, 'top10_overlap'):>10.4f}"
        )


def print_likelihood_table(rows: list[dict[str, Any]]) -> None:
    by_model_split: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_model_split.setdefault(str(row["model"]), {})[str(row["split"])] = row
    print(f"{'model':<14} {'train BIC':>12} {'test ELPD':>12} {'test ELPD/obs':>15} {'sigma':>9} {'k':>5}")
    for model in ["ndm", "esm", "global_fkpp", "local_fkpp"]:
        train = by_model_split.get(model, {}).get("train", {})
        test = by_model_split.get(model, {}).get("test", {})
        print(
            f"{model:<14} "
            f"{float_value(train, 'bic'):>12.1f} "
            f"{float_value(test, 'elpd'):>12.1f} "
            f"{float_value(test, 'elpd_per_observation'):>15.4f} "
            f"{float_value(train, 'sigma_train'):>9.4f} "
            f"{int_value(train, 'n_parameters'):>5}"
        )


def median(metrics: dict[str, dict[str, Any]], name: str) -> float:
    return float(metrics[name]["median"])


def float_value(row: dict[str, Any], name: str) -> float:
    value = row.get(name, float("nan"))
    return float(value)


def int_value(row: dict[str, Any], name: str) -> int:
    value = row.get(name, 0)
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
