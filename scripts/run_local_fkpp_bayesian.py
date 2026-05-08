#!/usr/bin/env python3
"""Run Bayesian Chaggar-style local FKPP with PyMC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.forecasting import (  # noqa: E402
    ForecastDataset,
    SubjectSplit,
    compute_aggregate_metrics,
    compute_pair_metrics,
    load_forecast_dataset,
    load_labeled_matrix,
    make_subject_split,
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.models.fkpp import LocalFKPPModel, fit_local_fkpp_components  # noqa: E402
from spread_toolbox.models.local_fkpp_bayes import (  # noqa: E402
    group_indices_by_rid,
    median_or,
    predict_by_subject,
    require_pymc,
    sample_subject_posterior,
)


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
    parser.add_argument("--draws", type=int, default=None, help="Posterior draws per sampled subject.")
    parser.add_argument("--tune", type=int, default=None, help="MCMC tuning steps per sampled subject.")
    parser.add_argument("--chains", type=int, default=None, help="MCMC chains per sampled subject.")
    parser.add_argument(
        "--max-train-subjects",
        type=int,
        default=None,
        help="Limit sampled training subjects for exploratory runs. Use 0 or a negative value for all train subjects.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run Bayesian local FKPP without writing outputs.",
    )
    args = parser.parse_args()

    require_pymc()

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

    random_seed = int(config.get("experiment", {}).get("random_seed", 20260507))
    component_quantile = float(modeling.get("local_fkpp_component_quantile", 0.99))
    components = fit_local_fkpp_components(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
        carrying_capacity_quantile=component_quantile,
        random_seed=random_seed,
    )
    model = LocalFKPPModel(
        laplacian,
        u0=components.u0,
        cc=components.cc,
        steps_per_year=int(modeling.get("local_fkpp_steps_per_year", 12)),
        laplacian_normalization=str(modeling.get("local_fkpp_laplacian_normalization", "spectral")),
    )

    parameter_bounds = modeling.get("parameter_bounds", {})
    rho_bounds = tuple(float(value) for value in parameter_bounds.get("rho", [0.0, 10.0]))
    alpha_bounds = tuple(float(value) for value in parameter_bounds.get("alpha", [0.0, 10.0]))
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

    draws = int(args.draws if args.draws is not None else modeling.get("local_fkpp_bayes_draws", 200))
    tune = int(args.tune if args.tune is not None else modeling.get("local_fkpp_bayes_tune", 200))
    chains = int(args.chains if args.chains is not None else modeling.get("local_fkpp_bayes_chains", 1))
    max_train_subjects = args.max_train_subjects
    if max_train_subjects is None:
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
            f"sampled {count}/{len(sampled_rids)} RID={rid} "
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
    metrics_summary = compute_aggregate_metrics(pair_metrics)

    report = {
        "model": "local_fkpp",
        "equation": "dS/dt = -rho_s L (S - u0) + alpha_s (S - u0)((cc - u0) - (S - u0))",
        "implementation_note": (
            "Exploratory empirical-Bayes PyMC sampler. Subject posterior sampling is run for selected "
            "training subjects only; unseen test subjects use train-derived fallback posterior medians."
        ),
        "fair_test_note": (
            "Because this is a subject-held-out split, test subjects cannot receive subject-specific "
            "posterior parameters without using their target scan. Test predictions therefore use fallback "
            "population parameters."
        ),
        "split": {
            "train_pairs": int(split.train_indices.size),
            "test_pairs": int(split.test_indices.size),
            "train_subjects": len(split.train_rids),
            "test_subjects": len(split.test_rids),
        },
        "sampling": {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "sampled_train_subjects": len(sampled_rids),
            "available_train_subjects": len(subject_indices),
            "residual_sigma_fixed": residual_sigma,
        },
        "population_fit": {
            "rho": population_fit.rho,
            "alpha": population_fit.alpha,
            "train_mse_clipped_suvr": population_fit.train_mse,
        },
        "fallback_parameters": {
            "rho": fallback_rho,
            "alpha": fallback_alpha,
        },
        "component_estimation": {
            "source": "training baseline and target tau only",
            "quantile": component_quantile,
            "u0_min": float(components.u0.min()),
            "u0_max": float(components.u0.max()),
            "cc_min": float(components.cc.min()),
            "cc_max": float(components.cc.max()),
        },
        "test_metrics": [
            row
            for row in metrics_summary
            if row["split"] == "test" and row["metric"] in {"mae", "rmse", "subject_spearman", "delta_spearman"}
        ],
    }

    print("\nBayesian local FKPP test metrics:")
    for row in report["test_metrics"]:
        print(f"test {row['metric']}: median={row['median']:.4f}, mean={row['mean']:.4f}, n={row['n']}")

    if not args.no_write:
        subject_posteriors_path = output_dir / outputs.get(
            "local_fkpp_subject_posteriors",
            "local_fkpp_subject_posteriors.csv",
        )
        pair_metrics_path = output_dir / outputs.get(
            "local_fkpp_pair_metrics",
            "local_fkpp_pair_metrics.csv",
        )
        metrics_summary_path = output_dir / outputs.get(
            "local_fkpp_metrics_summary",
            "local_fkpp_metrics_summary.csv",
        )
        report_path = output_dir / outputs.get("local_fkpp_report", "local_fkpp_report.json")

        write_csv_rows(subject_posteriors_path, subject_rows)
        write_csv_rows(pair_metrics_path, pair_metrics)
        write_csv_rows(metrics_summary_path, metrics_summary)
        write_json(report_path, report)

        print("\nWrote Bayesian local FKPP outputs:")
        print(f"subject_posteriors: {subject_posteriors_path}")
        print(f"pair_metrics: {pair_metrics_path}")
        print(f"metrics_summary: {metrics_summary_path}")
        print(f"report: {report_path}")

    return 0


def load_required_matrix(path: Path, dataset: ForecastDataset) -> np.ndarray:
    labels, matrix = load_labeled_matrix(path)
    if labels != dataset.region_labels:
        raise ValueError(f"Matrix labels do not match forecast dataset region labels: {path}")
    return matrix


if __name__ == "__main__":
    raise SystemExit(main())
