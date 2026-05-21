#!/usr/bin/env python3
"""Fit Bio-FKPP: full generalised biologically-modulated FKPP.

  dS_i/dt = -rho_eff_i * (L S)_i
             + alpha_eff_i * S_i*(1-S_i)
             - c_eff_i * S_i
             + seeding_i

where:
  rho_eff_i   = rho + delta_rho * thickness_i
  alpha_eff_i = alpha + beta_g*amyloid_i + beta_t*thickness_i + beta_a*apoe4_s
  c_eff_i     = lambda_c * thickness_i
  seeding_i   = gamma_a*amyloid_i*S0_i + gamma_t*thickness_i*S0_i

Two-stage:
  Stage 1: global FKPP backbone -> freeze rho, alpha
  Stage 2: jointly optimize (delta_rho, beta_g, beta_t, beta_a, lambda_c, gamma_a, gamma_t)
"""

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
    MinMaxStateScaler,
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
from spread_toolbox.models.bio_fkpp import BioFKPPModel  # noqa: E402


MODEL_NAME = "bio_fkpp"


def default_config_path() -> Path:
    experiment_dir = PROJECT_ROOT / "experiments" / "group_average_enigma"
    local_config = experiment_dir / "config.yaml"
    if local_config.exists():
        return local_config
    return experiment_dir / "config.example.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    laplacian = load_required_matrix(
        output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"), dataset
    )
    # Load adjacency for connectivity-biology seeding term (optional — skipped if missing)
    adj_path = output_dir / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv")
    adjacency = None
    if adj_path.exists():
        _, adjacency = load_labeled_matrix(adj_path)
    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=int(config.get("experiment", {}).get("random_seed", 20260507)),
    )

    result = run_bio_fkpp_model(dataset, split, laplacian, modeling, config, adjacency=adjacency)
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
    comparison_rows = test_comparison_rows(baseline_summary + metrics_summary)
    report = {
        "model": MODEL_NAME,
        "description": (
            "Full generalised Bio-FKPP with modulated diffusion, growth, clearance and seeding. "
            "Three biological modulation channels: rho(x,b), alpha(x,b), c(x,b)."
        ),
        "equation": (
            "dS_i/dt = -rho_eff_i*(LS)_i + alpha_eff_i*S_i*(1-S_i) - c_eff_i*S_i + seeding_i; "
            "rho_eff=rho+delta_rho*thick; alpha_eff=alpha+bg*amy+bt*thick+ba*apoe4; "
            "c_eff=lambda_c*thick; seeding=ga*amy*S0+gt*thick*S0"
        ),
        "split": split_summary(split),
        "fit_report": result["fit_report"],
        "test_comparison": comparison_rows,
    }

    print(json.dumps(report["split"], indent=2, sort_keys=True))
    print("\nBio-FKPP fit report:")
    print(json.dumps({k: v for k, v in report["fit_report"].items() if k != "covariates"}, indent=2))
    print("\nBio-FKPP test comparison:")
    print_comparison_table(comparison_rows)

    if not args.no_write:
        write_csv_rows(
            output_dir / outputs.get("bio_fkpp_pair_metrics", "bio_fkpp_pair_metrics.csv"),
            pair_metrics,
        )
        write_csv_rows(
            output_dir / outputs.get("bio_fkpp_metrics_summary", "bio_fkpp_metrics_summary.csv"),
            metrics_summary,
        )
        write_csv_rows(
            output_dir / outputs.get("bio_fkpp_likelihood_metrics", "bio_fkpp_likelihood_metrics.csv"),
            likelihood_metrics,
        )
        write_json(
            output_dir / outputs.get("bio_fkpp_report", "bio_fkpp_report.json"),
            {**report, "likelihood_metrics": likelihood_metrics},
        )
        print("\nWrote Bio-FKPP outputs.")

    return 0


def run_bio_fkpp_model(
    dataset: ForecastDataset,
    split: SubjectSplit,
    laplacian: np.ndarray,
    modeling: dict[str, Any],
    config: dict[str, Any],
    adjacency: np.ndarray | None = None,
) -> dict[str, Any]:
    scaler = MinMaxStateScaler.fit(dataset.baseline[split.train_indices], dataset.observed[split.train_indices])
    baseline_scaled = scaler.transform(dataset.baseline)
    observed_scaled = scaler.transform(dataset.observed)

    parameter_bounds = modeling.get("parameter_bounds", {})
    rho_bounds = tuple(float(v) for v in parameter_bounds.get("rho", [0.0, 10.0]))
    alpha_bounds = tuple(float(v) for v in parameter_bounds.get("alpha", [0.0, 10.0]))
    beta_bounds_list = modeling.get("bio_fkpp_beta_bounds", [-5.0, 5.0])
    beta_bounds = (float(beta_bounds_list[0]), float(beta_bounds_list[1]))

    model = BioFKPPModel(
        laplacian,
        adjacency=adjacency,
        steps_per_year=int(modeling.get("bio_fkpp_steps_per_year", 12)),
        laplacian_normalization=str(modeling.get("bio_fkpp_laplacian_normalization", "spectral")),
    )

    pair_covariates, regional_covariates, covariate_report = build_closure_covariates(
        dataset, split, config, PROJECT_ROOT
    )
    amyloid = regional_covariates.get("amyloid_suvr")
    thickness = regional_covariates.get("cortical_thickness")
    apoe4_dose = pair_covariates.get("apoe4_dose")

    delta_rho_bounds_list = modeling.get("bio_fkpp_delta_rho_bounds", [-2.0, 2.0])
    delta_rho_bounds = (float(delta_rho_bounds_list[0]), float(delta_rho_bounds_list[1]))
    lambda_c_bounds_list = modeling.get("bio_fkpp_lambda_c_bounds", [-5.0, 5.0])
    lambda_c_bounds = (float(lambda_c_bounds_list[0]), float(lambda_c_bounds_list[1]))
    gamma_bounds_list = modeling.get("bio_fkpp_gamma_bounds", [-5.0, 5.0])
    gamma_bounds = (float(gamma_bounds_list[0]), float(gamma_bounds_list[1]))

    fit = model.fit(
        baseline_scaled, observed_scaled, dataset.time_years,
        amyloid=amyloid,
        thickness=thickness,
        apoe4_dose=apoe4_dose,
        train_indices=split.train_indices,
        rho_bounds=(rho_bounds[0], rho_bounds[1]),
        alpha_bounds=(alpha_bounds[0], alpha_bounds[1]),
        beta_bounds=beta_bounds,
        gamma_bounds=gamma_bounds,
        delta_rho_bounds=delta_rho_bounds,
        lambda_c_bounds=lambda_c_bounds,
        maxiter=int(modeling.get("bio_fkpp_optimizer_maxiter", 120)),
    )

    n_pairs = len(dataset.pairs)
    rho_eff = model.build_rho_eff(
        n_pairs, fit.rho, thickness,
        delta_rho_thickness=fit.delta_rho_thickness,
    )
    alpha_eff = model.build_alpha_eff(
        n_pairs, fit.alpha, amyloid, thickness, apoe4_dose,
        beta_growth_amyloid=fit.beta_growth_amyloid,
        beta_growth_thickness=fit.beta_growth_thickness,
        beta_growth_apoe4=fit.beta_growth_apoe4,
    )
    clearance = model.build_clearance(
        n_pairs, thickness,
        lambda_clearance_thickness=fit.lambda_clearance_thickness,
    )
    seeding = model.build_seeding(
        baseline_scaled, amyloid, thickness,
        gamma_seeding_amyloid=fit.gamma_seeding_amyloid,
        gamma_seeding_thickness=fit.gamma_seeding_thickness,
        gamma_seeding_connectivity=fit.gamma_seeding_connectivity,
    )
    predicted_scaled = model.predict(
        baseline_scaled, dataset.time_years,
        rho_eff=rho_eff, alpha_eff=alpha_eff, clearance=clearance, seeding=seeding,
    )
    predicted = scaler.inverse_transform(predicted_scaled)

    pair_metrics = compute_pair_metrics(
        dataset.pairs, dataset.baseline, dataset.observed, predicted, split, MODEL_NAME
    )
    # rho (frozen) + alpha (frozen) + len(fitted_terms) + sigma
    n_likelihood_parameters = 3 + len(fit.fitted_terms)

    return {
        "predicted": predicted,
        "pair_metrics": pair_metrics,
        "fit_report": {
            "model": MODEL_NAME,
            "backbone": "global_fkpp",
            "rho": fit.rho,
            "alpha": fit.alpha,
            "delta_rho_thickness": fit.delta_rho_thickness,
            "beta_growth_amyloid": fit.beta_growth_amyloid,
            "beta_growth_thickness": fit.beta_growth_thickness,
            "beta_growth_apoe4": fit.beta_growth_apoe4,
            "lambda_clearance_thickness": fit.lambda_clearance_thickness,
            "gamma_seeding_amyloid": fit.gamma_seeding_amyloid,
            "gamma_seeding_thickness": fit.gamma_seeding_thickness,
            "gamma_seeding_connectivity": fit.gamma_seeding_connectivity,
            "fitted_terms": fit.fitted_terms,
            "stage1_train_mse_scaled": fit.stage1_train_mse,
            "stage2_train_mse_scaled": fit.stage2_train_mse,
            "mse_reduction_pct": fit.mse_reduction_pct,
            "laplacian_normalization": model.laplacian_normalization,
            "laplacian_scale": model.laplacian_scale,
            "optimizer_success": fit.optimizer_success,
            "optimizer_message": fit.optimizer_message,
            "optimizer_iterations": fit.optimizer_iterations,
            "covariates": covariate_report,
            "n_likelihood_parameters": n_likelihood_parameters,
        },
    }


def load_required_matrix(path: Path, dataset: ForecastDataset) -> np.ndarray:
    labels, matrix = load_labeled_matrix(path)
    if labels != dataset.region_labels:
        raise ValueError(f"Matrix labels do not match forecast dataset region labels: {path}")
    return matrix


def load_optional_rows(path: Path) -> list[dict[str, Any]]:
    return read_csv_rows(path) if path.exists() else []


def test_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {"mae", "rmse", "subject_spearman", "delta_spearman", "top5_overlap", "top10_overlap"}
    return [row for row in rows if row.get("split") == "test" and row.get("metric") in wanted]


def split_summary(split: SubjectSplit) -> dict[str, int]:
    return {
        "train_pairs": int(split.train_indices.size),
        "test_pairs": int(split.test_indices.size),
        "train_subjects": len(split.train_rids),
        "test_subjects": len(split.test_rids),
    }


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), {})[str(row["metric"])] = row
    print(f"{'model':<38} {'MAE':>8} {'RMSE':>8} {'subj rho':>10} {'delta rho':>10} {'top5':>8} {'top10':>8}")
    for model_name, metrics in sorted(by_model.items()):
        if metrics:
            print(
                f"{model_name:<38} "
                f"{_median(metrics, 'mae'):>8.4f} "
                f"{_median(metrics, 'rmse'):>8.4f} "
                f"{_median(metrics, 'subject_spearman'):>10.4f} "
                f"{_median(metrics, 'delta_spearman'):>10.4f} "
                f"{_median(metrics, 'top5_overlap'):>8.4f} "
                f"{_median(metrics, 'top10_overlap'):>8.4f}"
            )


def _median(metrics: dict[str, dict[str, Any]], name: str) -> float:
    return float(metrics[name]["median"])


if __name__ == "__main__":
    raise SystemExit(main())
