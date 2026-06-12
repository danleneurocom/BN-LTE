#!/usr/bin/env python3
"""Run PINN-style KPP reaction discovery with PySR symbolic distillation."""

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
from spread_toolbox.models.fkpp import fit_local_fkpp_components  # noqa: E402
from spread_toolbox.models.pinn_reaction import (  # noqa: E402
    GraphKPPReactionModel,
    distill_kpp_reaction_with_pysr,
    fisher_reaction,
    fit_pinn_kpp_reaction,
    symbolic_reaction_from_expression,
)


NN_MODEL_NAME = "pinn_kpp_nn"
SYMBOLIC_MODEL_NAME = "pinn_kpp_symbolic"


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
    laplacian = load_required_matrix(output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"), dataset)
    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=int(config.get("experiment", {}).get("random_seed", 20260507)),
    )

    result = run_pinn_symbolic_reaction(dataset, split, laplacian, modeling, config)
    pair_metrics = result["nn_pair_metrics"] + result["symbolic_pair_metrics"]
    metrics_summary = compute_aggregate_metrics(pair_metrics)
    likelihood_metrics = []
    for model_name, predicted, n_parameters in [
        (NN_MODEL_NAME, result["nn_predicted"], result["fit_report"]["nn_n_likelihood_parameters"]),
        (SYMBOLIC_MODEL_NAME, result["symbolic_predicted"], result["fit_report"]["symbolic_n_likelihood_parameters"]),
    ]:
        likelihood_metrics.extend(
            compute_gaussian_likelihood_metrics(
                dataset.observed,
                predicted,
                split,
                model_name,
                n_parameters=int(n_parameters),
                min_sigma=float(modeling.get("likelihood_min_sigma", 1.0e-6)),
            )
        )

    report = {
        "models": [NN_MODEL_NAME, SYMBOLIC_MODEL_NAME],
        "purpose": (
            "PINN+SR-inspired reaction discovery: learn a hard-constrained scalar KPP reaction f(c) "
            "inside the graph reaction-diffusion ODE, distill f(c)=c(1-c)q(c) with real PySR, "
            "and select symbolic candidates by projection error."
        ),
        "method_note": (
            "This follows the Zhang/Zou/Kuhl/Karniadakis idea more closely than the feature closure: "
            "the unknown object is the scalar reaction law f(c), not a linear correction over covariates."
        ),
        "split": {
            "train_pairs": int(split.train_indices.size),
            "test_pairs": int(split.test_indices.size),
            "train_subjects": len(split.train_rids),
            "test_subjects": len(split.test_rids),
        },
        "fit_report": result["fit_report"],
        "r2_report": result["r2_report"],
        "symbolic_equation": result["symbolic_equation"],
    }

    print(json.dumps(report["split"], indent=2, sort_keys=True))
    print("\nPINN-style KPP reaction comparison:")
    print_comparison_table(compute_aggregate_metrics(pair_metrics))
    print("\nSelected symbolic reaction:")
    print(result["symbolic_equation"])
    print("\nR2 report:")
    print(json.dumps(result["r2_report"], indent=2, sort_keys=True))

    if not args.no_write:
        pair_metrics_path = output_dir / outputs.get("pinn_reaction_pair_metrics", "pinn_reaction_pair_metrics.csv")
        metrics_summary_path = output_dir / outputs.get(
            "pinn_reaction_metrics_summary",
            "pinn_reaction_metrics_summary.csv",
        )
        likelihood_path = output_dir / outputs.get(
            "pinn_reaction_likelihood_metrics",
            "pinn_reaction_likelihood_metrics.csv",
        )
        pareto_path = output_dir / outputs.get("pinn_reaction_pareto_front", "pinn_reaction_pareto_front.csv")
        report_path = output_dir / outputs.get("pinn_reaction_report", "pinn_reaction_report.json")

        write_csv_rows(pair_metrics_path, pair_metrics)
        write_csv_rows(metrics_summary_path, metrics_summary)
        write_csv_rows(likelihood_path, likelihood_metrics)
        write_csv_rows(pareto_path, result["pareto_rows"])
        write_json(report_path, report)

        print("\nWrote PINN-style reaction outputs:")
        print(f"pair_metrics: {pair_metrics_path}")
        print(f"metrics_summary: {metrics_summary_path}")
        print(f"likelihood_metrics: {likelihood_path}")
        print(f"pareto_front: {pareto_path}")
        print(f"report: {report_path}")

    return 0


def run_pinn_symbolic_reaction(
    dataset: ForecastDataset,
    split: SubjectSplit,
    laplacian: np.ndarray,
    modeling: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    random_seed = int(config.get("experiment", {}).get("random_seed", 20260507))
    component_quantile = float(modeling.get("pinn_reaction_component_quantile", 0.99))
    hidden_layers = tuple(int(value) for value in modeling.get("pinn_reaction_hidden_layers", [8]))
    activation = str(modeling.get("pinn_reaction_activation", "tanh"))
    steps_per_year = int(modeling.get("pinn_reaction_steps_per_year", modeling.get("local_fkpp_steps_per_year", 12)))
    laplacian_normalization = str(
        modeling.get(
            "pinn_reaction_laplacian_normalization",
            modeling.get("local_fkpp_laplacian_normalization", "spectral"),
        )
    )

    components = fit_local_fkpp_components(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
        carrying_capacity_quantile=component_quantile,
        random_seed=random_seed,
    )
    clipped_baseline = components.clip(dataset.baseline)
    clipped_observed = components.clip(dataset.observed)

    parameter_bounds = modeling.get("parameter_bounds", {})
    rho_bounds = tuple(float(value) for value in parameter_bounds.get("rho", [0.0, 10.0]))
    alpha_bounds = tuple(float(value) for value in parameter_bounds.get("alpha", [0.0, 10.0]))
    fisher_model = GraphKPPReactionModel(
        laplacian,
        u0=components.u0,
        cc=components.cc,
        reaction=fisher_reaction(hidden_layers, activation),
        steps_per_year=steps_per_year,
        laplacian_normalization=laplacian_normalization,
    )
    fisher_fit = fisher_model.fit_global_parameters(
        clipped_baseline[split.train_indices],
        clipped_observed[split.train_indices],
        dataset.time_years[split.train_indices],
        rho_bounds=(rho_bounds[0], rho_bounds[1]),
        alpha_bounds=(alpha_bounds[0], alpha_bounds[1]),
        maxiter=int(modeling.get("pinn_reaction_optimizer_maxiter", 80)),
    )
    fisher_predicted = fisher_model.predict(clipped_baseline, dataset.time_years, rho=fisher_fit.rho, alpha=fisher_fit.alpha)

    ensemble_size = int(modeling.get("pinn_reaction_ensemble_size", 3))
    ensemble_reports = []
    pareto_rows = []
    best: dict[str, Any] | None = None
    pysr_cache_dir = resolve_project_path(config["paths"]["cache_dir"], PROJECT_ROOT) / "pinn_reaction_pysr"
    pysr_cache_dir.mkdir(parents=True, exist_ok=True)
    for member in range(ensemble_size):
        seed = random_seed + 1000 + member
        fit = fit_pinn_kpp_reaction(
            laplacian=laplacian,
            baseline=clipped_baseline,
            observed=clipped_observed,
            time_years=dataset.time_years,
            row_indices=split.train_indices,
            u0=components.u0,
            cc=components.cc,
            initial_rho=fisher_fit.rho,
            initial_alpha=fisher_fit.alpha,
            rho_bounds=(rho_bounds[0], rho_bounds[1]),
            alpha_bounds=(alpha_bounds[0], alpha_bounds[1]),
            hidden_layers=hidden_layers,
            activation=activation,
            steps_per_year=steps_per_year,
            laplacian_normalization=laplacian_normalization,
            iterations=int(modeling.get("pinn_reaction_spsa_iterations", 120)),
            learning_rate=float(modeling.get("pinn_reaction_spsa_learning_rate", 0.04)),
            perturbation=float(modeling.get("pinn_reaction_spsa_perturbation", 0.08)),
            gradient_clip=float(modeling.get("pinn_reaction_spsa_gradient_clip", 1.0)),
            l2_weight=float(modeling.get("pinn_reaction_l2_weight", 1.0e-4)),
            aux_weight=float(modeling.get("pinn_reaction_aux_weight", 1.0e-3)),
            objective_max_pairs=int(modeling.get("pinn_reaction_objective_max_pairs", 256)),
            random_seed=seed,
        )
        expression, member_rows = distill_kpp_reaction_with_pysr(
            fit.reaction,
            grid_size=int(modeling.get("pinn_reaction_distill_grid", 512)),
            niterations=int(modeling.get("pinn_reaction_pysr_niterations", 80)),
            populations=int(modeling.get("pinn_reaction_pysr_populations", 12)),
            population_size=int(modeling.get("pinn_reaction_pysr_population_size", 24)),
            maxsize=int(modeling.get("pinn_reaction_pysr_maxsize", 24)),
            parsimony=float(modeling.get("pinn_reaction_pysr_parsimony", 0.0032)),
            timeout_seconds=int(modeling.get("pinn_reaction_pysr_timeout_seconds", 180)),
            output_directory=str(pysr_cache_dir / f"member_{member}"),
            binary_operators=tuple(modeling.get("pinn_reaction_pysr_binary_operators", ["+", "-", "*", "/"])),
            unary_operators=tuple(modeling.get("pinn_reaction_pysr_unary_operators", ["square", "exp"])),
            random_seed=seed + 100,
        )
        scored_rows = score_symbolic_candidates(
            member_rows,
            dataset,
            split,
            laplacian,
            components,
            fit.rho,
            fit.alpha,
            steps_per_year,
            laplacian_normalization,
            clipped_baseline,
            clipped_observed,
        )
        for row in scored_rows:
            row["ensemble_member"] = member
            row["rho"] = fit.rho
            row["alpha"] = fit.alpha
        pareto_rows.extend(scored_rows)
        selected_row = min(scored_rows, key=lambda row: row["train_projection_mse"])
        ensemble_reports.append({"member": member, **fit.report, "selected_symbolic": selected_row})
        if best is None or selected_row["train_projection_mse"] < best["selected_row"]["train_projection_mse"]:
            best = {"fit": fit, "selected_row": selected_row, "initial_expression": expression}

    if best is None:
        raise ValueError("No PINN reaction ensemble member was fitted.")

    best_fit = best["fit"]
    symbolic_reaction = symbolic_reaction_from_expression(str(best["selected_row"]["reaction_expression"]))
    nn_model = GraphKPPReactionModel(
        laplacian,
        u0=components.u0,
        cc=components.cc,
        reaction=best_fit.reaction,
        steps_per_year=steps_per_year,
        laplacian_normalization=laplacian_normalization,
    )
    symbolic_model = GraphKPPReactionModel(
        laplacian,
        u0=components.u0,
        cc=components.cc,
        reaction=symbolic_reaction,
        steps_per_year=steps_per_year,
        laplacian_normalization=laplacian_normalization,
    )
    nn_predicted = nn_model.predict(clipped_baseline, dataset.time_years, rho=best_fit.rho, alpha=best_fit.alpha)
    symbolic_predicted = symbolic_model.predict(clipped_baseline, dataset.time_years, rho=best_fit.rho, alpha=best_fit.alpha)

    nn_pair_metrics = compute_pair_metrics(dataset.pairs, dataset.baseline, dataset.observed, nn_predicted, split, NN_MODEL_NAME)
    symbolic_pair_metrics = compute_pair_metrics(
        dataset.pairs,
        dataset.baseline,
        dataset.observed,
        symbolic_predicted,
        split,
        SYMBOLIC_MODEL_NAME,
    )
    return {
        "nn_predicted": nn_predicted,
        "symbolic_predicted": symbolic_predicted,
        "nn_pair_metrics": nn_pair_metrics,
        "symbolic_pair_metrics": symbolic_pair_metrics,
        "pareto_rows": pareto_rows,
        "symbolic_equation": str(best["selected_row"]["reaction_expression"]),
        "r2_report": build_r2_report(dataset, split, fisher_predicted, nn_predicted, symbolic_predicted),
        "fit_report": {
            "fisher_rho": fisher_fit.rho,
            "fisher_alpha": fisher_fit.alpha,
            "fisher_train_mse": fisher_fit.train_mse,
            "selected_member": int(best["selected_row"]["ensemble_member"]),
            "selected_train_projection_mse": float(best["selected_row"]["train_projection_mse"]),
            "selected_test_projection_mse": float(best["selected_row"]["test_projection_mse"]),
            "selected_symbolic_expression": str(best["selected_row"]["reaction_expression"]),
            "selected_rho": best_fit.rho,
            "selected_alpha": best_fit.alpha,
            "ensemble": ensemble_reports,
            "component_quantile": component_quantile,
            "laplacian_normalization": laplacian_normalization,
            "nn_n_likelihood_parameters": 3 + best_fit.reaction.parameter_count,
            "symbolic_n_likelihood_parameters": 5,
        },
    }


def score_symbolic_candidates(
    rows: list[dict[str, Any]],
    dataset: ForecastDataset,
    split: SubjectSplit,
    laplacian: np.ndarray,
    components: Any,
    rho: float,
    alpha: float,
    steps_per_year: int,
    laplacian_normalization: str,
    clipped_baseline: np.ndarray,
    clipped_observed: np.ndarray,
) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        reaction = symbolic_reaction_from_expression(str(row["reaction_expression"]))
        model = GraphKPPReactionModel(
            laplacian,
            u0=components.u0,
            cc=components.cc,
            reaction=reaction,
            steps_per_year=steps_per_year,
            laplacian_normalization=laplacian_normalization,
        )
        predicted = model.predict(clipped_baseline, dataset.time_years, rho=rho, alpha=alpha)
        train_mse = float(np.mean((predicted[split.train_indices] - clipped_observed[split.train_indices]) ** 2))
        test_mse = float(np.mean((predicted[split.test_indices] - clipped_observed[split.test_indices]) ** 2))
        scored.append({**row, "train_projection_mse": train_mse, "test_projection_mse": test_mse})
    return scored


def build_r2_report(
    dataset: ForecastDataset,
    split: SubjectSplit,
    fisher_predicted: np.ndarray,
    nn_predicted: np.ndarray,
    symbolic_predicted: np.ndarray,
) -> dict[str, Any]:
    return {
        "train_scalar_r2_fisher": scalar_r2(dataset.observed[split.train_indices], fisher_predicted[split.train_indices]),
        "train_scalar_r2_pinn_nn": scalar_r2(dataset.observed[split.train_indices], nn_predicted[split.train_indices]),
        "train_scalar_r2_pinn_symbolic": scalar_r2(
            dataset.observed[split.train_indices],
            symbolic_predicted[split.train_indices],
        ),
        "test_scalar_r2_fisher": scalar_r2(dataset.observed[split.test_indices], fisher_predicted[split.test_indices]),
        "test_scalar_r2_pinn_nn": scalar_r2(dataset.observed[split.test_indices], nn_predicted[split.test_indices]),
        "test_scalar_r2_pinn_symbolic": scalar_r2(dataset.observed[split.test_indices], symbolic_predicted[split.test_indices]),
    }


def scalar_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    y = observed[mask]
    yhat = predicted[mask]
    total = float(np.sum((y - np.mean(y)) ** 2))
    if total <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / total)


def load_required_matrix(path: Path, dataset: ForecastDataset) -> np.ndarray:
    labels, matrix = load_labeled_matrix(path)
    if labels != dataset.region_labels:
        raise ValueError(f"Matrix labels do not match forecast dataset region labels: {path}")
    return matrix


def print_comparison_table(rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["split"] == "test":
            by_model.setdefault(str(row["model"]), {})[str(row["metric"])] = row
    print(f"{'model':<24} {'MAE':>8} {'RMSE':>8} {'subj rho':>10} {'delta rho':>10}")
    for model in [NN_MODEL_NAME, SYMBOLIC_MODEL_NAME]:
        metrics = by_model.get(model, {})
        if metrics:
            print(
                f"{model:<24} "
                f"{float(metrics['mae']['median']):>8.4f} "
                f"{float(metrics['rmse']['median']):>8.4f} "
                f"{float(metrics['subject_spearman']['median']):>10.4f} "
                f"{float(metrics['delta_spearman']['median']):>10.4f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
