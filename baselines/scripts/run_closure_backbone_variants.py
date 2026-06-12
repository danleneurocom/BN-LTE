#!/usr/bin/env python3
"""Fit Bayesian closure variants on top of global FKPP and ESM backbones."""

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
from spread_toolbox.models.closure import (  # noqa: E402
    apply_closure_delta,
    build_closure_feature_library,
    fit_horseshoe_linear_closure,
)
from spread_toolbox.models.esm import EpidemicSpreadingModel  # noqa: E402
from spread_toolbox.models.fkpp import GraphFKPPModel  # noqa: E402


BACKBONE_TO_MODEL = {
    "global_fkpp": "global_fkpp_horseshoe_closure",
    "esm": "esm_horseshoe_closure",
}


def default_config_path() -> Path:
    experiment_dir = PROJECT_ROOT / "experiments" / "group_average_enigma"
    local_config = experiment_dir / "config.yaml"
    if local_config.exists():
        return local_config
    return experiment_dir / "config.example.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=sorted(BACKBONE_TO_MODEL),
        default=None,
        help="Closure backbones to run. Defaults to config modeling.closure_variant_backbones.",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})
    backbones = args.backbones or list(modeling.get("closure_variant_backbones", ["global_fkpp", "esm"]))

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    adjacency = load_required_matrix(output_dir / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv"), dataset)
    laplacian = load_required_matrix(output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"), dataset)
    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=int(config.get("experiment", {}).get("random_seed", 20260507)),
    )
    pair_covariates, regional_covariates, covariate_report = build_closure_covariates(
        dataset,
        split,
        config,
        PROJECT_ROOT,
    )

    results = []
    for backbone in backbones:
        print(f"\nFitting {BACKBONE_TO_MODEL[backbone]}...")
        results.append(
            run_backbone_closure(
                backbone,
                dataset,
                split,
                adjacency=adjacency,
                laplacian=laplacian,
                pair_covariates=pair_covariates,
                regional_covariates=regional_covariates,
                covariate_report=covariate_report,
                modeling=modeling,
                config=config,
            )
        )

    baseline_summary = load_optional_rows(
        output_dir / outputs.get("baseline_comparison_metrics_summary", "baseline_comparison_metrics_summary.csv")
    )
    baseline_likelihood = load_optional_rows(
        output_dir / outputs.get("baseline_comparison_likelihood_metrics", "baseline_comparison_likelihood_metrics.csv")
    )
    local_closure_summary = load_optional_rows(output_dir / outputs.get("closure_metrics_summary", "closure_metrics_summary.csv"))
    local_closure_likelihood = load_optional_rows(
        output_dir / outputs.get("closure_likelihood_metrics", "closure_likelihood_metrics.csv")
    )

    print(json.dumps(split_summary(split), indent=2, sort_keys=True))
    for result in results:
        model_name = result["model"]
        metrics_summary = compute_aggregate_metrics(result["pair_metrics"])
        likelihood_metrics = compute_gaussian_likelihood_metrics(
            dataset.observed,
            result["predicted"],
            split,
            model_name,
            n_parameters=int(result["fit_report"]["n_likelihood_parameters"]),
            min_sigma=float(modeling.get("likelihood_min_sigma", 1.0e-6)),
        )
        comparison_rows = test_comparison_rows(baseline_summary + local_closure_summary + metrics_summary)
        print(f"\n{model_name} test comparison:")
        print_comparison_table(comparison_rows, extra_model=model_name)
        print("\nR2 report:")
        print(json.dumps(result["r2_report"], indent=2, sort_keys=True))
        print("\nClosure terms:")
        print_terms(result["term_rows"])

        if not args.no_write:
            prefix = output_prefix_for_model(model_name)
            write_csv_rows(output_dir / outputs.get(f"{prefix}_pair_metrics", f"{prefix}_pair_metrics.csv"), result["pair_metrics"])
            write_csv_rows(
                output_dir / outputs.get(f"{prefix}_metrics_summary", f"{prefix}_metrics_summary.csv"),
                metrics_summary,
            )
            write_csv_rows(
                output_dir / outputs.get(f"{prefix}_likelihood_metrics", f"{prefix}_likelihood_metrics.csv"),
                likelihood_metrics,
            )
            write_csv_rows(output_dir / outputs.get(f"{prefix}_terms", f"{prefix}_terms.csv"), result["term_rows"])
            write_json(
                output_dir / outputs.get(f"{prefix}_report", f"{prefix}_report.json"),
                {
                    "model": model_name,
                    "purpose": f"Bayesian feature-conditioned closure on top of {result['fit_report']['backbone']} backbone.",
                    "split": split_summary(split),
                    "fit_report": result["fit_report"],
                    "r2_report": result["r2_report"],
                    "test_comparison": comparison_rows,
                    "likelihood_comparison": baseline_likelihood + local_closure_likelihood + likelihood_metrics,
                },
            )

    return 0


def load_required_matrix(path: Path, dataset: ForecastDataset) -> np.ndarray:
    labels, matrix = load_labeled_matrix(path)
    if labels != dataset.region_labels:
        raise ValueError(f"Matrix labels do not match forecast dataset region labels: {path}")
    return matrix


def run_backbone_closure(
    backbone: str,
    dataset: ForecastDataset,
    split: SubjectSplit,
    *,
    adjacency: np.ndarray,
    laplacian: np.ndarray,
    pair_covariates: dict[str, np.ndarray],
    regional_covariates: dict[str, np.ndarray],
    covariate_report: dict[str, Any],
    modeling: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    random_seed = int(config.get("experiment", {}).get("random_seed", 20260507))
    model_name = BACKBONE_TO_MODEL[backbone]
    scaler = MinMaxStateScaler.fit(dataset.baseline[split.train_indices], dataset.observed[split.train_indices])
    baseline_scaled = scaler.transform(dataset.baseline)
    observed_scaled = scaler.transform(dataset.observed)
    region_count = dataset.observed.shape[1]
    u0 = np.zeros(region_count, dtype=float)
    cc = np.ones(region_count, dtype=float)

    if backbone == "global_fkpp":
        backbone_predicted_scaled, fit_report = fit_global_fkpp_backbone(
            baseline_scaled,
            observed_scaled,
            dataset.time_years,
            split,
            laplacian,
            modeling,
        )
        n_backbone_parameters = 2
    elif backbone == "esm":
        backbone_predicted_scaled, fit_report = fit_esm_backbone(
            baseline_scaled,
            observed_scaled,
            dataset.time_years,
            split,
            adjacency,
            modeling,
        )
        n_backbone_parameters = 1
    else:
        raise ValueError(f"Unsupported closure backbone: {backbone}")

    feature_library = build_closure_feature_library(
        baseline_scaled,
        u0=u0,
        cc=cc,
        pair_covariates=pair_covariates,
        regional_covariates=regional_covariates,
    )
    residual_rate = (observed_scaled - backbone_predicted_scaled) / np.maximum(dataset.time_years, 1.0e-6)[:, None]
    closure = fit_horseshoe_linear_closure(
        feature_library,
        residual_rate,
        row_indices=split.train_indices,
        draws=int(modeling.get("closure_draws", 200)),
        tune=int(modeling.get("closure_tune", 300)),
        chains=int(modeling.get("closure_chains", 2)),
        target_accept=float(modeling.get("closure_target_accept", 0.95)),
        random_seed=random_seed + (101 if backbone == "global_fkpp" else 202),
        inclusion_threshold=float(modeling.get("closure_inclusion_threshold", 1.0e-4)),
        max_train_rows=int(modeling.get("closure_max_train_rows", 10000)),
    )
    closure_rate = closure.predict_rate(feature_library.values)
    predicted_scaled = apply_closure_delta(
        backbone_predicted_scaled,
        dataset.time_years,
        closure_rate,
        u0=u0,
        cc=cc,
    )
    predicted = scaler.inverse_transform(predicted_scaled)
    backbone_predicted = scaler.inverse_transform(backbone_predicted_scaled)

    pair_metrics = compute_pair_metrics(dataset.pairs, dataset.baseline, dataset.observed, predicted, split, model_name)
    term_rows = closure.term_rows()
    selected_count = int(sum(row["selected"] for row in term_rows if row["term"] != "1"))
    n_likelihood_parameters = n_backbone_parameters + 1 + 1 + selected_count
    full_report = {
        "model": model_name,
        "backbone": backbone,
        "state_space": "per-region train min-max scaled tau",
        **fit_report,
        "closure_train_mse_rate_scaled": closure.train_mse_rate,
        "closure_train_r2_rate_scaled": closure.train_r2_rate,
        "closure_feature_count": len(closure.feature_names),
        "closure_selected_features_pip_ge_0_95": selected_count,
        "closure_inclusion_threshold": closure.inclusion_threshold,
        "closure_sampling": closure.sampling_report,
        "covariates": covariate_report,
        "n_likelihood_parameters": n_likelihood_parameters,
    }
    return {
        "model": model_name,
        "predicted": predicted,
        "backbone_predicted": backbone_predicted,
        "pair_metrics": pair_metrics,
        "term_rows": term_rows,
        "r2_report": build_r2_report(dataset, split, backbone_predicted, predicted),
        "fit_report": full_report,
    }


def fit_global_fkpp_backbone(
    baseline_scaled: np.ndarray,
    observed_scaled: np.ndarray,
    time_years: np.ndarray,
    split: SubjectSplit,
    laplacian: np.ndarray,
    modeling: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    parameter_bounds = modeling.get("parameter_bounds", {})
    rho_bounds = tuple(float(value) for value in parameter_bounds.get("rho", [0.0, 10.0]))
    alpha_bounds = tuple(float(value) for value in parameter_bounds.get("alpha", [0.0, 10.0]))
    model = GraphFKPPModel(
        laplacian,
        steps_per_year=int(modeling.get("closure_fkpp_steps_per_year", modeling.get("fkpp_steps_per_year", 12))),
        laplacian_normalization=str(modeling.get("closure_fkpp_laplacian_normalization", "spectral")),
    )
    fit = model.fit_global_parameters(
        baseline_scaled[split.train_indices],
        observed_scaled[split.train_indices],
        time_years[split.train_indices],
        rho_bounds=(rho_bounds[0], rho_bounds[1]),
        alpha_bounds=(alpha_bounds[0], alpha_bounds[1]),
        maxiter=int(modeling.get("closure_fkpp_optimizer_maxiter", modeling.get("fkpp_optimizer_maxiter", 80))),
    )
    return (
        model.predict(baseline_scaled, time_years, rho=fit.rho, alpha=fit.alpha),
        {
            "backbone_rho": fit.rho,
            "backbone_alpha": fit.alpha,
            "backbone_train_mse_scaled": fit.train_mse,
            "laplacian_normalization": model.laplacian_normalization,
            "laplacian_scale": model.laplacian_scale,
        },
    )


def fit_esm_backbone(
    baseline_scaled: np.ndarray,
    observed_scaled: np.ndarray,
    time_years: np.ndarray,
    split: SubjectSplit,
    adjacency: np.ndarray,
    modeling: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    beta_bounds = tuple(float(value) for value in modeling.get("parameter_bounds", {}).get("beta", [0.0, 10.0]))
    model = EpidemicSpreadingModel(adjacency, steps_per_year=int(modeling.get("closure_esm_steps_per_year", modeling.get("esm_steps_per_year", 12))))
    fit = model.fit_global_beta(
        baseline_scaled[split.train_indices],
        observed_scaled[split.train_indices],
        time_years[split.train_indices],
        bounds=(beta_bounds[0], beta_bounds[1]),
    )
    return (
        model.predict(baseline_scaled, time_years, fit.beta),
        {
            "backbone_beta": fit.beta,
            "backbone_train_mse_scaled": fit.train_mse,
            "steps_per_year": model.steps_per_year,
        },
    )


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
    backbone_test = scalar_r2(dataset.observed[test], backbone_predicted[test])
    closure_test = scalar_r2(dataset.observed[test], closure_predicted[test])
    return {
        "train_scalar_r2_backbone": scalar_r2(dataset.observed[train], backbone_predicted[train]),
        "train_scalar_r2_closure": scalar_r2(dataset.observed[train], closure_predicted[train]),
        "test_scalar_r2_backbone": backbone_test,
        "test_scalar_r2_closure": closure_test,
        "test_scalar_r2_gain": closure_test - backbone_test,
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
    return [row for row in metrics_summary if row["split"] == "test" and row["metric"] in wanted_metrics]


def print_comparison_table(rows: list[dict[str, Any]], *, extra_model: str) -> None:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), {})[str(row["metric"])] = row
    print(
        f"{'model':<34} {'MAE med':>9} {'RMSE med':>9} {'subject Spearman med':>21} "
        f"{'delta Spearman med':>19} {'top5 med':>9} {'top10 med':>10}"
    )
    for model in ["ndm", "esm", "global_fkpp", "local_fkpp", "local_fkpp_horseshoe_closure", extra_model]:
        metrics = by_model.get(model, {})
        if not metrics:
            continue
        print(
            f"{model:<34} "
            f"{metric_value(metrics, 'mae', 'median'):>9.4f} "
            f"{metric_value(metrics, 'rmse', 'median'):>9.4f} "
            f"{metric_value(metrics, 'subject_spearman', 'median'):>21.4f} "
            f"{metric_value(metrics, 'delta_spearman', 'median'):>19.4f} "
            f"{metric_value(metrics, 'top5_overlap', 'median'):>9.4f} "
            f"{metric_value(metrics, 'top10_overlap', 'median'):>10.4f}"
        )


def print_terms(rows: list[dict[str, Any]], limit: int = 10) -> None:
    rows = sorted(rows, key=lambda row: abs(float(row["coefficient_mean"])), reverse=True)
    for row in rows[:limit]:
        print(f"{float(row['coefficient_mean']): .6f} [PIP={float(row['inclusion_probability']):.3f}] * {row['term']}")
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more terms")


def metric_value(metrics: dict[str, dict[str, Any]], metric: str, field: str) -> float:
    return float(metrics[metric][field])


def split_summary(split: SubjectSplit) -> dict[str, int]:
    return {
        "train_pairs": int(split.train_indices.size),
        "test_pairs": int(split.test_indices.size),
        "train_subjects": len(split.train_rids),
        "test_subjects": len(split.test_rids),
    }


def output_prefix_for_model(model_name: str) -> str:
    return model_name.replace("_horseshoe_closure", "_closure")


if __name__ == "__main__":
    raise SystemExit(main())
