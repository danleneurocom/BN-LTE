#!/usr/bin/env python3
"""Run an iterative BN-LTE prototype with stage-by-stage findings."""

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
    SubjectTrainValidationTestSplit,
    compute_aggregate_metrics,
    compute_pair_metrics,
    load_forecast_dataset,
    make_subject_train_validation_test_split,
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.models.bn_lte import (  # noqa: E402
    bootstrap_edge_probabilities,
    fit_bn_lte_model,
    select_regions_by_train_variance,
    stable_edges_from_bootstrap,
)


HIGHER_IS_BETTER = {
    "subject_spearman",
    "subject_pearson",
    "delta_spearman",
    "delta_pearson",
    "top5_overlap",
    "top10_overlap",
}


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
    parser.add_argument("--max-regions", type=int, default=None, help="Maximum regional tau variables for this prototype.")
    parser.add_argument("--bootstrap-iterations", type=int, default=None, help="Bootstrap iterations for edge stability.")
    parser.add_argument("--no-write", action="store_true", help="Run without writing output files.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})
    evaluation = config.get("evaluation", {})
    random_seed = int(config.get("experiment", {}).get("random_seed", 20260507))

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    split = make_subject_train_validation_test_split(
        dataset.pairs,
        validation_fraction=float(modeling.get("validation_fraction", 0.2)),
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=random_seed,
    )
    pair_groups = np.asarray([pair["RID"] for pair in dataset.pairs], dtype=object)

    max_regions = int(args.max_regions if args.max_regions is not None else modeling.get("bn_lte_max_regions", 12))
    region_indices = select_regions_by_train_variance(
        dataset.baseline,
        dataset.region_labels,
        train_indices=split.train_indices,
        max_regions=max_regions,
    )
    selected_labels = [dataset.region_labels[int(index)] for index in region_indices]
    baseline = dataset.baseline[:, region_indices]
    observed = dataset.observed[:, region_indices]

    root_names, root_values, root_report = build_root_covariates(dataset.pairs)
    ridge_alphas = tuple(float(value) for value in modeling.get("bn_lte_ridge_alphas", [1.0, 10.0, 100.0, 1000.0, 10000.0]))
    n_knots = int(modeling.get("bn_lte_spline_knots", 4))
    spline_degree = int(modeling.get("bn_lte_spline_degree", 3))
    max_parents = int(modeling.get("bn_lte_max_parents_per_child", 5))
    edge_threshold = float(modeling.get("bn_lte_edge_effect_threshold", 0.01))
    pip_threshold = float(modeling.get("bn_lte_pip_threshold", 0.5))
    bootstrap_iterations = int(
        args.bootstrap_iterations
        if args.bootstrap_iterations is not None
        else modeling.get("bn_lte_bootstrap_iterations", 50)
    )
    primary_metric = str(evaluation.get("primary_metric", "subject_spearman"))
    selection_stat = str(modeling.get("selection_stat", "median"))

    stage_results: list[dict[str, Any]] = []
    all_pair_metrics: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []

    persistence_predicted = baseline.copy()
    stage_results.append(
        evaluate_stage(
            name="00_persistence",
            description="No causal model: predict follow-up regional tau as the baseline regional tau.",
            dataset=dataset,
            baseline=baseline,
            observed=observed,
            predicted=persistence_predicted,
            split=split,
            primary_metric=primary_metric,
            selection_stat=selection_stat,
            previous_best=None,
            limitation="Reference floor only; it cannot test the proposal's time-varying causal claims.",
        )
    )
    all_pair_metrics.extend(stage_results[-1]["pair_metrics"])
    previous_best = stage_results[-1]

    stage_specs = [
        {
            "name": "01_pseudotime_self_history",
            "description": (
                "Train-only latent pseudotime with smooth baseline trajectories and autonomous regional "
                "self-history terms; no cross-region parents yet."
            ),
            "parent_mode": "none",
            "include_roots": False,
            "allowed_edges": None,
            "fix_if_not_improved": (
                "If this fails, pseudotime alone is not adding longitudinal signal beyond persistence; "
                "the next step tests whether ordered regional parents supply missing mechanism."
            ),
        },
        {
            "name": "02_progression_ordered_regional_edges",
            "description": (
                "Adds regional parent effects modulated by cubic splines over pseudotime. Parents are "
                "restricted to earlier, higher-burden regions to keep the graph acyclic and biologically oriented."
            ),
            "parent_mode": "progression_ordered",
            "include_roots": False,
            "allowed_edges": None,
            "fix_if_not_improved": (
                "If this fails, the extra regional edges are likely too noisy or too flexible; the later "
                "bootstrap stage will prune unstable edges rather than accepting all fitted parents."
            ),
        },
        {
            "name": "03_apoe_root_edges",
            "description": (
                "Adds available disease-agnostic APOE e4 dosage as a root covariate. The root can point "
                "into tau rates but cannot be predicted by tau variables."
            ),
            "parent_mode": "progression_ordered",
            "include_roots": bool(root_names),
            "allowed_edges": None,
            "fix_if_not_improved": (
                "If this fails, APOE is not improving held-out regional tau prediction in this restricted "
                "prototype; it remains useful as a causal prior in the full multimodal model."
            ),
        },
    ]

    fitted_for_bootstrap = None
    for spec in stage_specs:
        fit, result, edge_rows = run_bn_lte_stage(
            spec=spec,
            dataset=dataset,
            split=split,
            baseline=baseline,
            observed=observed,
            selected_labels=selected_labels,
            pair_groups=pair_groups,
            root_names=root_names,
            root_values=root_values,
            ridge_alphas=ridge_alphas,
            n_knots=n_knots,
            spline_degree=spline_degree,
            max_parents=max_parents,
            edge_threshold=edge_threshold,
            primary_metric=primary_metric,
            selection_stat=selection_stat,
            previous_best=previous_best,
        )
        if spec["name"] == "03_apoe_root_edges":
            fitted_for_bootstrap = fit
        stage_results.append(result)
        all_pair_metrics.extend(result["pair_metrics"])
        all_edge_rows.extend(edge_rows)
        if result["adopted_as_best"]:
            previous_best = result

    if fitted_for_bootstrap is not None and bootstrap_iterations > 0:
        bootstrap_rows = bootstrap_edge_probabilities(
            baseline=baseline,
            observed=observed,
            time_years=dataset.time_years,
            region_labels=selected_labels,
            train_indices=split.train_indices,
            pair_groups=pair_groups,
            iterations=bootstrap_iterations,
            random_seed=random_seed + 101,
            model_name="04_bootstrap_stability",
            parent_mode="progression_ordered",
            include_roots=bool(root_names),
            root_names=root_names,
            root_values=root_values,
            include_self_history=True,
            max_parents_per_child=max_parents,
            n_knots=n_knots,
            spline_degree=spline_degree,
            ridge_alphas=ridge_alphas,
            edge_effect_threshold=edge_threshold,
        )
        stable_edges = stable_edges_from_bootstrap(bootstrap_rows, pip_threshold=pip_threshold)
        if stable_edges:
            spec = {
                "name": "04_bootstrap_pruned_edges",
                "description": (
                    f"Refits the BN-LTE model using only regional edges with bootstrap inclusion probability "
                    f">= {pip_threshold:.2f}; root covariates remain available."
                ),
                "parent_mode": "progression_ordered",
                "include_roots": bool(root_names),
                "allowed_edges": stable_edges,
                "fix_if_not_improved": (
                    "If this still fails, the stable-edge graph is interpretable but should not replace the "
                    "best predictive stage for forecasting."
                ),
            }
            _, result, edge_rows = run_bn_lte_stage(
                spec=spec,
                dataset=dataset,
                split=split,
                baseline=baseline,
                observed=observed,
                selected_labels=selected_labels,
                pair_groups=pair_groups,
                root_names=root_names,
                root_values=root_values,
                ridge_alphas=ridge_alphas,
                n_knots=n_knots,
                spline_degree=spline_degree,
                max_parents=max_parents,
                edge_threshold=edge_threshold,
                primary_metric=primary_metric,
                selection_stat=selection_stat,
                previous_best=previous_best,
            )
            stage_results.append(result)
            all_pair_metrics.extend(result["pair_metrics"])
            all_edge_rows.extend(edge_rows)
            if result["adopted_as_best"]:
                previous_best = result
        else:
            skipped = skipped_stage(
                "04_bootstrap_pruned_edges",
                "No regional edges reached the bootstrap stability threshold, so pruning was not refit.",
                previous_best,
            )
            stage_results.append(skipped)

    metrics_summary = compute_aggregate_metrics(all_pair_metrics)
    report = {
        "proposal_alignment": {
            "implemented_scope": "ADNI regional tau BN-LTE prototype",
            "not_yet_implemented": [
                "full ADNI multimodal fluid/PET/MRI/cognition feature matrix",
                "UK Biobank replication and accelerometry/perfusion gates",
                "joint Bayesian MCMC over graph, pseudotime, and spline coefficients",
                "1,000-bootstrap production stability run",
            ],
            "reason": "The current repository contains longitudinal ADNI tau forecasting tables, not the complete proposed biobank matrix.",
        },
        "configuration": {
            "max_regions": int(max_regions),
            "selected_regions": selected_labels,
            "spline_knots": n_knots,
            "spline_degree": spline_degree,
            "max_parents_per_child": max_parents,
            "edge_effect_threshold": edge_threshold,
            "bootstrap_iterations": bootstrap_iterations,
            "pip_threshold": pip_threshold,
            "primary_metric": primary_metric,
            "selection_stat": selection_stat,
            "root_covariates": root_report,
        },
        "split": split_report(split),
        "stages": public_stage_rows(stage_results),
        "best_stage": {
            "name": previous_best["name"],
            "validation_score": previous_best["validation_score"],
            "test_score": previous_best["test_score"],
        },
    }

    markdown = render_markdown_findings(report, stage_results, bootstrap_rows)
    print(markdown)

    if not args.no_write:
        pair_metrics_path = output_dir / outputs.get("bn_lte_pair_metrics", "bn_lte_iteration_pair_metrics.csv")
        metrics_summary_path = output_dir / outputs.get("bn_lte_metrics_summary", "bn_lte_iteration_metrics_summary.csv")
        edge_effects_path = output_dir / outputs.get("bn_lte_edge_effects", "bn_lte_edge_effects.csv")
        bootstrap_path = output_dir / outputs.get("bn_lte_bootstrap_edges", "bn_lte_bootstrap_edges.csv")
        report_path = output_dir / outputs.get("bn_lte_report", "bn_lte_iteration_report.json")
        findings_path = output_dir / outputs.get("bn_lte_findings", "bn_lte_iteration_findings.md")

        write_csv_rows(pair_metrics_path, all_pair_metrics)
        write_csv_rows(metrics_summary_path, metrics_summary)
        write_csv_rows(edge_effects_path, all_edge_rows)
        write_csv_rows(bootstrap_path, bootstrap_rows)
        write_json(report_path, report)
        findings_path.parent.mkdir(parents=True, exist_ok=True)
        findings_path.write_text(markdown, encoding="utf-8")

        print("\nWrote BN-LTE iterative outputs:")
        print(f"pair_metrics: {pair_metrics_path}")
        print(f"metrics_summary: {metrics_summary_path}")
        print(f"edge_effects: {edge_effects_path}")
        print(f"bootstrap_edges: {bootstrap_path}")
        print(f"report: {report_path}")
        print(f"findings: {findings_path}")

    return 0


def run_bn_lte_stage(
    *,
    spec: dict[str, Any],
    dataset: ForecastDataset,
    split: SubjectTrainValidationTestSplit,
    baseline: np.ndarray,
    observed: np.ndarray,
    selected_labels: list[str],
    pair_groups: np.ndarray,
    root_names: list[str],
    root_values: np.ndarray,
    ridge_alphas: tuple[float, ...],
    n_knots: int,
    spline_degree: int,
    max_parents: int,
    edge_threshold: float,
    primary_metric: str,
    selection_stat: str,
    previous_best: dict[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    fit = fit_bn_lte_model(
        baseline=baseline,
        observed=observed,
        time_years=dataset.time_years,
        region_labels=selected_labels,
        train_indices=split.train_indices,
        pair_groups=pair_groups,
        model_name=spec["name"],
        parent_mode=spec["parent_mode"],
        include_roots=bool(spec["include_roots"]),
        root_names=root_names,
        root_values=root_values,
        allowed_edges=spec.get("allowed_edges"),
        include_self_history=True,
        max_parents_per_child=max_parents,
        n_knots=n_knots,
        spline_degree=spline_degree,
        ridge_alphas=ridge_alphas,
        cv_folds=5,
        edge_effect_threshold=edge_threshold,
    )
    predicted = fit.predict(baseline, dataset.time_years, root_values=root_values if spec["include_roots"] else None)
    result = evaluate_stage(
        name=spec["name"],
        description=spec["description"],
        dataset=dataset,
        baseline=baseline,
        observed=observed,
        predicted=predicted,
        split=split,
        primary_metric=primary_metric,
        selection_stat=selection_stat,
        previous_best=previous_best,
        limitation=spec["fix_if_not_improved"],
        model_report=fit.report(),
    )
    edge_rows = fit.edge_rows()
    for row in edge_rows:
        row["stage"] = spec["name"]
    return fit, result, edge_rows


def evaluate_stage(
    *,
    name: str,
    description: str,
    dataset: ForecastDataset,
    baseline: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    split: SubjectTrainValidationTestSplit,
    primary_metric: str,
    selection_stat: str,
    previous_best: dict[str, Any] | None,
    limitation: str,
    model_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pair_metrics = compute_pair_metrics(dataset.pairs, baseline, observed, predicted, split, name)
    summary = compute_aggregate_metrics(pair_metrics)
    validation_score = metric_value(summary, "validation", primary_metric, selection_stat)
    test_score = metric_value(summary, "test", primary_metric, selection_stat)
    adopted = previous_best is None or is_better(validation_score, previous_best["validation_score"], primary_metric)
    delta_vs_previous = None if previous_best is None else safe_delta(validation_score, previous_best["validation_score"])
    finding = stage_finding(
        name=name,
        adopted=adopted,
        validation_score=validation_score,
        test_score=test_score,
        previous_best=previous_best,
        primary_metric=primary_metric,
        selection_stat=selection_stat,
        limitation=limitation,
    )
    return {
        "name": name,
        "description": description,
        "validation_score": validation_score,
        "test_score": test_score,
        "delta_vs_previous_best": delta_vs_previous,
        "adopted_as_best": adopted,
        "finding": finding,
        "pair_metrics": pair_metrics,
        "metrics_summary": summary,
        "model_report": model_report or {},
    }


def skipped_stage(name: str, reason: str, previous_best: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": reason,
        "validation_score": float("nan"),
        "test_score": float("nan"),
        "delta_vs_previous_best": float("nan"),
        "adopted_as_best": False,
        "finding": f"{name}: skipped. {reason}",
        "pair_metrics": [],
        "metrics_summary": [],
        "model_report": {"skipped": True, "previous_best": previous_best["name"]},
    }


def build_root_covariates(pairs: list[dict[str, str]]) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    apoe_values = np.asarray([parse_apoe_e4_dosage(pair.get("apoe_genotype", "")) for pair in pairs], dtype=float)
    finite = np.isfinite(apoe_values)
    report = {
        "candidate": "apoe_e4_dosage",
        "available_rows": int(np.sum(finite)),
        "total_rows": len(pairs),
        "used": False,
    }
    if int(np.sum(finite)) < 10 or np.nanstd(apoe_values) <= 1.0e-12:
        return [], np.zeros((len(pairs), 0), dtype=float), report
    imputed = apoe_values.copy()
    imputed[~finite] = float(np.nanmean(apoe_values))
    report.update(
        {
            "used": True,
            "imputed_rows": int(np.sum(~finite)),
            "mean": float(np.mean(imputed)),
            "std": float(np.std(imputed)),
        }
    )
    return ["apoe_e4_dosage"], imputed.reshape(-1, 1), report


def parse_apoe_e4_dosage(value: str) -> float:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return float("nan")
    alleles = [token for token in text.replace("|", "/").split("/") if token]
    if not alleles:
        return float("nan")
    return float(sum(1 for allele in alleles if allele.strip() == "4"))


def metric_value(summary: list[dict[str, Any]], split: str, metric: str, stat: str) -> float:
    for row in summary:
        if row["split"] == split and row["metric"] == metric:
            return float(row[stat])
    return float("nan")


def is_better(candidate: float, incumbent: float, metric: str) -> bool:
    if not np.isfinite(candidate):
        return False
    if not np.isfinite(incumbent):
        return True
    if metric in HIGHER_IS_BETTER:
        return candidate > incumbent
    return candidate < incumbent


def safe_delta(candidate: float, incumbent: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(incumbent):
        return float("nan")
    return float(candidate - incumbent)


def stage_finding(
    *,
    name: str,
    adopted: bool,
    validation_score: float,
    test_score: float,
    previous_best: dict[str, Any] | None,
    primary_metric: str,
    selection_stat: str,
    limitation: str,
) -> str:
    score_text = f"validation {primary_metric} {selection_stat}={format_float(validation_score)}, test={format_float(test_score)}"
    if previous_best is None:
        return f"{name}: established the baseline ({score_text}). {limitation}"
    if adopted:
        return (
            f"{name}: improved on the previous best stage {previous_best['name']} "
            f"({score_text}) and is adopted for the next iteration."
        )
    return (
        f"{name}: did not improve on {previous_best['name']} "
        f"({score_text}; previous validation={format_float(previous_best['validation_score'])}). {limitation}"
    )


def split_report(split: SubjectTrainValidationTestSplit) -> dict[str, Any]:
    return {
        "train_pairs": int(split.train_indices.size),
        "validation_pairs": int(split.validation_indices.size),
        "test_pairs": int(split.test_indices.size),
        "train_subjects": len(split.train_rids),
        "validation_subjects": len(split.validation_rids),
        "test_subjects": len(split.test_rids),
    }


def public_stage_rows(stage_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": result["name"],
            "description": result["description"],
            "validation_score": result["validation_score"],
            "test_score": result["test_score"],
            "delta_vs_previous_best": result["delta_vs_previous_best"],
            "adopted_as_best": result["adopted_as_best"],
            "finding": result["finding"],
            "model_report": result["model_report"],
        }
        for result in stage_results
    ]


def render_markdown_findings(
    report: dict[str, Any],
    stage_results: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
) -> str:
    config = report["configuration"]
    lines = [
        "# BN-LTE Iterative Research Findings",
        "",
        "## Scope",
        "",
        "This is an ADNI regional tau prototype of the proposal, not the full ADNI + UKB multimodal BN-LTE.",
        "The implementation uses train-only pseudotime, spline-modulated effects, progression-ordered regional parents, APOE root covariates when available, and bootstrap edge stability.",
        "",
        "## Configuration",
        "",
        f"- Selected regions: {config['max_regions']} requested, {len(config['selected_regions'])} used.",
        f"- Primary validation rule: {config['primary_metric']} using {config['selection_stat']}.",
        f"- Spline knots: {config['spline_knots']}; max parents per child: {config['max_parents_per_child']}.",
        f"- Bootstrap iterations: {config['bootstrap_iterations']}; PIP threshold: {config['pip_threshold']}.",
        f"- Root covariates: {json.dumps(config['root_covariates'], sort_keys=True)}.",
        "",
        "## Stage Results",
        "",
        "| Stage | Adopted | Validation | Test | Finding |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for result in stage_results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result["name"],
                    "yes" if result["adopted_as_best"] else "no",
                    format_float(result["validation_score"]),
                    format_float(result["test_score"]),
                    result["finding"].replace("|", "/"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Best Stage",
            "",
            f"Best validation stage: `{report['best_stage']['name']}`.",
            f"Validation score: {format_float(report['best_stage']['validation_score'])}; test score: {format_float(report['best_stage']['test_score'])}.",
            "",
            "## Bootstrap Edge Stability",
            "",
        ]
    )
    if bootstrap_rows:
        top_rows = sorted(bootstrap_rows, key=lambda row: float(row["bootstrap_inclusion_probability"]), reverse=True)[:10]
        lines.extend(["Top bootstrap-stable edges:", ""])
        for row in top_rows:
            lines.append(
                f"- {row['parent']} -> {row['child']}: PIP proxy={float(row['bootstrap_inclusion_probability']):.3f}, "
                f"mean max effect={float(row['mean_max_abs_effect']):.5f}"
            )
    else:
        lines.append("Bootstrap stability was not run or produced no edge rows.")
    lines.extend(
        [
            "",
            "## Data Gates Before Full Proposal",
            "",
            "- Add fluid biomarkers, amyloid PET, MRI volumes, cognition, demographics, and scan/site harmonization fields to the feature matrix.",
            "- Add UK Biobank replication features, especially accelerometry and vascular/perfusion measures, before testing the vascular-glymphatic hypothesis.",
            "- Replace this bootstrap empirical-Bayes approximation with joint MCMC only after the feature matrix and constraints are locked.",
            "- Treat validation failure as evidence against adopting a step, not as a reason to manually tune on the test split.",
            "",
        ]
    )
    return "\n".join(lines)


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
