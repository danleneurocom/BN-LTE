"""End-to-end runner for BN-LTE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data import build_multimodal_pair_dataset
from .dynamic_scm import bootstrap_edge_stability, fit_dynamic_scm
from .pseudotime import fit_pseudotime
from .reporting import (
    compact_metric_summary,
    compute_rate_metrics,
    h1_decoupling_summary,
    make_subject_split,
    render_markdown_report,
    summarize_metric_rows,
    write_csv_rows,
    write_json,
)


def run_dynamic_bn_scm(
    *,
    project_root: str | Path,
    output_dir: str | Path = "outputs/bnlte",
    pseudotime_mode: str = "tau_free",
    target_prefix: str = "tau_rate:",
    max_parents_per_target: int = 8,
    bootstrap_iterations: int = 25,
    random_seed: int = 20260519,
    no_write: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    dataset = build_multimodal_pair_dataset(root)
    split = make_subject_split(dataset.metadata_rows, random_seed=random_seed)
    pseudotime = fit_pseudotime(dataset.feature_matrix, dataset.feature_names, split.train_indices, mode=pseudotime_mode)
    target_names = [name for name in dataset.target_names if name.startswith(target_prefix)] if target_prefix else list(dataset.target_names)
    if not target_names:
        raise ValueError(f"No target names matched prefix {target_prefix!r}.")

    fit = fit_dynamic_scm(
        dataset,
        pseudotime,
        split.train_indices,
        target_names=target_names,
        max_parents_per_target=max_parents_per_target,
    )
    predicted_rates = fit.predict_rates(dataset)
    target_indices = [dataset.target_index(name) for name in target_names]
    metric_rows = compute_rate_metrics(
        dataset.target_rates[:, target_indices],
        predicted_rates[:, target_indices],
        target_names,
        split,
    )
    edge_rows = fit.edge_effect_rows()
    bootstrap_rows = bootstrap_edge_stability(
        dataset,
        pseudotime,
        split.train_indices,
        target_names=target_names,
        iterations=bootstrap_iterations,
        random_seed=random_seed + 17,
        max_parents_per_target=max_parents_per_target,
        edge_effect_threshold=fit.edge_effect_threshold,
    )

    report = {
        "data": {
            **dataset.report,
            "feature_count": len(dataset.feature_names),
            "target_count": len(dataset.target_names),
        },
        "split": split.report(),
        "pseudotime": pseudotime.report(dataset.feature_matrix, dataset.metadata_rows),
        "model": fit.report(),
        "metric_summary": compact_metric_summary(metric_rows),
        "metric_rows_summary": summarize_metric_rows(metric_rows),
        "h1_decoupling": h1_decoupling_summary(edge_rows),
        "top_edges": edge_rows[:25],
        "bootstrap_top_edges": bootstrap_rows[:25],
        "configuration": {
            "pseudotime_mode": pseudotime_mode,
            "target_prefix": target_prefix,
            "max_parents_per_target": int(max_parents_per_target),
            "bootstrap_iterations": int(bootstrap_iterations),
            "random_seed": int(random_seed),
        },
    }

    if not no_write:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.mkdir(parents=True, exist_ok=True)
        write_json(output_path / "dynamic_bn_scm_report.json", report)
        write_csv_rows(output_path / "dynamic_bn_scm_rate_metrics.csv", metric_rows)
        write_csv_rows(output_path / "dynamic_bn_scm_edge_effects.csv", edge_rows)
        write_csv_rows(output_path / "dynamic_bn_scm_bootstrap_edges.csv", bootstrap_rows)
        (output_path / "dynamic_bn_scm_findings.md").write_text(render_markdown_report(report), encoding="utf-8")

    return report
