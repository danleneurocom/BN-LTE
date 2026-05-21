#!/usr/bin/env python3
"""Create one notebook containing all paper-validation results and figures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "paper_validation"
FIG = OUT / "figures"
NOTEBOOK = HERE / "bn_lte_paper_validation_all_results.ipynb"


TABLES = {
    "primary_summary": OUT / "primary_summary.csv",
    "primary_pair_metrics": OUT / "primary_pair_metrics.csv",
    "repeated_split_summary": OUT / "repeated_split_summary.csv",
    "repeated_split_metrics": OUT / "repeated_split_metrics.csv",
    "ablation_summary": OUT / "ablation_summary.csv",
    "ablation_metrics": OUT / "ablation_metrics.csv",
    "stage_stratified_metrics": OUT / "stage_stratified_metrics.csv",
    "bootstrap_stage_edge_stability": OUT / "bootstrap_stage_edge_stability.csv",
    "counterfactual_effects": OUT / "counterfactual_effects.csv",
    "dynamic_graph_stage_edges": OUT / "dynamic_graph_stage_edges.csv",
    "edge_curve_grid": OUT / "edge_curve_grid.csv",
}

FIGURES = [
    ("Figure 1. Repeated subject-level validation", FIG / "fig1_repeated_validation.png"),
    ("Figure 2. Ablation and negative-control experiments", FIG / "fig2_ablation_controls.png"),
    ("Figure 3. Stage-stratified performance", FIG / "fig3_stage_stratified_performance.png"),
    ("Figure 4. Stage-varying direct edge effects", FIG / "fig4_dynamic_edge_heatmap.png"),
    ("Figure 5. Dynamic causal graph snapshots", FIG / "fig5_dynamic_graph_early_mid_late.png"),
    ("Figure 6. Counterfactual response windows", FIG / "fig6_counterfactual_windows.png"),
    ("Figure 7. Multi-view tau burden: baseline, empirical follow-up, and model predictions", FIG / "fig7_brain_multiview_tau_burden.png"),
    ("Figure 8. Multi-view model error against empirical follow-up", FIG / "fig8_brain_multiview_prediction_error.png"),
    ("Figure 9. BN-LTE stage cascade against empirical data", FIG / "fig9_brain_stage_cascade_bn_lte.png"),
]


def main() -> int:
    missing = [str(path) for path in [*TABLES.values(), *[path for _, path in FIGURES]] if not path.exists()]
    if missing:
        raise SystemExit("Missing required paper-validation artifacts:\n" + "\n".join(missing))

    cells = [
        md(
            """# BN-LTE Paper Validation: All Results Notebook

This notebook consolidates every output from the paper-validation run:

- all experiment CSV tables,
- repeated validation summaries,
- ablation and negative-control results,
- stage-stratified performance,
- bootstrap edge stability,
- counterfactual SCM probes,
- dynamic graph tables,
- and all paper figures, including multi-view brain-surface visualizations.

The notebook is intentionally reproducible. The figures are embedded as markdown images, and the full tables are loaded from the generated CSV files so the source data remain traceable.
"""
        ),
        md("## 1. Setup"),
        code(
            """from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, HTML, Markdown, Image

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

candidates = [
    Path("outputs/paper_validation"),
    Path("BayesianNetwork-SCM/outputs/paper_validation"),
]
BASE = next((path for path in candidates if path.exists()), None)
if BASE is None:
    raise FileNotFoundError("Could not find outputs/paper_validation from the current notebook working directory.")
FIG = BASE / "figures"
REPORT = BASE / "paper_validation_report.json"

TABLE_PATHS = {
    "primary_summary": BASE / "primary_summary.csv",
    "primary_pair_metrics": BASE / "primary_pair_metrics.csv",
    "repeated_split_summary": BASE / "repeated_split_summary.csv",
    "repeated_split_metrics": BASE / "repeated_split_metrics.csv",
    "ablation_summary": BASE / "ablation_summary.csv",
    "ablation_metrics": BASE / "ablation_metrics.csv",
    "stage_stratified_metrics": BASE / "stage_stratified_metrics.csv",
    "bootstrap_stage_edge_stability": BASE / "bootstrap_stage_edge_stability.csv",
    "counterfactual_effects": BASE / "counterfactual_effects.csv",
    "dynamic_graph_stage_edges": BASE / "dynamic_graph_stage_edges.csv",
    "edge_curve_grid": BASE / "edge_curve_grid.csv",
}

tables = {name: pd.read_csv(path) for name, path in TABLE_PATHS.items()}
with REPORT.open("r", encoding="utf-8") as handle:
    report = json.load(handle)

def scroll_table(df, height=650):
    html = df.to_html(index=False, max_rows=None, max_cols=None, escape=False)
    display(HTML(
        f"<div style='max-height:{height}px; overflow:auto; border:1px solid #ddd; padding:8px'>"
        f"{html}</div>"
    ))

def display_full_table(name, height=650):
    df = tables[name]
    display(Markdown(f"**{name}**: {df.shape[0]:,} rows x {df.shape[1]:,} columns"))
    scroll_table(df, height=height)

def display_csv_link(name):
    path = TABLE_PATHS[name]
    display(Markdown(f"[Open CSV: `{name}`]({path.as_posix()})"))

manifest = pd.DataFrame(
    [
        {
            "table": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "path": str(TABLE_PATHS[name]),
        }
        for name, df in tables.items()
    ]
)
manifest"""
        ),
        md("## 2. Artifact Manifest"),
        code("scroll_table(manifest, height=420)"),
        md("## 3. Headline Comparison Tables"),
        md(
            """These are the paper-facing metrics. MAE/RMSE quantify absolute burden error. Delta Spearman/Pearson and group-map metrics quantify whether the model captures the spatial pattern of progression."""
        ),
        code(
            """headline_metrics = ["mae_suvr", "rate_mae", "subject_spearman", "delta_spearman", "delta_pearson"]
headline = tables["repeated_split_summary"].query("split == 'test' and metric in @headline_metrics")
headline = headline[["model", "metric", "median", "q25", "q75", "n"]].sort_values(["metric", "model"])
scroll_table(headline, height=520)"""
        ),
        md("### Primary Held-Out Split"),
        code(
            """primary = tables["primary_summary"].query("split == 'test' and metric in @headline_metrics")
primary = primary[["model", "metric", "median", "q25", "q75", "n"]].sort_values(["metric", "model"])
scroll_table(primary, height=520)"""
        ),
        md("### Group-Map Progression Metrics"),
        code(
            """# This table corresponds to the group-average brain surface comparison.
surface_values_candidates = [
    Path("outputs/figures/region_tau_burden_nilearn_values.csv"),
    Path("BayesianNetwork-SCM/outputs/figures/region_tau_burden_nilearn_values.csv"),
]
surface_values_path = next((path for path in surface_values_candidates if path.exists()), None)
if surface_values_path is None:
    display(Markdown("Group-map value table was not found. Run the nilearn burden comparison renderer first."))
else:
    surface = pd.read_csv(surface_values_path)
    empirical_s1 = surface["empirical_s1"].to_numpy(float)
    empirical_delta = (surface["empirical_s1"] - surface["baseline_s0"]).to_numpy(float)
    rows = []
    for model in ["BayesianNetwork-SCM", "NDM", "ESM", "SIR"]:
        pred_col = f"{model}_predicted_s1"
        if pred_col not in surface:
            continue
        pred_s1 = surface[pred_col].to_numpy(float)
        pred_delta = pred_s1 - surface["baseline_s0"].to_numpy(float)
        denom = np.linalg.norm(pred_delta) * np.linalg.norm(empirical_delta)
        rows.append(
            {
                "model": "BN-LTE" if model == "BayesianNetwork-SCM" else model,
                "group_map_mae": np.mean(np.abs(pred_s1 - empirical_s1)),
                "s1_map_spearman": pd.Series(pred_s1).rank().corr(pd.Series(empirical_s1).rank()),
                "delta_map_spearman": pd.Series(pred_delta).rank().corr(pd.Series(empirical_delta).rank()),
                "delta_cosine": np.dot(pred_delta, empirical_delta) / denom,
                "direction_accuracy": np.mean(np.sign(pred_delta) == np.sign(empirical_delta)),
            }
        )
    group_map_metrics = pd.DataFrame(rows)
    scroll_table(group_map_metrics, height=360)"""
        ),
        md("## 4. All Paper Figures"),
    ]

    for caption, path in FIGURES:
        rel = path.relative_to(HERE).as_posix()
        cells.append(md(f"### {caption}\n\n![{caption}]({rel})"))

    cells.extend(
        [
            md("## 5. Full Data Tables"),
            md(
                """Each cell below loads the complete CSV table. Large tables are displayed in a scrollable HTML container so no rows are truncated by pandas settings."""
            ),
        ]
    )
    for name in TABLES:
        cells.extend(
            [
                md(f"### Full Table: `{name}`"),
                code(f"display_csv_link('{name}')\ndisplay_full_table('{name}')"),
            ]
        )

    cells.extend(
        [
            md("## 6. Machine-Readable Report JSON"),
            code(
                """display(Markdown(f"Report path: `{REPORT}`"))
display(report.keys())
report"""
            ),
            md("## 7. Interpretation Guardrails"),
            md(
                """- Use MAE/RMSE as absolute burden forecasting metrics.
- Use Delta Spearman, Delta Pearson, Delta Cosine, group-map MAE, and top-k progression overlap for the progression-topology claim.
- Persistence-like models can win raw MAE because longitudinal tau changes are small.
- BN-LTE should be framed as a stage-aware progression and causal-forecasting model, not merely as a lowest-MAE smoother.
- Counterfactual results are model-implied SCM probes, not randomized intervention evidence.
"""
            ),
        ]
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(NOTEBOOK)
    print(f"cells={len(cells)}")
    return 0


def md(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(source),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(source),
    }


def lines(source: str) -> list[str]:
    text = source.strip("\n")
    return [line + "\n" for line in text.splitlines()]


if __name__ == "__main__":
    raise SystemExit(main())
