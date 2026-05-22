#!/usr/bin/env python3
"""Create one notebook containing all paper-validation results and figures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "paper_validation"
FIG = OUT / "figures"
EXT = HERE / "outputs" / "paper_extended"
EXT_FIG = EXT / "figures"
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
    "extended_progression_topology_pair_metrics": EXT / "progression_topology_pair_metrics.csv",
    "extended_progression_topology_summary": EXT / "progression_topology_summary.csv",
    "extended_group_map_progression_metrics": EXT / "group_map_progression_metrics.csv",
    "extended_braak_stage_deltas": EXT / "braak_stage_deltas.csv",
    "extended_braak_ordering_summary": EXT / "braak_ordering_summary.csv",
    "extended_fast_progressor_classification": EXT / "fast_progressor_classification.csv",
    "extended_pseudotime_loadings": EXT / "pseudotime_loadings.csv",
    "extended_pseudotime_group_contributions": EXT / "pseudotime_group_contributions.csv",
    "extended_edge_confidence_bands": EXT / "edge_confidence_bands.csv",
    "extended_edge_confidence_summary": EXT / "edge_confidence_summary.csv",
    "extended_subject_archetypes": EXT / "subject_archetypes.csv",
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
    ("Figure 10. Subject-level progression topology metrics", EXT_FIG / "fig10_progression_topology_metrics.png"),
    ("Figure 11. Group-average progression-map metrics", EXT_FIG / "fig11_group_map_progression_metrics.png"),
    ("Figure 12. Top-k regional progression overlap", EXT_FIG / "fig12_topk_overlap_distribution.png"),
    ("Figure 13. Braak-like anatomical progression ordering", EXT_FIG / "fig13_braak_ordering.png"),
    ("Figure 14. Fast-progressor classification", EXT_FIG / "fig14_fast_progressor_classification.png"),
    ("Figure 15. Pseudotime explainability", EXT_FIG / "fig15_pseudotime_explainability.png"),
    ("Figure 16. Bootstrap confidence bands for dynamic edges", EXT_FIG / "fig16_edge_confidence_bands.png"),
    ("Figure 17. Multi-view brain progression maps by model and stage", EXT_FIG / "fig17_brain_stage_delta_models.png"),
    ("Figure 18. Multi-view brain map of top progression overlap", EXT_FIG / "fig18_brain_topk_progression_overlap.png"),
    ("Figure 19. Multi-view subject archetype brain maps", EXT_FIG / "fig19_brain_subject_archetypes.png"),
    ("Figure 20. Bilateral brain progression butterfly", EXT_FIG / "fig20_brain_bilateral_delta_butterfly.png"),
    ("Figure 21. Anatomical brain-region progression heatmap", EXT_FIG / "fig21_brain_region_delta_heatmap.png"),
    ("Figure 22. Brain causal-flow schematic", EXT_FIG / "fig22_brain_causal_flow_schematic.png"),
    ("Figure 23. Stage-wise regional brain fingerprint", EXT_FIG / "fig23_brain_stage_radial_fingerprint.png"),
    ("Figure 24. Brain-surface heatmap of tau progression intensity", EXT_FIG / "fig24_brain_surface_progression_heatmap.png"),
    ("Figure 25. Brain-surface heatmap of prediction error", EXT_FIG / "fig25_brain_surface_prediction_error_heatmap.png"),
    ("Figure 26. Brain-surface heatmap of BN-LTE regional advantage", EXT_FIG / "fig26_brain_surface_bnlte_advantage_heatmap.png"),
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
- extended progression-topology, fast-progressor, pseudotime, and anatomical-ordering experiments,
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

validation_candidates = [
    Path("outputs/paper_validation"),
    Path("BayesianNetwork-SCM/outputs/paper_validation"),
]
extended_candidates = [
    Path("outputs/paper_extended"),
    Path("BayesianNetwork-SCM/outputs/paper_extended"),
]
BASE = next((path for path in validation_candidates if path.exists()), None)
EXT = next((path for path in extended_candidates if path.exists()), None)
if BASE is None:
    raise FileNotFoundError("Could not find outputs/paper_validation from the current notebook working directory.")
if EXT is None:
    raise FileNotFoundError("Could not find outputs/paper_extended from the current notebook working directory.")
FIG = BASE / "figures"
EXT_FIG = EXT / "figures"
REPORT = BASE / "paper_validation_report.json"
EXT_REPORT = EXT / "extended_paper_experiments_report.json"

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
    "extended_progression_topology_pair_metrics": EXT / "progression_topology_pair_metrics.csv",
    "extended_progression_topology_summary": EXT / "progression_topology_summary.csv",
    "extended_group_map_progression_metrics": EXT / "group_map_progression_metrics.csv",
    "extended_braak_stage_deltas": EXT / "braak_stage_deltas.csv",
    "extended_braak_ordering_summary": EXT / "braak_ordering_summary.csv",
    "extended_fast_progressor_classification": EXT / "fast_progressor_classification.csv",
    "extended_pseudotime_loadings": EXT / "pseudotime_loadings.csv",
    "extended_pseudotime_group_contributions": EXT / "pseudotime_group_contributions.csv",
    "extended_edge_confidence_bands": EXT / "edge_confidence_bands.csv",
    "extended_edge_confidence_summary": EXT / "edge_confidence_summary.csv",
    "extended_subject_archetypes": EXT / "subject_archetypes.csv",
}

tables = {name: pd.read_csv(path) for name, path in TABLE_PATHS.items()}
with REPORT.open("r", encoding="utf-8") as handle:
    report = json.load(handle)
with EXT_REPORT.open("r", encoding="utf-8") as handle:
    extended_report = json.load(handle)

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
        md("### Extended Subject-Level Progression Topology"),
        code(
            """topology_metrics = ["delta_spearman", "delta_cosine", "top3_overlap", "weighted_top3_capture", "direction_accuracy"]
topology = tables["extended_progression_topology_summary"].query("split == 'test' and metric in @topology_metrics")
topology = topology[["model", "metric", "median", "q25", "q75", "n"]].sort_values(["metric", "model"])
scroll_table(topology, height=560)"""
        ),
        md("### Group-Map Progression Metrics"),
        code(
            """group_map_metrics = tables["extended_group_map_progression_metrics"].query("stage == 'all_test'")
scroll_table(group_map_metrics, height=360)"""
        ),
        md("### Fast-Progressor Classification"),
        code(
            """fast_progressor = tables["extended_fast_progressor_classification"].sort_values("auroc", ascending=False)
scroll_table(fast_progressor, height=360)"""
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
                """Each cell below contains the complete CSV table rendered as scrollable HTML, with a link back to the source CSV for traceability."""
            ),
        ]
    )
    for name in TABLES:
        cells.extend(
            [
                md(f"### Full Table: `{name}`"),
                md(static_table_markdown(name, TABLES[name])),
            ]
        )

    cells.extend(
        [
            md("## 6. Machine-Readable Report JSON"),
            code(
                """display(Markdown(f"Validation report path: `{REPORT}`"))
display(report.keys())
display(Markdown(f"Extended report path: `{EXT_REPORT}`"))
display(extended_report.keys())
{"paper_validation_report": report, "extended_paper_experiments_report": extended_report}"""
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


def static_table_markdown(name: str, path: Path) -> str:
    df = pd.read_csv(path)
    rel = path.relative_to(HERE).as_posix()
    table = df.to_html(index=False, max_rows=None, max_cols=None, escape=True)
    return f"""[Open CSV: `{name}`]({rel})

**{name}**: {df.shape[0]:,} rows x {df.shape[1]:,} columns

<div style="max-height:650px; overflow:auto; border:1px solid #ddd; padding:8px">
{table}
</div>
"""


def lines(source: str) -> list[str]:
    text = source.strip("\n")
    return [line + "\n" for line in text.splitlines()]


if __name__ == "__main__":
    raise SystemExit(main())
