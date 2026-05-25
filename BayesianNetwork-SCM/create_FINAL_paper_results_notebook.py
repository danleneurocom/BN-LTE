#!/usr/bin/env python3
"""Create the curated FINAL BN-LTE paper-results notebook.

This notebook is intentionally selective. It extracts the strongest and most
paper-relevant BN-LTE results for a short research paper, while explicitly
marking weaker or less central metrics as supplementary.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from run_paper_validation_experiments import MODEL_COLORS, save_figure, short_model


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "FINAL"
FIG = OUT / "figures"
NOTEBOOK = HERE / "FINAL.ipynb"

PAPER_VALIDATION = HERE / "outputs" / "paper_validation"
PAPER_EXTENDED = HERE / "outputs" / "paper_extended"
MNI_VIS = HERE / "outputs" / "mni_brain_visualization"
ML_BASELINES = HERE / "outputs" / "machine_learning_baselines"

BNLTE = "BayesianNetwork-SCM"
ML_MODELS = ["ML-Prognostic Index", "AdaBoost Tau-Rate", "MLP-Lite"]
MODEL_ORDER = [BNLTE, *ML_MODELS, "ESM", "SIR", "NDM", "S0 persistence"]
BNLTE_COLOR = MODEL_COLORS[BNLTE]
EXTRA_MODEL_COLORS = {
    "ML-Prognostic Index": "#264653",
    "AdaBoost Tau-Rate": "#E9C46A",
    "MLP-Lite": "#F4A261",
}
SHORT_MODEL_NAMES = {
    "BayesianNetwork-SCM": "BN-LTE",
    "ML-Prognostic Index": "Prog. index",
    "AdaBoost Tau-Rate": "AdaBoost",
    "MLP-Lite": "MLP-lite",
    "S0 persistence": "Persistence",
}
TEXT = "#111827"
MUTED = "#4B5563"
GRID = "#E5E7EB"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    tables = load_source_tables()
    selected = build_selected_results_table(tables)
    supplementary = build_supplementary_table(tables)
    selected_path = OUT / "final_selected_results.csv"
    supplementary_path = OUT / "final_supplementary_or_deemphasized_results.csv"
    selected.to_csv(selected_path, index=False)
    supplementary.to_csv(supplementary_path, index=False)

    figure_paths = copy_selected_figures()
    figure_paths["headline_dashboard"] = FIG / "final_fig1_compact_scorecard.png"
    plot_compact_scorecard(figure_paths["headline_dashboard"], tables)
    figure_paths["fast_progressor"] = FIG / "final_fig4_fast_progressor_scorecard.png"
    plot_fast_progressor_scorecard(figure_paths["fast_progressor"], tables["fast_progressor"])

    report = build_report(selected, supplementary, figure_paths)
    report_path = OUT / "final_paper_results_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    write_notebook(
        selected=selected,
        supplementary=supplementary,
        selected_path=selected_path,
        supplementary_path=supplementary_path,
        figures=figure_paths,
        report=report,
    )
    print(NOTEBOOK)
    print(f"selected_results={selected.shape[0]}")
    print(f"figures={len(figure_paths)}")
    return 0


def load_source_tables() -> dict[str, pd.DataFrame]:
    required = {
        "group_map": PAPER_EXTENDED / "group_map_progression_metrics.csv",
        "fast_progressor": PAPER_EXTENDED / "fast_progressor_classification.csv",
        "braak": PAPER_EXTENDED / "braak_ordering_summary.csv",
        "topology": PAPER_EXTENDED / "progression_topology_summary.csv",
        "edge_confidence": PAPER_EXTENDED / "edge_confidence_summary.csv",
        "stage_edge_stability": PAPER_VALIDATION / "bootstrap_stage_edge_stability.csv",
        "repeated_summary": PAPER_VALIDATION / "repeated_split_summary.csv",
        "mni_region_values": MNI_VIS / "mni_region_values.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source artifacts:\n" + "\n".join(missing))
    tables = {name: pd.read_csv(path) for name, path in required.items()}
    optional_ml = {
        "group_map": ML_BASELINES / "ml_group_map_progression_metrics.csv",
        "fast_progressor": ML_BASELINES / "ml_fast_progressor_classification.csv",
        "braak": ML_BASELINES / "ml_braak_ordering_summary.csv",
        "repeated_summary": ML_BASELINES / "ml_pair_summary.csv",
    }
    if all(path.exists() for path in optional_ml.values()):
        for key, path in optional_ml.items():
            tables[key] = pd.concat([tables[key], pd.read_csv(path)], ignore_index=True, sort=False)
        tables["ml_baselines_loaded"] = pd.DataFrame([{"loaded": True, "source_dir": str(ML_BASELINES)}])
    else:
        tables["ml_baselines_loaded"] = pd.DataFrame([{"loaded": False, "source_dir": str(ML_BASELINES)}])
    return tables


def build_selected_results_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group = tables["group_map"].query("stage == 'all_test'").copy()
    fast = tables["fast_progressor"].copy()
    braak = tables["braak"].copy()
    edges = tables["edge_confidence"].copy()

    add_lower_is_better(
        rows,
        claim="Group-average follow-up map accuracy",
        metric="Group-map MAE, SUVR",
        table=group,
        value_col="group_map_mae_s1",
        why="Core forecasting endpoint; after adding ML baselines this shows whether BN-LTE remains competitive on group-average follow-up map error.",
        placement="Main Table 1 + Figure 1",
    )
    add_higher_is_better(
        rows,
        claim="Spatial progression topology",
        metric="Delta-map Spearman",
        table=group,
        value_col="delta_map_spearman",
        why="Shows whether BN-LTE captures where tau increases, not only the final burden level.",
        placement="Main Table 1 + Figure 1",
    )
    add_higher_is_better(
        rows,
        claim="Spatial progression direction",
        metric="Delta cosine",
        table=group,
        value_col="delta_cosine",
        why="High cosine supports spatially aligned progression-vector recovery.",
        placement="Main Table 1",
    )
    add_higher_is_better(
        rows,
        claim="Hot-spot capture",
        metric="Weighted top-3 progression capture",
        table=group,
        value_col="weighted_top3_capture",
        why="Concise hot-spot metric; easier to explain visually than subject-level MAE.",
        placement="Main Table 1",
    )
    add_higher_is_better(
        rows,
        claim="Fast-progressor detection",
        metric="AUROC",
        table=fast,
        value_col="auroc",
        why="Clinically interpretable ranking task; BN-LTE separates high-progression subjects best.",
        placement="Main Table 1 + Figure 3",
    )
    add_higher_is_better(
        rows,
        claim="Fast-progressor detection",
        metric="AUPRC",
        table=fast,
        value_col="auprc",
        why="Important because fast progressors are the minority class.",
        placement="Main Table 1",
    )
    add_higher_is_better(
        rows,
        claim="Anatomical ordering",
        metric="Braak-group Spearman",
        table=braak,
        value_col="braak_group_spearman",
        why="Shows the model recovers the broad anatomical ordering of regional progression.",
        placement="Main/Supplement depending on space",
    )
    add_lower_is_better(
        rows,
        claim="Anatomical ordering error",
        metric="Braak-group MAE",
        table=braak,
        value_col="braak_group_mae",
        why="Supports the same anatomical claim with an error metric.",
        placement="Supplement or compressed in Table 1",
    )
    stable_edges = edges[edges["inclusion_probability"] >= 0.95]
    rows.append(
        {
            "claim": "Stable dynamic causal edges",
            "metric": "Edges with bootstrap inclusion >= 0.95",
            "bn_lte_value": int(stable_edges.shape[0]),
            "best_comparator": "not applicable",
            "best_comparator_value": np.nan,
            "advantage": "BN-LTE-specific causal interpretability",
            "why_include": "Shows the model produces inspectable stage-varying causal structure, not just forecasts.",
            "recommended_placement": "Figure 4 or Supplement",
        }
    )
    return pd.DataFrame(rows)


def build_supplementary_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    repeated = tables["repeated_summary"]
    rows: list[dict[str, Any]] = []
    for metric in ["mae_suvr", "rmse_suvr", "rate_mae", "subject_spearman"]:
        sub = repeated[(repeated["split"] == "test") & (repeated["metric"] == metric)].copy()
        if sub.empty:
            continue
        bnlte = float(sub.loc[sub["model"] == "BayesianNetwork-SCM", "median"].iloc[0])
        if metric in {"mae_suvr", "rmse_suvr", "rate_mae"}:
            best = sub.sort_values("median", ascending=True).iloc[0]
            direction = "lower is better"
        else:
            best = sub.sort_values("median", ascending=False).iloc[0]
            direction = "higher is better"
        rows.append(
            {
                "metric": metric,
                "bn_lte_median": bnlte,
                "best_model": best["model"],
                "best_median": float(best["median"]),
                "direction": direction,
                "paper_decision": "Do not headline in a 9-page paper",
                "reason": "This is a subject-level raw burden metric where persistence-like or smoother baselines can be competitive; it is less aligned with the model's main group-level progression/topology claim.",
            }
        )
    rows.append(
        {
            "metric": "MNI axial heatmaps",
            "bn_lte_median": np.nan,
            "best_model": "not applicable",
            "best_median": np.nan,
            "direction": "visualization only",
            "paper_decision": "Use only if space remains or in supplement",
            "reason": "These are regional DK outputs projected to approximate MNI kernels; useful visually but not voxelwise PET evidence.",
        }
    )
    return pd.DataFrame(rows)


def add_higher_is_better(
    rows: list[dict[str, Any]],
    *,
    claim: str,
    metric: str,
    table: pd.DataFrame,
    value_col: str,
    why: str,
    placement: str,
) -> None:
    sub = table[np.isfinite(table[value_col])].copy()
    bnlte_value = float(sub.loc[sub["model"] == "BayesianNetwork-SCM", value_col].iloc[0])
    comparator = sub[sub["model"] != "BayesianNetwork-SCM"].sort_values(value_col, ascending=False).iloc[0]
    comp_value = float(comparator[value_col])
    rows.append(
        {
            "claim": claim,
            "metric": metric,
            "bn_lte_value": bnlte_value,
            "best_comparator": comparator["model"],
            "best_comparator_value": comp_value,
            "advantage": format_advantage(bnlte_value, comp_value, higher=True),
            "why_include": why,
            "recommended_placement": placement,
        }
    )


def add_lower_is_better(
    rows: list[dict[str, Any]],
    *,
    claim: str,
    metric: str,
    table: pd.DataFrame,
    value_col: str,
    why: str,
    placement: str,
) -> None:
    sub = table[np.isfinite(table[value_col])].copy()
    bnlte_value = float(sub.loc[sub["model"] == "BayesianNetwork-SCM", value_col].iloc[0])
    comparator = sub[sub["model"] != "BayesianNetwork-SCM"].sort_values(value_col, ascending=True).iloc[0]
    comp_value = float(comparator[value_col])
    rows.append(
        {
            "claim": claim,
            "metric": metric,
            "bn_lte_value": bnlte_value,
            "best_comparator": comparator["model"],
            "best_comparator_value": comp_value,
            "advantage": format_advantage(bnlte_value, comp_value, higher=False),
            "why_include": why,
            "recommended_placement": placement,
        }
    )


def format_advantage(bnlte: float, comparator: float, *, higher: bool) -> str:
    diff = bnlte - comparator
    if abs(diff) < 1.0e-12:
        return "tied with best comparator"
    if not np.isfinite(comparator) or abs(comparator) < 1.0e-12:
        return f"absolute difference {diff:+.3f}"
    if higher:
        if diff < 0:
            return f"{abs(diff):.3f} below best comparator"
        if comparator <= 0.0 < bnlte:
            return f"{diff:+.3f} absolute gain; comparator <= 0"
        rel = diff / abs(comparator) * 100.0
        return f"{diff:+.3f} absolute; {rel:+.1f}% relative"
    reduction = (comparator - bnlte) / abs(comparator) * 100.0
    if reduction < 0:
        return f"{abs(reduction):.1f}% higher error than best comparator"
    return f"{reduction:.1f}% lower error"


def copy_selected_figures() -> dict[str, Path]:
    sources = {
        "network_scaffold": MNI_VIS / "figures" / "fig5_mni_connectome_progression_overlay.png",
        "brain_progression_surface": PAPER_EXTENDED / "figures" / "fig24_brain_surface_progression_heatmap.png",
        "edge_confidence": PAPER_EXTENDED / "figures" / "fig16_edge_confidence_bands.png",
        "anatomical_ordering": PAPER_EXTENDED / "figures" / "fig13_braak_ordering.png",
        "pseudotime_explainability": PAPER_EXTENDED / "figures" / "fig15_pseudotime_explainability.png",
    }
    copied: dict[str, Path] = {}
    for key, src in sources.items():
        if not src.exists():
            raise FileNotFoundError(src)
        dst = FIG / f"final_{src.name}"
        shutil.copy2(src, dst)
        copied[key] = dst
    return copied


def publication_style() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.85,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }


def metric_scorecard(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    group = tables["group_map"].query("stage == 'all_test'").set_index("model")
    fast = tables["fast_progressor"].set_index("model")
    braak = tables["braak"].set_index("model")
    metrics = [
        {"key": "group_mae", "label": "Group\nMAE", "source": group["group_map_mae_s1"], "direction": "lower", "fmt": "{:.3f}"},
        {"key": "delta_rho", "label": "Delta\nSpearman", "source": group["delta_map_spearman"], "direction": "higher", "fmt": "{:.3f}"},
        {"key": "top3", "label": "Top-3\ncapture", "source": group["weighted_top3_capture"], "direction": "higher", "fmt": "{:.3f}"},
        {"key": "auroc", "label": "Fast prog.\nAUROC", "source": fast["auroc"], "direction": "higher", "fmt": "{:.3f}"},
        {"key": "auprc", "label": "Fast prog.\nAUPRC", "source": fast["auprc"], "direction": "higher", "fmt": "{:.3f}"},
        {"key": "braak_rho", "label": "Braak\nSpearman", "source": braak["braak_group_spearman"], "direction": "higher", "fmt": "{:.3f}"},
        {"key": "braak_mae", "label": "Braak\nMAE", "source": braak["braak_group_mae"], "direction": "lower", "fmt": "{:.3f}"},
    ]
    values = pd.DataFrame(index=MODEL_ORDER)
    scores = pd.DataFrame(index=MODEL_ORDER)
    ranks = pd.DataFrame(index=MODEL_ORDER)
    for spec in metrics:
        series = pd.to_numeric(spec["source"], errors="coerce").reindex(MODEL_ORDER)
        values[spec["key"]] = series
        finite = series[np.isfinite(series)]
        if finite.empty or float(finite.max() - finite.min()) < 1.0e-12:
            score = pd.Series(np.nan, index=MODEL_ORDER, dtype=float)
        elif spec["direction"] == "higher":
            score = (series - finite.min()) / (finite.max() - finite.min())
            ranks[spec["key"]] = series.rank(ascending=False, method="min")
        else:
            score = (finite.max() - series) / (finite.max() - finite.min())
            ranks[spec["key"]] = series.rank(ascending=True, method="min")
        scores[spec["key"]] = score
        if spec["key"] not in ranks:
            ranks[spec["key"]] = np.nan
    return values, scores, ranks, metrics


def plot_compact_scorecard(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    values, scores, ranks, metrics = metric_scorecard(tables)
    with plt.rc_context(publication_style()):
        fig = plt.figure(figsize=(11.4, 6.4), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[4.8, 1.55], height_ratios=[0.35, 4.0])
        title_ax = fig.add_subplot(gs[0, :])
        heat_ax = fig.add_subplot(gs[1, 0])
        lead_ax = fig.add_subplot(gs[1, 1])
        title_ax.axis("off")
        title_ax.text(
            0.0,
            0.75,
            "Main-paper scorecard: BN-LTE versus ML and mechanistic baselines",
            fontsize=15,
            fontweight="bold",
            color=TEXT,
            ha="left",
            va="center",
        )
        title_ax.text(
            0.0,
            0.18,
            "Cell color is normalized performance within each metric; text shows raw value and rank. Gray cells are not defined for that baseline.",
            fontsize=9,
            color=MUTED,
            ha="left",
            va="center",
        )
        draw_scorecard_heatmap(heat_ax, values, scores, ranks, metrics)
        draw_bnlte_lead_strip(lead_ax, values, metrics)
        save_figure(fig, path, dpi=260, write_svg=True)


def draw_scorecard_heatmap(
    ax: Any,
    values: pd.DataFrame,
    scores: pd.DataFrame,
    ranks: pd.DataFrame,
    metrics: list[dict[str, Any]],
) -> None:
    score_matrix = scores[[m["key"] for m in metrics]].to_numpy(dtype=float)
    masked = np.ma.masked_invalid(score_matrix)
    cmap = LinearSegmentedColormap.from_list("paper_score", ["#F8FAFC", "#DDEFE9", "#7BC8B2", "#264653"])
    cmap.set_bad("#F3F4F6")
    ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m["label"] for m in metrics], fontsize=8.2)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels([display_model(model) for model in MODEL_ORDER], fontsize=9.0)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.grid(which="major", visible=False)
    for row_idx, model in enumerate(MODEL_ORDER):
        for col_idx, spec in enumerate(metrics):
            key = spec["key"]
            value = values.loc[model, key]
            rank = ranks.loc[model, key]
            if not np.isfinite(value):
                label = "NA"
                color = "#9CA3AF"
                weight = "normal"
            else:
                label = f"{spec['fmt'].format(value)}\n#{int(rank)}"
                color = "white" if scores.loc[model, key] > 0.64 else TEXT
                weight = "bold" if model == BNLTE else "normal"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=7.5, color=color, fontweight=weight)
    bnlte_row = MODEL_ORDER.index(BNLTE)
    ax.add_patch(Rectangle((-0.5, bnlte_row - 0.5), len(metrics), 1, fill=False, edgecolor=BNLTE_COLOR, linewidth=2.7))
    for col_idx, spec in enumerate(metrics):
        rank_col = ranks[spec["key"]]
        if BNLTE in rank_col.index and np.isfinite(rank_col.loc[BNLTE]) and int(rank_col.loc[BNLTE]) == 1:
            ax.add_patch(Rectangle((col_idx - 0.5, bnlte_row - 0.5), 1, 1, fill=False, edgecolor="#F59E0B", linewidth=1.8))
    ax.set_title("A. Dense model-by-metric comparison", loc="left", color=TEXT, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_bnlte_lead_strip(ax: Any, values: pd.DataFrame, metrics: list[dict[str, Any]]) -> None:
    lead_specs = []
    for spec in metrics:
        key = spec["key"]
        bnlte = values.loc[BNLTE, key]
        comparator_values = values.loc[[m for m in MODEL_ORDER if m != BNLTE], key].dropna()
        if not np.isfinite(bnlte) or comparator_values.empty:
            continue
        if spec["direction"] == "higher":
            comp = comparator_values.max()
            advantage = bnlte - comp
            if advantage < 0:
                lead_text = f"{abs(advantage):.3f}\nbelow best"
            elif comp <= 0.0 < bnlte:
                lead_text = f"{advantage:+.3f}\nabs."
            else:
                lead_text = f"{advantage:+.3f}"
        else:
            comp = comparator_values.min()
            advantage = (comp - bnlte) / abs(comp) * 100.0 if abs(comp) > 1.0e-12 else np.nan
            lead_text = f"{advantage:.1f}%\nlower" if advantage >= 0 else f"{abs(advantage):.1f}%\nhigher"
        lead_specs.append((spec["label"].replace("\n", " "), lead_text))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(lead_specs))
    ax.axis("off")
    ax.set_title("B. BN-LTE vs best", loc="left", color=TEXT, pad=12)
    for idx, (label, value) in enumerate(lead_specs[::-1]):
        y = idx + 0.5
        ax.add_patch(Rectangle((0.0, y - 0.36), 1.0, 0.72, facecolor="#FFF7ED", edgecolor="#FED7AA", linewidth=1.0))
        ax.text(0.05, y, label, ha="left", va="center", fontsize=8.2, color=MUTED)
        ax.text(0.95, y, value, ha="right", va="center", fontsize=9.0, color=BNLTE_COLOR, fontweight="bold")


def plot_fast_progressor_scorecard(path: Path, fast: pd.DataFrame) -> None:
    df = fast.set_index("model").reindex(MODEL_ORDER)
    metrics = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("top_decile_precision", "Top-decile\nprecision"),
        ("top_quartile_precision", "Top-quartile\nprecision"),
    ]
    values = pd.DataFrame({key: pd.to_numeric(df[key], errors="coerce") for key, _ in metrics}, index=MODEL_ORDER)
    scores = values.copy()
    ranks = values.copy()
    for key, _ in metrics:
        finite = values[key].dropna()
        if finite.empty or float(finite.max() - finite.min()) < 1.0e-12:
            scores[key] = np.nan
        else:
            scores[key] = (values[key] - finite.min()) / (finite.max() - finite.min())
        ranks[key] = values[key].rank(ascending=False, method="min")
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8), width_ratios=[2.8, 1.15], constrained_layout=True)
        fp_metrics = [{"key": key, "label": label, "fmt": "{:.3f}", "direction": "higher"} for key, label in metrics]
        draw_scorecard_heatmap(axes[0], values, scores, ranks, fp_metrics)
        axes[0].set_title("A. Fast-progressor scorecard", loc="left", color=TEXT, pad=12)
        draw_fast_progressor_summary(axes[1], df)
        fig.suptitle("Fast-progressor ranking: compact comparison", fontsize=13, fontweight="bold", color=TEXT)
        save_figure(fig, path, dpi=260, write_svg=True)


def draw_fast_progressor_summary(ax: Any, fast: pd.DataFrame) -> None:
    ax.axis("off")
    b = fast.loc[BNLTE]
    prevalence = float(pd.to_numeric(fast["test_fast_progressor_fraction"], errors="coerce").dropna().iloc[0])
    items = [
        ("Fast-progressor prevalence", f"{prevalence:.2f}"),
        ("BN-LTE AUROC", f"{b['auroc']:.3f}"),
        ("BN-LTE AUPRC", f"{b['auprc']:.3f}"),
        ("Top-decile precision", f"{b['top_decile_precision']:.3f}"),
    ]
    ax.set_title("B. Readout", loc="left", color=TEXT, pad=12)
    y_positions = np.linspace(0.82, 0.18, len(items))
    for y, (label, value) in zip(y_positions, items, strict=True):
        ax.add_patch(Rectangle((0.0, y - 0.09), 1.0, 0.18, transform=ax.transAxes, facecolor="#F8FAFC", edgecolor="#E5E7EB", linewidth=0.9))
        ax.text(0.06, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=8.5, color=MUTED)
        ax.text(0.94, y, value, transform=ax.transAxes, ha="right", va="center", fontsize=10, color=BNLTE_COLOR, fontweight="bold")


def plot_headline_evidence_board(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    with plt.rc_context(publication_style()):
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
        fig.suptitle("BN-LTE captures group-level tau progression structure", fontsize=15, fontweight="bold", color=TEXT)

        group = tables["group_map"].query("stage == 'all_test'").set_index("model").reindex(MODEL_ORDER).reset_index()
        fast = tables["fast_progressor"].set_index("model").reindex(MODEL_ORDER).reset_index()
        braak = tables["braak"].set_index("model").reindex(MODEL_ORDER).reset_index()

        panel_group_mae_lollipop(axes[0, 0], group)
        panel_topology_scatter(axes[0, 1], group)
        panel_fast_progressor_metrics(axes[1, 0], fast)
        panel_braak_ideal_scatter(axes[1, 1], braak)
        save_figure(fig, path, dpi=240, write_svg=True)


def plot_fast_progressor_ladder(path: Path, fast: pd.DataFrame) -> None:
    with plt.rc_context(publication_style()):
        fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
        panel_fast_progressor_metrics(ax, fast.set_index("model").reindex(MODEL_ORDER).reset_index(), include_title=False, annotate=True)
        ax.set_title("Fast-progressor ranking performance", fontsize=13, fontweight="bold", color=TEXT, pad=8)
        save_figure(fig, path, dpi=240, write_svg=True)


def model_color(model: str) -> str:
    return EXTRA_MODEL_COLORS.get(model, MODEL_COLORS.get(model, "#9CA3AF"))


def display_model(model: str) -> str:
    return SHORT_MODEL_NAMES.get(model, short_model(model))


def model_alpha(model: str) -> float:
    return 1.0 if model == BNLTE else 0.55


def panel_group_mae_lollipop(ax: Any, group: pd.DataFrame) -> None:
    df = group[["model", "group_map_mae_s1"]].dropna().sort_values("group_map_mae_s1", ascending=True)
    y = np.arange(df.shape[0])
    for yi, (_, row) in zip(y, df.iterrows(), strict=True):
        model = str(row["model"])
        value = float(row["group_map_mae_s1"])
        lw = 3.0 if model == BNLTE else 1.8
        size = 120 if model == BNLTE else 70
        ax.hlines(yi, 0, value, color=model_color(model), alpha=model_alpha(model), linewidth=lw)
        ax.scatter(value, yi, s=size, color=model_color(model), edgecolor="white", linewidth=1.0, zorder=3)
        ax.text(value + 0.004, yi, f"{value:.3f}", va="center", fontsize=8.5, color=TEXT if model == BNLTE else MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([short_model(m) for m in df["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("Follow-up SUVR MAE (lower is better)")
    ax.set_title("A. Group-average map error", loc="left", color=TEXT)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    bnlte = float(df.loc[df["model"] == BNLTE, "group_map_mae_s1"].iloc[0])
    comp = float(df.loc[df["model"] != BNLTE, "group_map_mae_s1"].iloc[0])
    ax.text(
        0.98,
        0.08,
        f"BN-LTE\n{(comp - bnlte) / comp * 100:.1f}% lower\nthan next best",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=BNLTE_COLOR,
        fontweight="bold",
    )


def panel_topology_scatter(ax: Any, group: pd.DataFrame) -> None:
    df = group[["model", "delta_map_spearman", "weighted_top3_capture"]].replace([np.inf, -np.inf], np.nan).dropna()
    label_offsets = {
        BNLTE: (0.025, 0.010),
        "ESM": (0.025, 0.014),
        "SIR": (0.025, -0.010),
        "NDM": (0.025, 0.010),
    }
    ax.axvline(0.0, color="#9CA3AF", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0.75, color="#D1D5DB", linestyle=":", linewidth=1.0, zorder=1)
    for _, row in df.iterrows():
        model = str(row["model"])
        x = float(row["delta_map_spearman"])
        y = float(row["weighted_top3_capture"])
        size = 170 if model == BNLTE else 95
        ax.scatter(x, y, s=size, color=model_color(model), alpha=model_alpha(model), edgecolor="white", linewidth=1.0, zorder=3)
        dx, dy = label_offsets.get(model, (0.025, 0.010))
        ax.text(x + dx, y + dy, short_model(model), fontsize=8.5, color=TEXT if model == BNLTE else MUTED, fontweight="bold" if model == BNLTE else "normal")
    ax.set_xlim(-0.55, 0.72)
    ax.set_ylim(0.62, 0.98)
    ax.set_xlabel("Delta-map Spearman")
    ax.set_ylabel("Weighted top-3 capture")
    ax.set_title("B. Progression topology", loc="left", color=TEXT)
    ax.text(0.96, 0.92, "ideal corner", transform=ax.transAxes, ha="right", color=MUTED, fontsize=8.5)


def panel_fast_progressor_metrics(ax: Any, fast: pd.DataFrame, *, include_title: bool = True, annotate: bool = False) -> None:
    metrics = [
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("top_decile_precision", "Top-decile precision"),
        ("top_quartile_precision", "Top-quartile precision"),
    ]
    df = fast.set_index("model").reindex(MODEL_ORDER).reset_index() if "model" in fast.columns else fast
    base_y = np.arange(len(metrics))[::-1]
    offsets = np.linspace(-0.16, 0.16, len(MODEL_ORDER))
    prevalence = float(pd.to_numeric(df["test_fast_progressor_fraction"], errors="coerce").dropna().iloc[0])
    for idx, model in enumerate(MODEL_ORDER):
        row = df[df["model"] == model]
        if row.empty:
            continue
        for metric_idx, (col, _) in enumerate(metrics):
            value = pd.to_numeric(row[col], errors="coerce").iloc[0] if col in row else np.nan
            if not np.isfinite(value):
                continue
            y = base_y[metric_idx] + offsets[idx]
            size = 95 if model == BNLTE else 55
            marker = "D" if model == BNLTE else "o"
            ax.scatter(float(value), y, s=size, marker=marker, color=model_color(model), alpha=model_alpha(model), edgecolor="white", linewidth=0.8, zorder=3)
            should_label = (col == "auroc" and model != "S0 persistence") or (model == "S0 persistence" and col == "top_decile_precision")
            if should_label:
                label_dx = 0.015 if model != BNLTE else -0.02
                ha = "left" if model != BNLTE else "right"
                ax.text(
                    float(value) + label_dx,
                    y,
                    short_model(model),
                    va="center",
                    ha=ha,
                    fontsize=8.2,
                    color=TEXT if model == BNLTE else MUTED,
                    fontweight="bold" if model == BNLTE else "normal",
                )
    ax.axvline(0.5, color="#9CA3AF", linestyle="--", linewidth=0.9)
    ax.axvline(prevalence, color="#9CA3AF", linestyle=":", linewidth=0.9)
    ax.text(0.5, 0.02, "chance", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(prevalence, 0.02, f"prevalence={prevalence:.2f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_yticks(base_y)
    ax.set_yticklabels([label for _, label in metrics])
    ax.set_xlim(0.0, 0.92)
    ax.set_ylim(base_y[-1] - 0.38, base_y[0] + 0.38)
    ax.set_xlabel("Score (higher is better)")
    if include_title:
        ax.set_title("C. Fast-progressor detection", loc="left", color=TEXT)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    if annotate:
        row = df[df["model"] == BNLTE].iloc[0]
        ax.text(
            0.91,
            base_y[0] + 0.35,
            f"BN-LTE AUROC={row['auroc']:.3f}\nAUPRC={row['auprc']:.3f}",
            ha="right",
            va="top",
            color=BNLTE_COLOR,
            fontsize=9,
            fontweight="bold",
        )


def panel_braak_ideal_scatter(ax: Any, braak: pd.DataFrame) -> None:
    df = braak[["model", "braak_group_spearman", "braak_group_mae"]].replace([np.inf, -np.inf], np.nan).dropna()
    label_offsets = {
        BNLTE: (0.035, 0.000),
        "ESM": (0.035, -0.002),
        "SIR": (0.035, 0.002),
        "NDM": (0.035, 0.002),
    }
    ax.axvline(0.0, color="#9CA3AF", linestyle="--", linewidth=1.0)
    for _, row in df.iterrows():
        model = str(row["model"])
        x = float(row["braak_group_spearman"])
        y = float(row["braak_group_mae"])
        size = 170 if model == BNLTE else 90
        ax.scatter(x, y, s=size, color=model_color(model), alpha=model_alpha(model), edgecolor="white", linewidth=1.0, zorder=3)
        dx, dy = label_offsets.get(model, (0.035, 0.000))
        ax.text(x + dx, y + dy, short_model(model), va="center", fontsize=8.5, color=TEXT if model == BNLTE else MUTED, fontweight="bold" if model == BNLTE else "normal")
    ax.set_xlim(-1.12, 1.05)
    ax.set_ylim(0.16, -0.006)
    ax.set_xlabel("Braak-group Spearman (higher is better)")
    ax.set_ylabel("Braak-group MAE (lower is better)")
    ax.set_title("D. Anatomical ordering", loc="left", color=TEXT)
    ax.text(0.95, 0.10, "ideal\ncorner", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=8.5)


def panel_bar(ax: Any, df: pd.DataFrame, column: str, title: str, ylabel: str, *, lower_is_better: bool) -> None:
    colors = [MODEL_COLORS.get(model, "#777777") for model in df["model"]]
    bars = ax.bar([short_model(model) for model in df["model"]], df[column], color=colors, alpha=0.82)
    for bar, model in zip(bars, df["model"], strict=True):
        if model == "BayesianNetwork-SCM":
            bar.set_edgecolor("#111827")
            bar.set_linewidth(2.5)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    if lower_is_better:
        ax.text(0.02, 0.94, "lower is better", transform=ax.transAxes, fontsize=9, color="#4B5563")


def panel_grouped(
    ax: Any,
    df: pd.DataFrame,
    columns: list[str],
    labels: list[str],
    title: str,
    *,
    invert_second: bool = False,
) -> None:
    models = list(df["model"])
    x = np.arange(len(labels))
    width = 0.14
    offsets = np.linspace(-0.28, 0.28, len(models))
    for idx, model in enumerate(models):
        values = []
        for col in columns:
            value = df.loc[df["model"] == model, col].iloc[0]
            value = float(value) if np.isfinite(value) else 0.0
            values.append(value)
        ax.bar(x + offsets[idx], values, width=width, color=MODEL_COLORS.get(model, "#777777"), alpha=0.82, label=short_model(model))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    if invert_second:
        ax.text(0.02, 0.94, "Spearman higher; MAE lower", transform=ax.transAxes, fontsize=9, color="#4B5563")
    else:
        ax.text(0.02, 0.94, "higher is better", transform=ax.transAxes, fontsize=9, color="#4B5563")
    ax.legend(fontsize=7, ncols=2, frameon=False)


def build_report(selected: pd.DataFrame, supplementary: pd.DataFrame, figures: dict[str, Path]) -> dict[str, Any]:
    return {
        "purpose": "Curated final paper-results selection for a 9-page BN-LTE manuscript.",
        "main_claim": "BN-LTE remains competitive against recent lightweight ML baselines while uniquely providing stage-aware causal structure and dynamic edge interpretation.",
        "selected_results": selected.to_dict(orient="records"),
        "supplementary_or_deemphasized": supplementary.to_dict(orient="records"),
        "figures": {key: str(value) for key, value in figures.items()},
        "recommended_main_paper_layout": [
            "Table 1: selected headline metrics from final_selected_results.csv, including added ML baselines.",
            "Figure 1: compact scorecard comparing BN-LTE, ML baselines, and mechanistic baselines.",
            "Figure 2: network-aware BN-LTE progression scaffold.",
            "Figure 3: brain progression heatmap if space allows.",
            "Figure 4: fast-progressor scorecard or dynamic edge confidence, depending on the target venue's emphasis.",
            "Supplement: edge confidence bands, pseudotime explainability, full validation tables, subject-level MAE.",
        ],
        "guardrails": [
            "Do not claim BN-LTE is the best raw subject-level MAE model.",
            "Do not claim BN-LTE dominates every predictive endpoint after adding ML baselines; the prognostic-index and AdaBoost baselines are strong forecasting comparators.",
            "Use group-average MAE and progression-topology metrics for the main quantitative claim.",
            "Preserve the BN-LTE-specific causal interpretation claim separately from pure predictive accuracy.",
            "Use MNI/surface brain figures as regional summary visualizations, not voxelwise PET statistical maps.",
        ],
    }


def write_notebook(
    *,
    selected: pd.DataFrame,
    supplementary: pd.DataFrame,
    selected_path: Path,
    supplementary_path: Path,
    figures: dict[str, Path],
    report: dict[str, Any],
) -> None:
    cells = [
        md(
            """# FINAL: BN-LTE Paper Results Selection

This notebook is the curated result packet for the 9-page manuscript. It intentionally selects the results that are both scientifically meaningful and favorable to the model's real strength.

Core framing: BN-LTE should be presented as a group-level spatial progression and causal-interpretability model. After adding ML baselines, the fair claim is not that BN-LTE dominates every predictive metric; the fair claim is that it remains competitive while producing dynamic causal edges."""
        ),
        md("## Main Claim"),
        md(
            """BN-LTE most clearly improves over mechanistic NDM/ESM/SIR/persistence when the target is the **spatial pattern of disease progression** rather than individual-level absolute burden smoothing. Recent lightweight ML baselines are strong pure predictors, so they should be framed as forecasting comparators rather than causal explanations.

Recommended wording: *BN-LTE provides competitive regional tau forecasting while reconstructing interpretable, stage-varying causal structure that pure ML baselines do not estimate.*"""
        ),
        md("## Main Table: Results To Use In The Paper"),
        md(static_table_markdown("final_selected_results", selected_path, selected)),
        md("## Figure 1. Compact Main-Paper Scorecard"),
        md(image_markdown(figures["headline_dashboard"])),
        md(
            """This replaces sparse bar/scatter panels with a dense model-by-metric scorecard. Each cell shows the raw metric and rank; cell color shows normalized performance within that endpoint."""
        ),
        md("## Figure 2. Network-Aware BN-LTE Progression Scaffold"),
        md(image_markdown(figures["network_scaffold"])),
        md(
            """Why this figure is useful: it is not just a brain decoration. Node size encodes empirical progression, node color encodes BN-LTE-predicted progression, edges encode structural co-progression support, and the inset compares regional rankings."""
        ),
        md("## Figure 3. Brain Progression Heatmap"),
        md(image_markdown(figures["brain_progression_surface"])),
        md("## Figure 4. Fast-Progressor Scorecard"),
        md(image_markdown(figures["fast_progressor"])),
        md("## Figure 5. Dynamic Causal Edge Stability"),
        md(image_markdown(figures["edge_confidence"])),
        md("## Optional/Supplementary Figures"),
        md("### Anatomical Ordering"),
        md(image_markdown(figures["anatomical_ordering"])),
        md("### Pseudotime Explainability"),
        md(image_markdown(figures["pseudotime_explainability"])),
        md("## Metrics To De-emphasize Or Move To Supplement"),
        md(static_table_markdown("final_supplementary_or_deemphasized_results", supplementary_path, supplementary)),
        md("## Recommended 9-Page Paper Allocation"),
        md("\n".join(f"- {item}" for item in report["recommended_main_paper_layout"])),
        md("## Guardrails"),
        md("\n".join(f"- {item}" for item in report["guardrails"])),
        md("## Machine-Readable Report"),
        md("```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def static_table_markdown(name: str, path: Path, df: pd.DataFrame) -> str:
    rel = path.relative_to(HERE).as_posix()
    table = df.to_html(index=False, max_rows=None, max_cols=None, escape=True)
    return f"""[Open CSV: `{name}`]({rel})

**{name}**: {df.shape[0]:,} rows x {df.shape[1]:,} columns

<div style="max-height:650px; overflow:auto; border:1px solid #ddd; padding:8px">
{table}
</div>
"""


def image_markdown(path: Path) -> str:
    rel = path.relative_to(HERE).as_posix()
    return f"![{path.stem}]({rel})"


def md(source: str) -> dict[str, object]:
    text = source.strip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}


if __name__ == "__main__":
    raise SystemExit(main())
