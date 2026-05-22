#!/usr/bin/env python3
"""Create a BN-LTE MNI brain-visualization notebook with non-surface figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(HERE))

from bayesian_network_scm.data import build_multimodal_pair_dataset  # noqa: E402
from bayesian_network_scm.reporting import make_subject_split  # noqa: E402
from run_extended_paper_experiments import anatomical_region_order, region_value_dict  # noqa: E402
from run_paper_validation_experiments import (  # noqa: E402
    MODEL_COLORS,
    REGION_SHORT_NAMES,
    fit_all_prediction_models,
    is_finite,
    load_graph_resources,
    safe_correlation,
    save_figure,
    short_model,
    validate_dataset,
    validate_predictions,
    validate_split,
)


OUT = HERE / "outputs" / "mni_brain_visualization"
FIG = OUT / "figures"
NOTEBOOK = HERE / "bn_lte_mni_brain_visualization.ipynb"

RANDOM_SEED = 20260521
MAX_PARENTS = 6
MODELS = ["BayesianNetwork-SCM", "ESM", "SIR", "NDM", "S0 persistence"]

# Approximate region centroids in MNI152 space. These are not subject-specific PET
# centroids; they are visualization anchors for the DK/aparc regions modeled here.
MNI_COORDS: dict[str, tuple[float, float, float]] = {
    "L_entorhinal": (-24.0, -14.0, -28.0),
    "R_entorhinal": (24.0, -14.0, -28.0),
    "L_fusiform": (-36.0, -44.0, -20.0),
    "R_fusiform": (36.0, -44.0, -20.0),
    "L_inferiortemporal": (-52.0, -38.0, -20.0),
    "R_inferiortemporal": (52.0, -38.0, -20.0),
    "L_middletemporal": (-58.0, -44.0, 0.0),
    "R_middletemporal": (58.0, -44.0, 0.0),
    "L_inferiorparietal": (-44.0, -64.0, 38.0),
    "R_inferiorparietal": (44.0, -64.0, 38.0),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    print("Step 1/5: fit held-out prediction models")
    artifacts = compute_artifacts()
    print("Step 2/5: write regional MNI value table")
    values_path = OUT / "mni_region_values.csv"
    artifacts["values"].to_csv(values_path, index=False)
    print("Step 3/5: render non-surface brain figures")
    figures = render_figures(artifacts)
    print("Step 4/5: write report")
    report = {
        "purpose": "Non-surface MNI and glass-brain visualizations for BN-LTE paper figures.",
        "data": {
            "pairs": int(artifacts["dataset"].pair_count),
            "test_pairs": int(len(artifacts["split"].test_indices)),
            "regions": artifacts["regions"],
            "mni_coordinate_note": "Approximate MNI152 anchors for regional DK/aparc summaries; not voxel-level PET statistical maps.",
        },
        "figures": {key: str(value) for key, value in figures.items()},
        "tables": {"mni_region_values": str(values_path)},
        "metrics": artifacts["metrics"],
        "guardrails": [
            "These figures are visualization transforms of regional model outputs, not independently estimated voxelwise tau maps.",
            "Gaussian MNI kernels use fixed approximate centroids so visual hot spots should be interpreted at the regional level.",
            "Model comparison claims should use the numeric validation tables; these figures are anatomical summaries of those outputs.",
        ],
    }
    (OUT / "mni_brain_visualization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Step 5/5: create notebook")
    write_notebook(report, values_path)
    print(NOTEBOOK)
    print(f"figures={len(figures)}")
    return 0


def compute_artifacts() -> dict[str, Any]:
    dataset = build_multimodal_pair_dataset(PROJECT_ROOT)
    regions = list(dataset.report["selected_tau_regions"])
    target_names = [f"tau_rate:{region}" for region in regions]
    target_indices = [dataset.target_index(name) for name in target_names]
    validate_dataset(dataset, target_indices)
    split = make_subject_split(dataset.metadata_rows, random_seed=RANDOM_SEED)
    validate_split(split)
    graph = load_graph_resources(PROJECT_ROOT, dataset, regions)
    fitted = fit_all_prediction_models(
        dataset=dataset,
        graph=graph,
        split=split,
        selected_regions=regions,
        selected_target_names=target_names,
        selected_target_indices=target_indices,
        max_parents=MAX_PARENTS,
    )
    validate_predictions(fitted["predictions"], dataset, target_indices)
    test = np.asarray(split.test_indices, dtype=int)
    baseline = dataset.target_baseline[:, target_indices]
    observed = dataset.target_observed[:, target_indices]
    baseline_mean = np.nanmean(baseline[test], axis=0)
    empirical_s1 = np.nanmean(observed[test], axis=0)
    empirical_delta = empirical_s1 - baseline_mean
    predicted_s1 = {model: np.nanmean(fitted["predictions"][model][test], axis=0) for model in MODELS}
    predicted_delta = {model: predicted_s1[model] - baseline_mean for model in MODELS}
    errors = {model: np.abs(predicted_s1[model] - empirical_s1) for model in MODELS}
    competitor_stack = np.vstack([errors[model] for model in ["ESM", "SIR", "NDM", "S0 persistence"]])
    mean_competitor_error = np.nanmean(competitor_stack, axis=0)
    bnlte_advantage = mean_competitor_error - errors["BayesianNetwork-SCM"]
    values = build_region_value_table(
        regions=regions,
        baseline_mean=baseline_mean,
        empirical_s1=empirical_s1,
        empirical_delta=empirical_delta,
        predicted_s1=predicted_s1,
        predicted_delta=predicted_delta,
        errors=errors,
        bnlte_advantage=bnlte_advantage,
    )
    metrics = {
        "bnlte_delta_spearman_vs_empirical": safe_correlation(predicted_delta["BayesianNetwork-SCM"], empirical_delta, rank=True),
        "bnlte_group_followup_mae": float(np.nanmean(errors["BayesianNetwork-SCM"])),
        "esm_group_followup_mae": float(np.nanmean(errors["ESM"])),
        "sir_group_followup_mae": float(np.nanmean(errors["SIR"])),
        "ndm_group_followup_mae": float(np.nanmean(errors["NDM"])),
        "persistence_group_followup_mae": float(np.nanmean(errors["S0 persistence"])),
    }
    return {
        "dataset": dataset,
        "split": split,
        "graph": graph,
        "fitted": fitted,
        "regions": regions,
        "target_indices": target_indices,
        "baseline_mean": baseline_mean,
        "empirical_s1": empirical_s1,
        "empirical_delta": empirical_delta,
        "predicted_s1": predicted_s1,
        "predicted_delta": predicted_delta,
        "errors": errors,
        "bnlte_advantage": bnlte_advantage,
        "values": values,
        "metrics": metrics,
    }


def build_region_value_table(
    *,
    regions: list[str],
    baseline_mean: np.ndarray,
    empirical_s1: np.ndarray,
    empirical_delta: np.ndarray,
    predicted_s1: dict[str, np.ndarray],
    predicted_delta: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    bnlte_advantage: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for idx, region in enumerate(regions):
        x, y, z = MNI_COORDS[region]
        row: dict[str, Any] = {
            "region": region,
            "short_region": REGION_SHORT_NAMES.get(region, region),
            "mni_x": x,
            "mni_y": y,
            "mni_z": z,
            "baseline_mean_suvr": float(baseline_mean[idx]),
            "empirical_followup_suvr": float(empirical_s1[idx]),
            "empirical_delta_suvr": float(empirical_delta[idx]),
            "bnlte_advantage_vs_mean_competitor": float(bnlte_advantage[idx]),
        }
        for model in MODELS:
            prefix = "bnlte" if model == "BayesianNetwork-SCM" else short_model(model).lower().replace("-", "_").replace(" ", "_")
            row[f"{prefix}_followup_suvr"] = float(predicted_s1[model][idx])
            row[f"{prefix}_delta_suvr"] = float(predicted_delta[model][idx])
            row[f"{prefix}_abs_error_suvr"] = float(errors[model][idx])
        rows.append(row)
    return pd.DataFrame(rows)


def render_figures(artifacts: dict[str, Any]) -> dict[str, Path]:
    template = load_template()
    regions = artifacts["regions"]
    graph = artifacts["graph"]
    figures = {
        "axial_progression_montage": FIG / "fig1_mni_axial_progression_montage.png",
        "axial_error_montage": FIG / "fig2_mni_axial_prediction_error_montage.png",
        "axial_bnlte_advantage": FIG / "fig3_mni_axial_bnlte_advantage_montage.png",
        "glass_brain_marker_panels": FIG / "fig4_mni_glass_brain_marker_panels.png",
        "connectome_progression_overlay": FIG / "fig5_mni_connectome_progression_overlay.png",
        "paper_style_summary": FIG / "fig6_mni_paper_style_summary_montage.png",
    }
    progression_rows = [
        ("Empirical tau increase", np.maximum(artifacts["empirical_delta"], 0.0)),
        ("BN-LTE tau increase", np.maximum(artifacts["predicted_delta"]["BayesianNetwork-SCM"], 0.0)),
        ("ESM tau increase", np.maximum(artifacts["predicted_delta"]["ESM"], 0.0)),
        ("SIR tau increase", np.maximum(artifacts["predicted_delta"]["SIR"], 0.0)),
        ("NDM tau increase", np.maximum(artifacts["predicted_delta"]["NDM"], 0.0)),
    ]
    plot_axial_mni_montage(
        figures["axial_progression_montage"],
        template,
        regions,
        progression_rows,
        title="MNI axial heatmaps of regional tau progression",
        cmap="magma",
        vmin=0.0,
        vmax=max_value(progression_rows, default=0.02),
        colorbar_label="Positive tau SUVR increase",
        note="Regional DK tau deltas are projected to approximate MNI centroids and smoothed with fixed Gaussian kernels.",
    )
    error_rows = [(f"{short_model(model)} absolute error", artifacts["errors"][model]) for model in MODELS]
    plot_axial_mni_montage(
        figures["axial_error_montage"],
        template,
        regions,
        error_rows,
        title="MNI axial heatmaps of regional follow-up prediction error",
        cmap="inferno",
        vmin=0.0,
        vmax=max_value(error_rows, default=0.01),
        colorbar_label="Absolute SUVR error",
        note="Error is absolute difference between model-predicted and empirical group-mean follow-up tau burden.",
    )
    advantage_rows = []
    bnlte_error = artifacts["errors"]["BayesianNetwork-SCM"]
    for model in ["ESM", "SIR", "NDM", "S0 persistence"]:
        advantage_rows.append((f"BN-LTE advantage vs {short_model(model)}", artifacts["errors"][model] - bnlte_error))
    advantage_rows.append(("BN-LTE advantage vs mean comparator", artifacts["bnlte_advantage"]))
    advantage_bound = max_abs_value(advantage_rows, default=0.01)
    plot_axial_mni_montage(
        figures["axial_bnlte_advantage"],
        template,
        regions,
        advantage_rows,
        title="MNI axial heatmaps of BN-LTE regional advantage",
        cmap="BrBG",
        vmin=-advantage_bound,
        vmax=advantage_bound,
        colorbar_label="Comparator error minus BN-LTE error",
        note="Positive values mean BN-LTE has lower regional follow-up error than the comparator.",
    )
    plot_glass_brain_marker_panels(figures["glass_brain_marker_panels"], regions, artifacts)
    plot_connectome_progression_overlay(figures["connectome_progression_overlay"], regions, graph, artifacts)
    plot_paper_style_summary_montage(figures["paper_style_summary"], figures, artifacts)
    return figures


def load_template() -> Any:
    from nilearn import datasets

    return datasets.load_mni152_template(resolution=2)


def region_volume_img(
    template: Any,
    regions: list[str],
    values: np.ndarray,
    *,
    sigma_mm: float = 5.0,
    support_threshold: float = 0.18,
) -> Any:
    shape = template.shape
    affine = template.affine
    inv_affine = np.linalg.inv(affine)
    accum = np.zeros(shape, dtype=float)
    weights = np.zeros(shape, dtype=float)
    sigma_vox = sigma_mm / float(np.mean(np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))))
    radius = int(np.ceil(2.5 * sigma_vox))
    for region, value in zip(regions, np.asarray(values, dtype=float), strict=True):
        if not is_finite(value):
            continue
        center = np.asarray(MNI_COORDS[region] + (1.0,), dtype=float)
        i, j, k = np.round(inv_affine @ center)[:3].astype(int)
        i0, i1 = max(i - radius, 0), min(i + radius + 1, shape[0])
        j0, j1 = max(j - radius, 0), min(j + radius + 1, shape[1])
        k0, k1 = max(k - radius, 0), min(k + radius + 1, shape[2])
        xs, ys, zs = np.ogrid[i0:i1, j0:j1, k0:k1]
        kernel = np.exp(-((xs - i) ** 2 + (ys - j) ** 2 + (zs - k) ** 2) / (2.0 * sigma_vox**2))
        accum[i0:i1, j0:j1, k0:k1] += float(value) * kernel
        weights[i0:i1, j0:j1, k0:k1] += kernel
    data = np.zeros(shape, dtype=float)
    brain_mask = np.asarray(template.get_fdata(), dtype=float) > 1.0e-6
    mask = (weights >= support_threshold) & brain_mask
    data[mask] = accum[mask] / weights[mask]
    return nib.Nifti1Image(data, affine)


def plot_axial_mni_montage(
    path: Path,
    template: Any,
    regions: list[str],
    rows: list[tuple[str, np.ndarray]],
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    note: str,
) -> None:
    from nilearn import plotting

    cuts = [-30, -20, -10, 0, 16, 36]
    fig = plt.figure(figsize=(15.5, 1.85 * len(rows) + 1.75), facecolor="white")
    top = 0.80
    bottom = 0.08
    row_h = (top - bottom) / len(rows)
    threshold = max(abs(float(vmin)), abs(float(vmax))) * 0.025
    threshold = max(threshold, 1.0e-6)
    for row_idx, (label, values) in enumerate(rows):
        y = top - (row_idx + 1) * row_h
        img = region_volume_img(template, regions, values)
        plotting.plot_stat_map(
            img,
            bg_img=template,
            display_mode="z",
            cut_coords=cuts,
            figure=fig,
            axes=(0.13, y + 0.015, 0.74, row_h * 0.88),
            colorbar=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            threshold=threshold,
            annotate=False,
            draw_cross=False,
            black_bg=False,
            symmetric_cbar=False,
        )
        fig.text(0.018, y + row_h * 0.48, label, ha="left", va="center", fontsize=10.2, weight="bold")
    fig.suptitle(title, x=0.018, y=0.985, ha="left", fontsize=16, weight="bold")
    fig.text(0.018, 0.925, note, ha="left", va="top", fontsize=9.3, color="#4B5563")
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.905, 0.18, 0.018, 0.55])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(colorbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    save_figure(fig, path, dpi=180, write_svg=False)


def plot_glass_brain_marker_panels(path: Path, regions: list[str], artifacts: dict[str, Any]) -> None:
    from nilearn import plotting

    coords = np.asarray([MNI_COORDS[region] for region in regions], dtype=float)
    panels = [
        ("Empirical progression", np.maximum(artifacts["empirical_delta"], 0.0), "magma", 0.0, None),
        ("BN-LTE progression", np.maximum(artifacts["predicted_delta"]["BayesianNetwork-SCM"], 0.0), "magma", 0.0, None),
        ("BN-LTE absolute error", artifacts["errors"]["BayesianNetwork-SCM"], "inferno", 0.0, None),
        ("BN-LTE advantage", artifacts["bnlte_advantage"], "BrBG", None, None),
    ]
    positive_max = max(max_value([(name, values) for name, values, _, _, _ in panels[:3]], default=0.02), 0.02)
    advantage_bound = max(abs(float(v)) for v in artifacts["bnlte_advantage"] if is_finite(v))
    fig = plt.figure(figsize=(14, 9), facecolor="white")
    positions = [(0.03, 0.53, 0.44, 0.35), (0.53, 0.53, 0.44, 0.35), (0.03, 0.10, 0.44, 0.35), (0.53, 0.10, 0.44, 0.35)]
    for (title, values, cmap, vmin, _), axes in zip(panels, positions, strict=True):
        if cmap == "BrBG":
            node_vmin, node_vmax = -advantage_bound, advantage_bound
        else:
            node_vmin, node_vmax = vmin, positive_max
        plotting.plot_markers(
            values,
            coords,
            node_size=95,
            node_cmap=cmap,
            node_vmin=node_vmin,
            node_vmax=node_vmax,
            display_mode="ortho",
            figure=fig,
            axes=axes,
            title=title,
            annotate=True,
            black_bg=False,
            colorbar=True,
        )
    fig.suptitle("Glass-brain node heatmaps from held-out regional tau outputs", x=0.03, y=0.985, ha="left", fontsize=16, weight="bold")
    fig.text(0.03, 0.94, "Nodes are approximate MNI anchors for the ten modeled tau regions; colors encode regional scalar values.", fontsize=9.3, color="#4B5563")
    save_figure(fig, path, dpi=180, write_svg=False)


def plot_connectome_progression_overlay(path: Path, regions: list[str], graph: dict[str, Any], artifacts: dict[str, Any]) -> None:
    from nilearn import plotting

    selected = graph["selected_region_indices"]
    adjacency = np.asarray(graph["adjacency"])[np.ix_(selected, selected)].astype(float)
    np.fill_diagonal(adjacency, 0.0)
    if np.nanmax(adjacency) > 0:
        adjacency = adjacency / np.nanmax(adjacency)
    coords = np.asarray([MNI_COORDS[region] for region in regions], dtype=float)
    order = anatomical_region_order(regions)
    order_idx = [regions.index(region) for region in order]
    empirical = np.maximum(artifacts["empirical_delta"], 0.0)
    predicted = np.maximum(artifacts["predicted_delta"]["BayesianNetwork-SCM"], 0.0)
    support = structural_coprogression_support(adjacency, predicted)
    empirical_top = set(np.argsort(-empirical)[:3])
    predicted_top = set(np.argsort(-predicted)[:3])
    top_overlap = len(empirical_top & predicted_top)
    delta_spearman = safe_correlation(predicted, empirical, rank=True)
    fig = plt.figure(figsize=(16.5, 8.6), facecolor="white")
    ax0 = fig.add_axes([0.045, 0.50, 0.34, 0.34])
    image = ax0.imshow(support[np.ix_(order_idx, order_idx)], cmap="mako" if "mako" in plt.colormaps() else "viridis", vmin=0.0, vmax=max(float(np.nanmax(support)), 0.01))
    labels = [REGION_SHORT_NAMES.get(region, region) for region in order]
    ax0.set_xticks(np.arange(len(labels)))
    ax0.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax0.set_yticks(np.arange(len(labels)))
    ax0.set_yticklabels(labels, fontsize=8)
    ax0.set_title("Structural support for BN-LTE co-progression", fontsize=11, weight="bold")
    fig.colorbar(image, ax=ax0, fraction=0.046, pad=0.03, label="A_ij x sqrt(delta_i delta_j)")

    ax_bar = fig.add_axes([0.045, 0.14, 0.34, 0.24])
    sorted_idx = np.argsort(-empirical)
    y = np.arange(len(sorted_idx))
    ax_bar.barh(y + 0.18, empirical[sorted_idx], height=0.32, color="#111827", alpha=0.78, label="Empirical")
    ax_bar.barh(y - 0.18, predicted[sorted_idx], height=0.32, color=MODEL_COLORS["BayesianNetwork-SCM"], alpha=0.84, label="BN-LTE")
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([REGION_SHORT_NAMES.get(regions[idx], regions[idx]) for idx in sorted_idx], fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Positive tau SUVR increase", fontsize=9)
    ax_bar.set_title("Regional ranking: empirical vs BN-LTE", fontsize=11, weight="bold")
    ax_bar.grid(axis="x", alpha=0.22)
    ax_bar.legend(loc="lower right", fontsize=8, frameon=False)

    ax1 = (0.43, 0.12, 0.53, 0.72)
    node_norm = Normalize(vmin=0.0, vmax=max(float(np.nanmax(predicted)), 0.01))
    node_cmap = plt.get_cmap("magma")
    node_colors = [node_cmap(node_norm(float(value))) for value in predicted]
    node_sizes = 70 + 560 * empirical / max(float(np.nanmax(empirical)), 1.0e-8)
    plotting.plot_connectome(
        support,
        coords,
        display_mode="ortho",
        figure=fig,
        axes=ax1,
        title="BN-LTE progression on structural scaffold",
        edge_threshold="70%",
        edge_cmap="YlOrBr",
        edge_vmin=0.0,
        edge_vmax=max(float(np.nanmax(support)), 0.01),
        node_color=node_colors,
        node_size=node_sizes,
        colorbar=False,
        black_bg=False,
        edge_kwargs={"linewidth": 2.0, "alpha": 0.55},
        node_kwargs={"linewidths": 0.8, "edgecolors": "#111827"},
    )
    sm = ScalarMappable(norm=node_norm, cmap=node_cmap)
    sm.set_array([])
    cax = fig.add_axes([0.935, 0.17, 0.014, 0.22])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("BN-LTE predicted tau increase", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    legend_y = 0.078
    fig.text(0.43, legend_y, "Node size = empirical progression", fontsize=8.4, color="#111827", weight="bold")
    fig.text(0.61, legend_y, "Node color = BN-LTE progression", fontsize=8.4, color="#111827", weight="bold")
    fig.text(0.80, legend_y, "Edges = structural x\nBN-LTE co-progression", fontsize=8.4, color="#111827", weight="bold", linespacing=1.1)
    fig.suptitle("Network-aware BN-LTE progression scaffold", x=0.045, y=0.985, ha="left", fontsize=16, weight="bold")
    fig.text(
        0.045,
        0.925,
        f"Top-3 hotspot recovery: {top_overlap}/3; regional delta Spearman={delta_spearman:.3f}. "
        "The scaffold highlights where structurally connected regions also carry high BN-LTE-predicted tau progression.",
        fontsize=9.3,
        color="#4B5563",
    )
    save_figure(fig, path, dpi=180, write_svg=False)


def structural_coprogression_support(adjacency: np.ndarray, positive_delta: np.ndarray) -> np.ndarray:
    delta = np.maximum(np.asarray(positive_delta, dtype=float), 0.0)
    support = np.asarray(adjacency, dtype=float) * np.sqrt(np.outer(delta, delta))
    np.fill_diagonal(support, 0.0)
    if np.nanmax(support) > 0:
        support = support / np.nanmax(support)
    return support


def plot_paper_style_summary_montage(path: Path, figures: dict[str, Path], artifacts: dict[str, Any]) -> None:
    from PIL import Image

    tiles = [
        ("a  Axial progression", figures["axial_progression_montage"]),
        ("b  Axial error", figures["axial_error_montage"]),
        ("c  BN-LTE advantage", figures["axial_bnlte_advantage"]),
        ("d  Glass-brain nodes", figures["glass_brain_marker_panels"]),
        ("e  Structural scaffold", figures["connectome_progression_overlay"]),
    ]
    opened = [Image.open(path).convert("RGB") for _, path in tiles]
    thumb_w = 760
    thumbs = []
    for image in opened:
        ratio = thumb_w / image.width
        thumbs.append(image.resize((thumb_w, int(image.height * ratio))))
    top_margin = 170
    gap = 42
    left_h = thumbs[0].height + thumbs[1].height + thumbs[2].height + 2 * gap
    right_h = thumbs[3].height + thumbs[4].height + gap
    canvas_w = thumb_w * 2 + 110
    canvas_h = max(left_h, right_h) + top_margin + 45
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw_positions = [
        (40, top_margin),
        (40, top_margin + thumbs[0].height + gap),
        (40, top_margin + thumbs[0].height + thumbs[1].height + 2 * gap),
        (thumb_w + 80, top_margin),
        (thumb_w + 80, top_margin + thumbs[3].height + gap),
    ]
    for thumb, (x, y) in zip(thumbs, draw_positions, strict=True):
        canvas.paste(thumb, (x, y))
    fig, ax = plt.subplots(figsize=(canvas_w / 170, canvas_h / 170), constrained_layout=True)
    ax.imshow(canvas)
    ax.axis("off")
    for label, (x, y) in zip([label for label, _ in tiles], draw_positions, strict=True):
        ax.text(x, y - 28, label, fontsize=12, fontweight="bold", color="#111827")
    ax.text(40, 42, "BN-LTE non-surface brain visualization montage", fontsize=18, fontweight="bold", color="#111827")
    ax.text(
        40,
        78,
        f"BN-LTE group follow-up MAE={artifacts['metrics']['bnlte_group_followup_mae']:.4f}; delta Spearman={artifacts['metrics']['bnlte_delta_spearman_vs_empirical']:.3f}.",
        fontsize=10,
        color="#4B5563",
    )
    save_figure(fig, path, dpi=170, write_svg=False)


def max_value(rows: list[tuple[str, np.ndarray]], *, default: float) -> float:
    values = [float(value) for _, row in rows for value in np.asarray(row, dtype=float).ravel() if is_finite(value)]
    return max(values + [default])


def max_abs_value(rows: list[tuple[str, np.ndarray]], *, default: float) -> float:
    values = [abs(float(value)) for _, row in rows for value in np.asarray(row, dtype=float).ravel() if is_finite(value)]
    return max(values + [default])


def write_notebook(report: dict[str, Any], values_path: Path) -> None:
    figures = report["figures"]
    values = pd.read_csv(values_path)
    cells = [
        md(
            """# BN-LTE MNI Brain Visualization Notebook

This notebook is intentionally separate from the main validation notebook. It creates non-surface visualizations in the style of neuroimaging paper figure panels: MNI axial heatmaps, glass-brain marker views, a regional structural scaffold, and a compact montage.

Important interpretation guardrail: these are regional DK/aparc summaries projected to approximate MNI coordinates with Gaussian kernels. They are useful anatomical visualizations of the model outputs, not voxelwise PET statistical maps."""
        ),
        md("## Figure 1. MNI Axial Progression Heatmaps"),
        md(image_markdown(figures["axial_progression_montage"])),
        md("## Figure 2. MNI Axial Prediction-Error Heatmaps"),
        md(image_markdown(figures["axial_error_montage"])),
        md("## Figure 3. MNI Axial BN-LTE Advantage Heatmaps"),
        md(image_markdown(figures["axial_bnlte_advantage"])),
        md("## Figure 4. Glass-Brain Marker Heatmaps"),
        md(image_markdown(figures["glass_brain_marker_panels"])),
        md("## Figure 5. Network-Aware Brain Visualization"),
        md(image_markdown(figures["connectome_progression_overlay"])),
        md("## Figure 6. Paper-Style Summary Montage"),
        md(image_markdown(figures["paper_style_summary"])),
        md("## Regional MNI Value Table"),
        md(static_table_markdown("mni_region_values", values_path, values)),
        md("## Metrics"),
        md(static_json_block(report["metrics"])),
        md("## Guardrails"),
        md("\n".join(f"- {item}" for item in report["guardrails"])),
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


def image_markdown(path_text: str) -> str:
    path = Path(path_text)
    rel = path.relative_to(HERE).as_posix()
    return f"![{path.stem}]({rel})"


def static_table_markdown(name: str, path: Path, df: pd.DataFrame) -> str:
    rel = path.relative_to(HERE).as_posix()
    table = df.to_html(index=False, max_rows=None, max_cols=None, escape=True)
    return f"""[Open CSV: `{name}`]({rel})

**{name}**: {df.shape[0]:,} rows x {df.shape[1]:,} columns

<div style="max-height:650px; overflow:auto; border:1px solid #ddd; padding:8px">
{table}
</div>
"""


def static_json_block(payload: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```"


def md(source: str) -> dict[str, object]:
    text = source.strip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}


if __name__ == "__main__":
    raise SystemExit(main())
