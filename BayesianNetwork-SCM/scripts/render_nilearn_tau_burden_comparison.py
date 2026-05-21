#!/usr/bin/env python3
"""Render nilearn-style surface maps for regional tau-burden comparison.

This complements ``render_region_burden_comparison.py``.  The existing renderer
is dependency-free and schematic; this one uses nilearn's fsaverage5 surface and
the Destrieux surface atlas to create cortical surface panels.

The model outputs are still Desikan-Killiany/aparc target regions.  Nilearn does
not ship a DK/aparc surface atlas, so the selected DK regions are projected to
the closest Destrieux parcels and the mapping is written alongside the figures.
"""

from __future__ import annotations

import csv
import math
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
BN_SCM_DIR = PROJECT_ROOT / "BayesianNetwork-SCM"
OUT_DIR = BN_SCM_DIR / "outputs" / "figures"

sys.path.insert(0, str(HERE))

from render_region_burden_comparison import (  # noqa: E402
    BN_LTE2_DIR,
    MODEL_ORDER,
    SELECTED_REGIONS,
    build_predicted_s1,
    compute_test_s0_s1,
    load_region_metrics,
    load_test_rids,
)


MODEL_LABELS = {
    "BayesianNetwork-SCM": "BN-SCM / BN-LTE",
    "BN-LTE-2": "BN-LTE-2",
    "NDM": "NDM",
    "ESM": "ESM",
    "SIR": "SIR",
}

PANEL_ORDER = [
    ("S0 baseline", "baseline"),
    ("Empirical S1", "empirical"),
    ("BN-SCM / BN-LTE", "BayesianNetwork-SCM"),
    ("BN-LTE-2", "BN-LTE-2"),
    ("NDM", "NDM"),
    ("ESM", "ESM"),
    ("SIR", "SIR"),
]

VIEW_COLUMNS = [
    ("left", "lateral", "L lateral"),
    ("right", "lateral", "R lateral"),
    ("left", "ventral", "L ventral"),
    ("right", "ventral", "R ventral"),
]

# Approximate projection from the selected DK/aparc regions to the closest
# Destrieux surface parcels available through nilearn.
DK_TO_DESTRIEUX = {
    "entorhinal": ["G_oc-temp_med-Parahip", "S_collat_transv_ant"],
    "fusiform": ["G_oc-temp_lat-fusifor", "S_oc-temp_lat"],
    "inferiortemporal": ["G_temporal_inf", "S_temporal_inf"],
    "middletemporal": ["G_temporal_middle"],
    "inferiorparietal": ["G_pariet_inf-Angular", "G_pariet_inf-Supramar"],
}


def selected_region_stem(region: str) -> str:
    return region.split("_", 1)[1]


def load_surface_assets():
    try:
        from nilearn import datasets
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "nilearn is required for this renderer. Install with: "
            ".venv/bin/python -m pip install nilearn matplotlib"
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        atlas = datasets.fetch_atlas_surf_destrieux()
    return fsaverage, atlas


def build_surface_maps(values: dict[str, float], atlas) -> dict[str, np.ndarray]:
    label_to_index = {label: idx for idx, label in enumerate(atlas["labels"])}
    maps = {
        "left": np.full(np.asarray(atlas["map_left"]).shape, np.nan, dtype=float),
        "right": np.full(np.asarray(atlas["map_right"]).shape, np.nan, dtype=float),
    }
    atlas_maps = {"left": np.asarray(atlas["map_left"]), "right": np.asarray(atlas["map_right"])}

    for region in SELECTED_REGIONS:
        value = values.get(region, float("nan"))
        if not math.isfinite(value):
            continue
        hemi = "left" if region.startswith("L_") else "right"
        for label in DK_TO_DESTRIEUX.get(selected_region_stem(region), []):
            label_idx = label_to_index.get(label)
            if label_idx is None:
                continue
            maps[hemi][atlas_maps[hemi] == label_idx] = value
    return maps


def plot_surface_grid(
    *,
    output_path: Path,
    title: str,
    panel_values: list[tuple[str, dict[str, float]]],
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    note: str,
) -> None:
    from nilearn import plotting

    fsaverage, atlas = load_surface_assets()
    fig = plt.figure(figsize=(16, 2.25 * len(panel_values) + 1.4), facecolor="white")
    grid_bottom = 0.06
    grid_top = 0.84
    grid = fig.add_gridspec(
        len(panel_values),
        len(VIEW_COLUMNS),
        left=0.05,
        right=0.90,
        bottom=grid_bottom,
        top=grid_top,
        wspace=0.00,
        hspace=0.05,
    )

    for row_idx, (row_label, values) in enumerate(panel_values):
        maps = build_surface_maps(values, atlas)
        for col_idx, (hemi, view, col_label) in enumerate(VIEW_COLUMNS):
            ax = fig.add_subplot(grid[row_idx, col_idx], projection="3d")
            title_text = col_label if row_idx == 0 else ""
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plotting.plot_surf_stat_map(
                    fsaverage[f"infl_{hemi}"],
                    maps[hemi],
                    hemi=hemi,
                    view=view,
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    axes=ax,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar=False,
                    title=title_text,
                )
        row_step = (grid_top - grid_bottom) / len(panel_values)
        row_center = grid_top - (row_idx + 0.5) * row_step
        fig.text(0.012, row_center, row_label, ha="left", va="center", fontsize=11, weight="bold")

    fig.suptitle(title, x=0.05, y=0.985, ha="left", fontsize=16, weight="bold")
    fig.text(0.05, 0.935, textwrap.fill(note, width=150), ha="left", va="top", fontsize=9.5, color="#4B5563")

    norm = Normalize(vmin=vmin, vmax=vmax)
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.60])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(colorbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_absolute_panel_values(
    s0_mean: dict[str, float],
    s1_mean: dict[str, float],
    predicted_s1: dict[str, dict[str, float]],
) -> list[tuple[str, dict[str, float]]]:
    rows: list[tuple[str, dict[str, float]]] = []
    for label, key in PANEL_ORDER:
        if key == "baseline":
            rows.append((label, s0_mean))
        elif key == "empirical":
            rows.append((label, s1_mean))
        elif key in predicted_s1:
            rows.append((label, predicted_s1[key]))
    return rows


def build_error_panel_values(
    s1_mean: dict[str, float],
    predicted_s1: dict[str, dict[str, float]],
) -> list[tuple[str, dict[str, float]]]:
    rows = []
    for model in MODEL_ORDER:
        if model not in predicted_s1:
            continue
        rows.append(
            (
                MODEL_LABELS.get(model, model),
                {
                    region: predicted_s1[model].get(region, float("nan")) - s1_mean.get(region, float("nan"))
                    for region in SELECTED_REGIONS
                },
            )
        )
    return rows


def empirical_suvr_bounds(s0_mean: dict[str, float], s1_mean: dict[str, float]) -> tuple[float, float]:
    values = [*s0_mean.values(), *s1_mean.values()]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 0.9, 1.5
    lo = min(finite)
    hi = max(finite)
    pad = max((hi - lo) * 0.18, 0.03)
    return lo - pad, hi + pad


def error_bound(error_rows: list[tuple[str, dict[str, float]]]) -> float:
    values = []
    for label, row in error_rows:
        if label == "BN-LTE-2":
            continue
        values.extend(abs(value) for value in row.values() if math.isfinite(value))
    return max(values + [0.08])


def write_values_csv(
    path: Path,
    s0_mean: dict[str, float],
    s1_mean: dict[str, float],
    predicted_s1: dict[str, dict[str, float]],
) -> None:
    fieldnames = ["region", "baseline_s0", "empirical_s1", "empirical_delta"]
    for model in MODEL_ORDER:
        if model in predicted_s1:
            fieldnames.extend([f"{model}_predicted_s1", f"{model}_error_vs_empirical_s1"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for region in SELECTED_REGIONS:
            row = {
                "region": region,
                "baseline_s0": s0_mean.get(region, float("nan")),
                "empirical_s1": s1_mean.get(region, float("nan")),
                "empirical_delta": s1_mean.get(region, float("nan")) - s0_mean.get(region, float("nan")),
            }
            for model in MODEL_ORDER:
                if model not in predicted_s1:
                    continue
                pred = predicted_s1[model].get(region, float("nan"))
                row[f"{model}_predicted_s1"] = pred
                row[f"{model}_error_vs_empirical_s1"] = pred - s1_mean.get(region, float("nan"))
            writer.writerow(row)


def write_mapping_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dk_region", "destrieux_parcels"])
        writer.writeheader()
        for region in SELECTED_REGIONS:
            writer.writerow(
                {
                    "dk_region": region,
                    "destrieux_parcels": ";".join(DK_TO_DESTRIEUX.get(selected_region_stem(region), [])),
                }
            )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bn_scm_metrics = load_region_metrics(BN_SCM_DIR / "outputs" / "model_comparison_region_metrics.csv")
    bn_lte2_metrics = load_region_metrics(BN_LTE2_DIR / "outputs" / "model_comparison_region_metrics.csv")
    test_rids = load_test_rids(BN_LTE2_DIR / "outputs" / "model_comparison_report.json")
    s0_mean, s1_mean, dt_mean = compute_test_s0_s1(test_rids)
    predicted_s1 = build_predicted_s1(s0_mean, dt_mean, bn_scm_metrics, bn_lte2_metrics)

    absolute_rows = build_absolute_panel_values(s0_mean, s1_mean, predicted_s1)
    vmin, vmax = empirical_suvr_bounds(s0_mean, s1_mean)
    absolute_note = (
        f"Group mean over held-out test split; mean follow-up = {dt_mean:.2f} years. "
        "DK/aparc regions are approximated with nearest nilearn Destrieux surface parcels. "
        "Color scale is anchored to empirical S0/S1, so out-of-range model values are clipped."
    )
    plot_surface_grid(
        output_path=OUT_DIR / "region_tau_burden_nilearn_surface.png",
        title="Regional tau burden on nilearn fsaverage surface",
        panel_values=absolute_rows,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        colorbar_label="Tau SUVR",
        note=absolute_note,
    )

    error_rows = build_error_panel_values(s1_mean, predicted_s1)
    bound = error_bound(error_rows)
    error_note = (
        "Prediction error is predicted S1 minus empirical S1. "
        "Diverging color scale excludes BN-LTE-2 when setting the bound because BN-LTE-2 has large off-scale errors; "
        "true numeric values are in the CSV table."
    )
    plot_surface_grid(
        output_path=OUT_DIR / "region_tau_burden_nilearn_error_surface.png",
        title="Predicted follow-up tau error on nilearn fsaverage surface",
        panel_values=error_rows,
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        colorbar_label="Predicted S1 - empirical S1 SUVR",
        note=error_note,
    )

    write_values_csv(OUT_DIR / "region_tau_burden_nilearn_values.csv", s0_mean, s1_mean, predicted_s1)
    write_mapping_csv(OUT_DIR / "region_tau_burden_nilearn_mapping.csv")

    print("Wrote", OUT_DIR / "region_tau_burden_nilearn_surface.png")
    print("Wrote", OUT_DIR / "region_tau_burden_nilearn_error_surface.png")
    print("Wrote", OUT_DIR / "region_tau_burden_nilearn_values.csv")
    print("Wrote", OUT_DIR / "region_tau_burden_nilearn_mapping.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
