#!/usr/bin/env python3
"""Render per-region burden-change figures comparing BN-SCM, BN-LTE-2, NDM, ESM, SIR.

The figures are written to BayesianNetwork-SCM/outputs/figures/ and are pulled
into bn_scm_model_comparison_analysis.ipynb. They reuse the test-split subjects
shared by both model_comparison_report.json files.
"""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
BN_SCM_DIR = PROJECT_ROOT / "BayesianNetwork-SCM"
BN_LTE2_DIR = PROJECT_ROOT / "BayesianNetwork-SCM-2"
COHORT_DIR = PROJECT_ROOT / "experiments" / "group_average_enigma" / "output"
MAPPING_PATH = PROJECT_ROOT / "experiments" / "group_average_enigma" / "adni_to_enigma_aparc_mapping.csv"
OUT_DIR = BN_SCM_DIR / "outputs" / "figures"

SELECTED_REGIONS = [
    "L_entorhinal", "R_entorhinal",
    "L_fusiform", "R_fusiform",
    "L_inferiortemporal", "R_inferiortemporal",
    "L_middletemporal", "R_middletemporal",
    "L_inferiorparietal", "R_inferiorparietal",
]

MODEL_ORDER = ["BayesianNetwork-SCM", "BN-LTE-2", "NDM", "ESM", "SIR"]
MODEL_COLORS = {
    "Observed":            "#374151",
    "BayesianNetwork-SCM": "#D55E00",
    "BN-LTE-2":            "#9333EA",
    "NDM":                 "#0072B2",
    "ESM":                 "#009E73",
    "SIR":                 "#CC79A7",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def load_region_metrics(path: Path, split: str = "test") -> dict[str, dict[str, dict[str, float]]]:
    """{model: {region: {column: float}}} for one split."""
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in read_csv_rows(path):
        if row.get("split") != split:
            continue
        model = row["model"]
        region = row["region"]
        if region not in SELECTED_REGIONS:
            continue
        out[model][region] = {
            "rate_mae":            parse_float(row.get("rate_mae")),
            "rate_rmse":           parse_float(row.get("rate_rmse")),
            "observed_rate_mean":  parse_float(row.get("observed_rate_mean")),
            "predicted_rate_mean": parse_float(row.get("predicted_rate_mean")),
            "mae_suvr":            parse_float(row.get("mae_suvr")),
        }
    return out


def load_test_rids(path: Path) -> set[str]:
    import json
    with path.open() as handle:
        report = json.load(handle)
    rids = report.get("split", {}).get("test_rids")
    if rids:
        return {str(rid) for rid in rids}
    return set()


def load_cohort_pairs() -> list[dict[str, str]]:
    return read_csv_rows(COHORT_DIR / "cohort_forecast_pairs.csv")


def load_observations_by_loniuid() -> dict[str, dict[str, str]]:
    return {row["LONIUID"]: row for row in read_csv_rows(COHORT_DIR / "cohort_tau_observations.csv")}


def region_to_adni_column() -> dict[str, str]:
    mapping = {}
    for row in read_csv_rows(MAPPING_PATH):
        if row["enigma_label"] in SELECTED_REGIONS:
            mapping[row["enigma_label"]] = row["adni_tau_column"]
    return mapping


def compute_test_s0_s1(test_rids: set[str]) -> tuple[dict[str, float], dict[str, float], float]:
    """Return mean baseline tau (S0), mean follow-up tau (S1) per region, and mean dt."""

    pairs = load_cohort_pairs()
    observations = load_observations_by_loniuid()
    region_columns = region_to_adni_column()

    s0_values: dict[str, list[float]] = {region: [] for region in SELECTED_REGIONS}
    s1_values: dict[str, list[float]] = {region: [] for region in SELECTED_REGIONS}
    dt_values: list[float] = []

    for pair in pairs:
        rid = str(int(float(pair.get("RID", "0") or 0)))
        if test_rids and rid not in test_rids:
            continue
        dt = parse_float(pair.get("target_time_years"))
        if not math.isfinite(dt) or dt <= 0.0:
            continue
        baseline = observations.get(pair.get("baseline_loniuid", ""))
        target = observations.get(pair.get("target_loniuid", ""))
        if baseline is None or target is None:
            continue
        ok = True
        for region in SELECTED_REGIONS:
            column = region_columns[region]
            b = parse_float(baseline.get(column))
            t = parse_float(target.get(column))
            if not (math.isfinite(b) and math.isfinite(t)):
                ok = False
                break
        if not ok:
            continue
        dt_values.append(dt)
        for region in SELECTED_REGIONS:
            column = region_columns[region]
            s0_values[region].append(parse_float(baseline.get(column)))
            s1_values[region].append(parse_float(target.get(column)))

    s0_mean = {region: sum(v) / len(v) for region, v in s0_values.items() if v}
    s1_mean = {region: sum(v) / len(v) for region, v in s1_values.items() if v}
    dt_mean = sum(dt_values) / len(dt_values) if dt_values else float("nan")
    return s0_mean, s1_mean, dt_mean


def build_predicted_s1(s0_mean: dict[str, float], dt_mean: float,
                       bn_scm_metrics: dict[str, dict[str, dict[str, float]]],
                       bn_lte2_metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Returns {model: {region: predicted_S1}} using S1 = S0 + dt * predicted_rate."""
    predictions: dict[str, dict[str, float]] = {}
    for model in MODEL_ORDER:
        source = bn_lte2_metrics if model == "BN-LTE-2" else bn_scm_metrics
        if model not in source:
            continue
        predictions[model] = {}
        for region in SELECTED_REGIONS:
            rate = source[model].get(region, {}).get("predicted_rate_mean", float("nan"))
            if not math.isfinite(rate):
                predictions[model][region] = float("nan")
            else:
                predictions[model][region] = s0_mean[region] + dt_mean * rate
    return predictions


# ---------------------------------------------------------------------------
# SVG primitives (stdlib only)
# ---------------------------------------------------------------------------


def svg_header(width: float, height: float) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}</style>',
    ]


def svg_text(x, y, text, *, size=12, fill="#111827", weight="400", anchor="start"):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(text))}</text>'
    )


def svg_rect(x, y, width, height, *, fill, stroke=None, radius=0.0, opacity=1.0):
    stroke_attr = f' stroke="{stroke}"' if stroke else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'rx="{radius:.2f}" fill="{fill}"{stroke_attr} opacity="{opacity:.3f}"/>'
    )


def svg_line(x1, y1, x2, y2, stroke, width, *, opacity=1.0, dasharray=None):
    dash = f' stroke-dasharray="{dasharray}"' if dasharray else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width:.2f}" opacity="{opacity:.3f}"{dash}/>'
    )


def svg_circle(x, y, radius, *, fill, stroke=None, opacity=1.0):
    stroke_attr = f' stroke="{stroke}" stroke-width="0.6"' if stroke else ""
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}"{stroke_attr} opacity="{opacity:.3f}"/>'


def format_signed(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:+.{digits}f}"


# ---------------------------------------------------------------------------
# Figure A: grouped bar chart of annualized tau rate per region
# ---------------------------------------------------------------------------


def write_rate_grouped_bars(
    out_path: Path,
    observed: dict[str, float],
    predicted_rates: dict[str, dict[str, float]],
):
    width = 1280
    height = 640
    plot_x0 = 92
    plot_y0 = 130
    plot_w = width - plot_x0 - 360
    plot_h = height - plot_y0 - 120

    series = [("Observed", observed)] + [(m, {r: predicted_rates.get(m, {}).get(r, float("nan")) for r in SELECTED_REGIONS}) for m in MODEL_ORDER]

    # Compute y-range from everything except BN-LTE-2 so the empirical-scale models stay legible.
    inrange_values = [
        v
        for series_name, values in series
        for v in values.values()
        if series_name != "BN-LTE-2" and math.isfinite(v)
    ]
    if not inrange_values:
        inrange_values = [0.0]
    vmax = max(inrange_values + [0.0])
    vmin = min(inrange_values + [0.0])
    span = max(vmax - vmin, 1e-6)
    vmax += 0.30 * span
    vmin -= 0.30 * span
    span = vmax - vmin

    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, "Per-region annualized tau-SUVR change (test set)", size=22, weight="700"))
    parts.append(svg_text(36, 64, "Observed vs predicted annualized rates (SUVR/year). Same 156 held-out pairs across all models.", size=13, fill="#4B5563"))
    parts.append(svg_text(36, 84, "Bars below zero are predicted decreases; the gray reference is the empirical group-average rate.", size=12, fill="#6B7280"))
    parts.append(svg_text(36, 104, "BN-LTE-2 bars marked with a black tip are clipped to the axis range; their true values are off-chart (down to ~-0.5/yr).", size=12, fill="#9333EA"))

    parts.append(svg_rect(plot_x0, plot_y0, plot_w, plot_h, fill="#FFFFFF", stroke="#E5E7EB", radius=6))

    # y-axis ticks
    n_ticks = 6
    for i in range(n_ticks + 1):
        value = vmin + (vmax - vmin) * i / n_ticks
        y = plot_y0 + plot_h - (value - vmin) / span * plot_h
        parts.append(svg_line(plot_x0, y, plot_x0 + plot_w, y, "#E5E7EB", 1))
        parts.append(svg_text(plot_x0 - 10, y + 4, f"{value:+.3f}", size=10, fill="#6B7280", anchor="end"))

    # zero line
    zero_y = plot_y0 + plot_h - (0.0 - vmin) / span * plot_h
    parts.append(svg_line(plot_x0, zero_y, plot_x0 + plot_w, zero_y, "#9CA3AF", 1.5))

    n_regions = len(SELECTED_REGIONS)
    n_series = len(series)
    region_slot = plot_w / n_regions
    bar_pad = 6
    bar_w = max((region_slot - 2 * bar_pad) / n_series, 4.0)

    for r_idx, region in enumerate(SELECTED_REGIONS):
        slot_x = plot_x0 + r_idx * region_slot + bar_pad
        # vertical separator between region slots
        if r_idx > 0:
            parts.append(svg_line(plot_x0 + r_idx * region_slot, plot_y0, plot_x0 + r_idx * region_slot, plot_y0 + plot_h, "#F3F4F6", 1))
        for s_idx, (series_name, series_values) in enumerate(series):
            value = series_values.get(region, float("nan"))
            color = MODEL_COLORS[series_name]
            x = slot_x + s_idx * bar_w
            if not math.isfinite(value):
                parts.append(svg_rect(x, plot_y0 + plot_h - 2, bar_w * 0.9, 2, fill="#E5E7EB"))
                continue
            clipped = value < vmin or value > vmax
            visible_value = min(max(value, vmin), vmax)
            value_y = plot_y0 + plot_h - (visible_value - vmin) / span * plot_h
            top = min(value_y, zero_y)
            bottom = max(value_y, zero_y)
            parts.append(svg_rect(x, top, bar_w * 0.9, max(bottom - top, 1.0), fill=color, opacity=0.92))
            if clipped:
                cap_y = plot_y0 + plot_h - 4 if value < vmin else plot_y0
                parts.append(svg_rect(x, cap_y, bar_w * 0.9, 4, fill="#111827"))
        label_short = region.replace("_", " ").replace("L ", "L-").replace("R ", "R-")
        parts.append(svg_text(plot_x0 + r_idx * region_slot + region_slot / 2, plot_y0 + plot_h + 22, label_short, size=10, anchor="middle", fill="#374151"))

    parts.append(svg_text(40, plot_y0 + plot_h / 2, "annualized SUVR change (/yr)", size=12, fill="#374151", anchor="middle"))
    parts.append(f'<g transform="rotate(-90,40,{plot_y0 + plot_h / 2:.2f})"></g>')

    # legend
    legend_x = plot_x0 + plot_w + 32
    legend_y = plot_y0 + 8
    parts.append(svg_text(legend_x, legend_y, "Series", size=13, weight="700"))
    for idx, (series_name, _) in enumerate(series):
        ly = legend_y + 22 + idx * 26
        parts.append(svg_rect(legend_x, ly - 12, 16, 14, fill=MODEL_COLORS[series_name], radius=3))
        parts.append(svg_text(legend_x + 24, ly, series_name, size=12, fill="#111827"))

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure B: brain-map small multiples for S0, observed S1, observed Δ, predicted Δ per model
# ---------------------------------------------------------------------------


# DK schematic coordinates copied from run_hypothesis_experiments.py so the
# notebook does not depend on that script being importable.
_DK_COORDS_LEFT: dict[str, tuple[float, float]] = {
    "L_frontalpole": (0.05, 0.58), "L_superiorfrontal": (0.22, 0.88),
    "L_rostralmiddlefrontal": (0.18, 0.68), "L_caudalmiddlefrontal": (0.34, 0.74),
    "L_medialorbitofrontal": (0.10, 0.30), "L_lateralorbitofrontal": (0.16, 0.35),
    "L_parsorbitalis": (0.18, 0.43), "L_parstriangularis": (0.25, 0.52),
    "L_parsopercularis": (0.30, 0.58), "L_precentral": (0.42, 0.80),
    "L_paracentral": (0.48, 0.92), "L_postcentral": (0.50, 0.78),
    "L_superiorparietal": (0.65, 0.80), "L_inferiorparietal": (0.68, 0.55),
    "L_supramarginal": (0.55, 0.62), "L_precuneus": (0.74, 0.78),
    "L_lateraloccipital": (0.92, 0.50), "L_cuneus": (0.93, 0.68),
    "L_pericalcarine": (0.96, 0.45), "L_lingual": (0.93, 0.30),
    "L_superiortemporal": (0.45, 0.38), "L_middletemporal": (0.55, 0.30),
    "L_inferiortemporal": (0.62, 0.20), "L_bankssts": (0.55, 0.45),
    "L_transversetemporal": (0.42, 0.45), "L_fusiform": (0.70, 0.13),
    "L_temporalpole": (0.20, 0.22), "L_entorhinal": (0.32, 0.10),
    "L_parahippocampal": (0.50, 0.10), "L_caudalanteriorcingulate": (0.30, 0.65),
    "L_rostralanteriorcingulate": (0.18, 0.55), "L_posteriorcingulate": (0.55, 0.70),
    "L_isthmuscingulate": (0.72, 0.60), "L_insula": (0.38, 0.48),
}
_DK_COORDS_RIGHT = {("R" + key[1:]): (1.0 - x, y) for key, (x, y) in _DK_COORDS_LEFT.items()}
_DK_COORDS = {**_DK_COORDS_LEFT, **_DK_COORDS_RIGHT}


def mix_hex(a: str, b: str, t: float) -> str:
    t = min(1.0, max(0.0, t))
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(ca[i] + (cb[i] - ca[i]) * t)):02X}" for i in range(3))


def diverging_color(value: float, bound: float) -> str:
    if not math.isfinite(value):
        return "#CBD5E1"
    t = 0.5 + 0.5 * max(-1.0, min(1.0, value / max(bound, 1e-9)))
    if t < 0.5:
        return mix_hex("#2166AC", "#F7F7F7", t / 0.5)
    return mix_hex("#F7F7F7", "#B2182B", (t - 0.5) / 0.5)


def sequential_color(value: float, vmin: float, vmax: float) -> str:
    if not math.isfinite(value):
        return "#CBD5E1"
    t = min(1.0, max(0.0, (value - vmin) / max(vmax - vmin, 1e-9)))
    if t < 0.5:
        return mix_hex("#E8F2EE", "#E9C46A", t / 0.5)
    return mix_hex("#E9C46A", "#D55E00", (t - 0.5) / 0.5)


def write_brain_panels(
    out_path: Path,
    s0_mean: dict[str, float],
    s1_mean: dict[str, float],
    predicted_s1: dict[str, dict[str, float]],
):
    observed_delta = {r: s1_mean[r] - s0_mean[r] for r in SELECTED_REGIONS}
    suvr_values = list(s0_mean.values()) + list(s1_mean.values())
    s_min = min(suvr_values)
    s_max = max(suvr_values)

    # Use only the empirical delta and the well-scaled models to set the diverging bound,
    # so BN-LTE-2's outliers do not saturate the rest of the panels.
    inrange_delta = [observed_delta[r] for r in SELECTED_REGIONS]
    for model, model_preds in predicted_s1.items():
        if model == "BN-LTE-2":
            continue
        for region in SELECTED_REGIONS:
            value = model_preds.get(region, float("nan")) - s0_mean[region]
            if math.isfinite(value):
                inrange_delta.append(value)
    bound = max([abs(v) for v in inrange_delta if math.isfinite(v)] + [0.02])

    panels = [
        {"title": "S0 baseline tau (group mean)", "values": s0_mean, "mode": "sequential", "vmin": s_min, "vmax": s_max},
        {"title": "S1 empirical follow-up tau", "values": s1_mean, "mode": "sequential", "vmin": s_min, "vmax": s_max},
        {"title": "S1 - S0 empirical change", "values": observed_delta, "mode": "diverging", "vmin": -bound, "vmax": bound},
    ]
    for model in MODEL_ORDER:
        if model not in predicted_s1:
            continue
        delta = {r: predicted_s1[model].get(r, float("nan")) - s0_mean[r] for r in SELECTED_REGIONS}
        max_abs = max([abs(v) for v in delta.values() if math.isfinite(v)] + [0.0])
        title = f"S1 predicted - S0 ({model})"
        if max_abs > bound:
            title += f"  [clipped @ {max_abs:.2f}]"
        panels.append({
            "title": title,
            "values": delta,
            "mode": "diverging",
            "vmin": -bound,
            "vmax": bound,
        })

    ncols = 4
    panel_w = 320
    panel_h = 230
    margin = 36
    header = 88
    nrows = math.ceil(len(panels) / ncols)
    width = margin * 2 + ncols * panel_w
    height = header + nrows * panel_h + 36

    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, "Group-average regional tau burden: S0, S1, and predicted change per model", size=22, weight="700"))
    parts.append(svg_text(36, 62, "Test split, mean of 156 held-out pairs. Coloured nodes are the 10 BN-SCM target regions; un-modelled DK regions are grey.", size=12, fill="#4B5563"))
    parts.append(svg_text(36, 80, "Sequential color: SUVR magnitude. Diverging color: red = increase, blue = decrease.", size=12, fill="#6B7280"))

    for idx, panel in enumerate(panels):
        x0 = margin + (idx % ncols) * panel_w
        y0 = header + (idx // ncols) * panel_h
        parts.extend(_draw_brain_panel(x0, y0, panel_w - 18, panel_h - 18, panel))

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _draw_brain_panel(x0, y0, width, height, panel):
    parts = [
        svg_rect(x0, y0, width, height, fill="#FFFFFF", stroke="#E5E7EB", radius=6),
        svg_text(x0 + 12, y0 + 22, panel["title"], size=13, weight="700"),
    ]
    brain_y = y0 + 36
    hemi_w = (width - 42) / 2
    hemi_h = height - 70
    vmin = panel["vmin"]
    vmax = panel["vmax"]
    mode = panel["mode"]

    for side_idx, side in enumerate(["L", "R"]):
        hx = x0 + 14 + side_idx * (hemi_w + 14)
        cy = brain_y + hemi_h / 2
        parts.append(
            f'<ellipse cx="{hx + hemi_w / 2:.2f}" cy="{cy:.2f}" rx="{hemi_w / 2:.2f}" '
            f'ry="{hemi_h / 2.15:.2f}" fill="#F3F4F6" stroke="#CBD5E1" stroke-width="1"/>'
        )
        for region, coord in _DK_COORDS.items():
            if not region.startswith(side + "_"):
                continue
            rx, ry = coord
            px = hx + rx * hemi_w
            py = brain_y + (1.02 - ry) / 1.08 * hemi_h
            if region in panel["values"]:
                value = panel["values"][region]
                if mode == "sequential":
                    color = sequential_color(value, vmin, vmax)
                else:
                    color = diverging_color(value, max(abs(vmin), abs(vmax)))
                radius = 6.0
                stroke = "#111827"
                opacity = 0.95
            else:
                color = "#E5E7EB"
                radius = 3.0
                stroke = "#9CA3AF"
                opacity = 0.55
            parts.append(svg_circle(px, py, radius, fill=color, stroke=stroke, opacity=opacity))

    # color legend strip
    lx = x0 + 14
    ly = y0 + height - 22
    for i in range(80):
        value = vmin + (vmax - vmin) * i / 79
        if mode == "sequential":
            color = sequential_color(value, vmin, vmax)
        else:
            color = diverging_color(value, max(abs(vmin), abs(vmax)))
        parts.append(svg_rect(lx + i * 2, ly, 2, 8, fill=color))
    parts.append(svg_text(lx, ly + 18, f"{vmin:+.3f}" if mode == "diverging" else f"{vmin:.3f}", size=9, fill="#6B7280"))
    parts.append(svg_text(lx + 158, ly + 18, f"{vmax:+.3f}" if mode == "diverging" else f"{vmax:.3f}", size=9, anchor="end", fill="#6B7280"))
    return parts


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bn_scm_metrics = load_region_metrics(BN_SCM_DIR / "outputs" / "model_comparison_region_metrics.csv")
    bn_lte2_metrics = load_region_metrics(BN_LTE2_DIR / "outputs" / "model_comparison_region_metrics.csv")
    test_rids = load_test_rids(BN_LTE2_DIR / "outputs" / "model_comparison_report.json")

    s0_mean, s1_mean, dt_mean = compute_test_s0_s1(test_rids)
    predicted_s1 = build_predicted_s1(s0_mean, dt_mean, bn_scm_metrics, bn_lte2_metrics)

    observed = {
        region: bn_scm_metrics.get("BayesianNetwork-SCM", {}).get(region, {}).get("observed_rate_mean", float("nan"))
        for region in SELECTED_REGIONS
    }
    predicted_rates: dict[str, dict[str, float]] = {}
    for model in MODEL_ORDER:
        source = bn_lte2_metrics if model == "BN-LTE-2" else bn_scm_metrics
        if model not in source:
            continue
        predicted_rates[model] = {
            region: source[model].get(region, {}).get("predicted_rate_mean", float("nan"))
            for region in SELECTED_REGIONS
        }

    write_rate_grouped_bars(OUT_DIR / "region_burden_change_bars.svg", observed, predicted_rates)
    write_brain_panels(OUT_DIR / "region_burden_change_brain.svg", s0_mean, s1_mean, predicted_s1)

    print("Wrote", OUT_DIR / "region_burden_change_bars.svg")
    print("Wrote", OUT_DIR / "region_burden_change_brain.svg")
    print(f"Test mean dt = {dt_mean:.3f} years")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
