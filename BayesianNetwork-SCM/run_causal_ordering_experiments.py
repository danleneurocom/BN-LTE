#!/usr/bin/env python3
"""Dynamic A/T/N causal-ordering experiments for ADNI BN-SCM."""

from __future__ import annotations

import csv
import html
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_network_scm.atn_data import build_atn_rate_dataset  # noqa: E402
from bayesian_network_scm.constraints import CausalOrderingConstraints  # noqa: E402
from bayesian_network_scm.dynamic_scm import fit_dynamic_scm  # noqa: E402
from bayesian_network_scm.pseudotime import fit_pseudotime  # noqa: E402
from bayesian_network_scm.reporting import compute_rate_metrics, make_subject_split, summarize_metric_rows  # noqa: E402
from run_hypothesis_experiments import load_region_labels, selected_to_full, write_brain_panel_grid  # noqa: E402


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PINK = "#CC79A7"
GOLD = "#E69F00"
GRAY = "#6B7280"
PURPLE = "#6A4C93"
RED = "#B91C1C"
PATHWAY_COLORS = {
    "A_to_T": BLUE,
    "P_to_T": PURPLE,
    "A_x_N_to_T": GOLD,
    "N_to_T": PINK,
    "T_to_N": ORANGE,
    "A_to_N": GREEN,
    "T_to_C": RED,
    "N_to_C": "#0F766E",
    "self_tau": "#111827",
    "other": GRAY,
}


def run_causal_ordering_experiments(
    *,
    project_root: str | Path = PROJECT_ROOT,
    output_dir: str | Path = THIS_DIR / "outputs" / "causal_ordering",
    primary_tolerance_days: int = 365,
    sensitivity_tolerances: tuple[int, ...] = (180, 365, 730),
    random_seed: int = 20260519,
    min_train_finite: int = 40,
    min_total_coverage: float = 0.25,
    max_parents_per_target: int = 10,
    activation_threshold: float = 0.05,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_path = resolve_output_path(output_dir, root)
    figure_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("[A/T/N ordering] Building tolerance sensitivity datasets...")
    sensitivity_rows = []
    datasets = {}
    for tolerance in sensitivity_tolerances:
        dataset = build_atn_rate_dataset(root, max_date_distance_days=int(tolerance))
        datasets[int(tolerance)] = dataset
        sensitivity_rows.extend(dataset_sensitivity_rows(dataset, int(tolerance)))
    dataset = datasets.get(int(primary_tolerance_days)) or build_atn_rate_dataset(root, max_date_distance_days=int(primary_tolerance_days))
    split = make_subject_split(dataset.metadata_rows, random_seed=random_seed)

    print("[A/T/N ordering] Fitting train-only tau-free pseudotime...")
    pseudotime = fit_pseudotime(
        dataset.feature_matrix,
        dataset.feature_names,
        split.train_indices,
        mode="tau_free",
        min_train_coverage=0.35,
    )
    selected_targets = select_fit_targets(dataset, split.train_indices, min_train_finite=min_train_finite, min_total_coverage=min_total_coverage)
    print(f"[A/T/N ordering] Fitting Dynamic BN-SCM for {len(selected_targets)} targets...")
    fit = fit_dynamic_scm(
        dataset,
        pseudotime,
        split.train_indices,
        target_names=selected_targets,
        constraints=CausalOrderingConstraints(dataset.variable_specs),
        max_parents_per_target=max_parents_per_target,
        cv_folds=3,
        edge_effect_threshold=0.01,
    )
    predicted_rates = fit.predict_rates(dataset)
    rate_metric_rows = compute_rate_metrics(dataset.target_rates, predicted_rates, dataset.target_names, split)
    metric_summary_rows = summarize_metric_rows(rate_metric_rows)

    print("[A/T/N ordering] Computing standardized edge activation curves...")
    edge_rows = standardized_edge_rows(fit, dataset, split.train_indices, activation_threshold=activation_threshold)
    pathway_rows = summarize_pathways(edge_rows)
    stage_rows = stage_dominance_rows(edge_rows)
    hypothesis_rows = hypothesis_test_rows(edge_rows, pathway_rows, stage_rows)

    print("[A/T/N ordering] Writing tables and figures...")
    figures = {
        "coverage": str(figure_dir / "fig1_dataset_coverage.svg"),
        "event_timeline": str(figure_dir / "fig2_event_activation_timeline.svg"),
        "particle_cascade": str(figure_dir / "fig3_causal_particles.svg"),
        "stage_dominance": str(figure_dir / "fig4_stage_dominance.svg"),
        "brain_empirical_predicted": str(figure_dir / "fig5_brain_empirical_predicted.svg"),
        "hypothesis_status": str(figure_dir / "fig6_causal_hypothesis_status.svg"),
    }
    write_coverage_svg(Path(figures["coverage"]), dataset, sensitivity_rows)
    write_event_timeline_svg(Path(figures["event_timeline"]), pathway_rows)
    write_particle_svg(Path(figures["particle_cascade"]), edge_rows)
    write_stage_dominance_svg(Path(figures["stage_dominance"]), stage_rows)
    write_brain_ordering_svg(Path(figures["brain_empirical_predicted"]), root, dataset, predicted_rates, split)
    write_hypothesis_svg(Path(figures["hypothesis_status"]), hypothesis_rows)

    tables = {
        "dataset_sensitivity": str(output_path / "dataset_sensitivity.csv"),
        "rate_metrics": str(output_path / "rate_metrics.csv"),
        "metric_summary": str(output_path / "metric_summary.csv"),
        "standardized_edges": str(output_path / "standardized_edges.csv"),
        "pathway_summary": str(output_path / "pathway_summary.csv"),
        "stage_dominance": str(output_path / "stage_dominance.csv"),
        "hypothesis_tests": str(output_path / "hypothesis_tests.csv"),
    }
    csv_write(Path(tables["dataset_sensitivity"]), sensitivity_rows)
    csv_write(Path(tables["rate_metrics"]), rate_metric_rows)
    csv_write(Path(tables["metric_summary"]), metric_summary_rows)
    csv_write(Path(tables["standardized_edges"]), edge_rows)
    csv_write(Path(tables["pathway_summary"]), pathway_rows)
    csv_write(Path(tables["stage_dominance"]), stage_rows)
    csv_write(Path(tables["hypothesis_tests"]), hypothesis_rows)

    report = {
        "purpose": "Dynamic causal event-ordering experiment over amyloid, tau, neurodegeneration/atrophy, and cognition.",
        "primary_tolerance_days": int(primary_tolerance_days),
        "activation_threshold": float(activation_threshold),
        "dataset": {
            "pairs": dataset.pair_count,
            "subjects": len({row["RID"] for row in dataset.metadata_rows}),
            "feature_count": len(dataset.feature_names),
            "target_count": len(dataset.target_names),
            "fit_target_count": len(selected_targets),
            "rows": dataset.report["rows"],
            "target_groups": dataset.report["target_groups"],
            "notes": dataset.report["notes"],
        },
        "split": split.report(),
        "pseudotime": pseudotime.report(dataset.feature_matrix, dataset.metadata_rows),
        "model": fit.report(),
        "hypothesis_tests": hypothesis_rows,
        "top_event_order": pathway_rows[:12],
        "tables": tables,
        "figures": figures,
        "interpretation_guardrails": [
            "Edges are standardized ridge varying effects, not posterior causal inclusion probabilities.",
            "Event onset is the first pseudotime where absolute standardized effect exceeds the configured threshold.",
            "Non-tau rates use source-specific scan intervals matched to tau-PET baseline/follow-up dates.",
            "A/T/N cross-direction edges are intentionally allowed for event-ordering tests.",
        ],
    }
    json_write(output_path / "causal_ordering_report.json", report)
    print(f"[A/T/N ordering] Wrote report: {output_path / 'causal_ordering_report.json'}")
    return report


def select_fit_targets(dataset: Any, train_indices: np.ndarray, *, min_train_finite: int, min_total_coverage: float) -> list[str]:
    selected = []
    train_indices = np.asarray(train_indices, dtype=int)
    for idx, name in enumerate(dataset.target_names):
        total_coverage = float(np.mean(np.isfinite(dataset.target_rates[:, idx])))
        train_finite = int(np.count_nonzero(np.isfinite(dataset.target_rates[train_indices, idx])))
        if total_coverage >= float(min_total_coverage) and train_finite >= int(min_train_finite):
            selected.append(name)
    return selected


def standardized_edge_rows(fit: Any, dataset: Any, train_indices: np.ndarray, *, activation_threshold: float) -> list[dict[str, Any]]:
    basis = fit.spline_basis.transform(fit.z_grid)
    train_indices = np.asarray(train_indices, dtype=int)
    rows = []
    for target_fit in fit.target_fits:
        target_idx = target_fit.target_index
        y = dataset.target_rates[train_indices, target_idx]
        y_sd = finite_std(y)
        if not np.isfinite(y_sd) or y_sd <= 0.0:
            y_sd = 1.0
        self_curve = target_fit.self_effect_curve(basis)
        baseline_sd = finite_std(dataset.target_baseline[train_indices, target_idx])
        rows.append(
            edge_curve_row(
                parent="self_history",
                target=target_fit.target_name,
                raw_effect=self_curve,
                predictor_sd=baseline_sd,
                target_sd=y_sd,
                z_grid=fit.z_grid,
                threshold=activation_threshold,
            )
        )
        for parent in target_fit.parent_names:
            feature_idx = dataset.feature_index(parent)
            predictor_sd = finite_std(dataset.feature_matrix[train_indices, feature_idx])
            rows.append(
                edge_curve_row(
                    parent=parent,
                    target=target_fit.target_name,
                    raw_effect=target_fit.parent_effect_curve(parent, basis),
                    predictor_sd=predictor_sd,
                    target_sd=y_sd,
                    z_grid=fit.z_grid,
                    threshold=activation_threshold,
                )
            )
    return sorted(rows, key=lambda row: (-float(row["max_abs_std_effect"]), row["target"], row["parent"]))


def edge_curve_row(
    *,
    parent: str,
    target: str,
    raw_effect: np.ndarray,
    predictor_sd: float,
    target_sd: float,
    z_grid: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    scale = predictor_sd / target_sd if np.isfinite(predictor_sd) and predictor_sd > 0.0 and target_sd > 0.0 else 1.0 / target_sd
    std_curve = np.asarray(raw_effect, dtype=float) * float(scale)
    abs_curve = np.abs(std_curve)
    max_idx = int(np.argmax(abs_curve)) if abs_curve.size else 0
    active = abs_curve >= float(threshold)
    onset = float(z_grid[int(np.argmax(active))]) if np.any(active) else float("nan")
    pathway = classify_pathway(parent, target)
    return {
        "parent": parent,
        "target": target,
        "parent_process": process_of(parent),
        "target_process": process_of(target),
        "pathway": pathway,
        "max_abs_std_effect": float(np.max(abs_curve)) if abs_curve.size else 0.0,
        "mean_abs_std_effect": float(np.mean(abs_curve)) if abs_curve.size else 0.0,
        "active_fraction": float(np.mean(active)) if active.size else 0.0,
        "onset_z": onset,
        "z_at_max_abs_effect": float(z_grid[max_idx]) if abs_curve.size else float("nan"),
        "std_effect_at_z_min": float(std_curve[0]) if std_curve.size else float("nan"),
        "std_effect_at_z_mid": float(std_curve[std_curve.size // 2]) if std_curve.size else float("nan"),
        "std_effect_at_z_max": float(std_curve[-1]) if std_curve.size else float("nan"),
        "abs_effect_early": float(np.mean(abs_curve[z_grid <= 0.33])) if abs_curve.size else float("nan"),
        "abs_effect_middle": float(np.mean(abs_curve[(z_grid > 0.33) & (z_grid <= 0.66)])) if abs_curve.size else float("nan"),
        "abs_effect_late": float(np.mean(abs_curve[z_grid > 0.66])) if abs_curve.size else float("nan"),
        "activation_threshold": float(threshold),
    }


def process_of(name: str) -> str:
    text = name.lower()
    if name == "self_history":
        return "self"
    if text.startswith("interaction:"):
        return "interaction"
    if "cognitive_rate" in text or any(token in text for token in ("adas", "mmse", "ravlt", "cdrsb")):
        return "C"
    if any(token in text for token in ("atrophy", "ashs", "mri_", "hippocampus", "ventricle", "volume", "thickness")):
        return "N"
    if "pt217" in text or "ptau" in text:
        return "P"
    if "amyloid" in text or "centiloid" in text or "ab42" in text:
        return "A"
    if "tau" in text:
        return "T"
    if any(token in text for token in ("nfl", "gfap")):
        return "N"
    return "other"


def classify_pathway(parent: str, target: str) -> str:
    target_process = process_of(target)
    parent_process = process_of(parent)
    if parent == "self_history" and target_process == "T":
        return "self_tau"
    if parent_process == "interaction" and target_process == "T":
        return "A_x_N_to_T" if "amyloid" in parent.lower() and "mri" in parent.lower() else "other"
    if parent_process == "A" and target_process == "T":
        return "A_to_T"
    if parent_process == "P" and target_process == "T":
        return "P_to_T"
    if parent_process == "N" and target_process == "T":
        return "N_to_T"
    if parent_process == "T" and target_process == "N":
        return "T_to_N"
    if parent_process == "A" and target_process == "N":
        return "A_to_N"
    if parent_process == "T" and target_process == "C":
        return "T_to_C"
    if parent_process == "N" and target_process == "C":
        return "N_to_C"
    return "other"


def summarize_pathways(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pathway in sorted({row["pathway"] for row in edge_rows if row["pathway"] != "other"}):
        items = [row for row in edge_rows if row["pathway"] == pathway]
        active = [row for row in items if np.isfinite(float(row["onset_z"]))]
        top = max(items, key=lambda row: float(row["max_abs_std_effect"])) if items else {}
        rows.append(
            {
                "pathway": pathway,
                "edge_count": len(items),
                "active_edge_count": len(active),
                "earliest_onset_z": min((float(row["onset_z"]) for row in active), default=float("nan")),
                "median_onset_z": float(np.median([float(row["onset_z"]) for row in active])) if active else float("nan"),
                "mean_max_abs_std_effect": float(np.mean([float(row["max_abs_std_effect"]) for row in items])) if items else float("nan"),
                "top_parent": top.get("parent", ""),
                "top_target": top.get("target", ""),
                "top_max_abs_std_effect": top.get("max_abs_std_effect", float("nan")),
            }
        )
    return sorted(rows, key=lambda row: (float(row["earliest_onset_z"]) if np.isfinite(float(row["earliest_onset_z"])) else 9.0, -float(row["mean_max_abs_std_effect"])))


def stage_dominance_rows(edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pathways = ["A_to_T", "P_to_T", "A_x_N_to_T", "N_to_T", "self_tau", "T_to_N", "A_to_N", "T_to_C", "N_to_C"]
    rows = []
    for pathway in pathways:
        items = [row for row in edge_rows if row["pathway"] == pathway]
        for stage, key in [("early", "abs_effect_early"), ("middle", "abs_effect_middle"), ("late", "abs_effect_late")]:
            values = [float(row[key]) for row in items if np.isfinite(float(row[key]))]
            rows.append(
                {
                    "pathway": pathway,
                    "stage": stage,
                    "edge_count": len(items),
                    "mean_abs_std_effect": float(np.mean(values)) if values else float("nan"),
                    "median_abs_std_effect": float(np.median(values)) if values else float("nan"),
                }
            )
    return rows


def hypothesis_test_rows(edge_rows: list[dict[str, Any]], pathway_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pathway = {row["pathway"]: row for row in pathway_rows}
    stage = {(row["pathway"], row["stage"]): row for row in stage_rows}
    axn = pathway.get("A_x_N_to_T", {})
    a_to_t = pathway.get("A_to_T", {})
    self_tau_early = stage.get(("self_tau", "early"), {}).get("mean_abs_std_effect", float("nan"))
    self_tau_late = stage.get(("self_tau", "late"), {}).get("mean_abs_std_effect", float("nan"))
    a_to_t_early = stage.get(("A_to_T", "early"), {}).get("mean_abs_std_effect", float("nan"))
    a_to_t_late = stage.get(("A_to_T", "late"), {}).get("mean_abs_std_effect", float("nan"))
    t_to_n = pathway.get("T_to_N", {})
    n_to_t = pathway.get("N_to_T", {})
    return [
        {
            "hypothesis": "Atrophy-gated tau ignition",
            "status": "supported" if float(axn.get("active_edge_count", 0)) > 0 and float(axn.get("mean_max_abs_std_effect", 0.0)) >= 0.05 else "weak / not confirmed",
            "primary_evidence": f"A_x_N_to_T active_edges={int(axn.get('active_edge_count', 0))}; mean_max={format_float(axn.get('mean_max_abs_std_effect'))}; earliest_z={format_float(axn.get('earliest_onset_z'))}",
            "comparison": f"A_to_T mean_max={format_float(a_to_t.get('mean_max_abs_std_effect'))}",
        },
        {
            "hypothesis": "Amyloid-to-tau early, tau-self late phase switch",
            "status": "supported" if finite_gt(a_to_t_early, a_to_t_late) and finite_gt(self_tau_late, self_tau_early) else "weak / not confirmed",
            "primary_evidence": f"A_to_T early={format_float(a_to_t_early)}, late={format_float(a_to_t_late)}; self_tau early={format_float(self_tau_early)}, late={format_float(self_tau_late)}",
            "comparison": "Looks for amyloid weakening and tau self-history strengthening across pseudotime.",
        },
        {
            "hypothesis": "Tau-to-atrophy stronger than atrophy-to-tau",
            "status": "supported" if float(t_to_n.get("mean_max_abs_std_effect", 0.0)) > float(n_to_t.get("mean_max_abs_std_effect", 0.0)) else "not confirmed",
            "primary_evidence": f"T_to_N mean_max={format_float(t_to_n.get('mean_max_abs_std_effect'))}; N_to_T mean_max={format_float(n_to_t.get('mean_max_abs_std_effect'))}",
            "comparison": f"T_to_N earliest_z={format_float(t_to_n.get('earliest_onset_z'))}; N_to_T earliest_z={format_float(n_to_t.get('earliest_onset_z'))}",
        },
        {
            "hypothesis": "Amyloid is permissive more than directly neurodegenerative",
            "status": "supported" if float(a_to_t.get("mean_max_abs_std_effect", 0.0)) > float(pathway.get("A_to_N", {}).get("mean_max_abs_std_effect", 0.0)) else "not confirmed",
            "primary_evidence": f"A_to_T mean_max={format_float(a_to_t.get('mean_max_abs_std_effect'))}; A_to_N mean_max={format_float(pathway.get('A_to_N', {}).get('mean_max_abs_std_effect'))}",
            "comparison": "Compares direct amyloid effects on future tau rate versus future atrophy rate.",
        },
    ]


def dataset_sensitivity_rows(dataset: Any, tolerance: int) -> list[dict[str, Any]]:
    rows = []
    rows.append({"tolerance_days": tolerance, "group": "rows", "name": "usable_pairs", "finite_rows": dataset.pair_count, "coverage": 1.0})
    for key in ("amyloid", "foxlab_bsi", "picsl_ashs", "fsx6", "all_main_modalities", "all_cortical_modalities"):
        rows.append(
            {
                "tolerance_days": tolerance,
                "group": "matched_modalities",
                "name": key,
                "finite_rows": int(dataset.report["rows"].get(key, 0)),
                "coverage": float(dataset.report["rows"].get(key, 0) / max(dataset.pair_count, 1)),
            }
        )
    for name, cov in dataset.report["target_coverage"].items():
        rows.append(
            {
                "tolerance_days": tolerance,
                "group": "target",
                "name": name,
                "finite_rows": cov["finite_rows"],
                "coverage": cov["coverage"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# SVG figures
# ---------------------------------------------------------------------------


def write_coverage_svg(path: Path, dataset: Any, sensitivity_rows: list[dict[str, Any]]) -> None:
    groups = ["amyloid_rate", "tau_rate", "atrophy_rate", "ashs_rate", "mri_thickness_rate", "mri_volume_rate", "cognitive_rate"]
    coverage_by_group = []
    for group in groups:
        values = [
            cov["coverage"]
            for name, cov in dataset.report["target_coverage"].items()
            if name.startswith(group + ":")
        ]
        coverage_by_group.append(float(np.mean(values)) if values else 0.0)
    width, height = 1040, 430
    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, "A/T/N Rate-Target Dataset Coverage", size=24, weight="700"))
    parts.append(svg_text(36, 64, "Primary tolerance: nearest source baseline/follow-up within 365 days of tau-PET anchors.", size=13, fill="#4B5563"))
    x0, y0, bar_w, bar_h = 220, 105, 650, 28
    for idx, (group, coverage) in enumerate(zip(groups, coverage_by_group, strict=True)):
        y = y0 + idx * 42
        parts.append(svg_text(36, y + 18, group, size=13, fill="#111827"))
        parts.append(svg_rect(x0, y, bar_w, bar_h, fill="#F3F4F6", stroke="#E5E7EB", radius=4))
        parts.append(svg_rect(x0, y, bar_w * coverage, bar_h, fill=PATHWAY_COLORS.get("A_to_T" if "amyloid" in group else "T_to_N", BLUE), radius=4))
        parts.append(svg_text(x0 + bar_w + 12, y + 18, f"{coverage:.2f}", size=13, fill="#111827"))
    rows = [row for row in sensitivity_rows if row["group"] == "matched_modalities" and row["name"] in {"all_main_modalities", "all_cortical_modalities"}]
    parts.append(svg_text(36, 390, "Sensitivity cohort sizes: " + "; ".join(f"{r['name']}@{r['tolerance_days']}d={r['finite_rows']}" for r in rows), size=12, fill="#4B5563"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_event_timeline_svg(path: Path, pathway_rows: list[dict[str, Any]]) -> None:
    width, height = 1080, 460
    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, "Causal Event Activation Timeline", size=24, weight="700"))
    parts.append(svg_text(36, 64, "Onset is first pseudotime where absolute standardized effect exceeds 0.05 target SD per predictor SD.", size=13, fill="#4B5563"))
    axis_x0, axis_x1, axis_y = 220, 1000, 372
    parts.append(svg_line(axis_x0, axis_y, axis_x1, axis_y, "#111827", width=1.3))
    for tick in np.linspace(0, 1, 6):
        x = axis_x0 + tick * (axis_x1 - axis_x0)
        parts.append(svg_line(x, axis_y - 5, x, axis_y + 5, "#111827", width=1.0))
        parts.append(svg_text(x - 8, axis_y + 24, f"{tick:.1f}", size=11, fill="#374151"))
    for idx, row in enumerate(pathway_rows[:9]):
        y = 110 + idx * 30
        onset = float(row["earliest_onset_z"])
        strength = float(row["mean_max_abs_std_effect"])
        color = PATHWAY_COLORS.get(str(row["pathway"]), GRAY)
        x = axis_x0 + (0.0 if not np.isfinite(onset) else onset) * (axis_x1 - axis_x0)
        radius = min(16, 5 + 12 * min(strength, 1.0))
        parts.append(svg_text(36, y + 4, str(row["pathway"]), size=13, weight="700", fill=color))
        parts.append(svg_line(axis_x0, y, axis_x1, y, "#E5E7EB", width=1.0))
        if np.isfinite(onset):
            parts.append(svg_circle(x, y, radius, fill=color, opacity=0.82))
            parts.append(svg_text(x + radius + 8, y + 4, f"z={onset:.2f}, strength={strength:.2f}", size=11, fill="#374151"))
        else:
            parts.append(svg_text(axis_x0, y + 4, "not active", size=11, fill="#6B7280"))
    parts.append(svg_text((axis_x0 + axis_x1) / 2 - 42, 430, "Pseudotime Z", size=13, weight="700", fill="#111827"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_particle_svg(path: Path, edge_rows: list[dict[str, Any]]) -> None:
    active = [row for row in edge_rows if row["pathway"] != "other" and np.isfinite(float(row["onset_z"]))]
    active = sorted(active, key=lambda row: (-float(row["max_abs_std_effect"]), str(row["pathway"])))[:90]
    width, height = 1120, 620
    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, "Causal Particles Across Disease Pseudotime", size=24, weight="700"))
    parts.append(svg_text(36, 64, "Each particle is one active edge. Horizontal position is onset Z; size is standardized effect strength.", size=13, fill="#4B5563"))
    x0, x1 = 80, 1040
    y0, y1 = 120, 560
    for tick in np.linspace(0, 1, 6):
        x = x0 + tick * (x1 - x0)
        parts.append(svg_line(x, y0 - 20, x, y1 + 8, "#E5E7EB", width=1.0))
        parts.append(svg_text(x - 8, y1 + 30, f"{tick:.1f}", size=11, fill="#4B5563"))
    rng = np.random.default_rng(20260519)
    lane_map = {name: idx for idx, name in enumerate(["A_to_T", "P_to_T", "A_x_N_to_T", "N_to_T", "self_tau", "T_to_N", "A_to_N", "T_to_C", "N_to_C"])}
    lane_h = (y1 - y0) / max(len(lane_map), 1)
    for pathway, lane in lane_map.items():
        y = y0 + lane * lane_h + lane_h / 2
        parts.append(svg_text(36, y + 4, pathway, size=10, fill=PATHWAY_COLORS.get(pathway, GRAY)))
        parts.append(svg_line(x0, y, x1, y, "#F3F4F6", width=1.0))
    for row in active:
        pathway = str(row["pathway"])
        lane = lane_map.get(pathway, 0)
        onset = float(row["onset_z"])
        strength = float(row["max_abs_std_effect"])
        x = x0 + onset * (x1 - x0)
        y = y0 + lane * lane_h + lane_h / 2 + rng.normal(0, lane_h * 0.18)
        radius = min(13, 3.5 + 10 * min(strength, 1.0))
        parts.append(svg_circle(x, y, radius, fill=PATHWAY_COLORS.get(pathway, GRAY), stroke="#FFFFFF", opacity=0.78))
    parts.append(svg_text(512, 604, "Pseudotime Z", size=13, weight="700"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_stage_dominance_svg(path: Path, stage_rows: list[dict[str, Any]]) -> None:
    pathways = ["A_to_T", "P_to_T", "A_x_N_to_T", "N_to_T", "self_tau", "T_to_N", "A_to_N", "T_to_C", "N_to_C"]
    stages = ["early", "middle", "late"]
    values = {(row["pathway"], row["stage"]): float(row["mean_abs_std_effect"]) for row in stage_rows}
    max_value = max([value for value in values.values() if np.isfinite(value)] or [1.0])
    width, height = 1080, 480
    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, "Stage-Specific Pathway Dominance", size=24, weight="700"))
    x0, y0 = 190, 92
    cell_w, cell_h = 86, 34
    for j, stage in enumerate(stages):
        parts.append(svg_text(x0 + j * cell_w + 18, y0 - 18, stage, size=12, weight="700"))
    for i, pathway in enumerate(pathways):
        y = y0 + i * (cell_h + 8)
        parts.append(svg_text(36, y + 22, pathway, size=12, fill=PATHWAY_COLORS.get(pathway, GRAY), weight="700"))
        for j, stage in enumerate(stages):
            value = values.get((pathway, stage), float("nan"))
            opacity = 0.12 if not np.isfinite(value) else 0.18 + 0.76 * min(value / max_value, 1.0)
            parts.append(svg_rect(x0 + j * cell_w, y, cell_w - 6, cell_h, fill=PATHWAY_COLORS.get(pathway, GRAY), opacity=opacity, radius=4))
            parts.append(svg_text(x0 + j * cell_w + 14, y + 22, format_float(value), size=10, fill="#111827"))
    parts.append(svg_text(36, 452, "Values are mean absolute standardized effects across edges in each pathway and pseudotime stage.", size=12, fill="#4B5563"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_brain_ordering_svg(path: Path, root: Path, dataset: Any, predicted_rates: np.ndarray, split: Any) -> None:
    region_labels = load_region_labels(root)
    selected = dataset.report["selected_tau_regions"]
    test_idx = np.asarray(split.test_indices, dtype=int)
    panels = []
    tau_regions = [name for name in selected if f"tau_rate:{name}" in dataset.target_names]
    tau_indices = [dataset.target_index(f"tau_rate:{name}") for name in tau_regions]
    tau_baseline = np.nanmean(dataset.target_baseline[test_idx[:, None], tau_indices], axis=0)
    tau_obs = np.nanmean(dataset.target_rates[test_idx[:, None], tau_indices], axis=0)
    tau_pred = np.nanmean(predicted_rates[test_idx[:, None], tau_indices], axis=0)
    tau_bound = max_abs_bound([tau_obs, tau_pred, tau_pred - tau_obs], floor=0.02)
    panels.extend(
        [
            {"title": "Initial tau PET", "values": selected_to_full(region_labels, tau_regions, tau_baseline), "mode": "sequential", "vmin": float(np.nanmin(tau_baseline)), "vmax": float(np.nanmax(tau_baseline))},
            {"title": "Empirical tau rate", "values": selected_to_full(region_labels, tau_regions, tau_obs), "mode": "diverging", "vmin": -tau_bound, "vmax": tau_bound},
            {"title": "Predicted tau rate", "values": selected_to_full(region_labels, tau_regions, tau_pred), "mode": "diverging", "vmin": -tau_bound, "vmax": tau_bound},
        ]
    )
    thick_regions = [name for name in selected if f"mri_thickness_rate:{name}" in dataset.target_names]
    thick_indices = [dataset.target_index(f"mri_thickness_rate:{name}") for name in thick_regions]
    if thick_indices:
        thick_baseline = np.nanmean(dataset.target_baseline[test_idx[:, None], thick_indices], axis=0)
        thick_obs = -np.nanmean(dataset.target_rates[test_idx[:, None], thick_indices], axis=0)
        thick_pred = -np.nanmean(predicted_rates[test_idx[:, None], thick_indices], axis=0)
        thick_bound = max_abs_bound([thick_obs, thick_pred, thick_pred - thick_obs], floor=0.005)
        panels.extend(
            [
                {"title": "Initial cortical thickness", "values": selected_to_full(region_labels, thick_regions, thick_baseline), "mode": "sequential", "vmin": float(np.nanmin(thick_baseline)), "vmax": float(np.nanmax(thick_baseline))},
                {"title": "Empirical thinning rate", "values": selected_to_full(region_labels, thick_regions, thick_obs), "mode": "diverging", "vmin": -thick_bound, "vmax": thick_bound},
                {"title": "Predicted thinning rate", "values": selected_to_full(region_labels, thick_regions, thick_pred), "mode": "diverging", "vmin": -thick_bound, "vmax": thick_bound},
            ]
        )
    write_brain_panel_grid(path, panels, region_labels, "A/T/N Brain Maps: Initial State, Empirical Rate, BN-SCM Predicted Rate", ncols=3)


def write_hypothesis_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1120, 360
    parts = svg_header(width, height)
    parts.append(svg_text(36, 38, "Causal-Ordering Hypothesis Status", size=24, weight="700"))
    y0 = 96
    for idx, row in enumerate(rows):
        y = y0 + idx * 62
        status = str(row["status"])
        color = GREEN if status == "supported" else GOLD if "weak" in status else GRAY
        parts.append(svg_rect(36, y - 28, 1048, 50, fill="#FFFFFF", stroke="#E5E7EB", radius=6))
        parts.append(svg_rect(36, y - 28, 8, 50, fill=color, radius=4))
        parts.append(svg_text(58, y - 8, row["hypothesis"], size=14, weight="700"))
        parts.append(svg_text(408, y - 8, status, size=13, weight="700", fill=color))
        parts.append(svg_text(560, y - 8, row["primary_evidence"], size=11, fill="#374151"))
        parts.append(svg_text(560, y + 10, row["comparison"], size=10, fill="#6B7280"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}</style>',
    ]


def svg_text(x: float, y: float, text: Any, *, size: int = 12, fill: str = "#111827", weight: str = "400") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" font-weight="{weight}">{html.escape(str(text))}</text>'


def svg_rect(x: float, y: float, width: float, height: float, *, fill: str, stroke: str | None = None, radius: float = 0.0, opacity: float = 1.0) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="{radius:.2f}" fill="{fill}"{stroke_attr} opacity="{opacity:.3f}"/>'


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str, *, width: float = 1.0) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width:.2f}"/>'


def svg_circle(x: float, y: float, radius: float, *, fill: str, stroke: str | None = None, opacity: float = 1.0) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="0.8"' if stroke else ""
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}"{stroke_attr} opacity="{opacity:.3f}"/>'


# ---------------------------------------------------------------------------
# IO and numeric helpers
# ---------------------------------------------------------------------------


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def resolve_output_path(path_value: str | Path, root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def finite_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size >= 2 else float("nan")


def finite_gt(left: Any, right: Any) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return np.isfinite(left_value) and np.isfinite(right_value) and left_value > right_value


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.3f}" if abs(number) < 10 else f"{number:.2f}"


def max_abs_bound(values: list[np.ndarray], *, floor: float) -> float:
    finite = []
    for value in values:
        arr = np.asarray(value, dtype=float)
        finite.extend(np.abs(arr[np.isfinite(arr)]).tolist())
    return max(float(np.max(finite)) if finite else float(floor), float(floor))


def main() -> int:
    run_causal_ordering_experiments()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
