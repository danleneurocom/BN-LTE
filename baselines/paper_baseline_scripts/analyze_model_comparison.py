#!/usr/bin/env python3
"""Visual comparison of Dynamic BN-SCM against NDM, ESM, and SIR.

The comparison is intentionally tied to the BN-SCM experiment split and the
selected temporolimbic/parietal tau targets used by the causal model. NDM, ESM,
and SIR are fit on the full 68-region ENIGMA/aparc tau state and evaluated on
the same selected regions, so network models retain their full graph context.
"""

from __future__ import annotations

import csv
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from bayesian_network_scm.data import build_multimodal_pair_dataset  # noqa: E402
from bayesian_network_scm.dynamic_scm import fit_dynamic_scm  # noqa: E402
from bayesian_network_scm.pseudotime import fit_pseudotime  # noqa: E402
from bayesian_network_scm.reporting import make_subject_split  # noqa: E402
from spread_toolbox.forecasting import MinMaxStateScaler, load_forecast_dataset, load_labeled_matrix  # noqa: E402
from spread_toolbox.io_adni import load_yaml_config  # noqa: E402
from spread_toolbox.models.esm import EpidemicSpreadingModel  # noqa: E402
from spread_toolbox.models.ndm import NetworkDiffusionModel  # noqa: E402
from spread_toolbox.models.sir import GraphSIRModel  # noqa: E402


MODEL_ORDER = ["BayesianNetwork-SCM", "NDM", "ESM", "SIR", "S0 persistence"]
MODEL_COLORS = {
    "BayesianNetwork-SCM": "#D55E00",
    "NDM": "#0072B2",
    "ESM": "#009E73",
    "SIR": "#CC79A7",
    "S0 persistence": "#6B7280",
}
REGION_SHORT_NAMES = {
    "L_entorhinal": "L-Ent",
    "R_entorhinal": "R-Ent",
    "L_fusiform": "L-Fus",
    "R_fusiform": "R-Fus",
    "L_inferiortemporal": "L-IT",
    "R_inferiortemporal": "R-IT",
    "L_middletemporal": "L-MT",
    "R_middletemporal": "R-MT",
    "L_inferiorparietal": "L-IP",
    "R_inferiorparietal": "R-IP",
}


def run_analysis(
    *,
    project_root: str | Path = PROJECT_ROOT,
    output_dir: str | Path = THIS_DIR / "outputs",
    random_seed: int = 20260519,
    max_parents_per_target: int = 6,
) -> dict[str, Any]:
    """Fit all models on the shared split and write tables plus SVG figures."""

    root = Path(project_root).resolve()
    output_path = resolve_path(output_dir, root)
    figure_dir = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("Building BN-SCM multimodal dataset...")
    bn_dataset = build_multimodal_pair_dataset(root)
    split = make_subject_split(bn_dataset.metadata_rows, random_seed=random_seed)
    selected_regions = list(bn_dataset.report["selected_tau_regions"])
    selected_target_names = [f"tau_rate:{region}" for region in selected_regions]
    selected_target_indices = [bn_dataset.target_index(name) for name in selected_target_names]

    baseline_selected = bn_dataset.target_baseline[:, selected_target_indices]
    observed_selected = bn_dataset.target_observed[:, selected_target_indices]
    time_years = bn_dataset.time_years

    print("Fitting explainable pseudotime and Dynamic BN-SCM...")
    pseudotime = fit_pseudotime(
        bn_dataset.feature_matrix,
        bn_dataset.feature_names,
        split.train_indices,
        mode="tau_free",
    )
    bn_fit = fit_dynamic_scm(
        bn_dataset,
        pseudotime,
        split.train_indices,
        target_names=selected_target_names,
        max_parents_per_target=max_parents_per_target,
    )
    bn_predicted_rates = bn_fit.predict_rates(bn_dataset)[:, selected_target_indices]
    bn_predicted_observed = baseline_selected + time_years[:, None] * bn_predicted_rates

    print("Loading full 68-region forecast dataset and ENIGMA matrices...")
    config_path = root / "experiments" / "group_average_enigma" / "config.yaml"
    if not config_path.exists():
        config_path = root / "experiments" / "group_average_enigma" / "config.example.yaml"
    config = load_yaml_config(config_path)
    forecast_dataset = load_forecast_dataset(config, root)
    assert_aligned_pairs(forecast_dataset.pairs, bn_dataset.metadata_rows)
    output_root = root / config["paths"]["output_dir"]
    outputs = config.get("outputs", {})
    labels, adjacency = load_labeled_matrix(output_root / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv"))
    lap_labels, laplacian = load_labeled_matrix(output_root / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"))
    if labels != forecast_dataset.region_labels or lap_labels != forecast_dataset.region_labels:
        raise ValueError("ENIGMA matrix labels do not match the forecast dataset region labels.")
    selected_region_indices = [forecast_dataset.region_labels.index(region) for region in selected_regions]

    predictions: dict[str, np.ndarray] = {
        "BayesianNetwork-SCM": bn_predicted_observed,
        "S0 persistence": baseline_selected.copy(),
    }
    fit_reports: dict[str, dict[str, Any]] = {
        "BayesianNetwork-SCM": {
            "equation": "dX_j/dt = a_j(Z) + self_j(Z) X_j(0) + sum_l b_jl(Z) X_l(0)",
            "fit_scope": "selected tau rates only",
            "max_parents_per_target": int(max_parents_per_target),
            "pseudotime_mode": pseudotime.mode,
            "target_count": len(selected_target_names),
        },
        "S0 persistence": {
            "equation": "S(t) = S(0)",
            "fit_scope": "no learned parameters",
            "target_count": len(selected_target_names),
        },
    }

    print("Fitting NDM on the full graph...")
    ndm = NetworkDiffusionModel(laplacian)
    ndm_fit = ndm.fit_global_rho(
        forecast_dataset.baseline[split.train_indices],
        forecast_dataset.observed[split.train_indices],
        forecast_dataset.time_years[split.train_indices],
        bounds=parameter_bounds(config, "rho", (0.0, 10.0)),
    )
    ndm_full = ndm.predict(forecast_dataset.baseline, forecast_dataset.time_years, ndm_fit.rho)
    predictions["NDM"] = ndm_full[:, selected_region_indices]
    fit_reports["NDM"] = {
        "equation": "dS/dt = -rho L S",
        "fit_scope": "full 68-region tau graph",
        "rho": ndm_fit.rho,
        "train_mse": ndm_fit.train_mse,
        "optimizer_success": ndm_fit.optimizer_success,
        "optimizer_message": ndm_fit.optimizer_message,
    }

    print("Fitting ESM on the full graph...")
    scaler = MinMaxStateScaler.fit(
        forecast_dataset.baseline[split.train_indices],
        forecast_dataset.observed[split.train_indices],
    )
    baseline_scaled = scaler.transform(forecast_dataset.baseline)
    observed_scaled = scaler.transform(forecast_dataset.observed)
    esm = EpidemicSpreadingModel(
        adjacency,
        steps_per_year=int(config.get("modeling", {}).get("esm_steps_per_year", 12)),
    )
    esm_fit = esm.fit_global_beta(
        baseline_scaled[split.train_indices],
        observed_scaled[split.train_indices],
        forecast_dataset.time_years[split.train_indices],
        bounds=parameter_bounds(config, "beta", (0.0, 10.0)),
    )
    esm_full = scaler.inverse_transform(esm.predict(baseline_scaled, forecast_dataset.time_years, esm_fit.beta))
    predictions["ESM"] = esm_full[:, selected_region_indices]
    fit_reports["ESM"] = {
        "equation": "dS/dt = beta (1 - S) W S",
        "fit_scope": "full 68-region tau graph, min-max state scaled",
        "beta": esm_fit.beta,
        "train_mse_scaled": esm_fit.train_mse,
        "optimizer_success": esm_fit.optimizer_success,
        "optimizer_message": esm_fit.optimizer_message,
    }

    print("Fitting SIR on the full graph...")
    sir = GraphSIRModel(
        adjacency,
        steps_per_year=int(config.get("modeling", {}).get("sir_steps_per_year", 12)),
    )
    sir_fit = sir.fit_global_parameters(
        baseline_scaled[split.train_indices],
        observed_scaled[split.train_indices],
        forecast_dataset.time_years[split.train_indices],
        beta_bounds=parameter_bounds(config, "beta", (0.0, 10.0)),
        gamma_bounds=parameter_bounds(config, "gamma", (0.0, 10.0)),
        maxiter=int(config.get("modeling", {}).get("sir_optimizer_maxiter", 80)),
    )
    sir_full = scaler.inverse_transform(
        sir.predict(baseline_scaled, forecast_dataset.time_years, beta=sir_fit.beta, gamma=sir_fit.gamma)
    )
    predictions["SIR"] = sir_full[:, selected_region_indices]
    fit_reports["SIR"] = {
        "equation": "dI/dt = beta S (W I) - gamma I; observed tau = I",
        "fit_scope": "full 68-region tau graph, min-max state scaled",
        "beta": sir_fit.beta,
        "gamma": sir_fit.gamma,
        "train_mse_scaled": sir_fit.train_mse,
        "optimizer_success": sir_fit.optimizer_success,
        "optimizer_message": sir_fit.optimizer_message,
        "optimizer_iterations": sir_fit.optimizer_iterations,
        "optimizer_evaluations": sir_fit.optimizer_evaluations,
    }

    print("Computing comparison metrics...")
    pair_rows = []
    region_rows = []
    for model in MODEL_ORDER:
        pair_rows.extend(
            compute_pair_metric_rows(
                model,
                predictions[model],
                baseline_selected,
                observed_selected,
                time_years,
                bn_dataset.metadata_rows,
                split,
            )
        )
        region_rows.extend(
            compute_region_metric_rows(
                model,
                predictions[model],
                baseline_selected,
                observed_selected,
                time_years,
                selected_regions,
                split,
            )
        )
    summary_rows = summarize_pair_metrics(pair_rows)

    edge_curves = top_edge_curves(bn_fit, top_k=8)
    z_values = pseudotime.transform(bn_dataset.feature_matrix)
    figures = {
        "test_metric_bars": str(figure_dir / "model_comparison_test_metrics.svg"),
        "regional_rate_mae_heatmap": str(figure_dir / "regional_rate_mae_heatmap.svg"),
        "rate_scatter": str(figure_dir / "predicted_vs_observed_tau_rate.svg"),
        "bn_scm_edge_effects": str(figure_dir / "bn_scm_top_edge_effects.svg"),
        "pseudotime_diagnosis": str(figure_dir / "pseudotime_diagnosis.svg"),
    }
    write_metric_bars_svg(Path(figures["test_metric_bars"]), summary_rows)
    write_region_heatmap_svg(Path(figures["regional_rate_mae_heatmap"]), region_rows, selected_regions)
    write_rate_scatter_svg(
        Path(figures["rate_scatter"]),
        predictions,
        baseline_selected,
        observed_selected,
        time_years,
        split.test_indices,
    )
    write_edge_effect_svg(Path(figures["bn_scm_edge_effects"]), edge_curves)
    write_pseudotime_svg(Path(figures["pseudotime_diagnosis"]), z_values, bn_dataset.metadata_rows)

    csv_write(output_path / "model_comparison_pair_metrics.csv", pair_rows)
    csv_write(output_path / "model_comparison_summary.csv", summary_rows)
    csv_write(output_path / "model_comparison_region_metrics.csv", region_rows)

    report = {
        "purpose": (
            "Compare Dynamic BN-SCM against NDM, ESM, and SIR on the same BN-SCM "
            "subject split and selected regional tau targets."
        ),
        "comparison_scope": {
            "bn_scm_fit": "selected regional tau annualized rates",
            "graph_models_fit": "full 68-region ENIGMA/aparc tau states",
            "evaluation_targets": selected_regions,
            "metric_unit": "follow-up tau SUVR and annualized tau SUVR/year",
        },
        "split": split.report(),
        "data": {
            "pairs": bn_dataset.pair_count,
            "subjects": len({row["RID"] for row in bn_dataset.metadata_rows}),
            "selected_regions": selected_regions,
            "feature_count": len(bn_dataset.feature_names),
        },
        "pseudotime": pseudotime.report(bn_dataset.feature_matrix, bn_dataset.metadata_rows),
        "fit_reports": fit_reports,
        "test_metric_summary": nested_metric_summary(summary_rows, split_name="test"),
        "figures": figures,
        "tables": {
            "pair_metrics": str(output_path / "model_comparison_pair_metrics.csv"),
            "summary": str(output_path / "model_comparison_summary.csv"),
            "region_metrics": str(output_path / "model_comparison_region_metrics.csv"),
        },
        "limitations": [
            "This is a forecasting comparison, not proof of causal identifiability.",
            "BN-SCM currently uses ridge-estimated varying effects as a prototype rather than full posterior graph MCMC.",
            "Graph baselines are fit on full 68-region tau states, while BN-SCM is fit only on the selected regional rates.",
            "The selected-region evaluation emphasizes temporolimbic and inferior parietal tau spread, not whole-cortex accuracy.",
        ],
    }
    json_write(output_path / "model_comparison_report.json", report)
    print(f"Wrote analysis report: {output_path / 'model_comparison_report.json'}")
    return report


def resolve_path(path: str | Path, project_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_root / value


def parameter_bounds(config: dict[str, Any], name: str, default: tuple[float, float]) -> tuple[float, float]:
    values = config.get("modeling", {}).get("parameter_bounds", {}).get(name, default)
    return float(values[0]), float(values[1])


def assert_aligned_pairs(forecast_pairs: list[dict[str, str]], metadata_rows: list[dict[str, Any]]) -> None:
    if len(forecast_pairs) != len(metadata_rows):
        raise ValueError(f"Pair count mismatch: forecast={len(forecast_pairs)}, bn_scm={len(metadata_rows)}")
    for idx, (forecast_row, bn_row) in enumerate(zip(forecast_pairs, metadata_rows, strict=True)):
        if (
            str(forecast_row.get("RID", "")) != str(bn_row.get("RID", ""))
            or str(forecast_row.get("baseline_tau_date", "")) != str(bn_row.get("baseline_tau_date", ""))
            or str(forecast_row.get("target_tau_date", "")) != str(bn_row.get("target_tau_date", ""))
        ):
            raise ValueError(f"Forecast and BN-SCM pair ordering diverges at row {idx}.")


def compute_pair_metric_rows(
    model: str,
    predicted: np.ndarray,
    baseline: np.ndarray,
    observed: np.ndarray,
    time_years: np.ndarray,
    metadata_rows: list[dict[str, Any]],
    split: Any,
) -> list[dict[str, Any]]:
    split_labels = split_labels_by_index(split, observed.shape[0])
    rows = []
    for idx in range(observed.shape[0]):
        base = baseline[idx]
        y = observed[idx]
        pred = predicted[idx]
        dt = float(time_years[idx])
        obs_rate = (y - base) / dt
        pred_rate = (pred - base) / dt
        rows.append(
            {
                "model": model,
                "split": split_labels[idx],
                "RID": metadata_rows[idx]["RID"],
                "PTID": metadata_rows[idx]["PTID"],
                "TRACER": metadata_rows[idx]["TRACER"],
                "baseline_tau_date": metadata_rows[idx]["baseline_tau_date"],
                "target_tau_date": metadata_rows[idx]["target_tau_date"],
                "target_time_years": dt,
                "dx_nearest_baseline": metadata_rows[idx].get("dx_nearest_baseline", ""),
                "mae_suvr": finite_mean_abs(pred - y),
                "rmse_suvr": finite_rmse(pred - y),
                "rate_mae": finite_mean_abs(pred_rate - obs_rate),
                "rate_rmse": finite_rmse(pred_rate - obs_rate),
                "subject_spearman": safe_correlation(y, pred, rank=True),
                "delta_spearman": safe_correlation(y - base, pred - base, rank=True),
                "delta_pearson": safe_correlation(y - base, pred - base, rank=False),
            }
        )
    return rows


def compute_region_metric_rows(
    model: str,
    predicted: np.ndarray,
    baseline: np.ndarray,
    observed: np.ndarray,
    time_years: np.ndarray,
    regions: list[str],
    split: Any,
) -> list[dict[str, Any]]:
    rows = []
    for split_name, indices in split_items(split, observed.shape[0]):
        if split_name == "all":
            continue
        dt = time_years[indices, None]
        observed_rate = (observed[indices] - baseline[indices]) / dt
        predicted_rate = (predicted[indices] - baseline[indices]) / dt
        for region_idx, region in enumerate(regions):
            obs = observed[indices, region_idx]
            pred = predicted[indices, region_idx]
            obs_rate = observed_rate[:, region_idx]
            pred_rate = predicted_rate[:, region_idx]
            rows.append(
                {
                    "model": model,
                    "split": split_name,
                    "region": region,
                    "region_short": REGION_SHORT_NAMES.get(region, region),
                    "mae_suvr": finite_mean_abs(pred - obs),
                    "rate_mae": finite_mean_abs(pred_rate - obs_rate),
                    "rate_rmse": finite_rmse(pred_rate - obs_rate),
                    "rate_spearman": safe_correlation(obs_rate, pred_rate, rank=True),
                    "observed_rate_mean": finite_mean(obs_rate),
                    "predicted_rate_mean": finite_mean(pred_rate),
                }
            )
    return rows


def summarize_pair_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "mae_suvr",
        "rmse_suvr",
        "rate_mae",
        "rate_rmse",
        "subject_spearman",
        "delta_spearman",
        "delta_pearson",
    ]
    output = []
    for model in MODEL_ORDER:
        model_rows = [row for row in rows if row["model"] == model]
        for split_name in ("train", "validation", "test", "all"):
            split_rows = model_rows if split_name == "all" else [row for row in model_rows if row["split"] == split_name]
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in split_rows if is_finite(row[metric])], dtype=float)
                if values.size == 0:
                    continue
                output.append(
                    {
                        "model": model,
                        "split": split_name,
                        "metric": metric,
                        "n": int(values.size),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                    }
                )
    return output


def nested_metric_summary(rows: list[dict[str, Any]], *, split_name: str) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["split"] != split_name:
            continue
        model = str(row["model"])
        metric = str(row["metric"])
        summary.setdefault(model, {})[metric] = float(row["median"])
    return summary


def split_items(split: Any, row_count: int) -> list[tuple[str, np.ndarray]]:
    return [
        ("train", split.train_indices),
        ("validation", split.validation_indices),
        ("test", split.test_indices),
        ("all", np.arange(row_count, dtype=int)),
    ]


def split_labels_by_index(split: Any, row_count: int) -> list[str]:
    labels = ["unknown"] * row_count
    for name, indices in split_items(split, row_count):
        if name == "all":
            continue
        for idx in indices:
            labels[int(idx)] = name
    return labels


def finite_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    mask = np.isfinite(array)
    return float(np.mean(array[mask])) if np.any(mask) else float("nan")


def finite_mean_abs(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    mask = np.isfinite(array)
    return float(np.mean(np.abs(array[mask]))) if np.any(mask) else float("nan")


def finite_rmse(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    mask = np.isfinite(array)
    return float(np.sqrt(np.mean(array[mask] ** 2))) if np.any(mask) else float("nan")


def safe_correlation(a: np.ndarray, b: np.ndarray, *, rank: bool) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if rank:
        x = rankdata(x)
        y = rankdata(y)
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def top_edge_curves(bn_fit: Any, *, top_k: int) -> list[dict[str, Any]]:
    basis = bn_fit.spline_basis.transform(bn_fit.z_grid)
    rows = []
    for target_fit in bn_fit.target_fits:
        for parent in target_fit.parent_names:
            effect = target_fit.parent_effect_curve(parent, basis)
            rows.append(
                {
                    "parent": parent,
                    "target": target_fit.target_name,
                    "z": bn_fit.z_grid.copy(),
                    "effect": effect,
                    "max_abs_effect": float(np.max(np.abs(effect))) if effect.size else 0.0,
                }
            )
    rows.sort(key=lambda row: float(row["max_abs_effect"]), reverse=True)
    return rows[: int(top_k)]


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_metric_bars_svg(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("mae_suvr", "Follow-up MAE", "SUVR, lower is better"),
        ("rate_mae", "Rate MAE", "SUVR/year, lower is better"),
        ("subject_spearman", "Spatial Spearman", "within-subject tau rank"),
        ("delta_spearman", "Delta Spearman", "within-subject spread rank"),
    ]
    lookup = {
        (row["model"], row["split"], row["metric"]): float(row["median"])
        for row in summary_rows
        if row["split"] == "test" and is_finite(row["median"])
    }
    width = 1200
    height = 760
    margin = 46
    panel_gap = 34
    panel_w = (width - 2 * margin - panel_gap) / 2
    panel_h = (height - 118 - panel_gap) / 2
    parts = svg_header(width, height)
    parts.append(svg_text(36, 36, "Held-out Test Performance: BN-SCM vs Graph Spreading Models", size=22, weight="700"))
    parts.append(
        svg_text(
            36,
            62,
            "Bars show median pair-level performance across the shared BN-SCM test subjects and selected tau regions.",
            size=13,
            fill="#4B5563",
        )
    )
    for metric_idx, (metric, title, subtitle) in enumerate(metrics):
        col = metric_idx % 2
        row = metric_idx // 2
        x0 = margin + col * (panel_w + panel_gap)
        y0 = 96 + row * (panel_h + panel_gap)
        values = [lookup.get((model, "test", metric), float("nan")) for model in MODEL_ORDER]
        finite_values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
        max_value = float(np.max(finite_values)) if finite_values.size else 1.0
        min_value = float(np.min(finite_values)) if finite_values.size else 0.0
        if "spearman" in metric:
            axis_min = min(-0.05, min_value - 0.05)
            axis_max = max(0.15, max_value + 0.08)
        else:
            axis_min = 0.0
            axis_max = max(max_value * 1.12, 1.0e-6)
        parts.extend(draw_bar_panel(x0, y0, panel_w, panel_h, title, subtitle, MODEL_ORDER, values, axis_min, axis_max))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def draw_bar_panel(
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    models: list[str],
    values: list[float],
    axis_min: float,
    axis_max: float,
) -> list[str]:
    parts = [
        svg_rect(x0, y0, width, height, fill="#FFFFFF", stroke="#E5E7EB", radius=6),
        svg_text(x0 + 18, y0 + 28, title, size=16, weight="700"),
        svg_text(x0 + 18, y0 + 48, subtitle, size=12, fill="#6B7280"),
    ]
    label_w = 154
    bar_x = x0 + label_w
    bar_w = width - label_w - 72
    bar_h = 24
    gap = 16
    start_y = y0 + 76
    zero_x = bar_x + ((0.0 - axis_min) / max(axis_max - axis_min, 1.0e-12)) * bar_w
    parts.append(svg_line(zero_x, start_y - 8, zero_x, start_y + len(models) * (bar_h + gap) - gap + 8, "#9CA3AF", 1))
    for idx, (model, value) in enumerate(zip(models, values, strict=True)):
        y = start_y + idx * (bar_h + gap)
        parts.append(svg_text(x0 + 18, y + 17, model, size=12, fill="#111827", weight="700" if model == "BayesianNetwork-SCM" else "400"))
        if np.isfinite(value):
            value_x = bar_x + ((value - axis_min) / max(axis_max - axis_min, 1.0e-12)) * bar_w
            left = min(zero_x, value_x)
            w = max(abs(value_x - zero_x), 1.0)
            parts.append(svg_rect(left, y, w, bar_h, fill=MODEL_COLORS[model], radius=4, opacity=0.9))
            parts.append(svg_text(bar_x + bar_w + 10, y + 17, format_metric(value), size=12, fill="#111827"))
        else:
            parts.append(svg_text(bar_x + bar_w + 10, y + 17, "NA", size=12, fill="#6B7280"))
    parts.append(svg_text(bar_x, y0 + height - 14, format_metric(axis_min), size=10, fill="#6B7280"))
    parts.append(svg_text(bar_x + bar_w - 22, y0 + height - 14, format_metric(axis_max), size=10, fill="#6B7280"))
    return parts


def write_region_heatmap_svg(path: Path, region_rows: list[dict[str, Any]], selected_regions: list[str]) -> None:
    rows = [row for row in region_rows if row["split"] == "test"]
    lookup = {(row["model"], row["region"]): float(row["rate_mae"]) for row in rows}
    values = np.asarray([value for value in lookup.values() if np.isfinite(value)], dtype=float)
    vmin = float(np.quantile(values, 0.05)) if values.size else 0.0
    vmax = float(np.quantile(values, 0.95)) if values.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    width = 1180
    height = 460
    left = 170
    top = 106
    cell_w = 88
    cell_h = 42
    parts = svg_header(width, height)
    parts.append(svg_text(36, 36, "Regional Test Rate MAE", size=22, weight="700"))
    parts.append(svg_text(36, 62, "Lower values indicate more accurate annualized tau-rate forecasts in each selected region.", size=13, fill="#4B5563"))
    for col, region in enumerate(selected_regions):
        x = left + col * cell_w + cell_w / 2
        parts.append(svg_text(x, top - 16, REGION_SHORT_NAMES.get(region, region), size=11, anchor="middle", fill="#374151"))
    for row_idx, model in enumerate(MODEL_ORDER):
        y = top + row_idx * cell_h
        parts.append(svg_text(36, y + 27, model, size=12, fill="#111827", weight="700" if model == "BayesianNetwork-SCM" else "400"))
        for col, region in enumerate(selected_regions):
            value = lookup.get((model, region), float("nan"))
            x = left + col * cell_w
            color = color_scale(value, vmin, vmax)
            parts.append(svg_rect(x, y, cell_w - 2, cell_h - 2, fill=color, stroke="#FFFFFF"))
            parts.append(svg_text(x + cell_w / 2, y + 26, format_metric(value), size=10, anchor="middle", fill="#111827"))
    legend_x = left
    legend_y = top + len(MODEL_ORDER) * cell_h + 44
    parts.append(svg_text(36, legend_y + 14, "Rate MAE", size=12, weight="700"))
    for i in range(160):
        value = vmin + (vmax - vmin) * i / 159
        parts.append(svg_rect(legend_x + i * 2, legend_y, 2, 14, fill=color_scale(value, vmin, vmax)))
    parts.append(svg_text(legend_x, legend_y + 32, format_metric(vmin), size=10, fill="#4B5563"))
    parts.append(svg_text(legend_x + 320, legend_y + 32, format_metric(vmax), size=10, anchor="end", fill="#4B5563"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_rate_scatter_svg(
    path: Path,
    predictions: dict[str, np.ndarray],
    baseline: np.ndarray,
    observed: np.ndarray,
    time_years: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    observed_rate = (observed[test_indices] - baseline[test_indices]) / time_years[test_indices, None]
    predicted_rates = {
        model: (pred[test_indices] - baseline[test_indices]) / time_years[test_indices, None]
        for model, pred in predictions.items()
    }
    all_values = [observed_rate.reshape(-1)]
    all_values.extend(pred.reshape(-1) for pred in predicted_rates.values())
    values = np.concatenate(all_values)
    values = values[np.isfinite(values)]
    lower = float(np.quantile(values, 0.01)) if values.size else -0.05
    upper = float(np.quantile(values, 0.99)) if values.size else 0.05
    span = max(upper - lower, 0.02)
    lower -= 0.08 * span
    upper += 0.08 * span

    width = 1260
    height = 540
    panel_w = 224
    panel_h = 300
    gap = 18
    left = 48
    top = 112
    parts = svg_header(width, height)
    parts.append(svg_text(36, 36, "Predicted vs Observed Annualized Tau Rate", size=22, weight="700"))
    parts.append(svg_text(36, 62, "Each point is one held-out pair-region observation; identity line marks perfect rate prediction.", size=13, fill="#4B5563"))
    x_flat = observed_rate.reshape(-1)
    for idx, model in enumerate(MODEL_ORDER):
        x0 = left + idx * (panel_w + gap)
        y0 = top
        y_flat = predicted_rates[model].reshape(-1)
        parts.extend(draw_scatter_panel(x0, y0, panel_w, panel_h, model, x_flat, y_flat, lower, upper))
    parts.append(svg_text(width / 2, height - 32, "Observed tau rate (SUVR/year)", size=13, anchor="middle", fill="#374151"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def draw_scatter_panel(
    x0: float,
    y0: float,
    width: float,
    height: float,
    model: str,
    observed_rate: np.ndarray,
    predicted_rate: np.ndarray,
    lower: float,
    upper: float,
) -> list[str]:
    plot_x = x0 + 42
    plot_y = y0 + 36
    plot_w = width - 58
    plot_h = height - 70
    corr = safe_correlation(observed_rate, predicted_rate, rank=True)
    parts = [
        svg_rect(x0, y0, width, height, fill="#FFFFFF", stroke="#E5E7EB", radius=6),
        svg_text(x0 + 14, y0 + 23, model, size=14, weight="700", fill="#111827"),
        svg_text(x0 + width - 12, y0 + 23, f"rho={format_metric(corr)}", size=11, anchor="end", fill="#4B5563"),
        svg_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, "#9CA3AF", 1),
        svg_line(plot_x, plot_y, plot_x, plot_y + plot_h, "#9CA3AF", 1),
        svg_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y, "#111827", 1, opacity=0.45),
    ]
    mask = np.isfinite(observed_rate) & np.isfinite(predicted_rate)
    x = observed_rate[mask]
    y = predicted_rate[mask]
    for x_value, y_value in zip(x, y, strict=True):
        px = plot_x + (float(x_value) - lower) / max(upper - lower, 1.0e-12) * plot_w
        py = plot_y + plot_h - (float(y_value) - lower) / max(upper - lower, 1.0e-12) * plot_h
        if plot_x - 2 <= px <= plot_x + plot_w + 2 and plot_y - 2 <= py <= plot_y + plot_h + 2:
            parts.append(svg_circle(px, py, 1.25, fill=MODEL_COLORS[model], opacity=0.28))
    parts.append(svg_text(plot_x, plot_y + plot_h + 17, format_metric(lower), size=9, fill="#6B7280"))
    parts.append(svg_text(plot_x + plot_w, plot_y + plot_h + 17, format_metric(upper), size=9, anchor="end", fill="#6B7280"))
    parts.append(svg_text(x0 + 12, y0 + height - 12, "Predicted", size=10, fill="#6B7280"))
    return parts


def write_edge_effect_svg(path: Path, edge_curves: list[dict[str, Any]]) -> None:
    width = 1150
    height = 520
    plot_x = 78
    plot_y = 92
    plot_w = 725
    plot_h = 340
    all_effects = np.concatenate([row["effect"] for row in edge_curves]) if edge_curves else np.asarray([0.0])
    bound = max(float(np.quantile(np.abs(all_effects), 0.98)), 0.01)
    parts = svg_header(width, height)
    parts.append(svg_text(36, 36, "Dynamic BN-SCM Edge Effects Across Pseudotime", size=22, weight="700"))
    parts.append(svg_text(36, 62, "Top varying parent effects by maximum absolute effect. Curves are descriptive ridge effects, not posterior PIPs.", size=13, fill="#4B5563"))
    parts.append(svg_rect(plot_x, plot_y, plot_w, plot_h, fill="#FFFFFF", stroke="#E5E7EB", radius=6))
    parts.append(svg_line(plot_x, plot_y + plot_h / 2, plot_x + plot_w, plot_y + plot_h / 2, "#9CA3AF", 1, opacity=0.7))
    colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#6B7280", "#8B5CF6"]
    for idx, row in enumerate(edge_curves):
        z = np.asarray(row["z"], dtype=float)
        effect = np.asarray(row["effect"], dtype=float)
        points = []
        for z_value, e_value in zip(z, effect, strict=True):
            x = plot_x + float(z_value) * plot_w
            y = plot_y + plot_h / 2 - float(e_value) / bound * (plot_h / 2 - 18)
            points.append(f"{x:.2f},{y:.2f}")
        color = colors[idx % len(colors)]
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.2" opacity="0.92"/>')
        legend_y = 112 + idx * 43
        parts.append(svg_line(842, legend_y - 5, 878, legend_y - 5, color, 3))
        label = f"{short_parent(row['parent'])} -> {short_target(row['target'])}"
        parts.append(svg_text(888, legend_y, label, size=12, fill="#111827"))
        parts.append(svg_text(888, legend_y + 16, f"max |effect|={format_metric(row['max_abs_effect'])}", size=10, fill="#6B7280"))
    parts.append(svg_text(plot_x, plot_y + plot_h + 32, "Z=0", size=11, fill="#374151"))
    parts.append(svg_text(plot_x + plot_w, plot_y + plot_h + 32, "Z=1", size=11, anchor="end", fill="#374151"))
    parts.append(svg_text(plot_x + plot_w / 2, plot_y + plot_h + 32, "Pseudotime Z", size=12, anchor="middle", fill="#374151"))
    parts.append(svg_text(plot_x - 14, plot_y + 10, f"+{format_metric(bound)}", size=10, anchor="end", fill="#6B7280"))
    parts.append(svg_text(plot_x - 14, plot_y + plot_h / 2 + 4, "0", size=10, anchor="end", fill="#6B7280"))
    parts.append(svg_text(plot_x - 14, plot_y + plot_h - 4, f"-{format_metric(bound)}", size=10, anchor="end", fill="#6B7280"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_pseudotime_svg(path: Path, z_values: np.ndarray, metadata_rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[float]] = {}
    for z, row in zip(z_values, metadata_rows, strict=True):
        label = str(row.get("dx_nearest_baseline", "") or "unknown")
        groups.setdefault(label, []).append(float(z))
    ordered = sorted(groups, key=lambda label: float(np.median(groups[label])))
    width = 920
    height = 108 + 58 * len(ordered)
    x0 = 178
    plot_w = 660
    y0 = 86
    parts = svg_header(width, height)
    parts.append(svg_text(36, 36, "Explainable Disease Pseudotime Z", size=22, weight="700"))
    parts.append(svg_text(36, 62, "Median and interquartile range by nearest baseline diagnosis; Z was fit on train rows only.", size=13, fill="#4B5563"))
    parts.append(svg_line(x0, y0 - 10, x0 + plot_w, y0 - 10, "#9CA3AF", 1))
    for tick in np.linspace(0.0, 1.0, 6):
        x = x0 + tick * plot_w
        parts.append(svg_line(x, y0 - 16, x, y0 - 4, "#9CA3AF", 1))
        parts.append(svg_text(x, y0 - 24, f"{tick:.1f}", size=10, anchor="middle", fill="#6B7280"))
    for idx, label in enumerate(ordered):
        values = np.asarray(groups[label], dtype=float)
        y = y0 + idx * 58 + 28
        q25, med, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        x25 = x0 + q25 * plot_w
        x50 = x0 + med * plot_w
        x75 = x0 + q75 * plot_w
        parts.append(svg_text(36, y + 4, f"{label} (n={values.size})", size=12, fill="#111827"))
        parts.append(svg_line(x25, y, x75, y, "#0072B2", 8, opacity=0.35))
        parts.append(svg_circle(x50, y, 6.5, fill="#D55E00", opacity=0.95))
        for quantile in np.quantile(values, np.linspace(0.05, 0.95, 10)):
            x = x0 + float(quantile) * plot_w
            parts.append(svg_circle(x, y + 16, 2.2, fill="#6B7280", opacity=0.35))
    parts.append(svg_text(x0 + plot_w / 2, height - 20, "Pseudotime Z", size=12, anchor="middle", fill="#374151"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_header(width: float, height: float) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}</style>',
    ]


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 12,
    fill: str = "#111827",
    weight: str = "400",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(text))}</text>'
    )


def svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str | None = None,
    radius: float = 0.0,
    opacity: float = 1.0,
) -> str:
    stroke_attr = f' stroke="{stroke}"' if stroke else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'rx="{radius:.2f}" fill="{fill}"{stroke_attr} opacity="{opacity:.3f}"/>'
    )


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    stroke: str,
    width: float,
    *,
    opacity: float = 1.0,
) -> str:
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width:.2f}" opacity="{opacity:.3f}"/>'


def svg_circle(x: float, y: float, radius: float, *, fill: str, opacity: float = 1.0) -> str:
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" opacity="{opacity:.3f}"/>'


def color_scale(value: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(value):
        return "#F3F4F6"
    t = min(1.0, max(0.0, (float(value) - vmin) / max(vmax - vmin, 1.0e-12)))
    low = np.asarray([230, 245, 241], dtype=float)
    mid = np.asarray([254, 243, 199], dtype=float)
    high = np.asarray([248, 190, 174], dtype=float)
    if t <= 0.5:
        mix = low + (mid - low) * (t / 0.5)
    else:
        mix = mid + (high - mid) * ((t - 0.5) / 0.5)
    return "#" + "".join(f"{int(round(channel)):02X}" for channel in mix)


def short_parent(name: str) -> str:
    return {
        "plasma_ab42_ab40": "Aβ42/40",
        "amyloid_summary_suvr": "Amyloid PET",
        "amyloid_centiloids": "Centiloids",
        "plasma_pt217": "p-tau217",
        "apoe4_dose": "APOE4",
        "age_years": "Age",
    }.get(name, name.replace("tau_region:", "tau ").replace("_", " "))


def short_target(name: str) -> str:
    return name.replace("tau_rate:", "").replace("_", " ")


def format_metric(value: float) -> str:
    if not np.isfinite(float(value)):
        return "NA"
    value = float(value)
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    if abs(value) >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def main() -> int:
    run_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
