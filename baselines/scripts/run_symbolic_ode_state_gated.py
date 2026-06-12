#!/usr/bin/env python3
"""Unified state-gated symbolic ODE.

This runner tests a stricter mechanistic alternative to per-subject alpha fitting:
one global ODE equation whose mechanism terms are modulated by observed baseline
disease-state features.  There are no independently fitted subject coefficients.

Example form:
    dS_i/dt = Σ_m Σ_g beta_{m,g} * mechanism_m(S_i, C, A_i, T_i) * gate_g(subject)

where gates are baseline/covariate summaries such as Braak tau burden, amyloid
burden, APOE4, p-tau181, and Laplacian eigenmode loadings.  Candidates are chosen
by held-out validation delta-Spearman after dynamic ODE integration.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.adni_features import build_closure_covariates  # noqa: E402
from spread_toolbox.forecasting import (  # noqa: E402
    ForecastDataset,
    MinMaxStateScaler,
    SubjectSplit,
    SubjectTrainValidationTestSplit,
    compute_aggregate_metrics,
    compute_pair_metrics,
    load_forecast_dataset,
    load_labeled_matrix,
    make_subject_split,
    read_csv_rows,
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.models.symbolic_ode import (  # noqa: E402
    SymbolicODEModel,
    build_amortization_features,
)


MODEL_NAME = "symbolic_ode_state_gated"


@dataclass
class GatedFit:
    mechanisms: tuple[str, ...]
    gate_names: list[str]
    gate_indices: list[int]
    gate_mean: np.ndarray
    gate_scale: np.ndarray
    design_names: list[str]
    design_mean: np.ndarray
    design_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    fit_intercept: bool
    ridge_alpha: float
    train_r2: float
    train_mse: float
    regional_masks: dict[str, np.ndarray] | None = None


def default_config_path() -> Path:
    exp = PROJECT_ROOT / "experiments" / "group_average_enigma"
    hcp = exp / "config_hcp.yaml"
    local = exp / "config.yaml"
    return hcp if hcp.exists() else local if local.exists() else exp / "config.example.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--rate-intercept",
        action="store_true",
        help="Allow a free global rate intercept. Default is stricter: every rate contribution must pass through a mechanism term.",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})
    seed = int(config.get("experiment", {}).get("random_seed", 20260507))

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    split = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=seed,
    )
    selection_split = make_selection_split(
        dataset,
        split,
        validation_fraction=float(args.validation_fraction),
        random_seed=seed + 211,
    )

    _, adjacency = load_labeled_matrix(
        output_dir / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv")
    )
    _, laplacian = load_labeled_matrix(
        output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv")
    )

    scaler = MinMaxStateScaler.fit(
        dataset.baseline[split.train_indices],
        dataset.observed[split.train_indices],
    )
    bl_s = scaler.transform(dataset.baseline)
    ob_s = scaler.transform(dataset.observed)

    pair_cov, reg_cov, _ = build_closure_covariates(dataset, split, config, PROJECT_ROOT)
    amyloid = reg_cov.get("amyloid_suvr")
    thickness = reg_cov.get("cortical_thickness")
    apoe4 = pair_cov.get("apoe4_dose")
    ptau181 = pair_cov.get("plasma_ptau181")

    model = SymbolicODEModel(adjacency, steps_per_year=12)
    braak_idx = build_braak_indices(dataset.region_labels)
    regional_masks = build_braak_region_masks(dataset.region_labels)
    _, eigenvectors = np.linalg.eigh(laplacian)
    gate_matrix, all_gate_names = build_amortization_features(
        bl_s,
        dataset.time_years,
        amyloid,
        thickness,
        apoe4,
        ptau181,
        braak_idx,
        eigenvectors,
        model.adj_norm,
    )

    gate_sets = build_gate_sets(all_gate_names)
    mechanism_sets = [
        ("amyloid_growth", "fickian"),
        ("amyloid_growth", "fickian", "autonomous_tau"),
        ("amyloid_growth", "fickian", "fickian_x_tau", "autonomous_tau"),
        ("growth", "amyloid_growth", "fickian", "fickian_x_tau", "tau_decay"),
        (
            "growth_braak_I_II", "growth_braak_III_IV", "growth_braak_V_VI",
            "growth_braak_Other", "amyloid_growth", "fickian", "fickian_x_tau", "tau_decay",
        ),
        ("growth", "amyloid_growth", "thickness_growth", "fickian", "fickian_x_tau", "tau_decay"),
    ]
    ridge_alphas = (10.0, 100.0, 1000.0, 10000.0)

    print("\n[1/3] Validation selection for unified state-gated ODE")
    print(f"      subtrain pairs={selection_split.train_indices.size}  "
          f"validation pairs={selection_split.validation_indices.size}  "
          f"test pairs={selection_split.test_indices.size}")

    rows: list[dict[str, Any]] = []
    for mechanisms in mechanism_sets:
        for gate_set_name, gate_names in gate_sets.items():
            gate_indices = [all_gate_names.index(name) for name in gate_names if name in all_gate_names]
            for alpha in ridge_alphas:
                fit = fit_gated_candidate(
                    model,
                    bl_s,
                    ob_s,
                    dataset.time_years,
                    train_indices=selection_split.train_indices,
                    mechanisms=mechanisms,
                    gate_matrix=gate_matrix,
                    gate_indices=gate_indices,
                    all_gate_names=all_gate_names,
                    amyloid=amyloid,
                    thickness=thickness,
                    regional_masks=regional_masks,
                    ridge_alpha=alpha,
                    fit_intercept=bool(args.rate_intercept),
                )
                pred_s = predict_gated_candidate(
                    model,
                    bl_s,
                    dataset.time_years,
                    fit,
                    gate_matrix=gate_matrix,
                    amyloid=amyloid,
                    thickness=thickness,
                )
                pred = scaler.inverse_transform(pred_s)
                pair_metrics = compute_pair_metrics(
                    dataset.pairs, dataset.baseline, dataset.observed,
                    pred, selection_split, candidate_name(mechanisms, gate_set_name, alpha),
                )
                summary = summarize_split(pair_metrics, "validation")
                row = {
                    "candidate": candidate_name(mechanisms, gate_set_name, alpha),
                    "mechanisms": list(mechanisms),
                    "gate_set": gate_set_name,
                    "gate_names": [all_gate_names[i] for i in gate_indices],
                    "ridge_alpha": alpha,
                    "train_rate_r2": fit.train_r2,
                    "train_rate_mse": fit.train_mse,
                    **summary,
                }
                rows.append(row)
                print_candidate(row)

    best = choose_candidate(rows)
    print("\n[2/3] Selected state-gated equation family")
    print_candidate(best, prefix="      ")

    selected_mechanisms = tuple(str(x) for x in best["mechanisms"])
    selected_gate_indices = [
        all_gate_names.index(name) for name in best["gate_names"] if name in all_gate_names
    ]
    final_fit = fit_gated_candidate(
        model,
        bl_s,
        ob_s,
        dataset.time_years,
        train_indices=split.train_indices,
        mechanisms=selected_mechanisms,
        gate_matrix=gate_matrix,
        gate_indices=selected_gate_indices,
        all_gate_names=all_gate_names,
        amyloid=amyloid,
        thickness=thickness,
        regional_masks=regional_masks,
        ridge_alpha=float(best["ridge_alpha"]),
        fit_intercept=bool(args.rate_intercept),
    )
    final_pred_s = predict_gated_candidate(
        model,
        bl_s,
        dataset.time_years,
        final_fit,
        gate_matrix=gate_matrix,
        amyloid=amyloid,
        thickness=thickness,
    )
    final_pred = scaler.inverse_transform(final_pred_s)
    pair_metrics = compute_pair_metrics(
        dataset.pairs, dataset.baseline, dataset.observed, final_pred, split, MODEL_NAME
    )
    summary_rows = compute_aggregate_metrics(pair_metrics)

    print("\n[3/3] Final held-out test performance")
    for row in summary_rows:
        if row["split"] == "test" and row["metric"] in {
            "mae", "rmse", "subject_spearman", "delta_spearman", "top10_overlap"
        }:
            print(f"      {row['metric']:<18} mean={row['mean']:.4f} "
                  f"median={row['median']:.4f}")

    print("\n      Strongest standardized mechanism-gate terms:")
    for term in top_terms(final_fit, n=12):
        print(f"        {term['coefficient']:+.6f}  {term['term']}")

    report = {
        "model": MODEL_NAME,
        "equation_family": (
            "dS_i/dt = sum_m sum_g beta[m,g] * mechanism_m(S_i,A_i,T_i,C) * "
            "gate_g(subject baseline state)"
        ),
        "selection_metric": "validation mean delta_spearman after dynamic integration",
        "selected": best,
        "candidate_selection": rows,
        "final_fit": {
            "mechanisms": list(final_fit.mechanisms),
            "gate_names": final_fit.gate_names,
            "ridge_alpha": final_fit.ridge_alpha,
            "fit_intercept": final_fit.fit_intercept,
            "train_rate_r2": final_fit.train_r2,
            "train_rate_mse": final_fit.train_mse,
            "intercept": final_fit.intercept,
            "top_terms": top_terms(final_fit, n=25),
        },
    }

    if not args.no_write:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv_rows(output_dir / f"{MODEL_NAME}_pair_metrics.csv", pair_metrics)
        write_csv_rows(output_dir / f"{MODEL_NAME}_metrics_summary.csv", summary_rows)
        write_json(output_dir / f"{MODEL_NAME}_report.json", report)
        print(f"\nWrote {MODEL_NAME} outputs to {output_dir}")
    return 0


def make_selection_split(
    dataset: ForecastDataset,
    split: SubjectSplit,
    *,
    validation_fraction: float,
    random_seed: int,
) -> SubjectTrainValidationTestSplit:
    train_rids = sorted({dataset.pairs[int(i)]["RID"] for i in split.train_indices})
    rng = np.random.default_rng(random_seed)
    shuffled = np.asarray(train_rids, dtype=object)
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_rids = sorted(str(v) for v in shuffled[:validation_count])
    subtrain_rids = sorted(str(v) for v in shuffled[validation_count:])
    validation_set = set(validation_rids)
    subtrain_set = set(subtrain_rids)

    train_indices, validation_indices = [], []
    for i in split.train_indices:
        rid = dataset.pairs[int(i)]["RID"]
        if rid in validation_set:
            validation_indices.append(int(i))
        elif rid in subtrain_set:
            train_indices.append(int(i))
        else:
            raise ValueError(f"RID {rid} missing from selection split.")

    return SubjectTrainValidationTestSplit(
        train_indices=np.asarray(train_indices, dtype=int),
        validation_indices=np.asarray(validation_indices, dtype=int),
        test_indices=split.test_indices,
        train_rids=subtrain_rids,
        validation_rids=validation_rids,
        test_rids=split.test_rids,
    )


def build_braak_indices(region_labels: list[str]) -> dict[str, list[int]]:
    braak = {
        "I-II": ["L_entorhinal", "R_entorhinal", "L_parahippocampal", "R_parahippocampal"],
        "III-IV": [
            "L_fusiform", "R_fusiform", "L_inferiortemporal", "R_inferiortemporal",
            "L_middletemporal", "R_middletemporal", "L_isthmuscingulate", "R_isthmuscingulate",
            "L_posteriorcingulate", "R_posteriorcingulate", "L_insula", "R_insula",
        ],
        "V-VI": [
            "L_inferiorparietal", "R_inferiorparietal", "L_superiorparietal", "R_superiorparietal",
            "L_precuneus", "R_precuneus", "L_superiorfrontal", "R_superiorfrontal",
            "L_rostralmiddlefrontal", "R_rostralmiddlefrontal",
            "L_superiortemporal", "R_superiortemporal",
        ],
    }
    return {stage: [region_labels.index(r) for r in regions if r in region_labels]
            for stage, regions in braak.items()}


def build_braak_region_masks(region_labels: list[str]) -> dict[str, np.ndarray]:
    """Fixed regional vulnerability masks, not fitted per-region coefficients."""
    indices = build_braak_indices(region_labels)
    masks: dict[str, np.ndarray] = {}
    assigned = np.zeros(len(region_labels), dtype=bool)
    for stage, idxs in indices.items():
        key = stage.replace("-", "_")
        mask = np.zeros(len(region_labels), dtype=float)
        mask[idxs] = 1.0
        masks[key] = mask
        assigned[idxs] = True
    masks["Other"] = (~assigned).astype(float)
    return masks


def build_gate_sets(all_gate_names: list[str]) -> dict[str, list[str]]:
    disease = [
        "tau_mean", "tau_std", "tau_gini", "amyloid_mean",
        "amyloid_tau_spatial_corr", "apoe4_dose", "plasma_ptau181",
    ]
    spatial = [
        "tau_braak_I-II", "tau_braak_III-IV", "tau_braak_V-VI",
        "braak_early_ratio", "eigenmode_1_loading", "eigenmode_4_loading",
        "fickian_drive_magnitude", "tau_lr_asymmetry",
    ]
    return {
        "constant": [],
        "disease_state": [name for name in disease if name in all_gate_names],
        "spatial_state": [name for name in spatial if name in all_gate_names],
        "full_state": [name for name in disease + spatial if name in all_gate_names],
    }


def fit_gated_candidate(
    model: SymbolicODEModel,
    baseline_scaled: np.ndarray,
    observed_scaled: np.ndarray,
    time_years: np.ndarray,
    *,
    train_indices: np.ndarray,
    mechanisms: tuple[str, ...],
    gate_matrix: np.ndarray,
    gate_indices: list[int],
    all_gate_names: list[str],
    amyloid: np.ndarray | None,
    thickness: np.ndarray | None,
    regional_masks: dict[str, np.ndarray] | None = None,
    ridge_alpha: float,
    fit_intercept: bool,
    sample_weight: np.ndarray | None = None,
) -> GatedFit:
    bl = np.asarray(baseline_scaled, dtype=float)
    obs = np.asarray(observed_scaled, dtype=float)
    t = np.asarray(time_years, dtype=float)
    n_pairs, n_reg = bl.shape

    gate_raw = gate_matrix[:, gate_indices] if gate_indices else np.zeros((n_pairs, 0), dtype=float)
    gate_mean = gate_raw[train_indices].mean(axis=0) if gate_indices else np.zeros(0)
    gate_scale = gate_raw[train_indices].std(axis=0) if gate_indices else np.ones(0)
    gate_scale = np.where(np.isfinite(gate_scale) & (gate_scale > 1.0e-10), gate_scale, 1.0)
    gates = standardize_gates(gate_raw, gate_mean, gate_scale)

    term_values, design_names = build_design(
        model,
        bl,
        mechanisms,
        gates,
        [all_gate_names[i] for i in gate_indices],
        amyloid=amyloid,
        thickness=thickness,
        regional_masks=regional_masks,
    )
    y = ((obs - bl) / np.maximum(t, 1.0e-6)[:, None]).reshape(-1)
    weights_flat = flatten_sample_weight(sample_weight, n_pairs, n_reg)
    flat_mask = np.zeros(n_pairs * n_reg, dtype=bool)
    for i in np.asarray(train_indices, dtype=int):
        flat_mask[i * n_reg:(i + 1) * n_reg] = True
    X_flat = term_values.reshape(-1, term_values.shape[-1])
    finite = flat_mask & np.isfinite(y) & np.all(np.isfinite(X_flat), axis=1)
    if weights_flat is not None:
        finite &= np.isfinite(weights_flat) & (weights_flat > 0.0)
    X_train = X_flat[finite]
    y_train = y[finite]
    w_train = weights_flat[finite] if weights_flat is not None else None

    design_mean = X_train.mean(axis=0)
    design_scale = X_train.std(axis=0)
    design_scale = np.where(np.isfinite(design_scale) & (design_scale > 1.0e-10), design_scale, 1.0)
    X_sc = (X_train - design_mean[None, :]) / design_scale[None, :]

    ridge = Ridge(alpha=float(ridge_alpha), fit_intercept=bool(fit_intercept))
    ridge.fit(X_sc, y_train, sample_weight=w_train)
    y_pred = ridge.predict(X_sc)
    ss_res = float(np.sum((y_train - y_pred) ** 2))
    ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
    train_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    train_mse = float(np.mean((y_train - y_pred) ** 2))

    return GatedFit(
        mechanisms=mechanisms,
        gate_names=[all_gate_names[i] for i in gate_indices],
        gate_indices=list(gate_indices),
        gate_mean=gate_mean,
        gate_scale=gate_scale,
        design_names=design_names,
        design_mean=design_mean,
        design_scale=design_scale,
        coefficients=np.asarray(ridge.coef_, dtype=float),
        intercept=float(ridge.intercept_),
        fit_intercept=bool(fit_intercept),
        ridge_alpha=float(ridge_alpha),
        train_r2=train_r2,
        train_mse=train_mse,
        regional_masks=regional_masks,
    )


def predict_gated_candidate(
    model: SymbolicODEModel,
    baseline_scaled: np.ndarray,
    time_years: np.ndarray,
    fit: GatedFit,
    *,
    gate_matrix: np.ndarray,
    amyloid: np.ndarray | None,
    thickness: np.ndarray | None,
) -> np.ndarray:
    bl = np.asarray(baseline_scaled, dtype=float)
    t = np.asarray(time_years, dtype=float)
    n_pairs, n_reg = bl.shape
    gate_raw = gate_matrix[:, fit.gate_indices] if fit.gate_indices else np.zeros((n_pairs, 0), dtype=float)
    gates = standardize_gates(gate_raw, fit.gate_mean, fit.gate_scale)

    amy = fill_nan(np.asarray(amyloid, dtype=float), n_pairs, n_reg) if amyloid is not None else None
    thick = fill_nan(np.asarray(thickness, dtype=float), n_pairs, n_reg) if thickness is not None else None

    states = np.clip(bl, 0.0, 1.0).copy()
    remaining = t.copy()
    step_dt = 1.0 / model.steps_per_year

    def rate_fn(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
        term_values, _ = build_design(
            model,
            S,
            fit.mechanisms,
            gates[idx],
            fit.gate_names,
            amyloid=amy[idx] if amy is not None else None,
            thickness=thick[idx] if thick is not None else None,
            regional_masks=fit.regional_masks,
        )
        X = term_values.reshape(-1, term_values.shape[-1])
        X_sc = (X - fit.design_mean[None, :]) / fit.design_scale[None, :]
        rate = fit.intercept + X_sc @ fit.coefficients
        return np.clip(np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).reshape(S.shape)

    while np.any(remaining > 0.0):
        active = remaining > 0.0
        idx = np.where(active)[0]
        dt = np.minimum(step_dt, remaining[active])[:, None]
        s = states[active]
        k1 = rate_fn(s, idx)
        k2 = rate_fn(np.clip(s + 0.5 * dt * k1, 0.0, 1.0), idx)
        k3 = rate_fn(np.clip(s + 0.5 * dt * k2, 0.0, 1.0), idx)
        k4 = rate_fn(np.clip(s + dt * k3, 0.0, 1.0), idx)
        states[active] = np.clip(s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0)
        remaining[active] -= dt[:, 0]
    return states


def build_design(
    model: SymbolicODEModel,
    state: np.ndarray,
    mechanisms: tuple[str, ...],
    gates: np.ndarray,
    gate_names: list[str],
    *,
    amyloid: np.ndarray | None,
    thickness: np.ndarray | None,
    regional_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    S = np.asarray(state, dtype=float)
    n_pairs, n_reg = S.shape
    neighbour = S @ model.adj_norm.T
    fickian = neighbour - S
    neighbour_minus_self = S[:, None, :] - S[:, :, None]  # (pair, target_i, source_j)
    directed_inflow = np.sum(model.adj_norm[None, :, :] * np.maximum(neighbour_minus_self, 0.0), axis=2)
    directed_outflow = np.sum(model.adj_norm[None, :, :] * np.maximum(-neighbour_minus_self, 0.0), axis=2)
    growth = S * (1.0 - S)
    amy = fill_nan(np.asarray(amyloid, dtype=float), n_pairs, n_reg) if amyloid is not None else np.zeros_like(S)
    thick = fill_nan(np.asarray(thickness, dtype=float), n_pairs, n_reg) if thickness is not None else np.zeros_like(S)

    local_terms = {
        "growth": growth,
        "amyloid_growth": amy * growth,
        "thickness_growth": thick * growth,
        "fickian": fickian,
        "fickian_x_tau": fickian * S,
        "directed_inflow": directed_inflow,
        "directed_inflow_x_tau": directed_inflow * S,
        "directed_outflow": directed_outflow,
        "directed_outflow_x_tau": directed_outflow * S,
        "autonomous_tau": (S**2) * (1.0 - S),
        "tau_decay": -S,
    }
    if regional_masks:
        for name, raw_mask in regional_masks.items():
            mask = np.asarray(raw_mask, dtype=float)
            if mask.shape != (n_reg,):
                raise ValueError(f"Regional mask {name!r} must have shape {(n_reg,)}, got {mask.shape}.")
            local_terms[f"growth_braak_{name}"] = growth * mask[None, :]
            local_terms[f"amyloid_growth_braak_{name}"] = amy * growth * mask[None, :]
            local_terms[f"decay_braak_{name}"] = -S * mask[None, :]
    gate_columns = [np.ones(n_pairs, dtype=float)]
    gate_labels = ["1"]
    if gates.size:
        gate_columns.extend([gates[:, j] for j in range(gates.shape[1])])
        gate_labels.extend(gate_names)

    columns: list[np.ndarray] = []
    names: list[str] = []
    for mechanism in mechanisms:
        base = local_terms[mechanism]
        for gate, gate_label in zip(gate_columns, gate_labels, strict=True):
            columns.append(base * gate[:, None])
            names.append(mechanism if gate_label == "1" else f"{mechanism}*{gate_label}")
    return np.stack(columns, axis=-1), names


def standardize_gates(gate_raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if gate_raw.shape[1] == 0:
        return gate_raw.astype(float)
    return np.nan_to_num((gate_raw - mean[None, :]) / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0)


def flatten_sample_weight(sample_weight: np.ndarray | None, n_pairs: int, n_reg: int) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape == (n_pairs,):
        weights = np.repeat(weights[:, None], n_reg, axis=1)
    if weights.shape != (n_pairs, n_reg):
        raise ValueError(f"sample_weight must have shape {(n_pairs,)} or {(n_pairs, n_reg)}, got {weights.shape}.")
    finite_positive = np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite_positive):
        raise ValueError("sample_weight contains no finite positive entries.")
    safe = weights.copy()
    safe[~finite_positive] = np.nan
    median = float(np.nanmedian(safe))
    safe = np.where(finite_positive, safe, median)
    return (safe / np.mean(safe)).reshape(-1)


def summarize_split(pair_metrics: list[dict[str, Any]], split_name: str) -> dict[str, float]:
    rows = [r for r in pair_metrics if r["split"] == split_name]
    out: dict[str, float] = {}
    for metric in ["mae", "rmse", "subject_spearman", "delta_spearman", "delta_pearson", "top10_overlap"]:
        vals = np.asarray([r[metric] for r in rows if r[metric] == r[metric]], dtype=float)
        out[f"validation_{metric}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
        out[f"validation_{metric}_median"] = float(np.median(vals)) if vals.size else float("nan")
    return out


def choose_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda r: (
            np.nan_to_num(r["validation_delta_spearman_mean"], nan=-999.0),
            np.nan_to_num(r["validation_top10_overlap_mean"], nan=-999.0),
            -np.nan_to_num(r["validation_mae_mean"], nan=999.0),
        ),
    )


def print_candidate(row: dict[str, Any], *, prefix: str = "") -> None:
    print(
        f"{prefix}{row['candidate']:<84} "
        f"val_delta_mean={row['validation_delta_spearman_mean']:+.4f} "
        f"val_top10={row['validation_top10_overlap_mean']:.4f} "
        f"val_mae={row['validation_mae_mean']:.4f} "
        f"rate_R2={row['train_rate_r2']:.4f}"
    )


def candidate_name(mechanisms: tuple[str, ...], gate_set_name: str, alpha: float) -> str:
    return f"{'+'.join(mechanisms)}|gates={gate_set_name}|ridge={alpha:g}"


def top_terms(fit: GatedFit, *, n: int) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(fit.coefficients))[::-1][:n]
    return [
        {
            "term": fit.design_names[int(i)],
            "coefficient": float(fit.coefficients[int(i)]),
            "abs_coefficient": float(abs(fit.coefficients[int(i)])),
        }
        for i in order
    ]


def ablate_fit_terms(
    fit: GatedFit,
    *,
    drop_mechanisms: tuple[str, ...] = (),
    drop_gates: tuple[str, ...] = (),
    drop_substrings: tuple[str, ...] = (),
) -> GatedFit:
    """Return a copy of ``fit`` with selected coefficients zeroed.

    This is intended for mechanistic contribution analysis after a single
    equation family has been selected and fitted.  It does not refit the model;
    therefore it asks how much the learned equation depends on a term family,
    not whether another equation could compensate after retraining.
    """
    coeffs = fit.coefficients.copy()
    drop_mechanisms = tuple(drop_mechanisms)
    drop_gates = tuple(drop_gates)
    drop_substrings = tuple(drop_substrings)
    for i, term in enumerate(fit.design_names):
        pieces = term.split("*")
        mechanism = pieces[0]
        gates = tuple(pieces[1:])
        if mechanism in drop_mechanisms:
            coeffs[i] = 0.0
        elif any(gate in gates for gate in drop_gates):
            coeffs[i] = 0.0
        elif any(token in term for token in drop_substrings):
            coeffs[i] = 0.0
    return replace(fit, coefficients=coeffs)


def fill_nan(values: np.ndarray, n_pairs: int, n_reg: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n_pairs, n_reg):
        raise ValueError(f"Expected {(n_pairs, n_reg)}, got {arr.shape}.")
    result = arr.copy()
    col_median = np.nanmedian(result, axis=0)
    col_median = np.where(np.isfinite(col_median), col_median, 0.0)
    bad = ~np.isfinite(result)
    result[bad] = np.broadcast_to(col_median, result.shape)[bad]
    return result


if __name__ == "__main__":
    raise SystemExit(main())
