#!/usr/bin/env python3
"""Test each of the five paper hypotheses one by one against the state-gated baseline.

Hypotheses (from Symbolic_ODE.pdf):
  H1  Neuronal Activity-Dependent Seeding  -> hubness as E_i
  H2  Anisotropic Axonal Transport         -> state-asymmetric inflow/outflow split
  H3  Perfusion-Coupled Glymphatic Clearance -> thickness as CBF/WMH proxy
  H4  Heterotypic Polymorph Cross-Seeding  -> amyloid * ptau181_plasma synergy
  H5  Transcriptomic Resilience            -> Laplacian eigenmodes as AHBA proxy

Each run uses the same validation/test split as the baseline state-gated runner,
adds hypothesis-specific mechanisms / gates, performs validation selection, and
reports final test metrics for direct comparison with the baseline JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

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
    write_csv_rows,
    write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.models.symbolic_ode import (  # noqa: E402
    SymbolicODEModel,
    build_amortization_features,
)
from run_symbolic_ode_state_gated import (  # noqa: E402
    GatedFit,
    build_braak_indices,
    build_braak_region_masks,
    candidate_name,
    choose_candidate,
    fill_nan,
    flatten_sample_weight,
    make_selection_split,
    print_candidate,
    standardize_gates,
    summarize_split,
    top_terms,
)


# ── Hypothesis registry ──────────────────────────────────────────────────────


def baseline_mechanism_sets() -> list[tuple[str, ...]]:
    """Baseline mechanisms (same as run_symbolic_ode_state_gated.py)."""
    return [
        ("amyloid_growth", "fickian"),
        ("amyloid_growth", "fickian", "autonomous_tau"),
        ("amyloid_growth", "fickian", "fickian_x_tau", "autonomous_tau"),
        ("growth", "amyloid_growth", "fickian", "fickian_x_tau", "tau_decay"),
        (
            "growth_braak_I_II", "growth_braak_III_IV", "growth_braak_V_VI",
            "growth_braak_Other", "amyloid_growth", "fickian", "fickian_x_tau",
            "tau_decay",
        ),
        ("growth", "amyloid_growth", "thickness_growth", "fickian", "fickian_x_tau",
         "tau_decay"),
    ]


HYPOTHESIS_MECHANISM_SETS: dict[str, list[tuple[str, ...]]] = {
    # Each list is the EXTRA mechanism sets to add on top of baseline.
    # The script runs baseline + hypothesis-specific extras.
    "H1": [
        # Activity-only minimal
        ("amyloid_growth", "fickian", "activity_growth"),
        ("amyloid_growth", "fickian", "autonomous_tau", "activity_autonomous"),
        ("amyloid_growth", "fickian", "activity_growth", "activity_autonomous"),
        # Activity layered onto best baseline family
        ("growth", "amyloid_growth", "fickian", "fickian_x_tau", "tau_decay",
         "activity_growth", "activity_autonomous"),
    ],
    "H2": [
        # Replace symmetric fickian with directed split
        ("amyloid_growth", "directed_inflow", "directed_outflow"),
        ("amyloid_growth", "directed_inflow_x_tau", "directed_outflow_x_tau"),
        ("amyloid_growth", "fickian", "directed_inflow_x_tau",
         "directed_outflow_x_tau"),
        ("growth", "amyloid_growth", "directed_inflow", "directed_outflow",
         "tau_decay"),
        ("growth", "amyloid_growth", "directed_inflow_x_tau",
         "directed_outflow_x_tau", "tau_decay"),
    ],
    "H3": [
        # Thickness-modulated clearance
        ("amyloid_growth", "fickian", "thickness_clearance"),
        ("amyloid_growth", "fickian", "fickian_x_tau", "thickness_clearance"),
        ("growth", "amyloid_growth", "fickian", "fickian_x_tau",
         "thickness_clearance"),
        # With dual decay terms
        ("growth", "amyloid_growth", "fickian", "fickian_x_tau", "tau_decay",
         "thickness_clearance"),
    ],
    "H4": [
        # Cross-seeding mechanisms layered on baseline
        ("amyloid_growth", "fickian", "cross_seeding_growth"),
        ("amyloid_growth", "fickian", "autonomous_tau", "cross_seeding_growth"),
        ("amyloid_growth", "fickian", "fickian_x_tau", "cross_seeding_growth"),
        ("growth", "amyloid_growth", "fickian", "fickian_x_tau", "tau_decay",
         "cross_seeding_growth"),
    ],
    "H5": [
        # Transcriptomic resilience: eigenmode-gated autonomous seeding
        # Use 'autonomous_tau' mechanism with eigenmode gates as resilience
        ("amyloid_growth", "fickian", "autonomous_tau"),
        ("amyloid_growth", "fickian", "fickian_x_tau", "autonomous_tau"),
        ("growth", "amyloid_growth", "fickian", "fickian_x_tau", "autonomous_tau",
         "tau_decay"),
    ],
}


# Hypothesis-specific gate sets — different gates emphasise the hypothesis
HYPOTHESIS_GATE_OVERRIDES: dict[str, list[str]] = {
    # H1: add hubness as a gate (we add it to the gate matrix, not to all_gate_names from build_amortization_features)
    "H1": ["tau_braak_I-II", "tau_braak_III-IV", "tau_braak_V-VI",
           "amyloid_mean", "fickian_drive_magnitude",
           "hubness_mean", "hubness_max"],
    # H2: spatial features are paramount
    "H2": ["tau_braak_I-II", "tau_braak_III-IV", "tau_braak_V-VI",
           "braak_early_ratio", "eigenmode_1_loading", "eigenmode_4_loading",
           "fickian_drive_magnitude", "tau_lr_asymmetry"],
    # H3: thickness-related gates are emphasised
    "H3": ["tau_mean", "tau_std", "amyloid_mean", "amyloid_tau_spatial_corr",
           "apoe4_dose", "plasma_ptau181", "thickness_mean", "thickness_loss"],
    # H4: amyloid x ptau gates
    "H4": ["tau_mean", "amyloid_mean", "amyloid_tau_spatial_corr",
           "plasma_ptau181", "apoe4_x_amyloid", "amyloid_x_ptau"],
    # H5: eigenmode gates (used as transcriptomic-gradient proxies)
    "H5": ["eigenmode_0_loading", "eigenmode_1_loading", "eigenmode_2_loading",
           "eigenmode_3_loading", "eigenmode_4_loading",
           "tau_braak_I-II", "tau_braak_V-VI"],
}


# ── Extended local-term builder ──────────────────────────────────────────────


def build_design_extended(
    model: SymbolicODEModel,
    state: np.ndarray,
    mechanisms: tuple[str, ...],
    gates: np.ndarray,
    gate_names: list[str],
    *,
    amyloid: np.ndarray | None,
    thickness: np.ndarray | None,
    ptau181: np.ndarray | None,
    hubness: np.ndarray | None,
    regional_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extension of run_symbolic_ode_state_gated.build_design with hypothesis terms.

    New mechanisms (H1..H5):
        activity_growth, activity_autonomous   (H1 - hubness as E_i)
        directed_inflow, directed_outflow, directed_inflow_x_tau, directed_outflow_x_tau   (H2)
        thickness_clearance                                       (H3)
        cross_seeding_growth                                      (H4)
    """
    S = np.asarray(state, dtype=float)
    n_pairs, n_reg = S.shape

    neighbour = S @ model.adj_norm.T
    fickian = neighbour - S
    neighbour_minus_self = S[:, None, :] - S[:, :, None]
    directed_inflow = np.sum(
        model.adj_norm[None, :, :] * np.maximum(neighbour_minus_self, 0.0), axis=2
    )
    directed_outflow = np.sum(
        model.adj_norm[None, :, :] * np.maximum(-neighbour_minus_self, 0.0), axis=2
    )
    growth = S * (1.0 - S)

    amy = (
        fill_nan(np.asarray(amyloid, dtype=float), n_pairs, n_reg)
        if amyloid is not None else np.zeros_like(S)
    )
    thick = (
        fill_nan(np.asarray(thickness, dtype=float), n_pairs, n_reg)
        if thickness is not None else np.zeros_like(S)
    )

    # H1: hubness as activity proxy (regional, broadcast over pairs)
    if hubness is not None:
        hub = np.asarray(hubness, dtype=float)  # (n_reg,)
        hub_b = np.broadcast_to(hub[None, :], S.shape)
    else:
        hub_b = np.zeros_like(S)

    # H3: thickness clearance — γ_i = γ_0 · exp(κ·thick_i).
    # Nonlinear in thickness so NOT collinear with tau_decay (-S) or thickness_growth.
    # Atrophic region (thick<<0) -> exp factor < 1 -> weakened clearance.
    # Use clamped exponent so values stay bounded.
    thick_factor = np.exp(0.5 * np.clip(thick, -3.0, 3.0))

    # H4: cross-seeding via (amyloid * plasma_ptau181) — ptau is pair-level (n_pairs,)
    if ptau181 is not None:
        p = np.asarray(ptau181, dtype=float)
        p_b = np.broadcast_to(p[:, None], S.shape)
        cross_seed_factor = amy * p_b
    else:
        cross_seed_factor = np.zeros_like(S)

    local_terms: dict[str, np.ndarray] = {
        # baseline
        "growth": growth,
        "amyloid_growth": amy * growth,
        "thickness_growth": thick * growth,
        "fickian": fickian,
        "fickian_x_tau": fickian * S,
        "directed_inflow": directed_inflow,
        "directed_inflow_x_tau": directed_inflow * S,
        "directed_outflow": directed_outflow,
        "directed_outflow_x_tau": directed_outflow * S,
        "autonomous_tau": (S ** 2) * (1.0 - S),
        "tau_decay": -S,
        # H1
        "activity_growth": hub_b * growth,
        "activity_autonomous": hub_b * (S ** 2) * (1.0 - S),
        # H3
        "thickness_clearance": -thick_factor * S,
        # H4
        "cross_seeding_growth": cross_seed_factor * growth,
    }

    if regional_masks:
        for name, raw_mask in regional_masks.items():
            mask = np.asarray(raw_mask, dtype=float)
            if mask.shape != (n_reg,):
                raise ValueError(f"Regional mask {name!r} must have shape {(n_reg,)}.")
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
        if mechanism not in local_terms:
            raise KeyError(f"Unknown mechanism {mechanism!r}; "
                           f"available: {sorted(local_terms)}")
        base = local_terms[mechanism]
        for gate, label in zip(gate_columns, gate_labels, strict=True):
            columns.append(base * gate[:, None])
            names.append(mechanism if label == "1" else f"{mechanism}*{label}")
    return np.stack(columns, axis=-1), names


# ── Fit & predict using extended design ───────────────────────────────────────


def fit_candidate(
    model: SymbolicODEModel,
    bl_s: np.ndarray,
    ob_s: np.ndarray,
    time_years: np.ndarray,
    *,
    train_indices: np.ndarray,
    mechanisms: tuple[str, ...],
    gate_matrix: np.ndarray,
    gate_indices: list[int],
    all_gate_names: list[str],
    amyloid: np.ndarray | None,
    thickness: np.ndarray | None,
    ptau181: np.ndarray | None,
    hubness: np.ndarray | None,
    regional_masks: dict[str, np.ndarray],
    ridge_alpha: float,
    fit_intercept: bool,
) -> GatedFit:
    n_pairs, n_reg = bl_s.shape

    gate_raw = gate_matrix[:, gate_indices] if gate_indices else np.zeros((n_pairs, 0))
    gate_mean = gate_raw[train_indices].mean(axis=0) if gate_indices else np.zeros(0)
    gate_scale = (
        gate_raw[train_indices].std(axis=0) if gate_indices else np.ones(0)
    )
    gate_scale = np.where(np.isfinite(gate_scale) & (gate_scale > 1e-10),
                          gate_scale, 1.0)
    gates = standardize_gates(gate_raw, gate_mean, gate_scale)

    term_values, design_names = build_design_extended(
        model, bl_s, mechanisms, gates,
        [all_gate_names[i] for i in gate_indices],
        amyloid=amyloid, thickness=thickness,
        ptau181=ptau181, hubness=hubness,
        regional_masks=regional_masks,
    )
    y = ((ob_s - bl_s) / np.maximum(time_years, 1e-6)[:, None]).reshape(-1)
    flat_mask = np.zeros(n_pairs * n_reg, dtype=bool)
    for i in np.asarray(train_indices, dtype=int):
        flat_mask[i * n_reg:(i + 1) * n_reg] = True
    X_flat = term_values.reshape(-1, term_values.shape[-1])
    finite = flat_mask & np.isfinite(y) & np.all(np.isfinite(X_flat), axis=1)
    X_train = X_flat[finite]
    y_train = y[finite]

    design_mean = X_train.mean(axis=0)
    design_scale = X_train.std(axis=0)
    design_scale = np.where(np.isfinite(design_scale) & (design_scale > 1e-10),
                            design_scale, 1.0)
    X_sc = (X_train - design_mean[None, :]) / design_scale[None, :]

    ridge = Ridge(alpha=float(ridge_alpha), fit_intercept=bool(fit_intercept))
    ridge.fit(X_sc, y_train)
    y_pred = ridge.predict(X_sc)
    ss_res = float(np.sum((y_train - y_pred) ** 2))
    ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
    train_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
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


def predict_candidate(
    model: SymbolicODEModel,
    bl_s: np.ndarray,
    time_years: np.ndarray,
    fit: GatedFit,
    *,
    gate_matrix: np.ndarray,
    amyloid: np.ndarray | None,
    thickness: np.ndarray | None,
    ptau181: np.ndarray | None,
    hubness: np.ndarray | None,
) -> np.ndarray:
    n_pairs, n_reg = bl_s.shape
    gate_raw = (
        gate_matrix[:, fit.gate_indices] if fit.gate_indices
        else np.zeros((n_pairs, 0), dtype=float)
    )
    gates = standardize_gates(gate_raw, fit.gate_mean, fit.gate_scale)

    amy = (
        fill_nan(np.asarray(amyloid, dtype=float), n_pairs, n_reg)
        if amyloid is not None else None
    )
    thick = (
        fill_nan(np.asarray(thickness, dtype=float), n_pairs, n_reg)
        if thickness is not None else None
    )
    p = np.asarray(ptau181, dtype=float) if ptau181 is not None else None
    hub = np.asarray(hubness, dtype=float) if hubness is not None else None

    states = np.clip(bl_s, 0.0, 1.0).copy()
    remaining = time_years.copy()
    step_dt = 1.0 / model.steps_per_year

    def rate_fn(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
        term_values, _ = build_design_extended(
            model, S, fit.mechanisms, gates[idx], fit.gate_names,
            amyloid=amy[idx] if amy is not None else None,
            thickness=thick[idx] if thick is not None else None,
            ptau181=p[idx] if p is not None else None,
            hubness=hub,  # regional, no pair index
            regional_masks=fit.regional_masks,
        )
        X = term_values.reshape(-1, term_values.shape[-1])
        X_sc = (X - fit.design_mean[None, :]) / fit.design_scale[None, :]
        rate = fit.intercept + X_sc @ fit.coefficients
        return np.clip(
            np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0
        ).reshape(S.shape)

    while np.any(remaining > 0.0):
        active = remaining > 0.0
        idx = np.where(active)[0]
        dt = np.minimum(step_dt, remaining[active])[:, None]
        s = states[active]
        k1 = rate_fn(s, idx)
        k2 = rate_fn(np.clip(s + 0.5 * dt * k1, 0.0, 1.0), idx)
        k3 = rate_fn(np.clip(s + 0.5 * dt * k2, 0.0, 1.0), idx)
        k4 = rate_fn(np.clip(s + dt * k3, 0.0, 1.0), idx)
        states[active] = np.clip(
            s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0
        )
        remaining[active] -= dt[:, 0]
    return states


# ── Extra gates we add to gate_matrix ────────────────────────────────────────


def augment_gate_matrix(
    gate_matrix: np.ndarray,
    all_gate_names: list[str],
    *,
    bl_s: np.ndarray,
    thickness: np.ndarray | None,
    amyloid: np.ndarray | None,
    ptau181: np.ndarray | None,
    hubness: np.ndarray | None,
) -> tuple[np.ndarray, list[str]]:
    """Add hubness, thickness, amyloid_x_ptau gates to the matrix."""
    extra_cols: list[np.ndarray] = []
    extra_names: list[str] = []

    if hubness is not None:
        # Weighted hubness loading per pair = sum_i tau_i * hubness_i (focus of tau onto hubs)
        hub = np.asarray(hubness, dtype=float)
        extra_cols.append((bl_s @ hub)[:, None]); extra_names.append("hubness_loading")
        extra_cols.append(np.full((bl_s.shape[0], 1), float(hub.mean())))
        extra_names.append("hubness_mean")
        extra_cols.append(np.full((bl_s.shape[0], 1), float(hub.max())))
        extra_names.append("hubness_max")

    if thickness is not None:
        thick = np.asarray(thickness, dtype=float)
        extra_cols.append(thick.mean(axis=1, keepdims=True))
        extra_names.append("thickness_mean")
        # thickness loss approximated as -mean(thick); thick is standardised so smaller=lower thickness
        extra_cols.append((-thick.mean(axis=1, keepdims=True)))
        extra_names.append("thickness_loss")

    if amyloid is not None and ptau181 is not None:
        amy = np.asarray(amyloid, dtype=float)
        p = np.asarray(ptau181, dtype=float)
        # amyloid_x_ptau per-pair = (mean amyloid) * ptau
        extra_cols.append((np.nanmean(amy, axis=1) * p)[:, None])
        extra_names.append("amyloid_x_ptau")

    if not extra_cols:
        return gate_matrix, all_gate_names
    return np.hstack([gate_matrix] + extra_cols), all_gate_names + extra_names


# ── Main per-hypothesis runner ───────────────────────────────────────────────


def run_hypothesis(
    hypothesis: str,
    *,
    include_baseline_sets: bool = True,
    validation_fraction: float = 0.2,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run validation selection over baseline+hypothesis mechanism sets and report metrics."""
    config_path = config_path or default_config_path()
    config = load_yaml_config(config_path)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    modeling = config.get("modeling", {})
    seed = int(config.get("experiment", {}).get("random_seed", 20260507))

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    split = make_subject_split(
        dataset.pairs, test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=seed,
    )
    sel = make_selection_split(dataset, split,
                                validation_fraction=validation_fraction,
                                random_seed=seed + 211)

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
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

    gate_matrix, all_gate_names = build_amortization_features(
        bl_s, dataset.time_years, amyloid, thickness, apoe4, ptau181,
        braak_idx, eigenvectors, model.adj_norm,
    )
    # H5 needs eigenmode_0_loading (the trivial constant); add it if missing
    if "eigenmode_0_loading" not in all_gate_names:
        em0 = (bl_s @ eigenvectors[:, 0])[:, None]
        gate_matrix = np.hstack([gate_matrix, em0])
        all_gate_names = all_gate_names + ["eigenmode_0_loading"]

    # Hubness = weighted degree of each region
    hubness_raw = adjacency.sum(axis=1).astype(float)  # (n_reg,)
    hubness = (hubness_raw - hubness_raw.mean()) / (hubness_raw.std() + 1e-12)

    gate_matrix, all_gate_names = augment_gate_matrix(
        gate_matrix, all_gate_names,
        bl_s=bl_s, thickness=thickness, amyloid=amyloid,
        ptau181=ptau181, hubness=hubness,
    )

    # Pre-filter to existing gates only
    def filter_gate_names(gnames: list[str]) -> list[str]:
        return [n for n in gnames if n in all_gate_names]

    # Build candidate space ───────────────────────────────────────────────────
    if hypothesis == "baseline":
        mechanism_sets = baseline_mechanism_sets()
        gate_set_options = {
            "constant": [],
            "spatial_state": filter_gate_names([
                "tau_braak_I-II", "tau_braak_III-IV", "tau_braak_V-VI",
                "braak_early_ratio", "eigenmode_1_loading", "eigenmode_4_loading",
                "fickian_drive_magnitude", "tau_lr_asymmetry",
            ]),
        }
    else:
        mech_sets = list(HYPOTHESIS_MECHANISM_SETS[hypothesis])
        if include_baseline_sets:
            # Always include baseline so we know whether the extra mechanism HELPS
            mech_sets = baseline_mechanism_sets() + mech_sets
        mechanism_sets = mech_sets
        # Hypothesis-specific gate set + the standard spatial_state for comparison
        gate_set_options = {
            "constant": [],
            "spatial_state": filter_gate_names([
                "tau_braak_I-II", "tau_braak_III-IV", "tau_braak_V-VI",
                "braak_early_ratio", "eigenmode_1_loading", "eigenmode_4_loading",
                "fickian_drive_magnitude", "tau_lr_asymmetry",
            ]),
            "hypothesis_state": filter_gate_names(HYPOTHESIS_GATE_OVERRIDES[hypothesis]),
        }

    ridge_alphas = (10.0, 100.0, 1000.0, 10000.0)
    print(f"\n[{hypothesis}] validation selection")
    print(f"  candidates = {len(mechanism_sets)} x {len(gate_set_options)} x "
          f"{len(ridge_alphas)} = "
          f"{len(mechanism_sets)*len(gate_set_options)*len(ridge_alphas)}")
    print(f"  subtrain={sel.train_indices.size}  validation={sel.validation_indices.size}"
          f"  test={sel.test_indices.size}")

    rows: list[dict[str, Any]] = []
    for mechanisms in mechanism_sets:
        for gate_set_name, gate_names in gate_set_options.items():
            gate_indices = [all_gate_names.index(name) for name in gate_names]
            for alpha in ridge_alphas:
                fit = fit_candidate(
                    model, bl_s, ob_s, dataset.time_years,
                    train_indices=sel.train_indices, mechanisms=mechanisms,
                    gate_matrix=gate_matrix, gate_indices=gate_indices,
                    all_gate_names=all_gate_names, amyloid=amyloid,
                    thickness=thickness, ptau181=ptau181, hubness=hubness,
                    regional_masks=regional_masks, ridge_alpha=alpha,
                    fit_intercept=False,
                )
                pred_s = predict_candidate(
                    model, bl_s, dataset.time_years, fit,
                    gate_matrix=gate_matrix, amyloid=amyloid,
                    thickness=thickness, ptau181=ptau181, hubness=hubness,
                )
                pred = scaler.inverse_transform(pred_s)
                pair_metrics = compute_pair_metrics(
                    dataset.pairs, dataset.baseline, dataset.observed, pred, sel,
                    candidate_name(mechanisms, gate_set_name, alpha),
                )
                summary = summarize_split(pair_metrics, "validation")
                row = {
                    "candidate": candidate_name(mechanisms, gate_set_name, alpha),
                    "mechanisms": list(mechanisms),
                    "gate_set": gate_set_name,
                    "gate_names": gate_names,
                    "ridge_alpha": alpha,
                    "train_rate_r2": fit.train_r2,
                    "train_rate_mse": fit.train_mse,
                    **summary,
                }
                rows.append(row)
                # Compact line
                print(f"   val Δρ={summary['validation_delta_spearman_mean']:+.4f}  "
                      f"top10={summary['validation_top10_overlap_mean']:.4f}  "
                      f"MAE={summary['validation_mae_mean']:.4f}  "
                      f"trR2={fit.train_r2:.4f}  "
                      f"{row['candidate']}")

    best = choose_candidate(rows)
    print("\n  Selected:")
    print_candidate(best, prefix="    ")

    selected_mechanisms = tuple(str(x) for x in best["mechanisms"])
    selected_gate_indices = [
        all_gate_names.index(n) for n in best["gate_names"]
    ]
    final_fit = fit_candidate(
        model, bl_s, ob_s, dataset.time_years,
        train_indices=split.train_indices, mechanisms=selected_mechanisms,
        gate_matrix=gate_matrix, gate_indices=selected_gate_indices,
        all_gate_names=all_gate_names, amyloid=amyloid, thickness=thickness,
        ptau181=ptau181, hubness=hubness, regional_masks=regional_masks,
        ridge_alpha=float(best["ridge_alpha"]), fit_intercept=False,
    )
    final_pred_s = predict_candidate(
        model, bl_s, dataset.time_years, final_fit,
        gate_matrix=gate_matrix, amyloid=amyloid, thickness=thickness,
        ptau181=ptau181, hubness=hubness,
    )
    final_pred = scaler.inverse_transform(final_pred_s)
    model_name = f"symbolic_ode_{hypothesis}"
    pair_metrics = compute_pair_metrics(
        dataset.pairs, dataset.baseline, dataset.observed, final_pred, split,
        model_name,
    )
    summary_rows = compute_aggregate_metrics(pair_metrics)

    test_summary = {row["metric"]: row for row in summary_rows if row["split"] == "test"}
    print(f"\n  TEST  MAE={test_summary['mae']['mean']:.4f}  "
          f"Δρ={test_summary['delta_spearman']['mean']:+.4f}  "
          f"subjρ={test_summary['subject_spearman']['mean']:.4f}  "
          f"top10={test_summary['top10_overlap']['mean']:.4f}")

    report = {
        "model": model_name,
        "hypothesis": hypothesis,
        "selected": best,
        "candidate_selection": rows,
        "final_fit": {
            "mechanisms": list(final_fit.mechanisms),
            "gate_names": final_fit.gate_names,
            "ridge_alpha": final_fit.ridge_alpha,
            "intercept": final_fit.intercept,
            "train_rate_r2": final_fit.train_r2,
            "train_rate_mse": final_fit.train_mse,
            "top_terms": top_terms(final_fit, n=25),
        },
        "test_metrics": {
            metric: {"mean": float(row["mean"]),
                     "median": float(row["median"]),
                     "std": float(row["std"])}
            for metric, row in test_summary.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / f"{model_name}_pair_metrics.csv", pair_metrics)
    write_csv_rows(output_dir / f"{model_name}_metrics_summary.csv", summary_rows)
    write_json(output_dir / f"{model_name}_report.json", report)
    print(f"  Wrote {model_name}_*.csv/json to {output_dir}")
    return report


def default_config_path() -> Path:
    exp = PROJECT_ROOT / "experiments" / "group_average_enigma"
    return exp / "config_hcp.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hypothesis",
        choices=["H1", "H2", "H3", "H4", "H5", "baseline", "all"],
        required=True,
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--no-baseline-sets", action="store_true",
                        help="Test only hypothesis-specific mechanism sets, "
                             "without baseline mechanism sets for comparison.")
    args = parser.parse_args()

    hyps = (["baseline", "H1", "H2", "H3", "H4", "H5"]
            if args.hypothesis == "all" else [args.hypothesis])
    summary: dict[str, dict[str, Any]] = {}
    for hyp in hyps:
        rep = run_hypothesis(
            hyp, include_baseline_sets=(not args.no_baseline_sets) and (hyp != "baseline"),
            validation_fraction=args.validation_fraction,
            config_path=args.config,
        )
        summary[hyp] = {
            "selected_candidate": rep["selected"]["candidate"],
            "test_metrics": rep["test_metrics"],
        }
    # Print compact comparison table
    print("\n\n=== Comparison summary ===")
    metrics = ["mae", "rmse", "subject_spearman", "delta_spearman", "top10_overlap"]
    header = "  " + "  ".join(f"{m:>16s}" for m in metrics) + "    selected"
    print(header)
    for hyp, info in summary.items():
        cells = "  ".join(
            f"{info['test_metrics'][m]['mean']:>16.4f}" for m in metrics
        )
        print(f"{hyp:>3s} {cells}    {info['selected_candidate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
