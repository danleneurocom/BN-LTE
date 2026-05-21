#!/usr/bin/env python3
"""H1 ablation: isolate the contribution of hubness as a gate.

The original H1 run swapped multiple gates (added amyloid_mean + hubness_mean +
hubness_max, dropped eigenmode_1/4, braak_early_ratio, tau_lr_asymmetry). To
attribute the +0.030 test delta_spearman gain to hubness specifically, this
script tests three configurations on the same mechanism set as the baseline
winner:

  A. spatial_state                                  (baseline winner gates)
  B. spatial_state + hubness_mean + hubness_max     (baseline + hubness only)
  C. hubness_mean + hubness_max only                (hubness alone)

If A << B and A << C, hubness is the driver. If A << B but A ≈ C, hubness combined
with structural gates matters but hubness alone is weak.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from spread_toolbox.adni_features import build_closure_covariates
from spread_toolbox.forecasting import (
    MinMaxStateScaler, compute_aggregate_metrics, compute_pair_metrics,
    load_forecast_dataset, load_labeled_matrix, make_subject_split,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path
from spread_toolbox.models.symbolic_ode import (
    SymbolicODEModel, build_amortization_features,
)
from run_symbolic_ode_state_gated import (
    build_braak_indices, build_braak_region_masks,
    make_selection_split,
)
from run_symbolic_ode_hypotheses import (
    augment_gate_matrix, fit_candidate, predict_candidate,
)


def main() -> int:
    config = load_yaml_config(PROJECT_ROOT / "experiments" / "group_average_enigma" / "config_hcp.yaml")
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs = config.get("outputs", {})
    seed = 20260507

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    split = make_subject_split(dataset.pairs, test_fraction=0.2, random_seed=seed)
    sel = make_selection_split(dataset, split, validation_fraction=0.2,
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
    _, eigenvectors = np.linalg.eigh(laplacian)

    gate_matrix, all_gate_names = build_amortization_features(
        bl_s, dataset.time_years, amyloid, thickness, apoe4, ptau181,
        braak_idx, eigenvectors, model.adj_norm,
    )

    hubness_raw = adjacency.sum(axis=1).astype(float)
    hubness = (hubness_raw - hubness_raw.mean()) / (hubness_raw.std() + 1e-12)

    gate_matrix, all_gate_names = augment_gate_matrix(
        gate_matrix, all_gate_names,
        bl_s=bl_s, thickness=thickness, amyloid=amyloid,
        ptau181=ptau181, hubness=hubness,
    )

    # The baseline winner mechanism set
    mechanisms = (
        "growth_braak_I_II", "growth_braak_III_IV", "growth_braak_V_VI",
        "growth_braak_Other", "amyloid_growth", "fickian", "fickian_x_tau",
        "tau_decay",
    )

    spatial_state = [
        "tau_braak_I-II", "tau_braak_III-IV", "tau_braak_V-VI",
        "braak_early_ratio", "eigenmode_1_loading", "eigenmode_4_loading",
        "fickian_drive_magnitude", "tau_lr_asymmetry",
    ]

    h1_hypothesis = ["tau_braak_I-II", "tau_braak_III-IV", "tau_braak_V-VI",
                     "amyloid_mean", "fickian_drive_magnitude",
                     "hubness_mean", "hubness_max"]
    h1_hypothesis_no_hub = ["tau_braak_I-II", "tau_braak_III-IV", "tau_braak_V-VI",
                            "amyloid_mean", "fickian_drive_magnitude"]
    ablations = {
        "A_spatial_only":  spatial_state,
        "F_H1_full":  h1_hypothesis,
        "G_H1_minus_hub":  h1_hypothesis_no_hub,
        "H_just_amyloid_mean":  ["tau_braak_I-II", "tau_braak_III-IV", "tau_braak_V-VI",
                                 "amyloid_mean", "fickian_drive_magnitude",
                                 "braak_early_ratio", "eigenmode_1_loading",
                                 "eigenmode_4_loading", "tau_lr_asymmetry"],
    }

    print(f"\nH1 ablation — mechanism = {mechanisms}\n")
    print(f"{'name':>22s}  {'ridge':>6s}  {'val Δρ':>8s}  {'test MAE':>9s}  {'test Δρ':>8s}  {'test top10':>10s}")
    results: dict[str, Any] = {}
    for name, gnames in ablations.items():
        gnames = [n for n in gnames if n in all_gate_names]
        gindices = [all_gate_names.index(n) for n in gnames]
        results[name] = []
        for alpha in (1000.0, 10000.0):
            fit = fit_candidate(
                model, bl_s, ob_s, dataset.time_years,
                train_indices=sel.train_indices, mechanisms=mechanisms,
                gate_matrix=gate_matrix, gate_indices=gindices,
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
            pm = compute_pair_metrics(
                dataset.pairs, dataset.baseline, dataset.observed, pred, sel,
                f"h1abl_{name}_r{int(alpha)}",
            )
            val_rows = [r for r in pm if r["split"] == "validation"]
            v_d = np.array([r["delta_spearman"] for r in val_rows if r["delta_spearman"] == r["delta_spearman"]]).mean()

            # Refit on full training and evaluate on test
            fit_full = fit_candidate(
                model, bl_s, ob_s, dataset.time_years,
                train_indices=split.train_indices, mechanisms=mechanisms,
                gate_matrix=gate_matrix, gate_indices=gindices,
                all_gate_names=all_gate_names, amyloid=amyloid,
                thickness=thickness, ptau181=ptau181, hubness=hubness,
                regional_masks=regional_masks, ridge_alpha=alpha,
                fit_intercept=False,
            )
            pred_s_full = predict_candidate(
                model, bl_s, dataset.time_years, fit_full,
                gate_matrix=gate_matrix, amyloid=amyloid,
                thickness=thickness, ptau181=ptau181, hubness=hubness,
            )
            pred_full = scaler.inverse_transform(pred_s_full)
            pm_full = compute_pair_metrics(
                dataset.pairs, dataset.baseline, dataset.observed, pred_full, split,
                f"h1abl_{name}_r{int(alpha)}_test",
            )
            test_rows = [r for r in pm_full if r["split"] == "test"]
            t_mae = np.array([r["mae"] for r in test_rows]).mean()
            t_d = np.array([r["delta_spearman"] for r in test_rows if r["delta_spearman"] == r["delta_spearman"]]).mean()
            t_top10 = np.array([r["top10_overlap"] for r in test_rows]).mean()

            print(f"{name:>22s}  {int(alpha):>6d}  {v_d:>+8.4f}  {t_mae:>9.4f}  {t_d:>+8.4f}  {t_top10:>10.4f}")
            results[name].append({
                "ridge_alpha": alpha, "val_delta_spearman": v_d,
                "test_mae": t_mae, "test_delta_spearman": t_d, "test_top10": t_top10,
                "gates": gnames,
            })

    (output_dir / "symbolic_ode_H1_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {output_dir / 'symbolic_ode_H1_ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
