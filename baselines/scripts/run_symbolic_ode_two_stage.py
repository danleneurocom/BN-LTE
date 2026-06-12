#!/usr/bin/env python3
"""Symbolic ODE — Improvements 2 + 4 + 1 combined.

Full model:
    dS_i/dt = alpha1_i * f1(S, A) + alpha2_i * f2(S, connectivity)

Stage 1 — analytical amyloid-saturation (Improvement 2):
    f1 = beta0 + beta1 * amyloid_x_tau * (C - tau)
    Fits 3 parameters by L-BFGS-B on training data (seconds).
    This is the saturation form validated across two PySR runs (C≈3.9).

Stage 2 — PySR on residual after f1 removed (Improvement 4):
    f2 = PySR( observed_rate - f1 )
    With amyloid term gone, PySR is free to discover connectivity/Fickian terms.
    Uses lower parsimony (0.003) than original discovery (0.01).

Per-subject (alpha1_i, alpha2_i) via OLS + rich amortization (Improvement 1):
    alpha1 ∝ amyloid burden, APOE4, disease stage
    alpha2 ∝ disease stage, HCP eigenmode loading
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.adni_features import build_closure_covariates            # noqa: E402
from spread_toolbox.forecasting import (                                       # noqa: E402
    ForecastDataset, MinMaxStateScaler, SubjectSplit,
    compute_aggregate_metrics, compute_pair_metrics,
    load_forecast_dataset, load_labeled_matrix,
    make_subject_split, read_csv_rows, write_csv_rows, write_json,
)
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path    # noqa: E402
from spread_toolbox.models.symbolic_ode import (                             # noqa: E402
    SymbolicODEModel, build_amortization_features,
    fit_amortize_two,
)

MODEL_NAME = "symbolic_ode_two_stage"


def default_config_path() -> Path:
    exp = PROJECT_ROOT / "experiments" / "group_average_enigma"
    local = exp / "config.yaml"
    return local if local.exists() else exp / "config.example.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--pysr-model-selection", choices=("best", "accuracy", "score"), default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config     = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs    = config.get("outputs", {})
    modeling   = config.get("modeling", {})
    seed       = int(config.get("experiment", {}).get("random_seed", 20260507))

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    split   = make_subject_split(
        dataset.pairs, test_fraction=float(modeling.get("test_fraction", 0.2)),
        random_seed=seed,
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
    amyloid   = reg_cov.get("amyloid_suvr")
    thickness = reg_cov.get("cortical_thickness")
    apoe4     = pair_cov.get("apoe4_dose")
    ptau181   = pair_cov.get("plasma_ptau181")
    pair_groups = np.array([str(p["RID"]) for p in dataset.pairs])
    region_labels = dataset.region_labels

    # Braak-stage region indices
    BRAAK = {
        "I-II":   ["L_entorhinal","R_entorhinal","L_parahippocampal","R_parahippocampal"],
        "III-IV": ["L_fusiform","R_fusiform","L_inferiortemporal","R_inferiortemporal",
                   "L_middletemporal","R_middletemporal","L_isthmuscingulate","R_isthmuscingulate",
                   "L_posteriorcingulate","R_posteriorcingulate","L_insula","R_insula"],
        "V-VI":   ["L_inferiorparietal","R_inferiorparietal","L_superiorparietal","R_superiorparietal",
                   "L_precuneus","R_precuneus","L_superiorfrontal","R_superiorfrontal",
                   "L_rostralmiddlefrontal","R_rostralmiddlefrontal",
                   "L_superiortemporal","R_superiortemporal"],
    }
    braak_idx = {s: [region_labels.index(r) for r in rs if r in region_labels]
                 for s, rs in BRAAK.items()}
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues = np.clip(eigenvalues, 0, None)

    model = SymbolicODEModel(adjacency, steps_per_year=12)

    # ── Step 1: Two-stage fit (Stage 1 analytical + Stage 2 PySR on residual) ─
    print("\n[1/4] Two-stage symbolic ODE fit...")
    ts_fit = model.fit_two_stage(
        bl_s, ob_s, dataset.time_years,
        train_indices=split.train_indices,
        amyloid=amyloid, thickness=thickness,
        pysr_niterations     =int(modeling.get("symbolic_ode_pysr_niterations",   300)),
        pysr_populations     =int(modeling.get("symbolic_ode_pysr_populations",    20)),
        pysr_population_size =int(modeling.get("symbolic_ode_pysr_population_size",33)),
        pysr_maxsize         =int(modeling.get("symbolic_ode_pysr_maxsize",        15)),
        pysr_parsimony       =float(modeling.get("symbolic_ode_pysr_parsimony",  0.003)),
        pysr_timeout_seconds =int(modeling.get("symbolic_ode_pysr_timeout_seconds",300)),
        pysr_model_selection =args.pysr_model_selection
                              or str(modeling.get("symbolic_ode_pysr_model_selection", "best")),
        pysr_batching        =bool(modeling.get("symbolic_ode_pysr_batching",    True)),
        pysr_batch_size      =int(modeling.get("symbolic_ode_pysr_batch_size",   2048)),
        max_train_rows       =int(modeling.get("symbolic_ode_max_train_rows",   40000)),
        random_seed=seed,
    )
    print(f"\n   Stage 1 (analytical): f1 = {ts_fit.stage1_beta0:.5f} "
          f"+ {ts_fit.stage1_beta1:.5f} * amyloid_x_tau * ({ts_fit.stage1_cap:.3f} - tau)")
    print(f"   Stage 1 R² = {ts_fit.stage1_train_r2:.4f}")
    print(f"   Stage 2 (PySR residual): f2 = {ts_fit.stage2_expression}")
    print(f"   Stage 2 R² = {ts_fit.stage2_train_r2:.4f}  "
          f"(combined Stage1+Stage2 R² ≈ {ts_fit.stage1_train_r2 + ts_fit.stage2_train_r2:.4f})")
    fickian_found = "fickian" in ts_fit.stage2_expression.lower()
    print(f"   Fickian connectivity in Stage 2: {'YES ✓' if fickian_found else 'NO — check Pareto'}")
    if not fickian_found and ts_fit.stage2_pareto_front:
        print("   Stage 2 Pareto front:")
        for row in sorted(ts_fit.stage2_pareto_front, key=lambda r: r.get("complexity", 0))[:8]:
            print(f"     c={row['complexity']:2d}  loss={row['loss']:.6f}  {row['equation']}")

    # ── Step 2: Per-subject (alpha1_i, alpha2_i) fitting ──────────────────────
    print(f"\n[2/4] Fitting per-subject (alpha1, alpha2) on {split.train_indices.size} training pairs...")
    alpha1_train, alpha2_train = model.fit_per_subject_two_component(
        bl_s, ob_s, dataset.time_years, ts_fit,
        indices=split.train_indices,
        amyloid=amyloid, thickness=thickness,
    )
    print(f"   alpha1 (amyloid-growth):  mean={alpha1_train.mean():.3f}  median={np.median(alpha1_train):.3f}")
    print(f"   alpha2 (autonomous seed): mean={alpha2_train.mean():.3f}  median={np.median(alpha2_train):.3f}")

    # Correlate with biology
    _print_alpha_correlations(alpha1_train, alpha2_train, bl_s, split, dataset,
                               amyloid, apoe4, ptau181, braak_idx, eigenvectors)

    # ── Step 3: Rich amortization ─────────────────────────────────────────────
    print(f"\n[3/4] Amortizing (alpha1, alpha2) from biology features...")
    X_amort, feat_names = build_amortization_features(
        bl_s, dataset.time_years, amyloid, thickness, apoe4, ptau181,
        braak_idx, eigenvectors, model.adj_norm,
    )
    alpha1_pred, alpha2_pred, amort_report = fit_amortize_two(
        alpha1_train, alpha2_train, X_amort,
        train_indices=split.train_indices,
        pair_groups=pair_groups,
        feat_names=feat_names,
        random_seed=seed,
    )
    print(f"   alpha1 R²={amort_report['r2_alpha1']:.4f}  "
          f"alpha2 R²={amort_report['r2_alpha2']:.4f}")
    print(f"   Top alpha1 predictors: {amort_report['top_alpha1']}")
    print(f"   Top alpha2 predictors: {amort_report['top_alpha2']}")

    # ── Step 4: Predict and evaluate ─────────────────────────────────────────
    print(f"\n[4/4] Predicting with two-component amortized model (alpha1, alpha2)...")
    ts_pred = scaler.inverse_transform(
        model.predict_two_stage(
            bl_s, dataset.time_years, ts_fit,
            alpha1_pred, alpha2_pred,
            amyloid=amyloid, thickness=thickness,
        )
    )
    # Also compute global (alpha1=alpha2=1) for comparison
    global_pred = scaler.inverse_transform(
        model.predict_two_stage(
            bl_s, dataset.time_years, ts_fit,
            np.ones(len(dataset.pairs)), np.ones(len(dataset.pairs)),
            amyloid=amyloid, thickness=thickness,
        )
    )

    global_pm = compute_pair_metrics(dataset.pairs, dataset.baseline, dataset.observed,
                                     global_pred, split, "symbolic_ode_two_stage_global")
    ts_pm     = compute_pair_metrics(dataset.pairs, dataset.baseline, dataset.observed,
                                     ts_pred, split, MODEL_NAME)
    global_summary = compute_aggregate_metrics(global_pm)
    ts_summary     = compute_aggregate_metrics(ts_pm)

    comparison_files = {
        "FKPP+IR":  output_dir / "individualized_residual_metrics_summary.csv",
        "NDM+IR":   output_dir / "ndm_individualized_residual_metrics_summary.csv",
        "Bio-FKPP": output_dir / "bio_fkpp_metrics_summary.csv",
    }
    all_rows = list(global_summary) + list(ts_summary)
    for name, path in comparison_files.items():
        if path.exists():
            for r in read_csv_rows(path):
                r["model"] = name
                all_rows.append(r)

    print()
    _print_table(all_rows)

    report = {
        "model": MODEL_NAME,
        "stage1": {
            "equation": f"f1 = {ts_fit.stage1_beta0:.5f} + {ts_fit.stage1_beta1:.5f} * amyloid_x_tau * ({ts_fit.stage1_cap:.3f} - tau)",
            "r2_train": ts_fit.stage1_train_r2,
        },
        "stage2": {
            "expression": ts_fit.stage2_expression,
            "r2_train": ts_fit.stage2_train_r2,
            "fickian_discovered": fickian_found,
            "pareto_front": ts_fit.stage2_pareto_front,
        },
        "amortization": amort_report,
        "alpha1_stats": {"mean": float(alpha1_train.mean()), "std": float(alpha1_train.std())},
        "alpha2_stats": {"mean": float(alpha2_train.mean()), "std": float(alpha2_train.std())},
    }

    if not args.no_write:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv_rows(output_dir / f"{MODEL_NAME}_pair_metrics.csv",    ts_pm)
        write_csv_rows(output_dir / f"{MODEL_NAME}_metrics_summary.csv", ts_summary)
        write_csv_rows(output_dir / "symbolic_ode_two_stage_global_metrics_summary.csv", global_summary)
        write_json(output_dir / f"{MODEL_NAME}_report.json", report)
        print("Wrote outputs.")
    return 0


def _print_alpha_correlations(alpha1, alpha2, bl_s, split, dataset, amyloid, apoe4, ptau181,
                               braak_idx, eigenvectors):
    from scipy import stats as sp_stats
    tr = split.train_indices
    bl_tr = bl_s[tr]
    t_tr  = dataset.time_years[tr]
    print("\n   Correlations:  (* p<0.05)")
    print(f"   {'Feature':<35} {'r(α1)':>8} {'r(α2)':>8}")
    print(f"   {'-'*53}")
    def row(name, vals):
        r1, p1 = sp_stats.spearmanr(alpha1, vals)
        r2, p2 = sp_stats.spearmanr(alpha2, vals)
        s1 = '*' if p1 < 0.05 else ' '
        s2 = '*' if p2 < 0.05 else ' '
        print(f"   {name:<35} {r1:>+7.3f}{s1} {r2:>+7.3f}{s2}")
    for stage, idx in braak_idx.items():
        if idx: row(f"tau_braak_{stage}", bl_tr[:, idx].mean(1))
    row("amyloid_mean",   np.nanmean(amyloid[tr], 1) if amyloid is not None else np.zeros(len(tr)))
    row("follow_up_time", t_tr)
    if apoe4 is not None:
        row("apoe4_dose", np.nan_to_num(np.asarray(apoe4[tr], float), nan=0.))
    if ptau181 is not None:
        pt = np.asarray(ptau181[tr], float)
        row("plasma_ptau181", np.where(np.isfinite(pt), pt, np.nanmedian(pt)))
    for k in range(3):
        row(f"eigenmode_{k}_loading", bl_tr @ eigenvectors[:, k])


def _print_table(rows):
    by_model: dict[str, dict] = {}
    for r in rows:
        if r.get("split") == "test":
            by_model.setdefault(r["model"], {})[r.get("metric", "")] = r
    def med(m, k):
        v = m.get(k, {}); return float(v.get("median", float("nan"))) if isinstance(v, dict) else float("nan")
    order  = ["symbolic_ode_two_stage_global", MODEL_NAME, "FKPP+IR", "NDM+IR", "Bio-FKPP"]
    labels = {
        "symbolic_ode_two_stage_global": "Two-stage ODE (α1=α2=1, global)",
        MODEL_NAME:                      "Two-stage ODE (amortized α)  ← new",
        "FKPP+IR": "FKPP+IR", "NDM+IR": "NDM+IR", "Bio-FKPP": "Bio-FKPP",
    }
    print(f"{'Model':<46} {'subj_ρ':>8} {'delta_ρ':>9} {'MAE':>8}")
    print("-" * 74)
    for m in order:
        if m in by_model:
            print(f"  {labels.get(m,m):<44} "
                  f"{med(by_model[m],'subject_spearman'):>8.4f} "
                  f"{med(by_model[m],'delta_spearman'):>9.4f} "
                  f"{med(by_model[m],'mae'):>8.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
