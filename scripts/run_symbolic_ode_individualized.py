#!/usr/bin/env python3
"""Symbolic ODE + per-subject two-component weights — Improvement 1 (fixed).

Key insight from first attempt:
  A scalar alpha_i multiplying f_sym CANNOT improve delta_spearman — multiplying a
  vector by a positive scalar never changes rank ordering.

Fix: fit TWO per-subject weights (alpha1_i, alpha2_i) for TWO spatially orthogonal terms:
    dS/dt = alpha1_i * amyloid*tau*(1-tau)          [amyloid-growth, spatial pattern ∝ amyloid]
          + alpha2_i * Fickian_gradient(tau) * tau   [connectivity, spatial pattern ∝ HCP routing]

These have DIFFERENT spatial patterns, so changing their relative weight shifts which
regions are predicted to gain tau — directly improving delta_spearman.

Fitting: closed-form OLS per subject (no iterative optimisation).
Amortization: multi-target ridge from biologically rich features:
  Braak-stage tau means, tau concentration (Gini), amyloid-tau spatial correlation,
  plasma p-tau181, APOE4, follow-up time, HCP eigenmode loadings.
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
    SymbolicODEModel, build_amortization_features, fit_amortize_two,
)

MODEL_NAME = "symbolic_ode_two_component"


def default_config_path() -> Path:
    exp = PROJECT_ROOT / "experiments" / "group_average_enigma"
    local = exp / "config.yaml"
    return local if local.exists() else exp / "config.example.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config     = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs    = config.get("outputs", {})
    modeling   = config.get("modeling", {})
    seed       = int(config.get("experiment", {}).get("random_seed", 20260507))

    dataset = load_forecast_dataset(config, PROJECT_ROOT)
    split   = make_subject_split(
        dataset.pairs,
        test_fraction=float(modeling.get("test_fraction", 0.2)),
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

    pair_cov, reg_cov, cov_report = build_closure_covariates(
        dataset, split, config, PROJECT_ROOT
    )
    amyloid   = reg_cov.get("amyloid_suvr")
    thickness = reg_cov.get("cortical_thickness")
    apoe4     = pair_cov.get("apoe4_dose")
    ptau181   = pair_cov.get("plasma_ptau181")
    pair_groups = np.array([str(p["RID"]) for p in dataset.pairs])
    region_labels = dataset.region_labels

    # ── Braak-stage region indices ────────────────────────────────────────────
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
    braak_idx = {stage: [region_labels.index(r) for r in regions if r in region_labels]
                 for stage, regions in BRAAK.items()}

    # HCP eigenmodes
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues = np.clip(eigenvalues, 0, None)

    # ── Step 1: Fit Symbolic ODE (group-level PySR) ───────────────────────────
    print("\n[1/4] Fitting group-level Symbolic ODE (PySR)...")
    model   = SymbolicODEModel(adjacency, steps_per_year=12)
    sym_fit = model.fit(
        bl_s, ob_s, dataset.time_years,
        train_indices=split.train_indices,
        amyloid=amyloid, thickness=thickness,
        pysr_niterations     =int(modeling.get("symbolic_ode_pysr_niterations",   300)),
        pysr_populations     =int(modeling.get("symbolic_ode_pysr_populations",    20)),
        pysr_population_size =int(modeling.get("symbolic_ode_pysr_population_size",33)),
        pysr_maxsize         =int(modeling.get("symbolic_ode_pysr_maxsize",        15)),
        pysr_parsimony       =float(modeling.get("symbolic_ode_pysr_parsimony",  0.01)),
        pysr_timeout_seconds =int(modeling.get("symbolic_ode_pysr_timeout_seconds",300)),
        pysr_batching        =bool(modeling.get("symbolic_ode_pysr_batching",    True)),
        pysr_batch_size      =int(modeling.get("symbolic_ode_pysr_batch_size",   2048)),
        max_train_rows       =int(modeling.get("symbolic_ode_max_train_rows",   40000)),
        random_seed=seed,
    )
    print(f"   Equation (group): dS/dt = {sym_fit.symbolic_expression}")

    # Global single-alpha predictions (baseline)
    global_pred = scaler.inverse_transform(
        model.predict(bl_s, dataset.time_years, sym_fit,
                      amyloid=amyloid, thickness=thickness)
    )

    # ── Step 2: Fit per-subject (alpha1_i, alpha2_i) on training pairs ────────
    print(f"\n[2/4] Fitting per-subject (alpha1, alpha2) on {split.train_indices.size} training pairs...")
    alpha1_train, alpha2_train = model.fit_per_subject_two_component(
        bl_s, ob_s, dataset.time_years, sym_fit,
        indices=split.train_indices,
        amyloid=amyloid, thickness=thickness,
    )
    print(f"   alpha1 (amyloid-growth):     mean={alpha1_train.mean():.3f}  "
          f"std={alpha1_train.std():.3f}  median={np.median(alpha1_train):.3f}")
    print(f"   alpha2 (Fickian-connectivity): mean={alpha2_train.mean():.3f}  "
          f"std={alpha2_train.std():.3f}  median={np.median(alpha2_train):.3f}")
    print(f"   alpha2 > 0 (connectivity-driven): {(alpha2_train > 0).sum()} / {len(alpha2_train)}")

    # Correlate (alpha1, alpha2) with biology to understand what predicts them
    _print_alpha_correlations(
        alpha1_train, alpha2_train,
        bl_s, split, dataset, amyloid, apoe4, ptau181,
        braak_idx, eigenvectors,
    )

    # ── Step 3: Rich amortization feature matrix ─────────────────────────────
    print(f"\n[3/4] Building rich amortization features and fitting ridge regression...")
    X_amort, feat_names = build_amortization_features(
        bl_s, dataset.time_years, amyloid, thickness, apoe4, ptau181,
        braak_idx, eigenvectors, model.adj_norm,
    )
    alpha1_pred_all, alpha2_pred_all, amort_report = fit_amortize_two(
        alpha1_train, alpha2_train, X_amort,
        train_indices=split.train_indices,
        pair_groups=pair_groups,
        feat_names=feat_names,
        random_seed=seed,
    )
    print(f"   alpha1 amortization R²={amort_report['r2_alpha1']:.4f}")
    print(f"   alpha2 amortization R²={amort_report['r2_alpha2']:.4f}")
    print(f"   Top predictors of alpha1: {amort_report['top_alpha1']}")
    print(f"   Top predictors of alpha2: {amort_report['top_alpha2']}")

    # ── Step 4: Predict with amortized (alpha1, alpha2) ──────────────────────
    print(f"\n[4/4] Predicting with two-component amortized model...")
    two_pred = scaler.inverse_transform(
        model.predict_two_component(
            bl_s, dataset.time_years, sym_fit,
            alpha1_pred_all, alpha2_pred_all,
            amyloid=amyloid, thickness=thickness,
        )
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    global_pm = compute_pair_metrics(
        dataset.pairs, dataset.baseline, dataset.observed,
        global_pred, split, "symbolic_ode_global"
    )
    two_pm = compute_pair_metrics(
        dataset.pairs, dataset.baseline, dataset.observed,
        two_pred, split, MODEL_NAME
    )
    global_summary = compute_aggregate_metrics(global_pm)
    two_summary    = compute_aggregate_metrics(two_pm)

    # ── Comparison table ──────────────────────────────────────────────────────
    comparison_files = {
        "FKPP+IR":  output_dir / "individualized_residual_metrics_summary.csv",
        "NDM+IR":   output_dir / "ndm_individualized_residual_metrics_summary.csv",
        "Bio-FKPP": output_dir / "bio_fkpp_metrics_summary.csv",
    }
    all_rows = list(global_summary) + list(two_summary)
    for name, path in comparison_files.items():
        if path.exists():
            for r in read_csv_rows(path):
                r["model"] = name
                all_rows.append(r)

    print()
    _print_table(all_rows)

    report = {
        "model": MODEL_NAME,
        "equation": sym_fit.symbolic_expression,
        "two_component": {
            "alpha1_amyloid_growth_mean": float(alpha1_train.mean()),
            "alpha2_fickian_connectivity_mean": float(alpha2_train.mean()),
            "alpha2_positive_pct": float((alpha2_train > 0).mean()),
        },
        "amortization": amort_report,
    }

    if not args.no_write:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_csv_rows(output_dir / f"{MODEL_NAME}_pair_metrics.csv",    two_pm)
        write_csv_rows(output_dir / f"{MODEL_NAME}_metrics_summary.csv", two_summary)
        write_csv_rows(output_dir / "symbolic_ode_global_metrics_summary.csv", global_summary)
        write_json(output_dir / f"{MODEL_NAME}_report.json", report)
        # Save per-subject weights
        weight_rows = []
        for k, i in enumerate(split.train_indices):
            p = dataset.pairs[i]
            weight_rows.append({
                "RID": str(p["RID"]), "split": "train",
                "alpha1_fitted": float(alpha1_train[k]),
                "alpha2_fitted": float(alpha2_train[k]),
                "alpha1_amortized": float(alpha1_pred_all[i]),
                "alpha2_amortized": float(alpha2_pred_all[i]),
            })
        for i in split.test_indices:
            p = dataset.pairs[i]
            weight_rows.append({
                "RID": str(p["RID"]), "split": "test",
                "alpha1_fitted": float("nan"), "alpha2_fitted": float("nan"),
                "alpha1_amortized": float(alpha1_pred_all[i]),
                "alpha2_amortized": float(alpha2_pred_all[i]),
            })
        write_csv_rows(output_dir / "symbolic_ode_two_component_weights.csv", weight_rows)
        print("Wrote outputs.")
    return 0


# ── Rich amortization feature builder ────────────────────────────────────────

def _build_amortization_features(
    bl_s, time_years, amyloid, thickness, apoe4, ptau181,
    braak_idx, eigenvectors, adj_norm,
) -> tuple[np.ndarray, list[str]]:
    """Build biologically rich (n_pairs, n_features) feature matrix for amortizing alpha."""
    from scipy import stats as sp_stats

    n = bl_s.shape[0]
    cols, names = [], []

    # 1. Braak-stage mean tau (captures disease stage better than global mean)
    for stage, idx in braak_idx.items():
        if idx:
            cols.append(bl_s[:, idx].mean(axis=1, keepdims=True))
            names.append(f"tau_braak_{stage}")
    cols.append(bl_s.mean(axis=1, keepdims=True)); names.append("tau_mean")
    cols.append(bl_s.std(axis=1, keepdims=True));  names.append("tau_std")
    cols.append(bl_s.max(axis=1, keepdims=True));  names.append("tau_max")

    # 2. Tau spatial Gini coefficient (concentration: early vs late stage)
    def gini(x):
        x = np.sort(np.abs(x)); n = len(x)
        return (2*np.sum(np.arange(1,n+1)*x)/(n*np.sum(x)) - (n+1)/n) if x.sum()>0 else 0.
    gini_vals = np.array([gini(bl_s[i]) for i in range(n)])[:, None]
    cols.append(gini_vals); names.append("tau_gini")

    # 3. Braak early/late ratio (stage progression proxy)
    b12 = bl_s[:, braak_idx["I-II"]].mean(axis=1)
    b56 = bl_s[:, braak_idx["V-VI"]].mean(axis=1)
    cols.append((b12 / (b12 + b56 + 1e-8))[:, None]); names.append("braak_early_ratio")

    # 4. Amyloid features
    if amyloid is not None:
        amy = np.asarray(amyloid, dtype=float)
        cols.append(np.nanmean(amy, axis=1, keepdims=True)); names.append("amyloid_mean")
        cols.append(np.nanmax(amy,  axis=1, keepdims=True)); names.append("amyloid_max")
        # Amyloid specifically in Braak I-II (where tau catalysis is most relevant)
        amy_b12 = np.nanmean(amy[:, braak_idx["I-II"]], axis=1, keepdims=True)
        cols.append(amy_b12); names.append("amyloid_braak_I_II")

        # Amyloid-tau spatial correlation per subject
        atcorr = np.array([
            sp_stats.pearsonr(bl_s[i], amy[i])[0]
            if np.std(bl_s[i]) > 1e-8 and np.std(amy[i]) > 1e-8 else 0.
            for i in range(n)
        ])
        cols.append(np.nan_to_num(atcorr, nan=0.)[:, None])
        names.append("amyloid_tau_spatial_corr")

        # APOE4 × amyloid interaction
        if apoe4 is not None:
            a4 = np.nan_to_num(np.asarray(apoe4, dtype=float), nan=0.)
            amy_m = np.nanmean(amy, axis=1)
            cols.append((a4 * amy_m)[:, None]); names.append("apoe4_x_amyloid")

    # 5. APOE4 dose
    if apoe4 is not None:
        a4 = np.nan_to_num(np.asarray(apoe4, dtype=float), nan=0.)
        cols.append(a4[:, None]); names.append("apoe4_dose")

    # 6. Plasma p-tau181 (direct tau phosphorylation biomarker)
    if ptau181 is not None:
        pt = np.asarray(ptau181, dtype=float)
        pt_median = float(np.nanmedian(pt))
        pt_filled = np.where(np.isfinite(pt), pt, pt_median)
        cols.append(pt_filled[:, None]); names.append("plasma_ptau181")

    # 7. Follow-up time
    cols.append(time_years[:, None]); names.append("follow_up_t")

    # 8. HCP eigenmode projections (slow modes = global spreading modes)
    for k in range(min(5, eigenvectors.shape[1])):
        loading = (bl_s @ eigenvectors[:, k])[:, None]
        cols.append(loading); names.append(f"eigenmode_{k}_loading")

    # 9. Connectivity-weighted tau gradient magnitude (Fickian drive strength)
    neighbour_tau = bl_s @ adj_norm.T
    fickian_mag = np.abs(neighbour_tau - bl_s).mean(axis=1, keepdims=True)
    cols.append(fickian_mag); names.append("fickian_drive_magnitude")

    # 10. Tau asymmetry (L vs R mean difference)
    n_reg = bl_s.shape[1]
    lh_mean = bl_s[:, :n_reg//2].mean(axis=1)
    rh_mean = bl_s[:, n_reg//2:].mean(axis=1)
    cols.append(np.abs(lh_mean - rh_mean)[:, None]); names.append("tau_lr_asymmetry")

    X = np.hstack(cols)
    return X, names


def _fit_amortize_two(
    alpha1_train, alpha2_train, X_all,
    train_indices, pair_groups, feat_names, random_seed,
    alphas_ridge=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    cv_folds=5,
):
    """Multi-target ridge: predict (alpha1, alpha2) from rich feature matrix."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    X_tr = X_all[train_indices]
    Y_tr = np.stack([alpha1_train, alpha2_train], axis=1)   # (n_train, 2)
    groups = np.asarray(pair_groups, dtype=str)[train_indices]
    unique_g = np.unique(groups)
    n_folds  = min(cv_folds, unique_g.size)

    # Standardise
    fm = X_tr.mean(0); fs = X_tr.std(0)
    fs = np.where((fs > 1e-10) & np.isfinite(fs), fs, 1.0)
    X_tr_sc = (X_tr - fm) / fs

    best_alpha_r, best_cv_mse = alphas_ridge[0], float("inf")
    if n_folds >= 2:
        for ar in alphas_ridge:
            fold_mse = []
            for tr_i, va_i in GroupKFold(n_splits=n_folds).split(X_tr_sc, Y_tr, groups):
                m = Ridge(alpha=ar).fit(X_tr_sc[tr_i], Y_tr[tr_i])
                fold_mse.append(float(np.mean((m.predict(X_tr_sc[va_i]) - Y_tr[va_i])**2)))
            mse = float(np.mean(fold_mse))
            if mse < best_cv_mse:
                best_cv_mse, best_alpha_r = mse, ar

    final = Ridge(alpha=best_alpha_r).fit(X_tr_sc, Y_tr)
    # Compute per-target R² manually (sklearn score() expects same n_outputs as fitted)
    Y_pred_tr = final.predict(X_tr_sc)   # (n_train, 2)
    def _r2(y_true, y_pred):
        ss_res = float(np.sum((y_true - y_pred)**2))
        ss_tot = float(np.sum((y_true - y_true.mean())**2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r2_1 = _r2(Y_tr[:, 0], Y_pred_tr[:, 0])
    r2_2 = _r2(Y_tr[:, 1], Y_pred_tr[:, 1])

    X_sc = (X_all - fm) / fs
    preds = final.predict(X_sc)                                # (n_all, 2)
    alpha1_pred = np.clip(preds[:, 0],  0.0, 20.0)
    alpha2_pred = np.clip(preds[:, 1], -5.0,  5.0)

    # Top predictors (by |coefficient| sum across both targets)
    coef_importance = np.abs(final.coef_).sum(axis=0)
    top_idx = np.argsort(coef_importance)[::-1][:5]
    top_all = [feat_names[i] for i in top_idx]
    top1    = [feat_names[i] for i in np.argsort(np.abs(final.coef_[0]))[::-1][:3]]
    top2    = [feat_names[i] for i in np.argsort(np.abs(final.coef_[1]))[::-1][:3]]

    report = {
        "ridge_alpha": best_alpha_r,
        "r2_alpha1":   r2_1,
        "r2_alpha2":   r2_2,
        "top_features_combined": top_all,
        "top_alpha1":  top1,
        "top_alpha2":  top2,
        "n_features":  len(feat_names),
        "feature_names": feat_names,
    }
    return alpha1_pred, alpha2_pred, report


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_alpha_correlations(
    alpha1_train, alpha2_train,
    bl_s, split, dataset, amyloid, apoe4, ptau181,
    braak_idx, eigenvectors,
):
    from scipy import stats as sp_stats

    tr = split.train_indices
    bl_tr = bl_s[tr]
    t_tr  = dataset.time_years[tr]
    print("\n   Correlations with per-subject weights (training pairs):")
    print(f"   {'Feature':<35} {'r(alpha1)':>10} {'r(alpha2)':>10}")
    print(f"   {'-'*57}")

    def row(name, vals):
        r1, p1 = sp_stats.spearmanr(alpha1_train, vals)
        r2, p2 = sp_stats.spearmanr(alpha2_train, vals)
        s1 = '*' if p1 < 0.05 else ''
        s2 = '*' if p2 < 0.05 else ''
        print(f"   {name:<35} {r1:>+8.3f}{s1:<2} {r2:>+8.3f}{s2}")

    for stage, idx in braak_idx.items():
        if idx: row(f"tau_braak_{stage}", bl_tr[:, idx].mean(1))
    row("tau_mean",         bl_tr.mean(1))
    row("follow_up_time",   t_tr)
    if amyloid is not None:
        row("amyloid_mean", np.nanmean(amyloid[tr], axis=1))
    if apoe4 is not None:
        a4 = np.nan_to_num(np.asarray(apoe4, dtype=float)[tr], nan=0.)
        row("apoe4_dose", a4)
    if ptau181 is not None:
        pt = np.asarray(ptau181, dtype=float)[tr]
        pt = np.where(np.isfinite(pt), pt, np.nanmedian(pt))
        row("plasma_ptau181", pt)
    for k in range(3):
        row(f"eigenmode_{k}_loading", bl_tr @ eigenvectors[:, k])
    print("   (* p<0.05)")


def _print_table(rows: list[dict]) -> None:
    by_model: dict[str, dict] = {}
    for r in rows:
        if r.get("split") == "test":
            by_model.setdefault(r["model"], {})[r.get("metric", "")] = r

    def med(m, k):
        v = m.get(k, {})
        return float(v.get("median", float("nan"))) if isinstance(v, dict) else float("nan")

    order = ["symbolic_ode_global", MODEL_NAME, "FKPP+IR", "NDM+IR", "Bio-FKPP"]
    labels = {
        "symbolic_ode_global": "Symbolic ODE (global)",
        MODEL_NAME:            "Symbolic ODE (2-component)  ← new",
        "FKPP+IR":             "FKPP+IR",
        "NDM+IR":              "NDM+IR",
        "Bio-FKPP":            "Bio-FKPP",
    }
    print(f"{'Model':<44} {'subj_ρ':>8} {'delta_ρ':>9} {'MAE':>8}")
    print("-" * 72)
    for m in order:
        if m in by_model:
            print(f"  {labels.get(m,m):<42} "
                  f"{med(by_model[m],'subject_spearman'):>8.4f} "
                  f"{med(by_model[m],'delta_spearman'):>9.4f} "
                  f"{med(by_model[m],'mae'):>8.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
