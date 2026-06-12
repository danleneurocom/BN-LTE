"""Symbolic ODE Discovery — backbone-free, parsimonious, universally generalisable.

Goal: discover the unified spreading equation directly from data, without assuming
any backbone ODE form (no FKPP, no NDM pre-assumed).

Pipeline:

  Step 1  Compute the raw observed rate per (pair, region):
              rate[i,j] = (S1[i,j] - S0[i,j]) / dt[i]
          This is the FULL dynamics, not a residual from any assumed model.

  Step 2  Build MINIMAL universally-available features per (pair, region):
              S0                        — baseline tau level
              S0*(1-S0)                 — logistic (saturation) drive
              fickian_gradient          — Σ_j C_ji (S0_j - S0_i)   [Fickian diffusion]
              fickian_x_state           — fickian_gradient * S0_i  [connectivity term]
              amyloid * S0              — amyloid-tau interaction   [optional]
              thickness * S0            — structural vulnerability  [optional]
          Total: 4 core + 2 optional = 6 features maximum.

          DELIBERATE design choices for generalisability:
          - No AHBA gene expression (not available in most datasets)
          - No plasma biomarkers (not universally available)
          - No APOE4 (not always genotyped)
          - No Laplacian (uses raw Fickian gradient instead — physically cleaner)
          Amyloid and thickness degrade gracefully: if unavailable, set to zero.

  Step 3  Run PySR with HIGH parsimony on (features, rate) — training data only.
          PySR discovers the parsimonious unified equation.
          Target parsimony: 3–5 term expression.

  Step 4  Integrate the discovered ODE dynamically via RK4:
              dS_i/dt = f_sym(S_i, fickian_i(t), amyloid_i, thickness_i)
          The Fickian gradient is RECOMPUTED at each RK4 step from current S(t),
          giving true dynamic coupling — not a static correction.

The discovered expression is a self-contained ODE. No backbone parameters (rho,
alpha) are needed. The equation works on ANY dataset with tau PET + connectivity
+ optional amyloid/thickness.

Example of expected output form (matches the literature equation):
    dS/dt ≈ β₁ · amyloid · S + β₂ · Σ_j C_ji (S_j - S_i) · S
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SymbolicODEFitResult:
    """Result of backbone-free symbolic ODE discovery."""
    symbolic_expression: str        # discovered unified equation (sympy string)
    feature_names: list[str]        # features used
    feature_mean: np.ndarray        # training standardisation mean
    feature_scale: np.ndarray       # training standardisation scale
    residual_train_r2: float        # R² on raw rate (training)
    residual_train_mse: float       # MSE on raw rate (training)
    n_parameters: int               # complexity of discovered expression
    pareto_front: list[dict[str, Any]] = field(default_factory=list)
    _pysr_model: Any = field(default=None, repr=False)


@dataclass
class TwoStageSymbolicODEFitResult:
    """Result of two-stage symbolic ODE:
      Stage 1 — Analytical amyloid-saturation term (3 params, no PySR):
                  f1(tau, amyloid) = beta0 + beta1 * amyloid_x_tau * (C - tau)
                  where tau and amyloid_x_tau are in standardised feature space.
      Stage 2 — PySR on residual rate after f1 is removed (discovers connectivity terms).
    """
    # Stage 1 — analytical amyloid-saturation
    stage1_beta0: float          # constant offset
    stage1_beta1: float          # amyloid coupling strength
    stage1_cap: float            # saturation cap C in standardised space
    stage1_train_r2: float       # R² of f1 on raw rate
    stage1_train_mse: float

    # Stage 2 — PySR on residual
    stage2_expression: str       # discovered expression (sympy string)
    stage2_train_r2: float       # R² of f2 on residual rate
    stage2_train_mse: float
    stage2_n_parameters: int
    stage2_pareto_front: list[dict[str, Any]] = field(default_factory=list)

    # Shared feature standardisation (both stages use the same feature space)
    feature_names: list[str] = field(default_factory=list)
    feature_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_scale: np.ndarray = field(default_factory=lambda: np.array([]))

    _stage2_pysr_model: Any = field(default=None, repr=False)


class SymbolicODEModel:
    """Backbone-free symbolic ODE discovery for tau spreading.

    Discovers a parsimonious unified equation directly from the observed
    rate of change, using only universally-available features.
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        *,
        steps_per_year: int = 12,
    ):
        adj = np.asarray(adjacency, dtype=float)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"Adjacency must be square, got {adj.shape}.")
        self.adjacency = adj
        self.steps_per_year = int(steps_per_year)
        # Row-stochastic adjacency for Fickian gradient computation
        degree = adj.sum(axis=1)
        degree_safe = np.where(degree > 0, degree, 1.0)
        self.adj_norm: np.ndarray = adj / degree_safe[:, None]   # (n, n)
        self.n_regions = adj.shape[0]

    # ── Feature builder ───────────────────────────────────────────────────────

    def build_features(
        self,
        state: np.ndarray,                  # (n_pairs, n_regions) — S at any time
        *,
        amyloid: np.ndarray | None = None,  # (n_pairs, n_regions)
        thickness: np.ndarray | None = None,# (n_pairs, n_regions)
        feature_mean: np.ndarray | None = None,
        feature_scale: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
        """Build (n_pairs * n_regions, n_features) feature matrix.

        Core features (always present, universal):
          1. S                     — current tau level
          2. S*(1-S)               — logistic drive (saturation)
          3. fickian_gradient      — Σ_j C_ji (S_j - S_i)   [Fickian diffusion]
          4. fickian_x_state       — fickian_gradient * S_i  [connectivity × state]

        Optional features (set to zero if unavailable):
          5. amyloid * S           — amyloid-tau interaction
          6. thickness * S         — structural vulnerability × tau
        """
        S = np.asarray(state, dtype=float)
        n_pairs, n_reg = S.shape

        # Core feature 3: Fickian gradient Σ_j C_ji (S_j - S_i)
        # adj_norm is row-stochastic: (adj_norm @ S.T).T[i] = Σ_j C_ji * S_j / Σ_j C_ji
        # fickian = mean_neighbour_tau - S  (concentration gradient)
        mean_neighbour = S @ self.adj_norm.T    # (n_pairs, n_regions)
        fickian = mean_neighbour - S            # (n_pairs, n_regions)  Σ_j C_ji(S_j-S_i)

        feat_cols = [
            S,                      # 1: tau level
            S * (1.0 - S),          # 2: logistic drive
            fickian,                # 3: Fickian gradient
            fickian * S,            # 4: Fickian × state (the connectivity spreading term)
        ]
        # Use tau-prefixed names to avoid SymPy reserved symbols (S, E, N, I, etc.)
        feat_names = ["tau", "tau_logistic", "fickian", "fickian_x_tau"]

        # Optional: amyloid × tau
        if amyloid is not None:
            amy = _fill_nan(np.asarray(amyloid, dtype=float), n_pairs, n_reg)
            feat_cols.append(amy * S)
            feat_names.append("amyloid_x_tau")

        # Optional: thickness × tau
        if thickness is not None:
            thick = _fill_nan(np.asarray(thickness, dtype=float), n_pairs, n_reg)
            feat_cols.append(thick * S)
            feat_names.append("thickness_x_tau")

        X = np.stack(feat_cols, axis=-1).reshape(-1, len(feat_names))  # (N, F)

        if feature_mean is not None and feature_scale is not None:
            X = (X - feature_mean[None, :]) / feature_scale[None, :]
            return X, feat_names, feature_mean, feature_scale
        else:
            fm = np.mean(X, axis=0)
            fs = np.std(X, axis=0)
            fs = np.where((fs > 1e-10) & np.isfinite(fs), fs, 1.0)
            X = (X - fm[None, :]) / fs[None, :]
            return X, feat_names, fm, fs

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        baseline_scaled: np.ndarray,
        observed_scaled: np.ndarray,
        time_years: np.ndarray,
        *,
        train_indices: np.ndarray,
        amyloid: np.ndarray | None = None,
        thickness: np.ndarray | None = None,
        pysr_niterations: int = 300,
        pysr_populations: int = 20,
        pysr_population_size: int = 33,
        pysr_maxsize: int = 15,
        pysr_parsimony: float = 0.01,        # high parsimony → simple expression
        pysr_binary_operators: list[str] | None = None,
        pysr_unary_operators: list[str] | None = None,
        pysr_batching: bool = True,
        pysr_batch_size: int = 2048,
        pysr_timeout_seconds: int = 300,
        pysr_model_selection: str = "best",
        max_train_rows: int = 40000,
        random_seed: int = 20260507,
    ) -> SymbolicODEFitResult:
        """Discover the unified spreading ODE from the raw observed rate.

        Target: (S1 - S0) / dt   (raw, no backbone subtracted)
        Features: S0, S0*(1-S0), fickian_gradient, fickian_x_state,
                  amyloid*S0, thickness*S0  (last two optional)

        High parsimony enforces a simple 3–5 term equation.
        """
        from pysr import PySRRegressor

        bl  = np.asarray(baseline_scaled, dtype=float)
        obs = np.asarray(observed_scaled, dtype=float)
        t   = np.asarray(time_years, dtype=float)
        tr  = np.asarray(train_indices, dtype=int)

        # ── Step 1: raw observed rate ─────────────────────────────────────────
        safe_t = np.maximum(t, 1e-6)
        raw_rate = (obs - bl) / safe_t[:, None]     # (n_pairs, n_regions)
        print("  [1/3] Target: raw observed rate (S1-S0)/dt — no backbone assumed")
        print(f"        Rate range: [{raw_rate.min():.4f}, {raw_rate.max():.4f}] /yr")

        # ── Step 2: build features from baseline ─────────────────────────────
        print("  [2/3] Building universal features (Fickian gradient, biology)...")
        X_flat, feat_names, fm, fs = self.build_features(
            bl, amyloid=amyloid, thickness=thickness
        )
        y_flat = raw_rate.reshape(-1)
        n_reg  = bl.shape[1]

        # Training mask
        flat_mask = np.zeros(len(y_flat), dtype=bool)
        for i in tr:
            flat_mask[i * n_reg:(i + 1) * n_reg] = True
        finite_mask = flat_mask & np.isfinite(y_flat) & np.all(np.isfinite(X_flat), axis=1)

        X_tr = X_flat[finite_mask]
        y_tr = y_flat[finite_mask]
        print(f"        Features: {feat_names}")
        print(f"        Training rows: {X_tr.shape[0]}  (before subsample)")

        if max_train_rows > 0 and X_tr.shape[0] > max_train_rows:
            rng  = np.random.default_rng(random_seed)
            sel  = rng.choice(X_tr.shape[0], size=max_train_rows, replace=False)
            X_tr, y_tr = X_tr[sel], y_tr[sel]
            print(f"        Subsampled to {X_tr.shape[0]} rows.")

        # ── Step 3: PySR ──────────────────────────────────────────────────────
        print(f"  [3/3] PySR — backbone-free equation discovery "
              f"(parsimony={pysr_parsimony}, maxsize={pysr_maxsize})...")
        bin_ops = pysr_binary_operators or ["+", "-", "*", "/"]
        un_ops  = pysr_unary_operators  or ["square"]

        model = PySRRegressor(
            niterations=pysr_niterations,
            binary_operators=bin_ops,
            unary_operators=un_ops,
            maxsize=pysr_maxsize,
            parsimony=pysr_parsimony,
            populations=pysr_populations,
            population_size=pysr_population_size,
            batching=pysr_batching,
            batch_size=pysr_batch_size,
            timeout_in_seconds=pysr_timeout_seconds,
            model_selection=pysr_model_selection,
            random_state=random_seed,
            verbosity=0,
            progress=True,
        )
        model.fit(X_tr, y_tr, variable_names=feat_names)

        best_expr = _safe_expression(model)
        y_pred    = model.predict(X_tr)
        ss_res = float(np.sum((y_tr - y_pred) ** 2))
        ss_tot = float(np.sum((y_tr - float(np.mean(y_tr))) ** 2))
        r2  = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mse = float(np.mean((y_tr - y_pred) ** 2))

        # Complexity of best expression (number of nodes)
        complexity = 1
        try:
            best_row = model.get_best()
            complexity = int(best_row.get("complexity", 1))
        except Exception:
            pass

        print(f"        Discovered: {best_expr}")
        print(f"        R²={r2:.4f}  MSE={mse:.6f}  complexity={complexity}")

        return SymbolicODEFitResult(
            symbolic_expression=best_expr,
            feature_names=feat_names,
            feature_mean=fm,
            feature_scale=fs,
            residual_train_r2=r2,
            residual_train_mse=mse,
            n_parameters=complexity,
            pareto_front=_pareto_rows(model),
            _pysr_model=model,
        )

    # ── Predict (dynamic RK4 integration) ────────────────────────────────────

    def predict(
        self,
        baseline_scaled: np.ndarray,
        time_years: np.ndarray,
        fit: SymbolicODEFitResult,
        *,
        amyloid: np.ndarray | None = None,
        thickness: np.ndarray | None = None,
    ) -> np.ndarray:
        """Integrate the discovered ODE dynamically.

        dS_i/dt = f_sym(S_i(t), fickian_i(t), amyloid_i, thickness_i)

        The Fickian gradient is RECOMPUTED at each RK4 step from the current
        state S(t), giving proper dynamic network coupling.
        Static features (amyloid, thickness) are fixed at baseline values.
        """
        if fit._pysr_model is None:
            raise RuntimeError(
                "No PySR model stored in fit result.\n"
                "This happens if: (a) fit() was not called, or (b) the PySR session "
                "was lost (e.g. kernel restart). Re-run the fit() cell."
            )

        bl = np.asarray(baseline_scaled, dtype=float)
        t  = np.asarray(time_years, dtype=float)
        n_pairs, n_reg = bl.shape

        # Precompute static biology arrays (fixed throughout integration)
        amy   = _fill_nan(np.asarray(amyloid,   dtype=float), n_pairs, n_reg) \
                if amyloid   is not None else None
        thick = _fill_nan(np.asarray(thickness, dtype=float), n_pairs, n_reg) \
                if thickness is not None else None

        # Maximum plausible rate in scaled units (±1 SUVR/yr is already extreme)
        _rate_clip = 1.0

        def rate_fn(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
            """Compute dS/dt for active pairs using current S."""
            amy_sub   = amy[idx]   if amy   is not None else None
            thick_sub = thick[idx] if thick is not None else None
            X, _, _, _ = self.build_features(
                S, amyloid=amy_sub, thickness=thick_sub,
                feature_mean=fit.feature_mean, feature_scale=fit.feature_scale,
            )
            rate_flat = fit._pysr_model.predict(X)          # (n_active * n_reg,)
            # Guard: NaN/inf from expressions evaluated outside training range
            rate_flat = np.nan_to_num(rate_flat, nan=0.0, posinf=0.0, neginf=0.0)
            rate_flat = np.clip(rate_flat, -_rate_clip, _rate_clip)
            return rate_flat.reshape(S.shape)

        states    = np.clip(bl, 0.0, 1.0).copy()
        remaining = t.copy()
        step_dt   = 1.0 / self.steps_per_year

        while np.any(remaining > 0.0):
            active = remaining > 0.0
            dt     = np.minimum(step_dt, remaining[active])[:, None]
            s      = states[active]

            k1 = rate_fn(s,                                      np.where(active)[0])
            k2 = rate_fn(np.clip(s + 0.5 * dt * k1, 0.0, 1.0), np.where(active)[0])
            k3 = rate_fn(np.clip(s + 0.5 * dt * k2, 0.0, 1.0), np.where(active)[0])
            k4 = rate_fn(np.clip(s + dt       * k3, 0.0, 1.0), np.where(active)[0])

            states[active] = np.clip(
                s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0
            )
            remaining[active] -= dt[:, 0]

        return states

    # ── Two-stage fit (Improvements 2 + 4 combined) ──────────────────────────

    def fit_two_stage(
        self,
        baseline_scaled: np.ndarray,
        observed_scaled: np.ndarray,
        time_years: np.ndarray,
        *,
        train_indices: np.ndarray,
        amyloid: np.ndarray | None = None,
        thickness: np.ndarray | None = None,
        # Stage 2 PySR settings — lower parsimony so connectivity terms emerge
        pysr_niterations: int = 300,
        pysr_populations: int = 20,
        pysr_population_size: int = 33,
        pysr_maxsize: int = 15,
        pysr_parsimony: float = 0.003,
        pysr_binary_operators: list[str] | None = None,
        pysr_unary_operators: list[str] | None = None,
        pysr_batching: bool = True,
        pysr_batch_size: int = 2048,
        pysr_timeout_seconds: int = 300,
        pysr_model_selection: str = "best",
        max_train_rows: int = 40000,
        random_seed: int = 20260507,
    ) -> TwoStageSymbolicODEFitResult:
        """Two-stage symbolic ODE discovery.

        Stage 1: Fit analytical amyloid-saturation form on training data.
            f1 = beta0 + beta1 * amyloid_x_tau_std * (C - tau_std)
            Uses standardised feature space (same as PySR features).
            Fits 3 parameters in seconds via L-BFGS-B.

        Stage 2: Run PySR on residual rate = observed_rate - f1.
            With the dominant amyloid term removed, PySR is free to discover
            the Fickian connectivity term and other corrections.
            Lower parsimony (0.003) than Stage 1 discovery (0.01) so additional
            terms are not suppressed.
        """
        from scipy.optimize import minimize
        from pysr import PySRRegressor

        bl  = np.asarray(baseline_scaled, dtype=float)
        obs = np.asarray(observed_scaled, dtype=float)
        t   = np.asarray(time_years, dtype=float)
        tr  = np.asarray(train_indices, dtype=int)

        # ── Shared feature standardisation ────────────────────────────────────
        print("  [Stage 1/2] Building standardised feature space...")
        X_flat, feat_names, fm, fs = self.build_features(
            bl, amyloid=amyloid, thickness=thickness
        )
        n_reg   = bl.shape[1]
        y_flat  = ((obs - bl) / np.maximum(t, 1e-6)[:, None]).reshape(-1)

        # Training mask (pairs in tr)
        flat_mask = np.zeros(len(y_flat), dtype=bool)
        for i in tr:
            flat_mask[i * n_reg:(i + 1) * n_reg] = True
        finite_mask = flat_mask & np.isfinite(y_flat) & np.all(np.isfinite(X_flat), axis=1)
        X_tr, y_tr = X_flat[finite_mask], y_flat[finite_mask]
        print(f"     Training rows: {X_tr.shape[0]}  features: {feat_names}")

        # Feature column indices (must match build_features order)
        # 0=tau, 1=tau_logistic, 2=fickian, 3=fickian_x_tau, 4=amyloid_x_tau, 5=thickness_x_tau
        TAU_IDX, AMY_TAU_IDX = 0, 4

        # ── Stage 1: Analytical amyloid-saturation fit ────────────────────────
        print("  [Stage 1/2] Fitting analytical f1 = β₀ + β₁·amyloid_x_tau·(C − tau)...")
        tau_tr     = X_tr[:, TAU_IDX]
        amy_tau_tr = X_tr[:, AMY_TAU_IDX]

        def f1_eval(params: np.ndarray, tau: np.ndarray, amt: np.ndarray) -> np.ndarray:
            b0, b1, C = float(params[0]), float(params[1]), float(params[2])
            return b0 + b1 * amt * (C - tau)

        def stage1_loss(params: np.ndarray) -> float:
            pred = f1_eval(params, tau_tr, amy_tau_tr)
            return float(np.mean((pred - y_tr) ** 2))

        # Warm-start: use values from prior PySR runs (beta0≈0.003, beta1≈0.002, C≈3.9)
        x0     = np.array([0.003, 0.002, 3.9])
        bounds = [(-0.05, 0.05), (-0.1, 0.1), (1.5, 8.0)]
        s1_result = minimize(stage1_loss, x0, method="L-BFGS-B", bounds=bounds,
                             options={"maxiter": 200})
        b0, b1, C = float(s1_result.x[0]), float(s1_result.x[1]), float(s1_result.x[2])

        # Stage 1 diagnostics
        f1_tr  = f1_eval(s1_result.x, tau_tr, amy_tau_tr)
        ss_res = float(np.sum((y_tr - f1_tr) ** 2))
        ss_tot = float(np.sum((y_tr - y_tr.mean()) ** 2))
        s1_r2  = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        s1_mse = float(np.mean((y_tr - f1_tr) ** 2))
        print(f"     f1: β₀={b0:.5f}  β₁={b1:.5f}  C={C:.3f}  "
              f"R²={s1_r2:.4f}  MSE={s1_mse:.6f}")

        # ── Stage 2: PySR on residual rate ────────────────────────────────────
        print("  [Stage 2/2] PySR on residual rate (amyloid term removed)...")
        f1_all     = f1_eval(s1_result.x, X_flat[:, TAU_IDX], X_flat[:, AMY_TAU_IDX])
        resid_flat = y_flat - f1_all
        resid_tr   = resid_flat[finite_mask]

        print(f"     Residual range: [{resid_tr.min():.4f}, {resid_tr.max():.4f}]  "
              f"std={resid_tr.std():.4f}")

        # Subsample for speed
        if max_train_rows > 0 and X_tr.shape[0] > max_train_rows:
            rng = np.random.default_rng(random_seed)
            sel = rng.choice(X_tr.shape[0], size=max_train_rows, replace=False)
            X_s2, y_s2 = X_tr[sel], resid_tr[sel]
            print(f"     Subsampled to {len(y_s2)} rows for PySR.")
        else:
            X_s2, y_s2 = X_tr, resid_tr

        bin_ops = pysr_binary_operators or ["+", "-", "*", "/"]
        un_ops  = pysr_unary_operators  or ["square"]

        s2_model = PySRRegressor(
            niterations=pysr_niterations,
            binary_operators=bin_ops,
            unary_operators=un_ops,
            maxsize=pysr_maxsize,
            parsimony=pysr_parsimony,    # lower → connectivity term more likely to emerge
            populations=pysr_populations,
            population_size=pysr_population_size,
            batching=pysr_batching,
            batch_size=pysr_batch_size,
            timeout_in_seconds=pysr_timeout_seconds,
            model_selection=pysr_model_selection,
            random_state=random_seed,
            verbosity=0,
            progress=True,
        )
        s2_model.fit(X_s2, y_s2, variable_names=feat_names)

        s2_expr = _safe_expression(s2_model)
        y_s2_pred = s2_model.predict(X_s2)
        y_s2_pred = np.nan_to_num(y_s2_pred, nan=0.0)
        s2_ss_res = float(np.sum((y_s2 - y_s2_pred) ** 2))
        s2_ss_tot = float(np.sum((y_s2 - y_s2.mean()) ** 2))
        s2_r2  = 1.0 - s2_ss_res / s2_ss_tot if s2_ss_tot > 0 else float("nan")
        s2_mse = float(np.mean((y_s2 - y_s2_pred) ** 2))

        # Complexity
        s2_complexity = 1
        try:
            s2_complexity = int(s2_model.get_best().get("complexity", 1))
        except Exception:
            pass

        print(f"     f2: '{s2_expr}'  R²={s2_r2:.4f}  MSE={s2_mse:.6f}")
        print(f"     Fickian term discovered: "
              f"{'YES' if 'fickian' in s2_expr else 'NO — check Pareto front'}")

        return TwoStageSymbolicODEFitResult(
            stage1_beta0=b0, stage1_beta1=b1, stage1_cap=C,
            stage1_train_r2=s1_r2, stage1_train_mse=s1_mse,
            stage2_expression=s2_expr,
            stage2_train_r2=s2_r2, stage2_train_mse=s2_mse,
            stage2_n_parameters=s2_complexity,
            stage2_pareto_front=_pareto_rows(s2_model),
            feature_names=feat_names,
            feature_mean=fm,
            feature_scale=fs,
            _stage2_pysr_model=s2_model,
        )

    def predict_two_stage(
        self,
        baseline_scaled: np.ndarray,
        time_years: np.ndarray,
        fit: TwoStageSymbolicODEFitResult,
        alpha1_all: np.ndarray,           # (n_pairs,) amyloid-saturation weight
        alpha2_all: np.ndarray,           # (n_pairs,) residual (autonomous) weight
        *,
        amyloid: np.ndarray | None = None,
        thickness: np.ndarray | None = None,
    ) -> np.ndarray:
        """Integrate dS/dt = alpha1_i*f1 + alpha2_i*f2 via RK4.

        f1 = beta0 + beta1*amyloid_x_tau*(C − tau)   [analytical, Stage 1]
        f2 = PySR-discovered expression on residual   [Stage 2]
        """
        if fit._stage2_pysr_model is None:
            raise RuntimeError("No Stage-2 PySR model. Re-run fit_two_stage().")

        bl  = np.asarray(baseline_scaled, dtype=float)
        t   = np.asarray(time_years, dtype=float)
        n_pairs, n_reg = bl.shape

        amy   = _fill_nan(np.asarray(amyloid,   dtype=float), n_pairs, n_reg) \
                if amyloid   is not None else None
        thick = _fill_nan(np.asarray(thickness, dtype=float), n_pairs, n_reg) \
                if thickness is not None else None
        _rate_clip = 1.0
        b0, b1, C_cap = fit.stage1_beta0, fit.stage1_beta1, fit.stage1_cap

        def rate_fn(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
            a1 = alpha1_all[idx][:, None]
            a2 = alpha2_all[idx][:, None]
            amy_sub   = amy[idx]   if amy   is not None else None
            thick_sub = thick[idx] if thick is not None else None

            X, _, _, _ = self.build_features(
                S, amyloid=amy_sub, thickness=thick_sub,
                feature_mean=fit.feature_mean, feature_scale=fit.feature_scale,
            )
            tau_std     = X[:, 0].reshape(S.shape)
            amy_tau_std = X[:, 4].reshape(S.shape)

            # Stage 1: analytical amyloid-saturation
            f1 = b0 + b1 * amy_tau_std * (C_cap - tau_std)
            # Stage 2: PySR autonomous seeding
            f2 = np.nan_to_num(
                fit._stage2_pysr_model.predict(X), nan=0.0, posinf=0.0, neginf=0.0
            ).reshape(S.shape)

            rate = a1 * f1 + a2 * f2
            return np.clip(np.nan_to_num(rate, nan=0.0), -_rate_clip, _rate_clip)

        states    = np.clip(bl, 0.0, 1.0).copy()
        remaining = t.copy()
        step_dt   = 1.0 / self.steps_per_year

        while np.any(remaining > 0.0):
            active = remaining > 0.0
            dt     = np.minimum(step_dt, remaining[active])[:, None]
            s      = states[active]
            k1 = rate_fn(s,                                   np.where(active)[0])
            k2 = rate_fn(np.clip(s + 0.5*dt*k1, 0., 1.),    np.where(active)[0])
            k3 = rate_fn(np.clip(s + 0.5*dt*k2, 0., 1.),    np.where(active)[0])
            k4 = rate_fn(np.clip(s + dt*k3,      0., 1.),    np.where(active)[0])
            states[active] = np.clip(
                s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4), 0., 1.
            )
            remaining[active] -= dt[:, 0]
        return states

    # ── Per-subject α_i fitting ───────────────────────────────────────────────

    def predict_with_alpha(
        self,
        baseline_scaled: np.ndarray,    # (1, n_regions) single subject
        time_years: np.ndarray,          # (1,)
        fit: SymbolicODEFitResult,
        alpha: float,
        *,
        amyloid: np.ndarray | None = None,
        thickness: np.ndarray | None = None,
    ) -> np.ndarray:
        """Integrate ODE for one subject with a scaled rate: dS/dt = alpha * f_sym.

        alpha > 1  → faster spreading than population average
        alpha < 1  → slower spreading
        alpha = 1  → same as predict()
        """
        bl = np.asarray(baseline_scaled, dtype=float)
        t  = np.asarray(time_years, dtype=float)
        n_pairs, n_reg = bl.shape

        amy   = _fill_nan(np.asarray(amyloid,   dtype=float), n_pairs, n_reg) \
                if amyloid   is not None else None
        thick = _fill_nan(np.asarray(thickness, dtype=float), n_pairs, n_reg) \
                if thickness is not None else None
        _rate_clip = 1.0
        _alpha = float(alpha)

        def rate_fn(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
            amy_sub   = amy[idx]   if amy   is not None else None
            thick_sub = thick[idx] if thick is not None else None
            X, _, _, _ = self.build_features(
                S, amyloid=amy_sub, thickness=thick_sub,
                feature_mean=fit.feature_mean, feature_scale=fit.feature_scale,
            )
            rate_flat = _alpha * fit._pysr_model.predict(X)
            rate_flat = np.nan_to_num(rate_flat, nan=0.0, posinf=0.0, neginf=0.0)
            rate_flat = np.clip(rate_flat, -_rate_clip, _rate_clip)
            return rate_flat.reshape(S.shape)

        states    = np.clip(bl, 0.0, 1.0).copy()
        remaining = t.copy()
        step_dt   = 1.0 / self.steps_per_year

        while np.any(remaining > 0.0):
            active = remaining > 0.0
            dt     = np.minimum(step_dt, remaining[active])[:, None]
            s      = states[active]
            k1 = rate_fn(s,                                      np.where(active)[0])
            k2 = rate_fn(np.clip(s + 0.5 * dt * k1, 0.0, 1.0), np.where(active)[0])
            k3 = rate_fn(np.clip(s + 0.5 * dt * k2, 0.0, 1.0), np.where(active)[0])
            k4 = rate_fn(np.clip(s + dt       * k3, 0.0, 1.0), np.where(active)[0])
            states[active] = np.clip(
                s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0
            )
            remaining[active] -= dt[:, 0]
        return states

    def fit_per_subject_alpha(
        self,
        baseline_scaled: np.ndarray,
        observed_scaled: np.ndarray,
        time_years: np.ndarray,
        fit: SymbolicODEFitResult,
        *,
        indices: np.ndarray,
        amyloid: np.ndarray | None = None,
        thickness: np.ndarray | None = None,
        alpha_bounds: tuple[float, float] = (0.0, 20.0),
    ) -> np.ndarray:
        """Fit one scalar alpha_i per subject (pair) to minimise MSE.

        Returns alpha_values: (len(indices),) array of per-subject rates.
        """
        from scipy.optimize import minimize_scalar

        bl  = np.asarray(baseline_scaled, dtype=float)
        obs = np.asarray(observed_scaled, dtype=float)
        t   = np.asarray(time_years, dtype=float)
        alpha_values = np.ones(len(indices), dtype=float)

        for k, i in enumerate(indices):
            amy_i   = amyloid[i:i+1]   if amyloid   is not None else None
            thick_i = thickness[i:i+1] if thickness is not None else None

            def mse(alpha: float) -> float:
                pred = self.predict_with_alpha(
                    bl[i:i+1], t[i:i+1], fit, alpha,
                    amyloid=amy_i, thickness=thick_i,
                )
                return float(np.mean((pred - obs[i:i+1]) ** 2))

            result = minimize_scalar(mse, bounds=alpha_bounds, method="bounded")
            alpha_values[k] = float(result.x)

        return alpha_values

    def fit_per_subject_two_component(
        self,
        baseline_scaled: np.ndarray,
        observed_scaled: np.ndarray,
        time_years: np.ndarray,
        fit: "SymbolicODEFitResult | TwoStageSymbolicODEFitResult",
        *,
        indices: np.ndarray,
        amyloid: np.ndarray | None = None,
        thickness: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit per-subject (alpha1_i, alpha2_i) via closed-form OLS per subject.

        Two-component model:
            dS/dt = alpha1_i * f_sym(tau, amyloid)          [amyloid-growth]
                  + alpha2_i * fickian_gradient(tau) * tau   [Fickian connectivity]

        Fitting is closed-form OLS: no iterative optimisation needed.
        Returns alpha1_values (n_indices,) and alpha2_values (n_indices,).
        """
        bl  = np.asarray(baseline_scaled, dtype=float)
        obs = np.asarray(observed_scaled, dtype=float)
        t   = np.asarray(time_years, dtype=float)
        n_pairs, n_reg = bl.shape

        # Keep None as None — build_features uses None to decide which columns to include
        # so it must match the feature space used during fit()
        amy   = _fill_nan(np.asarray(amyloid,   dtype=float), n_pairs, n_reg) \
                if amyloid   is not None else None
        thick = _fill_nan(np.asarray(thickness, dtype=float), n_pairs, n_reg) \
                if thickness is not None else None
        _has_amy   = amyloid   is not None
        _has_thick = thickness is not None

        alpha1_vals = np.zeros(len(indices))
        alpha2_vals = np.zeros(len(indices))

        # Determine fit type once
        is_two_stage = isinstance(fit, TwoStageSymbolicODEFitResult)

        for k, i in enumerate(indices):
            rate_i = (obs[i] - bl[i]) / max(t[i], 1e-6)    # (n_reg,) observed rate

            X_i, _, _, _ = self.build_features(
                bl[i:i+1],
                amyloid=amy[i:i+1]   if _has_amy   else None,
                thickness=thick[i:i+1] if _has_thick else None,
                feature_mean=fit.feature_mean, feature_scale=fit.feature_scale,
            )

            if is_two_stage:
                # f1: analytical amyloid-saturation (Stage 1)
                tau_std     = X_i[:, 0]                       # (n_reg,)
                amy_tau_std = X_i[:, 4]
                f1_i = (fit.stage1_beta0
                        + fit.stage1_beta1 * amy_tau_std * (fit.stage1_cap - tau_std))
                f1_i = np.nan_to_num(f1_i, nan=0.0)
                # f2: PySR residual term (Stage 2)
                f2_i = fit._stage2_pysr_model.predict(X_i).ravel()
                f2_i = np.nan_to_num(f2_i, nan=0.0)
            else:
                # f1: PySR full-rate equation
                f1_i = fit._pysr_model.predict(X_i).ravel()
                f1_i = np.nan_to_num(f1_i, nan=0.0)
                # f2: Fickian gradient × tau  (hardcoded connectivity component)
                neighbour_tau = bl[i] @ self.adj_norm.T
                f2_i = (neighbour_tau - bl[i]) * bl[i]

            # OLS: [alpha1, alpha2] = pinv([f1 | f2]) @ rate
            F = np.stack([f1_i, f2_i], axis=1)      # (n_reg, 2)
            FtF = F.T @ F + 1e-6 * np.eye(2)
            Ftr = F.T @ rate_i
            try:
                coeffs = np.linalg.solve(FtF, Ftr)
            except np.linalg.LinAlgError:
                coeffs = np.array([1.0, 0.0])
            alpha1_vals[k] = float(np.clip(coeffs[0],  0.0, 20.0))
            alpha2_vals[k] = float(np.clip(coeffs[1], -5.0,  5.0))

        return alpha1_vals, alpha2_vals

    def predict_two_component(
        self,
        baseline_scaled: np.ndarray,
        time_years: np.ndarray,
        fit: SymbolicODEFitResult,
        alpha1_all: np.ndarray,          # (n_pairs,) amyloid-growth weight
        alpha2_all: np.ndarray,          # (n_pairs,) Fickian weight
        *,
        amyloid: np.ndarray | None = None,
        thickness: np.ndarray | None = None,
    ) -> np.ndarray:
        """Integrate dS/dt = alpha1_i*f_sym + alpha2_i*fickian*tau via RK4."""
        if fit._pysr_model is None:
            raise RuntimeError("No PySR model stored. Re-run fit().")

        bl = np.asarray(baseline_scaled, dtype=float)
        t  = np.asarray(time_years, dtype=float)
        n_pairs, n_reg = bl.shape

        amy   = _fill_nan(np.asarray(amyloid,   dtype=float), n_pairs, n_reg) \
                if amyloid   is not None else None
        thick = _fill_nan(np.asarray(thickness, dtype=float), n_pairs, n_reg) \
                if thickness is not None else None
        _rate_clip = 1.0

        def rate_fn(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
            a1 = alpha1_all[idx][:, None]   # (n_active, 1)
            a2 = alpha2_all[idx][:, None]
            amy_sub   = amy[idx]   if amy   is not None else None
            thick_sub = thick[idx] if thick is not None else None
            X, _, _, _ = self.build_features(
                S, amyloid=amy_sub, thickness=thick_sub,
                feature_mean=fit.feature_mean, feature_scale=fit.feature_scale,
            )
            f1 = fit._pysr_model.predict(X)
            f1 = np.nan_to_num(f1, nan=0.0).reshape(S.shape)

            # Fickian: recomputed at current S (dynamic coupling)
            neighbour = S @ self.adj_norm.T
            f2 = (neighbour - S) * S

            rate = a1 * f1 + a2 * f2
            return np.clip(
                np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0),
                -_rate_clip, _rate_clip,
            )

        states    = np.clip(bl, 0.0, 1.0).copy()
        remaining = t.copy()
        step_dt   = 1.0 / self.steps_per_year

        while np.any(remaining > 0.0):
            active = remaining > 0.0
            dt     = np.minimum(step_dt, remaining[active])[:, None]
            s      = states[active]
            k1 = rate_fn(s,                                      np.where(active)[0])
            k2 = rate_fn(np.clip(s + 0.5*dt*k1, 0.0, 1.0),    np.where(active)[0])
            k3 = rate_fn(np.clip(s + 0.5*dt*k2, 0.0, 1.0),    np.where(active)[0])
            k4 = rate_fn(np.clip(s + dt*k3,      0.0, 1.0),    np.where(active)[0])
            states[active] = np.clip(
                s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4), 0.0, 1.0
            )
            remaining[active] -= dt[:, 0]
        return states

    def amortize_alpha(
        self,
        alpha_values: np.ndarray,           # (n_train,) fitted per-subject α
        baseline_scaled: np.ndarray,        # (n_all, n_regions)
        train_indices: np.ndarray,
        *,
        amyloid: np.ndarray | None = None,
        thickness: np.ndarray | None = None,
        apoe4_dose: np.ndarray | None = None,
        time_years: np.ndarray,
        alphas_ridge: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
        cv_folds: int = 5,
        pair_groups: np.ndarray | None = None,
        random_seed: int = 20260507,
    ) -> tuple[np.ndarray, dict]:
        """Predict α_i for ALL pairs from summary features via ridge regression.

        Features used (universally available):
          - mean baseline tau (tau burden)
          - std baseline tau (tau heterogeneity)
          - mean amyloid (if available)
          - APOE4 dose (if available)
          - follow-up time (dt)

        Returns:
          alpha_pred: (n_all,) predicted alpha for every pair
          report: fit diagnostics
        """
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold

        bl = np.asarray(baseline_scaled, dtype=float)
        t  = np.asarray(time_years, dtype=float)
        n  = bl.shape[0]

        # Build summary features per pair
        feat_cols, feat_names = [], []
        feat_cols.append(bl.mean(axis=1, keepdims=True));       feat_names.append("tau_mean")
        feat_cols.append(bl.std(axis=1, keepdims=True));        feat_names.append("tau_std")
        feat_cols.append(bl.max(axis=1, keepdims=True));        feat_names.append("tau_max")
        feat_cols.append(t[:, None]);                            feat_names.append("follow_up_t")

        if amyloid is not None:
            amy = np.asarray(amyloid, dtype=float)
            feat_cols.append(np.nanmean(amy, axis=1, keepdims=True)); feat_names.append("amyloid_mean")
            feat_cols.append(np.nanmax(amy,  axis=1, keepdims=True)); feat_names.append("amyloid_max")

        if apoe4_dose is not None:
            a4 = np.asarray(apoe4_dose, dtype=float)[:, None]
            a4 = np.nan_to_num(a4, nan=0.0)
            feat_cols.append(a4); feat_names.append("apoe4_dose")

        X = np.hstack(feat_cols)              # (n_all, n_features)
        y = alpha_values                       # (n_train,) — only training subjects

        X_tr = X[train_indices]
        y_tr = y                               # already indexed

        # Standardise
        feat_mean  = X_tr.mean(0)
        feat_scale = X_tr.std(0)
        feat_scale = np.where((feat_scale > 1e-10) & np.isfinite(feat_scale), feat_scale, 1.0)
        X_tr_sc = (X_tr - feat_mean) / feat_scale

        # Group-CV ridge
        groups = (np.asarray(pair_groups, dtype=str)[train_indices]
                  if pair_groups is not None
                  else np.arange(len(train_indices)).astype(str))
        unique_groups = np.unique(groups)
        n_folds = min(cv_folds, unique_groups.size)
        best_alpha_r, best_cv_mse = alphas_ridge[0], float("inf")
        if n_folds >= 2:
            splitter = GroupKFold(n_splits=n_folds)
            for ar in alphas_ridge:
                fold_mse = []
                for tr_i, va_i in splitter.split(X_tr_sc, y_tr, groups):
                    m = Ridge(alpha=ar).fit(X_tr_sc[tr_i], y_tr[tr_i])
                    fold_mse.append(float(np.mean((m.predict(X_tr_sc[va_i]) - y_tr[va_i])**2)))
                mse = float(np.mean(fold_mse))
                if mse < best_cv_mse:
                    best_cv_mse, best_alpha_r = mse, ar

        final = Ridge(alpha=best_alpha_r).fit(X_tr_sc, y_tr)
        train_r2 = float(final.score(X_tr_sc, y_tr))

        # Predict for ALL pairs
        X_sc   = (X - feat_mean) / feat_scale
        alpha_pred = np.clip(final.predict(X_sc), 0.0, 20.0)

        report = {
            "features": feat_names,
            "ridge_alpha": best_alpha_r,
            "train_r2_alpha": train_r2,
            "alpha_train_mean": float(y_tr.mean()),
            "alpha_train_std":  float(y_tr.std()),
            "alpha_pred_mean":  float(alpha_pred[train_indices].mean()),
            "alpha_pred_std":   float(alpha_pred[train_indices].std()),
        }
        return alpha_pred, report


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fill_nan(arr: np.ndarray, n_pairs: int, n_reg: int) -> np.ndarray:
    if arr.shape != (n_pairs, n_reg):
        raise ValueError(f"Expected ({n_pairs},{n_reg}), got {arr.shape}.")
    result = arr.copy()
    col_med = np.nanmedian(result, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    nan_mask = ~np.isfinite(result)
    result[nan_mask] = np.broadcast_to(col_med, result.shape)[nan_mask]
    return result


def _safe_expression(model: Any) -> str:
    try:
        eq = model.sympy()
        return str(eq) if eq is not None else "<no expression>"
    except Exception:
        try:
            return str(model.get_best()["equation"])
        except Exception:
            return "<expression unavailable>"


def _pareto_rows(model: Any) -> list[dict]:
    rows = []
    try:
        for _, row in model.equations_.iterrows():
            rows.append({
                "equation":   str(row.get("equation",  "")),
                "loss":       float(row.get("loss",    float("nan"))),
                "complexity": int(row.get("complexity", 0)),
                "score":      float(row.get("score",   float("nan"))),
            })
    except Exception:
        pass
    return rows


# ── Shared amortization utilities (used by run scripts) ──────────────────────

def build_amortization_features(
    bl_s: np.ndarray,
    time_years: np.ndarray,
    amyloid,
    thickness,
    apoe4,
    ptau181,
    braak_idx: dict,
    eigenvectors: np.ndarray,
    adj_norm: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build biologically rich (n_pairs, n_features) feature matrix for amortizing alpha.

    Features (universally available unless noted):
      Braak-stage tau means, global tau stats, Gini concentration, Braak ratio,
      amyloid stats + spatial correlation with tau, APOE4, plasma p-tau181,
      follow-up time, HCP eigenmode loadings, Fickian drive magnitude, L-R asymmetry.
    """
    from scipy import stats as _sp_stats

    n = bl_s.shape[0]
    cols, names = [], []

    # Braak-stage mean tau
    for stage, idx in braak_idx.items():
        if idx:
            cols.append(bl_s[:, idx].mean(axis=1, keepdims=True))
            names.append(f"tau_braak_{stage}")
    cols.append(bl_s.mean(axis=1, keepdims=True)); names.append("tau_mean")
    cols.append(bl_s.std(axis=1, keepdims=True));  names.append("tau_std")
    cols.append(bl_s.max(axis=1, keepdims=True));  names.append("tau_max")

    # Tau spatial Gini coefficient
    def _gini(x):
        x = np.sort(np.abs(x)); n_ = len(x)
        return (2*np.sum(np.arange(1,n_+1)*x)/(n_*np.sum(x)) - (n_+1)/n_) if x.sum()>0 else 0.
    cols.append(np.array([_gini(bl_s[i]) for i in range(n)])[:, None])
    names.append("tau_gini")

    # Braak early/late ratio
    b12 = bl_s[:, braak_idx["I-II"]].mean(axis=1) if braak_idx.get("I-II") else np.zeros(n)
    b56 = bl_s[:, braak_idx["V-VI"]].mean(axis=1) if braak_idx.get("V-VI") else np.zeros(n)
    cols.append((b12 / (b12 + b56 + 1e-8))[:, None]); names.append("braak_early_ratio")

    # Amyloid features
    if amyloid is not None:
        amy = np.asarray(amyloid, dtype=float)
        cols.append(np.nanmean(amy, axis=1, keepdims=True)); names.append("amyloid_mean")
        cols.append(np.nanmax(amy,  axis=1, keepdims=True)); names.append("amyloid_max")
        if braak_idx.get("I-II"):
            cols.append(np.nanmean(amy[:, braak_idx["I-II"]], axis=1, keepdims=True))
            names.append("amyloid_braak_I_II")
        atcorr = np.array([
            _sp_stats.pearsonr(bl_s[i], amy[i])[0]
            if np.std(bl_s[i]) > 1e-8 and np.std(amy[i]) > 1e-8 else 0.
            for i in range(n)
        ])
        cols.append(np.nan_to_num(atcorr, nan=0.)[:, None])
        names.append("amyloid_tau_spatial_corr")
        if apoe4 is not None:
            a4 = np.nan_to_num(np.asarray(apoe4, dtype=float), nan=0.)
            cols.append((a4 * np.nanmean(amy, axis=1))[:, None]); names.append("apoe4_x_amyloid")

    if apoe4 is not None:
        a4 = np.nan_to_num(np.asarray(apoe4, dtype=float), nan=0.)
        cols.append(a4[:, None]); names.append("apoe4_dose")

    if ptau181 is not None:
        pt = np.asarray(ptau181, dtype=float)
        pt_filled = np.where(np.isfinite(pt), pt, float(np.nanmedian(pt)))
        cols.append(pt_filled[:, None]); names.append("plasma_ptau181")

    cols.append(time_years[:, None]); names.append("follow_up_t")

    for k in range(min(5, eigenvectors.shape[1])):
        cols.append((bl_s @ eigenvectors[:, k])[:, None])
        names.append(f"eigenmode_{k}_loading")

    neighbour_tau = bl_s @ adj_norm.T
    cols.append(np.abs(neighbour_tau - bl_s).mean(axis=1, keepdims=True))
    names.append("fickian_drive_magnitude")

    n_reg = bl_s.shape[1]
    lh_mean = bl_s[:, :n_reg//2].mean(axis=1)
    rh_mean = bl_s[:, n_reg//2:].mean(axis=1)
    cols.append(np.abs(lh_mean - rh_mean)[:, None]); names.append("tau_lr_asymmetry")

    return np.hstack(cols), names


def fit_amortize_two(
    alpha1_train: np.ndarray,
    alpha2_train: np.ndarray,
    X_all: np.ndarray,
    *,
    train_indices: np.ndarray,
    pair_groups: np.ndarray,
    feat_names: list[str],
    random_seed: int = 20260507,
    alphas_ridge: tuple = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    cv_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Multi-target ridge regression: predict (alpha1, alpha2) from biology features.

    Returns (alpha1_pred_all, alpha2_pred_all, report).
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    X_tr = X_all[train_indices]
    Y_tr = np.stack([alpha1_train, alpha2_train], axis=1)
    groups = np.asarray(pair_groups, dtype=str)[train_indices]
    n_folds = min(cv_folds, np.unique(groups).size)

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
    Y_pred_tr = final.predict(X_tr_sc)

    def _r2(yt, yp):
        ss_res = float(np.sum((yt-yp)**2)); ss_tot = float(np.sum((yt-yt.mean())**2))
        return 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")

    X_sc = (X_all - fm) / fs
    preds = final.predict(X_sc)
    alpha1_pred = np.clip(preds[:, 0],  0.0, 20.0)
    alpha2_pred = np.clip(preds[:, 1], -5.0,  5.0)

    coef_imp = np.abs(final.coef_).sum(axis=0)
    top_idx  = np.argsort(coef_imp)[::-1][:5]
    report = {
        "ridge_alpha":          best_alpha_r,
        "r2_alpha1":            _r2(Y_tr[:, 0], Y_pred_tr[:, 0]),
        "r2_alpha2":            _r2(Y_tr[:, 1], Y_pred_tr[:, 1]),
        "top_features_combined":[feat_names[i] for i in top_idx],
        "top_alpha1":           [feat_names[i] for i in np.argsort(np.abs(final.coef_[0]))[::-1][:3]],
        "top_alpha2":           [feat_names[i] for i in np.argsort(np.abs(final.coef_[1]))[::-1][:3]],
        "n_features":           len(feat_names),
        "feature_names":        feat_names,
    }
    return alpha1_pred, alpha2_pred, report
