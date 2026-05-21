# Methodology: Dynamic Bayesian-Network Structural Causal Model (Dynamic BN-SCM)

This section documents the methodology *as implemented* in the
`BayesianNetwork-SCM/bayesian_network_scm/` package and exercised by
`run_dynamic_bn_scm.py` and `run_hypothesis_experiments.py`. All numbers and
defaults below are taken directly from the source code.

## 1. Design Overview

The model is a leakage-controlled, longitudinal structural causal model for
multimodal Alzheimer's disease progression. The primary object is **not** a
same-time cross-sectional DAG, which is known to be vulnerable to reverse
orientation and unmeasured confounding. Instead, for each subject we form a
**baseline-to-follow-up pair** and model:

```
baseline state X(t0) -> annualized rate R = (Y(t1) - Y(t0)) / dt
```

with all candidate parents pinned to the baseline visit `t0`. Future target
values can therefore never enter the predictor set, which gives the graph an
unambiguous temporal orientation and prevents target leakage.

For an outcome `j` (a regional tau rate or a cognitive rate) and a subject-pair
`i`, the structural equation is

```
R_ij = a_j(Z_i)
     + c_j(Z_i) * Y_ij(t0)
     + sum_{l in pa(j)} gamma_jl * b_jl(Z_i) * X_il(t0)
     + epsilon_ij                                             (Eq. 1)
```

where

- `R_ij = (Y_ij(t1) - Y_ij(t0)) / dt_i` is the annualized rate target,
- `Z_i ∈ [0, 1]` is a train-only **explainable disease pseudotime** (§3),
- `a_j(Z)` is a disease-stage-varying intercept (trajectory),
- `c_j(Z) * Y_ij(t0)` is a pseudotime-varying **self-history / autonomous**
  progression term,
- `b_jl(Z)` is the pseudotime-varying direct effect from baseline parent `l`
  on target `j`, restricted to the biologically admissible parent set `pa(j)`,
- `gamma_jl` is an edge-inclusion proxy obtained by thresholding the
  bootstrap-stability of `|b_jl(Z)|` over `Z` (§7).

The corresponding biomarker forecast is the explicit Euler step

```
Y_ij(t1) = Y_ij(t0) + dt_i * R_ij                             (Eq. 2)
```

Equation (1) is a varying-coefficient extension of a standard linear SCM:
fixing `Z` recovers an ordinary linear regression of rate on baseline state,
whereas allowing the coefficients to vary with `Z` lets the same causal
parents have stage-dependent effect magnitudes (e.g., early-only or late-only
mechanisms).

## 2. Data Construction

The pair table is assembled in
[bayesian_network_scm/data.py](../bayesian_network_scm/data.py) by
`build_multimodal_pair_dataset()` from the ENIGMA-format ADNI cohort exports
under `experiments/group_average_enigma/output/`.

**Pairs.** Each row is one (baseline, follow-up) tau-PET pair. We require
`baseline_loniuid` and `target_loniuid` to resolve in
`cohort_tau_observations.csv`, finite baseline date and follow-up date, and
`dt_i = target_time_years > 0`. Tau follow-up is always present because the
cohort was constructed from longitudinal FTP tau PET pairs.

**Baseline predictors.** For every pair `i`, baseline predictors are taken
from the visit nearest in time to the baseline tau-PET date within
`max_date_distance_days = 1095` (3 years):

| Layer              | Features (30 total)                                                                                              |
|--------------------|------------------------------------------------------------------------------------------------------------------|
| root               | `age_years`, `sex_female`, `education_years`, `apoe4_dose`                                                       |
| fluid              | `plasma_pt217`, `plasma_ab42_ab40`, `plasma_nfl`, `plasma_gfap`                                                  |
| pathology          | `amyloid_summary_suvr`, `amyloid_centiloids`, `amyloid_positive`, `tau_meta_temporal`, plus 10 regional tau SUVR |
| neurodegeneration  | `mri_hippocampus_volume`, `mri_amygdala_volume`, `mri_temporal_cortical_volume`, `mri_temporal_cortical_thickness` |
| clinical           | `adas13`, `mmse`, `ravlt_immediate`, `cdrsb`                                                                     |

`apoe4_dose` is the count of ε4 alleles in the APOE genotype. MRI volumes and
thicknesses are means over left/right entorhinal, fusiform, inferiortemporal,
and middletemporal cortical labels parsed from the FreeSurfer dictionary;
hippocampal and amygdalar SVs are bilateral averages. Rows with overall
FreeSurfer QC `FAIL` and amyloid scans with QC fail flags are excluded.

**Regional tau predictors and targets.** Ten temporo-parietal Schaefer/aparc
labels are used:
`L/R_entorhinal`, `L/R_fusiform`, `L/R_inferiortemporal`,
`L/R_middletemporal`, `L/R_inferiorparietal`, mapped to ADNI tau columns via
`experiments/group_average_enigma/adni_to_enigma_aparc_mapping.csv`. The
target vector contains 15 entries:

- `tau_rate:meta_temporal` (composite SUVR rate),
- `tau_rate:<region>` for each of the 10 regions above,
- `cognitive_rate:{adas13, mmse, ravlt_immediate, cdrsb}`.

Each target is the simple annualized rate
`R_ij = (Y_ij(t1) - Y_ij(t0)) / dt_i`. Cognitive rates are computed only when
both nearest-to-baseline and nearest-to-follow-up scores are finite within
the 1095-day window.

**Subject-level split.** `make_subject_split()` in `reporting.py` partitions
the **unique RIDs** (not pairs) into 60 % train / 20 % validation / 20 % test
with random seed `20260519`. All pairs from a given subject are kept inside a
single fold, which is essential for leakage control: the same subject never
appears as both a training row and a held-out row.

## 3. Explainable Pseudotime `Z`

`fit_pseudotime()` in
[bayesian_network_scm/pseudotime.py](../bayesian_network_scm/pseudotime.py)
defines `Z` so the model can capture stage-dependent dynamics without using
calendar time or future information.

**Feature selection.** Given a *mode* (`tau_free`, `global`, `clinical_free`,
`pt217_free`), only features admissible for the mode with finite-value
coverage `≥ 0.5` on the training rows and non-zero training variance are
retained. `tau_free` (the default) excludes all `tau`, `pT217`, and `pTau`
features so that `Z` is not endogenous to the tau targets of interest.

**Standardization and imputation.** Missing entries are filled with the
column-wise training median; columns are then mean-centered and scaled to
unit standard deviation (with a guard `scale > 1e-12`).

**Linear pseudotime via SVD.** Let `X̃_train ∈ R^{n_train × p}` be the
standardized training feature matrix. We take the leading right singular
vector `w` of `X̃_train`:

```
X̃_train = U Σ V^T,   w = V[:, 0]                              (Eq. 3)
```

For any (training or held-out) row, the raw score is the projection
`s_i = X̃_i^T w`. Scores are rescaled to `[0, 1]` using the empirical 1st and
99th training percentiles `(s_lower, s_upper)`:

```
Z_i = clip( (s_i - s_lower) / (s_upper - s_lower), 0, 1 )       (Eq. 4)
```

This is the SVD-based, linear analog of a single principal axis of disease
progression; using only the leading component makes `Z` one-dimensional and
interpretable.

**Orientation.** Because singular vectors are sign-ambiguous, `w` is flipped
so its correlation with a sign-coded disease-burden proxy is non-negative.
The burden proxy is the mean of z-scored features signed by
`burden_sign()` (positive: `tau`, `pT217`, `amyloid`, `centiloid`, `nfl`,
`gfap`, `adas13`, `cdrsb`; negative: `mmse`, `ravlt`, `volume`, `thickness`,
`ab42_ab40`). This guarantees that higher `Z` corresponds to higher disease
burden, while leaving the geometry of `Z` itself unchanged.

**Reportable diagnostics.** The model exposes feature loadings `w`,
per-subject contributions `w_m * z(X_im)`, a diagnosis-group ordering of
median `Z` (CN < MCI < AD is the expected sanity check), and the explained
variance ratio of the leading singular value.

## 4. Biological Graph Constraints

`CausalConstraints` in
[bayesian_network_scm/constraints.py](../bayesian_network_scm/constraints.py)
restricts the set of admissible baseline parents for each rate target.

**Variable layers.** Every variable is assigned a layer with rank
`root (0) < fluid (1) < pathology (2) < neurodegeneration (3) < clinical
(4)`. Layer inference uses keyword rules (e.g., `plasma`/`pt217`/`nfl`/`gfap`
→ `fluid`; `amyloid`/`tau` → `pathology`; `mri`/`volume`/`thickness` →
`neurodegeneration`; `adas`/`mmse`/`ravlt`/`cdrsb` → `clinical`).

**Hard constraints (`can_parent(parent, child)`).** Parent `l` may direct an
edge to child `j` only if:

1. `parent ≠ child`;
2. parent is not a clinical sink and child is not a root;
3. either `layer(parent) < layer(child)` (cross-layer downstream), or
4. parent and child are in the same layer **and** the same-layer rule below
   is satisfied.

**Same-layer rules.**

- `root`, `clinical`, `neurodegeneration`: no within-layer edges.
- `fluid`: ordered as `amyloid/ab42 < pT217/pTau < gfap < nfl`; only
  parent-to-later-stage child edges are allowed.
- `pathology`: only `amyloid → tau` is allowed; tau cannot drive amyloid;
  no edges within amyloid or within tau.

This encodes the canonical AD cascade
`root → fluid → amyloid → tau → neurodegeneration → clinical`, with diagnosis
left out of the parent set entirely and reserved for validation and
stratification.

## 5. Varying-Coefficient Design Matrix

`build_design_matrix()` in `dynamic_scm.py` constructs the per-target design
that operationalizes Eq. (1).

**B-spline basis on `Z`.** Let `B(Z) ∈ R^{n × K}` be a cubic B-spline basis
with `K` columns, built by
[`sklearn.preprocessing.SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)
with `degree = 3`, `n_knots = 4`, `include_bias = True`, and constant
extrapolation. The varying intercept and the varying parent effects are
expanded as

```
a_j(Z)   = sum_{k=1..K} alpha_jk    * B_k(Z)                  (Eq. 5)
c_j(Z)   = sum_{k=1..K} kappa_jk    * B_k(Z)
b_jl(Z) = sum_{k=1..K} theta_jlk   * B_k(Z)
```

If the train set has too few unique `Z` values to support the basis, the
implementation falls back to a polynomial basis of width
`min(4, degree + 1)`.

**Design columns.** For each target `j` with chosen parent set `pa(j)`, the
design row `x_i` is the concatenation of three blocks:

1. **Trajectory:** the `K` basis values `B_k(Z_i)` → gives `a_j(Z)`.
2. **Self-history:** `Y_ij(t0) * B_k(Z_i)` for `k = 1..K` → gives
   `c_j(Z) * Y_ij(t0)`.
3. **Edge-by-stage interactions:** `X_il(t0) * B_k(Z_i)` for each parent
   `l ∈ pa(j)` and `k = 1..K` → gives the `b_jl(Z) * X_il(t0)` terms.

The total design dimension for target `j` is therefore
`D_j = K * (2 + |pa(j)|)`. The recovered effect curves (§7) are computed by
contracting these coefficients back through `B(Z_grid)`.

## 6. Parent Selection and Ridge Estimation

**Candidate parents.** For target `j`, `candidate_parents()` returns every
feature `l` that satisfies the layer/role constraints of §4 *and* has
training coverage `≥ 0.25`. Features are then ranked by the absolute Pearson
correlation between `X_l(t0)` and `R_j` on the training rows. A small set of
biological priority features is prepended ahead of the correlation ranking
(for `tau_rate:*`: `plasma_pt217`, `amyloid_summary_suvr`,
`amyloid_centiloids`, `plasma_ab42_ab40`, `apoe4_dose`, `age_years`; for
`cognitive_rate:*`: `tau_meta_temporal`, `mri_hippocampus_volume`,
`amyloid_summary_suvr`, `plasma_nfl`, `age_years`). The final list is
truncated to `max_parents_per_target` (default 8 in
`run_dynamic_bn_scm.py`, 6 in `run_hypothesis_experiments.py`).

**Preprocessing.** Within `fit_ridge()`, rows with non-finite targets are
dropped, design entries are imputed with column-wise training medians, and
columns are mean-centered and scaled by their training standard deviation
(with a `> 1e-12` guard).

**Ridge solution.** Let `Φ_train ∈ R^{n × D}` be the standardized training
design and `r_train` the centered training rate target. The estimator solves

```
beta_hat = argmin_{beta}  || r_train - Φ_train beta ||² + alpha * ||beta||²
         = (Φ_train^T Φ_train + alpha * I_D)^{-1} Φ_train^T r_train  (Eq. 6)
```

with intercept `beta_0 = mean(r_train)`. Ridge regularization is used in
place of an explicit second-order random-walk prior because the prototype
estimates effects pointwise rather than sampling a posterior; the L2 penalty
plays the role of the inverse-variance hyperprior on the spline coefficients
while remaining convex and closed-form.

**Penalty selection.** The penalty is chosen by **subject-grouped 5-fold
cross-validation** using
[`sklearn.model_selection.GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
with groups equal to each pair's `RID`. This guarantees that no subject
appears in both the inner train and inner validation folds. The grid is
`alpha ∈ {1, 10, 100, 1000, 10000}` and the alpha minimizing mean-squared
prediction error across folds is retained.

**Effect-curve reconstruction.** Once `beta_hat` is fit, the effect of parent
`l` at any `Z` is recovered as

```
b_jl(Z) = sum_{k=1..K} B_k(Z) * beta_hat[col(l, k)] / scale[col(l, k)]  (Eq. 7)
```

with the analogous formula for `c_j(Z)`. The grid `Z = linspace(0, 1, 101)`
is used to evaluate all curves.

## 7. Edge Stability and Inclusion (`gamma_jl`)

A first inclusion screen is the **effect-magnitude threshold** `tau_eff`:
edge `l → j` is "selected by effect" if `max_Z |b_jl(Z)| ≥ tau_eff`, with
`tau_eff = 0.01` by default.

A frequentist analog of an edge posterior inclusion probability (PIP) is then
obtained by **subject-block bootstrap**
(`bootstrap_edge_stability()`):

1. Group training pairs by `RID`; let `G` be the set of training subjects.
2. For `b = 1, ..., B` iterations (default `B = 25`), sample `|G|` subjects
   from `G` with replacement, take all of each sampled subject's training
   pairs to form a bootstrap training set, and re-fit the full Dynamic BN-SCM
   on this sample (with `cv_folds = 0` for speed).
3. For each candidate parent–target pair `(l, j)`, record whether
   `max_Z |b_jl^{(b)}(Z)| ≥ tau_eff`.

The empirical **bootstrap inclusion probability** is

```
PIP_jl^{boot} = (1 / B) * sum_{b=1..B} I{ max_Z |b_jl^{(b)}(Z)| >= tau_eff }   (Eq. 8)
```

and is used as the implementation's proxy for `gamma_jl`. Bootstrap blocks
are at the subject level so re-sampled pairs respect the subject-level
dependence structure of the cohort.

## 8. Forecasting and Evaluation

For any held-out pair, the predicted SUVR (or score) follow-up is obtained
from Eq. (2) using the model's predicted rate `R̂_ij` together with the
observed `dt_i` and `Y_ij(t0)`. Evaluation metrics are computed per
(target, split) pane and aggregated across targets:

- per-pair rate error metrics `mae`, `rmse`, `pearson`, `spearman` between
  observed and predicted rates,
- per-subject regional `delta_spearman` between observed and predicted
  baseline-to-follow-up change vectors,
- a compact summary of validation/test tau-rate MAE and Spearman (mean and
  median across targets) for ranking and model comparison.

## 9. Pre-Specified Hypothesis Tests

`run_hypothesis_experiments.py` runs three pre-registered analyses on top of
the base `tau_free` fit.

### 9.1 H1 — pT217-to-Tau Decoupling

For every tau-rate target `j` and the parent `l = plasma_pT217`, the effect
curve `b_pT217,j(Z)` is sliced into three windows
`{early: Z ≤ 0.30, mid: 0.30 < Z < 0.70, late: Z ≥ 0.70}`. We report

- the means of `|b_pT217,j(Z)|` in each window,
- the indicator
  `late_less_than_early = (mean_late < mean_early)`,
- a "near-zero late" indicator `mean_late < 0.005`,
- a strict decoupling pattern that additionally requires the self-history
  effect to rise at late `Z`,
- the decoupling coordinate

  ```
  Z_decouple = min { Z >= 0.30 : |b_pT217,j(Z)| < 0.005 }       (Eq. 9)
  ```

  The prototype reports these as descriptive summaries of the ridge effect
  curves; turning them into a formal posterior probability would require a
  Bayesian sampler over `theta_jl`, which is explicitly out of scope for the
  current implementation.

### 9.2 H2 — Transcriptomic Resilience Gating

The intended structural extension introduces a regional gene-expression
modifier `SR_r` of the amyloid → tau edge:

```
R_ij,r = ... + b_{Aβ,tau,r}(Z_i) * amyloid_i(t0)
              + d_{SR,r}(Z_i)     * amyloid_i(t0) * SR_r + ...   (Eq. 10)
```

with expected sign `d_{SR,r}(Z) < 0` (regions with higher resilience
expression have lower amyloid-driven tau acceleration). `evaluate_h2_data_gate()`
scans the project tree for AHBA / Allen / gene-expression files and reports
H2 as `not_testable_current_data` if none are found. The code intentionally
refuses to fit a structural-proxy H2 model until a region-by-DK/aparc
expression matrix is available.

### 9.3 H3 — PART-like vs AD-Continuum Spatial Route

Each pair is assigned an A/T status using the implemented data:

- `A+` if the baseline `amyloid_positive` feature `≥ 0.5`, else `A-`,
- `T+` if baseline `tau_meta_temporal ≥ q_75` of the **training cognitively
  normal subset**, else `T-`.

Subjects with non-finite amyloid or tau are labelled `unclassified`. The
two analyzed groups are `A-T+` (PART-like) and `A+T+` (AD continuum). For
each group, regional rate vectors are computed by averaging both the
observed and the BN-SCM-predicted regional rates across pairs. Two quantities
are reported:

```
route_similarity   = spearman( mean rate vector_PART,  mean rate vector_AD )    (Eq. 11)
kinetic_ratio_AD/PART = rms( mean rate vector_AD ) / rms( mean rate vector_PART )   (Eq. 12)
```

H3 is declared "supported" only when both `route_similarity ≥ 0.70` and
`kinetic_ratio ≥ 1.20` on the test split, encoding the "shared spatial route
with kinetic acceleration" claim. Equation (11) is computed both for
observed-rate route similarity and for the BN-SCM-predicted-rate route, so
that the model's spatial inductive bias can be checked against the data.

## 10. Sensitivity Analyses

Two robustness sweeps are run alongside H1–H3.

**Pseudotime sensitivity.** The full Dynamic BN-SCM is re-fit four times,
once per pseudotime mode (`tau_free`, `global`, `clinical_free`,
`pt217_free`), reporting the median test rate MAE/RMSE and the median test
delta-Spearman. This isolates the effect of `Z` from the effect of the
parent set.

**Parent ablations.** Starting from the `tau_free` model, candidate features
are deleted in groups
(`{plasma_pt217}`, `{amyloid_summary_suvr, amyloid_centiloids,
amyloid_positive}`, `{plasma_ab42_ab40}`, `{apoe4_dose}`) and the model is
re-fit and re-evaluated. Differences in test rate MAE versus the full model
quantify how much each modality contributes to held-out tau-rate forecasting.

## 11. Leakage Controls and Limitations

By construction, the implementation enforces the following:

- predictors are sampled at or before `t0`, never at `t1`;
- the subject split partitions unique RIDs;
- pseudotime, ridge centering/scaling, alpha selection, parent ranking, and
  bootstrap blocks are all fit using **training rows only**;
- ridge alpha selection uses `GroupKFold` with `RID` groups so that no
  subject's pairs straddle the inner cross-validation boundary;
- diagnosis is used only for stratification and reporting, never as a
  parent.

Stated limitations carried directly in the code are:

1. The estimator is a penalized (ridge) varying-coefficient regression with
   subject-block bootstrap stability, not a full posterior MCMC over the
   graph; reported edge probabilities should be read as bootstrap inclusion
   proxies, not Bayesian PIPs.
2. H2 is gated off until a DK/aparc-aligned regional gene-expression matrix
   is provided.
3. Brain-map panels visualize only the ten temporo-parietal target regions;
   un-modelled DK regions are greyed out.
4. The H3 `T+` threshold is currently derived from the training CN 75th
   percentile of `tau_meta_temporal`; a project-level tau-positivity
   threshold would be substituted if configured.
