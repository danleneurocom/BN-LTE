# Dynamic BN-SCM Methodology

## Objective

The goal is to reconstruct non-stationary causal cascades in ADNI by estimating
how baseline biomarkers cause future biomarker change along an explainable
disease pseudotime axis `Z`.

The design deliberately avoids a same-time cross-sectional DAG as the primary
object. Same-time DAG discovery is too vulnerable to reverse orientation and
latent confounding for this cohort. The primary model is longitudinal:

```text
baseline state X(t0) -> annualized rate R = (Y(t1) - Y(t0)) / dt
```

## Data Unit

One row is a subject/tracer baseline-to-target pair:

```text
i = subject-pair
t0 = baseline tau PET date
t1 = target tau PET date
dt_i = years between t0 and t1
```

The row contains baseline features:

```text
X_i(t0): roots, fluids, amyloid PET, tau PET, MRI, cognition
```

and rate targets:

```text
R_ij = (Y_ij(t1) - Y_ij(t0)) / dt_i
```

For the first implementation, tau PET rates are always available because the
existing cohort was built from longitudinal tau PET. Other rates are included
only when baseline and follow-up observations can be matched without target
leakage.

## Causal Graph Architecture

Variables are assigned to biological layers:

```text
root:
  age, sex, education, APOE4

fluid:
  plasma Aβ42/40, pT217, NfL, GFAP

pathology:
  amyloid PET, tau PET

neurodegeneration:
  hippocampal/amygdala/temporal volumes and cortical thickness

clinical:
  ADAS13, MMSE, AVLT/RAVLT-like memory scores, CDRSB, diagnosis
```

Allowed high-level direction:

```text
root -> fluid/pathology/neurodegeneration/clinical
fluid -> pathology/neurodegeneration/clinical
amyloid -> tau
tau -> neurodegeneration/clinical
neurodegeneration -> clinical
```

Forbidden:

```text
incoming edges into root nodes
outgoing edges from clinical sink nodes
future target variables causing baseline variables
cognition causing molecular or imaging pathology
atrophy causing genotype, sex, or chronological age
```

Diagnosis is used for validation and stratification, not as a causal parent in
the primary graph.

## Explainable Pseudotime

Pseudotime is fitted on training rows only:

```text
Z_i = scale(w^T standardize(X_i(t0)))
```

`Z` is an ordinal disease-stage coordinate, not calendar time. It is only
identifiable up to monotone transformation, so it is oriented so that high `Z`
corresponds to higher disease burden.

Reportable explanations:

```text
feature loadings w
subject-level feature contributions w_m * zscore(X_im)
clinical ordering CN < MCI < AD
biomarker trajectories over Z
predictive utility for future biomarker rates
```

Use sensitivity versions:

```text
global:       multimodal disease markers
tau_free:     excludes tau and pT217 when testing pT217 -> tau
clinical_free excludes diagnosis/cognitive scores from Z
```

## Dynamic Structural Equation

For outcome `Y_j`:

```text
R_ij = a_j(Z_i)
     + c_j(Z_i) Y_ij(t0)
     + sum_{l in pa(j)} gamma_jl b_jl(Z_i) X_il(t0)
     + sum_{m in M_j} d_jm(Z_i) I_im(t0)
     + epsilon_ij
```

where:

```text
R_ij       annualized future rate
a_j(Z)    disease-stage intercept trajectory
c_j(Z)    self-history/autonomous progression effect
b_jl(Z)   pseudotime-varying direct effect from parent l to target j
gamma_jl  edge inclusion variable or bootstrap stability proxy
I_im      biologically specified interaction term
epsilon   Gaussian residual in the prototype; Student-t can be added later
```

Forecast:

```text
Y_ij(t1) = Y_ij(t0) + dt_i * R_ij
```

## Spline Parameterization

Varying effects use cubic B-splines:

```text
a_j(Z)   = sum_k alpha_jk B_k(Z)
b_jl(Z) = sum_k theta_jlk B_k(Z)
```

Prototype regularization uses ridge penalties. Full Bayesian regularization
should use a second-order random-walk prior:

```text
theta_jl | tau_jl ~ Normal(0, [tau_jl K + lambda I]^-1)
tau_jl          ~ Gamma(a_tau, b_tau)
```

Under the standard precision parameterization, the conditional update for
`tau_jl` is Gamma-like. Do not claim a GIG update unless the hierarchy is
explicitly changed to produce one.

## Hypothesis Tests

### H1: pT217-to-Tau Decoupling

Target:

```text
neocortical or regional tau PET rate
```

Key term:

```text
b_pT217,tau(Z) * pT217(t0)
```

Evidence:

```text
b_pT217,tau(Z) > 0 at early Z
b_pT217,tau(Z) approaches 0 at late Z
self-history tau effect c_tau(Z) increases at late Z
```

Decoupling coordinate:

```text
Z_decouple = min Z such that P(|b_pT217,tau(Z)| < delta) > threshold
```

The current prototype estimates this with bootstrap effect curves rather than
full posterior probabilities.

### H2: Transcriptomic Resilience Gating

AHBA expression is regional and static, so it is modeled as a regional effect
modifier, not a subject node:

```text
tau_rate_ir = ...
            + b_Aβ,tau,r(Z_i) amyloid_ir(t0)
            + d_SR,r(Z_i) amyloid_ir(t0) * SR_r
            + ...
```

Expected sign:

```text
d_SR,r(Z) < 0
```

until real AHBA features are configured, this hypothesis remains gated off or
uses clearly labeled structural-proxy sensitivity checks only.

### H3: PART vs AD Continuum

Split subjects by baseline amyloid/tau status:

```text
A-T-
A-T+  PART-like
A+T-
A+T+  AD continuum
```

Compare:

```text
edge PIP/rank similarity
spatial tau-rate similarity
effect magnitude ratios over Z
```

The claim should be "shared spatial route with different kinetics" unless graph
similarity and rate-acceleration tests both pass.

## Acceptance Criteria

Each iteration must report:

```text
train/validation/test split at subject level
feature coverage and imputation counts
pseudotime loadings and diagnosis ordering
validation and test rate metrics
edge stability by bootstrap
known untestable hypotheses and missing data gates
```

No stage is promoted because it merely runs. A stage is promoted only if it is
technically valid, avoids leakage, and improves prediction, stability, or
mechanistic interpretability under pre-specified validation rules.
