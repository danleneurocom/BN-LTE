# BN-LTE Methodology And Iterative Research Plan

This plan translates the proposal "Mapping the Non-Stationary Causal Architecture of Alzheimer's Disease Progression Using Bayesian Networks with Latent Time Embedding" into an executable, auditable workflow. It separates the full target study from the current repository prototype so the work proceeds scientifically rather than by blind model stacking.

## 1. Research Objective

The target model is a Bayesian Network with Latent Time Embedding (BN-LTE): a non-stationary structural causal model where biomarker relationships change smoothly over a learned disease pseudotime axis `Z`.

The central scientific question is not only "can we predict follow-up tau?" It is:

1. Can heterogeneous subjects be mapped onto a biologically meaningful disease continuum?
2. Do causal edge strengths vary over that continuum?
3. Are proposed mechanisms, such as soluble-to-fibrillar tau decoupling, vascular clearance gating, and transcriptomic resilience, supported out of sample?
4. Does each methodological addition improve validation performance, graph stability, or mechanistic interpretability without test-set tuning?

## 2. Data Reality Check

The proposal requires ADNI plus UK Biobank multimodal data:

- ADNI: demographics, APOE, plasma or CSF biomarkers, amyloid PET, tau PET, FDG-PET, structural MRI, cognition, diagnosis.
- UK Biobank: accelerometry, vascular/perfusion features, structural MRI, dMRI connectomes, fMRI, genetics, PRS.
- Optional spatial biology: Allen Human Brain Atlas regional expression features.

The current repository currently provides a longitudinal ADNI tau forecasting cohort with regional tau SUVRs, APOE genotype in the forecast-pair table, diagnosis labels, and connectome-derived regional structure. Therefore, the implemented code is a tau-only BN-LTE prototype. It can validate the research loop and some regional tau dynamics, but it cannot yet truthfully test the full vascular, soluble tau, or transcriptomic hypotheses.

## 3. Full Study Methodology

### 3.1 Cohort Assembly

Construct a subject-visit feature matrix with one row per subject visit. Preserve subject identifiers, visit dates, scanner/site fields, tracer, and cohort source.

Required feature groups:

- Fixed or near-fixed roots: sex, birth year or age-at-scan, education, APOE e4 dosage, PRS.
- Fluid biomarkers: p-tau217, p-tau181 if p-tau217 unavailable, NfL, GFAP, amyloid beta ratios if available.
- Imaging biomarkers: amyloid PET composites, regional tau PET SUVRs, FDG-PET, structural MRI regional volumes or thickness, ASL perfusion if available.
- Lifestyle and vascular gates: accelerometry movement volume, sleep duration, blood pressure or vascular burden proxies, perfusion.
- Clinical sinks: MMSE, ADAS-Cog-13, CDR-SB, diagnosis.
- Regional biology modifiers: AHBA-derived regional gene modules for chaperone, synaptic vesicle, and vulnerability pathways.

Hard exclusion rules:

- Do not use future target scans to construct baseline features.
- Do not mix tracers without explicit harmonization or tracer covariates.
- Do not fit ComBat or scaling parameters on validation or test rows.
- Keep subject-level splits, not visit-level random splits.

### 3.2 Harmonization

Fit harmonization only on training data and apply learned transformations to validation and test. Scanner, site, tracer, age, and sex should be represented explicitly in the harmonization design matrix. For ADNI plus UKB, include cohort as a batch effect while preserving disease-relevant covariates.

Quality gates:

- Report missingness by feature group and cohort.
- Reject features with excessive missingness unless they are central hypotheses and can be modeled with explicit missingness handling.
- Report pre/post harmonization distributions.
- Confirm biological variance is not removed by checking retained association with disease stage markers.

### 3.3 Latent Pseudotime `Z`

The pseudotime model must be fitted on training data only. Recommended sequence:

1. Start with a robust low-dimensional disease-burden score using amyloid PET, tau PET, and cognitive or neurodegeneration markers.
2. Orient the score so higher `Z` corresponds to higher disease burden.
3. Scale training scores to `[0, 1]` using robust quantiles.
4. Project validation and test subjects using training parameters only.
5. Validate `Z` against diagnosis ordering, biomarker monotonicity, and longitudinal change.

Acceptance criteria:

- `Z` should order CN, MCI, and AD in the expected direction without using diagnosis as a target.
- Baseline `Z` should predict future biomarker change better than chronological age alone.
- The pseudotime transform must be frozen before graph fitting.

### 3.4 Background Knowledge Constraints

Use minimal disease-agnostic constraints:

- Root nodes: sex, genotype, PRS, baseline ancestry PCs, and other fixed attributes have no incoming edges.
- Sink nodes: cognitive scores and clinical diagnosis have no outgoing edges in the causal discovery graph.
- Temporal orientation: baseline variables may predict future change; future variables must not predict baseline variables.
- Regional tau prototype: when inferring regional tau parents, order parent candidates by training-only progression evidence to avoid circular regional graphs.

These constraints are not optional polishing. They prevent physically impossible conclusions such as atrophy causing APOE genotype.

### 3.5 Non-Stationary Structural Causal Model

For full BN-LTE, fit:

```text
X_j = a_j(Z) + sum_l b_jl(Z) X_l + epsilon_j
epsilon_j ~ Normal(0, sigma_j^2)
```

where `a_j(Z)` is the baseline trajectory and `b_jl(Z)` is the pseudotime-dependent direct effect from parent `l` to child `j`.

Use penalized cubic B-splines:

```text
a_j(Z) = sum_k s_jk alpha_k(Z)
b_jl(Z) = sum_k t_jlk beta_k(Z)
```

Regularization:

- Use second-order random-walk priors on spline coefficients.
- Use positive roughness penalties with hyperpriors.
- Use sparsity priors or spike-and-slab indicators for edges.
- In the prototype, ridge regularization plus group bootstrap approximates this behavior but must not be described as full Bayesian graph posterior inference.

### 3.6 Posterior Inference

Full target implementation:

1. Initialize `Z`, graph structure, spline coefficients, residual variances, and roughness parameters from stable penalized regressions.
2. Use blocked MCMC updates:
   - update graph parent indicators under constraints,
   - update spline coefficients conditional on graph,
   - update roughness penalties,
   - update residual variances,
   - optionally update `Z` if joint pseudotime inference is enabled.
3. Monitor convergence with multiple chains, effective sample size, R-hat, posterior predictive checks, and graph stability.
4. Report posterior inclusion probability (PIP) for each edge as a function of `Z`.

Prototype implementation:

- Uses train-only pseudotime.
- Uses spline-modulated ridge regressions for future tau rates.
- Uses subject-level group bootstrap as a PIP proxy.
- Uses validation performance and edge stability to decide whether a step is adopted.

## 4. Hypothesis Operationalization

### H1: Soluble-To-Fibrillar Tau Decoupling

Full data requirement:

- Plasma or CSF p-tau217.
- Neocortical or regional tau PET.
- Longitudinal tau PET or downstream neurodegeneration.

Test:

- Estimate `b_p-tau217 -> tauPET(Z)`.
- Identify whether the edge is strong at low `Z` and declines after a tau burden threshold.
- Quantify rupture point by posterior probability that the edge effect falls below a clinically meaningful threshold.

Current prototype:

- Cannot test soluble-to-fibrillar decoupling because p-tau217 is not in the current model matrix.
- Can only evaluate autonomous regional tau self-history terms and regional tau-to-tau progression.

### H2: Perfusion-Glymphatic Gating

Full data requirement:

- Physical activity or accelerometry.
- Perfusion or vascular burden markers.
- Amyloid and tau rates.

Test:

- Estimate whether activity/perfusion has negative effects on amyloid and tau accumulation rates.
- Test whether activity/perfusion reduces the `amyloid -> tau` edge strength over `Z`.

Current prototype:

- Not testable until vascular or activity variables are joined.

### H3: Transcriptomic Resilience

Full data requirement:

- Regional gene-expression modules aligned to the parcellation.
- Regional amyloid/tau/neurodegeneration outcomes.

Test:

- Model gene modules as regional modifiers of `amyloid -> tau` and `tau -> neurodegeneration`.
- Validate that high resilience-module expression attenuates downstream effects.

Current prototype:

- Not testable until AHBA-derived regional covariates are available.

## 5. Iterative Improvement Protocol

Each improvement must be evaluated before adopting the next stage.

Decision rule:

- Select by validation split only.
- Preserve a final held-out test split for reporting, not tuning.
- Prefer primary metric from config, currently `subject_spearman`.
- For error metrics such as MAE/RMSE, lower is better; for correlations and top-k overlap, higher is better.
- If a stage fails to improve validation performance, document the likely failure mode and do not promote it as the best predictive model.

Stages implemented in `scripts/run_bn_lte_iterative.py`:

1. `00_persistence`: baseline forecast with no causal model.
2. `01_pseudotime_self_history`: train-only pseudotime, spline trajectory, and autonomous regional self-history terms.
3. `02_progression_ordered_regional_edges`: adds pseudotime-varying regional parent effects with acyclic progression ordering.
4. `03_apoe_root_edges`: adds APOE e4 dosage as an exogenous root when available.
5. `04_bootstrap_pruned_edges`: bootstraps edge stability and refits only stable regional edges.

For every stage, the runner writes:

- pair-level train/validation/test metrics,
- aggregate metrics,
- edge effect summaries,
- bootstrap inclusion probabilities,
- a JSON report,
- a Markdown findings report.

## 6. Failure Analysis Rules

If pseudotime fails:

- Check whether `Z` is dominated by noise, tracer effects, or region scaling.
- Compare `Z` against diagnosis and baseline meta-temporal tau.
- Try a multimodal `Z` only after the full feature matrix is available.

If regional edges fail:

- Check overfitting by comparing train and validation metrics.
- Reduce max parents per child.
- Increase ridge alpha.
- Use bootstrap pruning.
- Do not manually tune on the test set.

If root covariates fail:

- Check missingness and variance.
- Keep the root constraint for causal validity even if prediction does not improve.
- Do not infer that APOE has no disease relevance from this tau-only prototype.

If bootstrap pruning fails:

- Treat stable edges as mechanistic candidates, not as a better forecaster.
- Report uncertainty and keep the previous validation-best model.

## 7. Reporting Standards

Every run should report:

- exact config path and output directory,
- subject counts in train, validation, and test,
- selected regions and why they were selected,
- all stage metrics,
- adopted versus rejected stages,
- edge stability with thresholds,
- known data limitations,
- next data gate required for each proposal hypothesis.

This is the minimum standard for a truthful iterative research process.
