# BayesianNetwork-SCM

Standalone prototype for a dynamic Bayesian-network structural causal model
for ADNI multimodal Alzheimer's disease progression.

This folder intentionally leaves the existing `src/spread_toolbox/models/bn_lte.py`
prototype untouched. The model here implements the revised methodology:
baseline multimodal variables at `t0` predict annualized biomarker change from
`t0` to `t1`, with train-only explainable disease pseudotime `Z` and hard
biological graph constraints.

## Core Model

For outcome `Y_j`, each subject-pair row is modeled as:

```text
R_ij = (Y_ij(t1) - Y_ij(t0)) / dt_i

R_ij = a_j(Z_i)
     + c_j(Z_i) Y_ij(t0)
     + sum_l gamma_jl b_jl(Z_i) X_il(t0)
     + sum_m d_jm(Z_i) I_im(t0)
     + epsilon_ij
```

where `Z_i` is a train-only baseline disease pseudotime, `gamma_jl` is an
edge-inclusion indicator or stability proxy, and all candidate parents are
baseline variables. This prevents future target leakage and gives the graph a
clear temporal orientation.

## Folder Layout

```text
bayesian_network_scm/
  data.py          multimodal ADNI pair-table assembly
  pseudotime.py    explainable train-only pseudotime
  constraints.py   root/sink/layered DAG constraints
  dynamic_scm.py   constrained varying-coefficient SCM prototype
  reporting.py     markdown/json/csv report helpers
  runner.py        end-to-end command-line workflow
docs/
  methodology.md   mathematical and research design specification
tests/
  test_*.py        focused synthetic and data-contract tests
```

## Intended Use

Run from the repository root with the project virtualenv:

```bash
.venv/bin/python BayesianNetwork-SCM/run_dynamic_bn_scm.py --no-write
```

The implementation is deliberately staged. The first reliable output is a
penalized spline/ridge and bootstrap-stability prototype. A full MCMC graph
posterior should only be added after the multimodal feature matrix, constraints,
and validation behavior are locked.
