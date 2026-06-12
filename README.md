# BN-LTE: Bayesian Networks with Latent Time Embedding for Stage-Aware Causal Modeling of Alzheimer’s Disease Progression

This repository contains the runnable BN-LTE pipeline for longitudinal regional tau-PET progression modeling on ADNI-derived inputs. The active model code is isolated from baseline and manuscript artifacts so the main experiment can be reproduced from the command line.

## Repository Layout

```text
BNLTE/
  run_bnlte.py
  bayesian_network_scm/
BRAIN DATA/ADNI/
experiments/group_average_enigma/
  adni_to_enigma_aparc_mapping.csv
  output/
baselines/
requirements.txt
environment.yml
```

`BNLTE/` contains the model, data assembly, pseudotime embedding, constrained transition fitting, and reporting code. `BRAIN DATA/ADNI/` and `experiments/group_average_enigma/output/` provide the cached input tables used by the BN-LTE runner. `baselines/` keeps legacy comparator code outside the active BN-LTE path.

## Environment

Use Python 3.10 or newer. Conda setup:

```bash
conda env create -f environment.yml
conda activate spread-toolbox
```

Pip setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run BN-LTE

```bash
python BNLTE/run_bnlte.py \
  --project-root . \
  --output-dir outputs/bnlte \
  --pseudotime-mode tau_free \
  --target-prefix tau_rate: \
  --max-parents 8 \
  --bootstrap-iterations 25
```
```

The full run writes `dynamic_bn_scm_report.json`, `dynamic_bn_scm_rate_metrics.csv`, `dynamic_bn_scm_edge_effects.csv`, `dynamic_bn_scm_bootstrap_edges.csv`, and `dynamic_bn_scm_findings.md` under the selected output directory.
