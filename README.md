# SPREAD Toolbox: A Connectome-based Toolkit for Modelling and Forecasting Neurodegenerative Disease Spread

## Environment

Use Python 3.10 for the ENIGMA group-average forecasting workflow.

Recommended:

```bash
conda env create -f environment.yml
conda activate spread-toolbox
python -m ipykernel install --user --name spread-toolbox --display-name "Python (spread-toolbox)"
```

Pip fallback, if Python 3.10 is available without Conda:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
