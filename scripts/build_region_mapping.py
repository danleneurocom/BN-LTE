#!/usr/bin/env python3
"""
Build ADNI tau-column to ENIGMA aparc region mapping.
Compares tau in brain regions with connections between those same brain regions.
ENIGMA gives us connectome matrix between regions:
entorhinal ↔ inferior temporal
entorhinal ↔ precuneus
precuneus ↔ temporal

Example
ADNI tau column name        ENIGMA aparc label
CTX_ENTORHINAL_SUVR   →     entorhinal
CTX_PRECUNEUS_SUVR    →     precuneus
...
"""


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402
from spread_toolbox.region_mapping import (  # noqa: E402
    build_adni_enigma_aparc_mapping,
    write_mapping_outputs,
)


def default_config_path() -> Path:
    experiment_dir = PROJECT_ROOT / "experiments" / "group_average_enigma"
    local_config = experiment_dir / "config.yaml"
    if local_config.exists():
        return local_config
    return experiment_dir / "config.example.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to config YAML. Defaults to config.yaml if present, otherwise config.example.yaml.",
    )
    parser.add_argument(
        "--tau-observations",
        type=Path,
        help="Override path to cohort_tau_observations.csv.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Build and validate the mapping without writing files.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    tau_observations = args.tau_observations or output_dir / config["outputs"].get(
        "tau_observations_table", "cohort_tau_observations.csv"
    )

    result = build_adni_enigma_aparc_mapping(tau_observations)
    print(json.dumps(result.summary, indent=2, sort_keys=True))

    if not args.no_write:
        output_paths = write_mapping_outputs(result, config, PROJECT_ROOT)
        print("\nWrote region mapping outputs:")
        for name, path in output_paths.items():
            print(f"{name}: {path}")

    if not result.summary["is_complete"]:
        print("\nRegion mapping is incomplete; inspect the report before modelling.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
