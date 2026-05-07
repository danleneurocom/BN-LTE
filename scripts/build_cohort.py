#!/usr/bin/env python3
"""Build the ADNI longitudinal tau cohort for a forecasting experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.io_adni import (  # noqa: E402
    build_longitudinal_tau_cohort,
    load_yaml_config,
    write_cohort_outputs,
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
        "--no-write",
        action="store_true",
        help="Build the cohort in memory and print row counts without writing outputs.",
    )
    parser.add_argument(
        "--adni-dir",
        type=Path,
        help="Override paths.adni_dir from the config. Useful when the repo has moved.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.adni_dir:
        config.setdefault("paths", {})["adni_dir"] = str(args.adni_dir.expanduser().resolve())
    result = build_longitudinal_tau_cohort(config, PROJECT_ROOT)

    print(json.dumps(result.row_counts, indent=2, sort_keys=True))
    if not args.no_write:
        output_paths = write_cohort_outputs(result, config, PROJECT_ROOT)
        print("\nWrote cohort outputs:")
        for name, path in output_paths.items():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
