#!/usr/bin/env python3
"""Prepare the mapped ENIGMA adjacency matrix and graph Laplacian."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.connectome import prepare_connectome, write_connectome_outputs  # noqa: E402
from spread_toolbox.io_adni import load_yaml_config  # noqa: E402


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
        help="Prepare and validate the connectome without writing files.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    result = prepare_connectome(config, PROJECT_ROOT)
    print(json.dumps(result.report, indent=2, sort_keys=True))

    if not args.no_write:
        output_paths = write_connectome_outputs(result, config, PROJECT_ROOT)
        print("\nWrote connectome outputs:")
        for name, path in output_paths.items():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
