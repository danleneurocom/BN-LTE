#!/usr/bin/env python3
"""Run the BN-LTE tau-progression model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from bayesian_network_scm.runner import run_dynamic_bn_scm  # noqa: E402
from bayesian_network_scm.reporting import render_markdown_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=THIS_DIR.parent)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR.parent / "outputs" / "bnlte")
    parser.add_argument("--pseudotime-mode", default="tau_free", choices=["tau_free", "global", "clinical_free", "pt217_free"])
    parser.add_argument("--target-prefix", default="tau_rate:")
    parser.add_argument("--max-parents", type=int, default=8)
    parser.add_argument("--bootstrap-iterations", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=20260519)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    report = run_dynamic_bn_scm(
        project_root=args.project_root,
        output_dir=args.output_dir,
        pseudotime_mode=args.pseudotime_mode,
        target_prefix=args.target_prefix,
        max_parents_per_target=args.max_parents,
        bootstrap_iterations=args.bootstrap_iterations,
        random_seed=args.random_seed,
        no_write=args.no_write,
    )
    print(render_markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
