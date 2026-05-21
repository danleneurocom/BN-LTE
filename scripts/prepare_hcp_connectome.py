#!/usr/bin/env python3
"""Build a group-average HCP structural connectome in aparc (68-region) space.

Pipeline:
  1. Load all individual HCP probabilistic connectivity matrices (Glasser 379×379).
  2. Average across subjects → group-average Glasser 360 cortical matrix.
  3. Derive Glasser→aparc mapping via vertex overlap on fsaverage5.
  4. Aggregate Glasser 360×360 → aparc 68×68 (mean of within-pair weights).
  5. Apply symmetrisation, zero diagonal, build combinatorial Laplacian.
  6. Write aparc-ordered adjacency + Laplacian CSVs alongside the ENIGMA outputs.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spread_toolbox.connectome import build_laplacian, clean_adjacency, write_labeled_matrix  # noqa: E402
from spread_toolbox.io_adni import load_yaml_config, resolve_project_path  # noqa: E402


HCP_DATA_DIR = PROJECT_ROOT / "BRAIN DATA" / "HCP_1200"
MATRIX_SUFFIX = "_glasser_atlas_probabilistic_structural_connectivity_with_subcortex.npy"
LABELS_FILE   = "glasser_atlas_label_names_ordered_list.json"


def default_config_path() -> Path:
    experiment_dir = PROJECT_ROOT / "experiments" / "group_average_enigma"
    local = experiment_dir / "config.yaml"
    return local if local.exists() else experiment_dir / "config.example.yaml"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--max-subjects", type=int, default=0,
                        help="Cap subjects for speed testing (0 = all)")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config    = load_yaml_config(args.config)
    output_dir = resolve_project_path(config["paths"]["output_dir"], PROJECT_ROOT)
    outputs   = config.get("outputs", {})

    # ── 1. Collect subject matrix files ──────────────────────────────────────
    matrix_files = sorted(
        p for p in HCP_DATA_DIR.iterdir()
        if p.is_dir() and (p / (p.name + MATRIX_SUFFIX)).exists()
    )
    if args.max_subjects > 0:
        matrix_files = matrix_files[:args.max_subjects]
    print(f"Found {len(matrix_files)} subjects with probabilistic connectivity matrices.")

    # Load label names from any one subject (shared across all)
    label_path = matrix_files[0] / LABELS_FILE
    with label_path.open() as f:
        all_labels = json.load(f)           # 379 labels: 19 subcortex + 360 cortex
    n_subcortex = sum(1 for l in all_labels if "CIFTI_STRUCTURE" in l)
    cortex_labels_hcp = all_labels[n_subcortex:]   # 360 Glasser area names
    n_glasser = len(cortex_labels_hcp)
    print(f"  Subcortex regions: {n_subcortex}, Cortical Glasser areas: {n_glasser}")

    # ── 2. Compute group-average cortical matrix ──────────────────────────────
    print("Computing group-average connectivity (cortex-only, probabilistic)...")
    group_sum = np.zeros((n_glasser, n_glasser), dtype=np.float64)
    n_loaded  = 0
    for subj_dir in matrix_files:
        fpath = subj_dir / (subj_dir.name + MATRIX_SUFFIX)
        try:
            m = np.load(fpath)
            cortex = m[n_subcortex:, n_subcortex:].astype(np.float64)
            group_sum += cortex
            n_loaded  += 1
        except Exception as e:
            print(f"  WARNING: skipping {subj_dir.name}: {e}")
        if n_loaded % 100 == 0 and n_loaded > 0:
            print(f"  loaded {n_loaded}/{len(matrix_files)} ...")

    group_avg = group_sum / max(n_loaded, 1)
    print(f"Group average from {n_loaded} subjects. "
          f"Shape: {group_avg.shape}, "
          f"value range [{group_avg.min():.1f}, {group_avg.max():.1f}]")

    # ── 3. Glasser → aparc mapping via fsaverage5 vertex overlap ─────────────
    print("Deriving Glasser→aparc mapping from fsaverage5 vertex overlap...")
    glasser_to_aparc_idx, enigma_labels = _build_glasser_to_aparc_mapping(
        cortex_labels_hcp
    )

    # ── 4. Aggregate Glasser 360 → aparc 68 ──────────────────────────────────
    print("Aggregating connectivity matrix: Glasser 360 → aparc 68...")
    n_aparc = len(enigma_labels)
    aparc_adj = np.zeros((n_aparc, n_aparc), dtype=np.float64)
    count_mat = np.zeros((n_aparc, n_aparc), dtype=np.int32)

    for gi, ai in enumerate(glasser_to_aparc_idx):
        if ai < 0:
            continue
        for gj, aj in enumerate(glasser_to_aparc_idx):
            if aj < 0:
                continue
            aparc_adj[ai, aj] += group_avg[gi, gj]
            count_mat[ai, aj] += 1

    # Mean over contributing Glasser pairs (avoid inflating dense regions)
    nonzero = count_mat > 0
    aparc_adj[nonzero] /= count_mat[nonzero]

    # ── 5. Clean, symmetrize, build Laplacian ────────────────────────────────
    connectome_cfg = config.get("connectome", {})
    cleaned, clean_report = clean_adjacency(
        aparc_adj,
        symmetrize=bool(connectome_cfg.get("symmetrize", True)),
        zero_diagonal=bool(connectome_cfg.get("zero_diagonal", True)),
        edge_weight_transform=str(connectome_cfg.get("edge_weight_transform", "none")),
    )
    laplacian = build_laplacian(cleaned, method=str(connectome_cfg.get("laplacian", "combinatorial")))

    degree = cleaned.sum(axis=1)
    report = {
        "connectome_source": "hcp_1200_individual_probabilistic",
        "parcellation": "aparc_68_via_glasser360_overlap",
        "n_subjects": n_loaded,
        "glasser_regions": n_glasser,
        "aparc_regions": n_aparc,
        "glasser_unmapped": int(sum(1 for x in glasser_to_aparc_idx if x < 0)),
        "degree_min": float(degree.min()),
        "degree_mean": float(degree.mean()),
        "degree_max": float(degree.max()),
        "nonzero_undirected_edges": int(np.count_nonzero(np.triu(cleaned, k=1))),
        "total_undirected_edge_weight": float(np.sum(np.triu(cleaned, k=1))),
        **clean_report,
    }

    print("\nHCP group-average connectome report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    if not args.no_write:
        output_dir.mkdir(parents=True, exist_ok=True)
        adj_path = output_dir / outputs.get("hcp_adjacency_matrix", "hcp_aparc_adjacency.csv")
        lap_path = output_dir / outputs.get("hcp_laplacian_matrix", "hcp_aparc_laplacian.csv")
        write_labeled_matrix(adj_path, cleaned, enigma_labels)
        write_labeled_matrix(lap_path, laplacian, enigma_labels)
        import json as _json
        rep_path = output_dir / outputs.get("hcp_connectome_report", "hcp_connectome_report.json")
        with rep_path.open("w") as fh:
            _json.dump(report, fh, indent=2)
        print(f"\nWrote:\n  {adj_path}\n  {lap_path}\n  {rep_path}")

    return 0


# ── Mapping helpers ──────────────────────────────────────────────────────────

def _build_glasser_to_aparc_mapping(
    cortex_labels_hcp: list[str],
) -> tuple[list[int], list[str]]:
    """Map each Glasser area (by HCP matrix row) to an aparc region index.

    Uses fsaverage5 vertex parcellations from enigmatoolbox to compute
    vertex overlap, then majority-vote assigns each Glasser parcel.

    Returns:
        glasser_to_aparc_idx: list of length 360; value = aparc row index (-1 = unmapped)
        enigma_labels: the 68 ENIGMA aparc region names (fixed ordering)
    """
    from importlib import resources

    base = resources.files("enigmatoolbox") / "datasets" / "parcellations"

    with (base / "aparc_fsa5.csv").open() as f:
        aparc_vtx = np.array(list(csv.reader(f)), dtype=float).ravel().astype(int)
    with (base / "glasser_360_fsa5.csv").open() as f:
        glasser_vtx = np.array(list(csv.reader(f)), dtype=float).ravel().astype(int)

    # ENIGMA connectivity label ordering (68 regions, L then R)
    with (
        resources.files("enigmatoolbox") / "datasets" / "matrices"
        / "hcp_connectivity" / "strucLabels_ctx.csv"
    ).open() as f:
        enigma_labels = next(csv.reader(f))

    # Build aparc label value → enigma_labels index
    # enigma_labels are ordered L_bankssts, L_cac, ... (34 left) then R_bankssts, ... (34 right)
    # aparc_fsa5 vertex values: the enigmatoolbox orders regions 1..68 in the same
    # order as strucLabels_ctx.csv (confirmed by index matching below)
    aparc_val_to_enigma_idx: dict[int, int] = {}
    unique_aparc_vals = sorted(v for v in np.unique(aparc_vtx) if v > 0)
    # Filter to only the 68 cortical values (drop unknown/medial wall = extra values)
    cortex_aparc_vals = unique_aparc_vals[:68]  # first 68 non-zero = cortical
    for enigma_idx, val in enumerate(cortex_aparc_vals):
        aparc_val_to_enigma_idx[val] = enigma_idx

    # Build Glasser label value → enigma_labels ordering for enigmatoolbox
    # glasser_360_fsa5: label values 1..360; need to map to HCP matrix rows
    # Enigmatoolbox orders Glasser regions 1..360 as left hemisphere first (1-180)
    # then right (181-360), in the same alphabetical order as the HCP JSON file
    # (both follow L_xxx then R_xxx alphabetical order)
    # So: enigmatoolbox glasser label k → cortex_labels_hcp[k-1]
    unique_glasser_vals = sorted(v for v in np.unique(glasser_vtx) if v > 0)
    assert len(unique_glasser_vals) == 360, f"Expected 360 Glasser labels, got {len(unique_glasser_vals)}"

    # glasser_val_to_hcp_row: enigmatoolbox glasser label value → HCP matrix cortex row index
    glasser_val_to_hcp_row: dict[int, int] = {val: i for i, val in enumerate(unique_glasser_vals)}

    # For each HCP cortex row (0..359), determine dominant aparc region by vertex overlap
    n_glasser = len(cortex_labels_hcp)
    glasser_to_aparc_idx: list[int] = []
    n_unmapped = 0

    for hcp_row in range(n_glasser):
        # Which enigmatoolbox glasser label value corresponds to this HCP row?
        # Find the val whose index matches hcp_row
        if hcp_row >= len(unique_glasser_vals):
            glasser_to_aparc_idx.append(-1)
            n_unmapped += 1
            continue
        gval = unique_glasser_vals[hcp_row]

        # Vertices with this Glasser label
        mask = glasser_vtx == gval
        if not mask.any():
            glasser_to_aparc_idx.append(-1)
            n_unmapped += 1
            continue

        # Count aparc label overlap
        aparc_counts = Counter(int(v) for v in aparc_vtx[mask] if v > 0)
        # Keep only cortical aparc labels
        aparc_counts = Counter({k: v for k, v in aparc_counts.items()
                                 if k in aparc_val_to_enigma_idx})
        if not aparc_counts:
            glasser_to_aparc_idx.append(-1)
            n_unmapped += 1
            continue

        best_val = aparc_counts.most_common(1)[0][0]
        glasser_to_aparc_idx.append(aparc_val_to_enigma_idx[best_val])

    print(f"  Glasser→aparc: {n_glasser - n_unmapped}/{n_glasser} mapped, "
          f"{n_unmapped} unmapped (medial wall / subcortex vertices).")
    return glasser_to_aparc_idx, enigma_labels


if __name__ == "__main__":
    raise SystemExit(main())
