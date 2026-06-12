"""
Connectome loading, alignment, cleaning, and Laplacian construction.
Builds the ENIGMA network matrices: A, the adjacency/connectivity matrix where each entry is the structural connection
strength between two brain regions, and D, the degree matrix where each diagonal entry is the total connectivity 
strength of one region. From these, we compute the graph Laplacian L = D - A, 
which is the network-spread operator used by NDM/FKPP.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .io_adni import resolve_project_path


@dataclass
class ConnectomePreparationResult:
    adjacency: np.ndarray
    laplacian: np.ndarray
    labels: list[str]
    mapping_rows: list[dict[str, Any]]
    degree_rows: list[dict[str, Any]]
    report: dict[str, Any]


def load_enigma_structural_connectome(parcellation: str = "aparc") -> tuple[np.ndarray, list[str]]:
    """Load ENIGMA/HCP cortical structural connectivity and labels."""
    base = resources.files("enigmatoolbox") / "datasets" / "matrices" / "hcp_connectivity"
    if parcellation == "aparc":
        matrix_name = "strucMatrix_ctx.csv"
        labels_name = "strucLabels_ctx.csv"
    else:
        matrix_name = f"strucMatrix_ctx_{parcellation}.csv"
        labels_name = f"strucLabels_ctx_{parcellation}.csv"

    with (base / matrix_name).open("r", encoding="utf-8") as handle:
        matrix = np.loadtxt(handle, delimiter=",")
    with (base / labels_name).open("r", encoding="utf-8") as handle:
        labels = next(csv.reader(handle))
    return matrix, [label.strip() for label in labels if label.strip()]


def read_mapping(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def prepare_connectome(config: dict[str, Any], project_root: str | Path) -> ConnectomePreparationResult:
    connectome_config = config.get("connectome", {})
    region_config = config.get("region_mapping", {})

    source = connectome_config.get("source", "enigma")
    parcellation = connectome_config.get("parcellation", "aparc")
    if source != "enigma":
        raise ValueError(f"Unsupported connectome source for this step: {source}")
    if connectome_config.get("include_subcortex", False):
        raise ValueError("Step 6 currently prepares cortical-only ENIGMA aparc connectivity.")

    full_adjacency, full_labels = load_enigma_structural_connectome(parcellation)
    mapping_path = resolve_project_path(region_config["mapping_file"], project_root)
    mapping_rows = read_mapping(mapping_path)
    ordered_mapping = sorted(mapping_rows, key=lambda row: int(row["enigma_index"]))

    indices = [int(row["enigma_index"]) for row in ordered_mapping]
    mapped_labels = [row["enigma_label"] for row in ordered_mapping]
    for idx, mapped_label in zip(indices, mapped_labels):
        if idx < 0 or idx >= len(full_labels):
            raise IndexError(f"Mapping index {idx} is outside ENIGMA label range.")
        if full_labels[idx] != mapped_label:
            raise ValueError(f"Mapping label mismatch at index {idx}: {mapped_label} != {full_labels[idx]}")

    raw_subset = np.asarray(full_adjacency[np.ix_(indices, indices)], dtype=float)
    cleaned, clean_report = clean_adjacency(
        raw_subset,
        symmetrize=bool(connectome_config.get("symmetrize", True)),
        zero_diagonal=bool(connectome_config.get("zero_diagonal", True)),
        edge_weight_transform=str(connectome_config.get("edge_weight_transform", "none")),
    )
    laplacian = build_laplacian(cleaned, method=str(connectome_config.get("laplacian", "combinatorial")))
    labels = mapped_labels
    degree = cleaned.sum(axis=1)

    degree_rows = [
        {
            "enigma_index": int(row["enigma_index"]),
            "enigma_label": row["enigma_label"],
            "adni_tau_column": row["adni_tau_column"],
            "degree": float(degree[pos]),
        }
        for pos, row in enumerate(ordered_mapping)
    ]

    report = {
        "connectome_source": source,
        "parcellation": parcellation,
        "include_subcortex": bool(connectome_config.get("include_subcortex", False)),
        "full_matrix_shape": list(full_adjacency.shape),
        "prepared_matrix_shape": list(cleaned.shape),
        "mapped_region_count": len(labels),
        "mapping_file": str(mapping_path),
        "symmetrize": bool(connectome_config.get("symmetrize", True)),
        "zero_diagonal": bool(connectome_config.get("zero_diagonal", True)),
        "edge_weight_transform": str(connectome_config.get("edge_weight_transform", "none")),
        "laplacian": str(connectome_config.get("laplacian", "combinatorial")),
        "label_order_matches_mapping": labels == mapped_labels,
        "row_sums_laplacian_max_abs": float(np.max(np.abs(laplacian.sum(axis=1)))) if laplacian.size else 0.0,
        "degree_min": float(np.min(degree)) if degree.size else 0.0,
        "degree_mean": float(np.mean(degree)) if degree.size else 0.0,
        "degree_max": float(np.max(degree)) if degree.size else 0.0,
        "nonzero_undirected_edges": int(np.count_nonzero(np.triu(cleaned, k=1))),
        "total_undirected_edge_weight": float(np.sum(np.triu(cleaned, k=1))),
    }
    report.update(clean_report)

    return ConnectomePreparationResult(
        adjacency=cleaned,
        laplacian=laplacian,
        labels=labels,
        mapping_rows=ordered_mapping,
        degree_rows=degree_rows,
        report=report,
    )


def clean_adjacency(
    adjacency: np.ndarray,
    *,
    symmetrize: bool,
    zero_diagonal: bool,
    edge_weight_transform: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(adjacency, dtype=float).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {matrix.shape}.")

    report = {
        "nan_count_before": int(np.isnan(matrix).sum()),
        "negative_edge_count_before": int(np.sum(matrix < 0)),
        "diagonal_max_abs_before": float(np.max(np.abs(np.diag(np.nan_to_num(matrix))))),
        "symmetry_error_max_abs_before": float(np.nanmax(np.abs(matrix - matrix.T))),
    }

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(matrix < 0):
        raise ValueError("Adjacency matrix contains negative edge weights after NaN cleanup.")

    if edge_weight_transform == "none":
        pass
    elif edge_weight_transform == "log1p":
        matrix = np.log1p(matrix)
    else:
        raise ValueError(f"Unsupported edge_weight_transform: {edge_weight_transform}")

    if symmetrize:
        matrix = (matrix + matrix.T) / 2.0
    if zero_diagonal:
        np.fill_diagonal(matrix, 0.0)

    report.update(
        {
            "nan_count_after": int(np.isnan(matrix).sum()),
            "negative_edge_count_after": int(np.sum(matrix < 0)),
            "diagonal_max_abs_after": float(np.max(np.abs(np.diag(matrix)))),
            "symmetry_error_max_abs_after": float(np.max(np.abs(matrix - matrix.T))),
            "edge_weight_min": float(np.min(matrix)),
            "edge_weight_mean": float(np.mean(matrix)),
            "edge_weight_max": float(np.max(matrix)),
        }
    )
    return matrix, report


def build_laplacian(adjacency: np.ndarray, method: str = "combinatorial") -> np.ndarray:
    if method != "combinatorial":
        raise ValueError(f"Unsupported Laplacian method: {method}")
    degree = np.diag(adjacency.sum(axis=1))
    return degree - adjacency


def write_connectome_outputs(
    result: ConnectomePreparationResult,
    config: dict[str, Any],
    project_root: str | Path,
) -> dict[str, Path]:
    output_dir = resolve_project_path(config["paths"]["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = config.get("outputs", {})

    paths = {
        "adjacency_matrix": output_dir / outputs.get("adjacency_matrix", "enigma_aparc_adjacency.csv"),
        "laplacian_matrix": output_dir / outputs.get("laplacian_matrix", "enigma_aparc_laplacian.csv"),
        "connectome_labels": output_dir / outputs.get("connectome_labels", "enigma_aparc_labels.csv"),
        "connectome_degrees": output_dir / outputs.get("connectome_degrees", "enigma_aparc_degrees.csv"),
        "connectome_report": output_dir / outputs.get("connectome_report", "connectome_report.json"),
    }

    write_labeled_matrix(paths["adjacency_matrix"], result.adjacency, result.labels)
    write_labeled_matrix(paths["laplacian_matrix"], result.laplacian, result.labels)
    write_label_table(paths["connectome_labels"], result.mapping_rows)
    write_table(paths["connectome_degrees"], result.degree_rows)
    with paths["connectome_report"].open("w", encoding="utf-8") as handle:
        json.dump(result.report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return paths


def write_labeled_matrix(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["region"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{value:.12g}" for value in row])


def write_label_table(path: Path, mapping_rows: list[dict[str, Any]]) -> None:
    rows = [
        {
            "matrix_index": pos,
            "enigma_index": int(row["enigma_index"]),
            "enigma_label": row["enigma_label"],
            "hemisphere": row["hemisphere"],
            "aparc_region": row["aparc_region"],
            "adni_tau_column": row["adni_tau_column"],
        }
        for pos, row in enumerate(mapping_rows)
    ]
    write_table(path, rows)


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = union_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def union_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    return fieldnames
