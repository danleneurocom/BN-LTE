"""Region mapping between ADNI Berkeley tau columns and connectome labels."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .io_adni import resolve_project_path


ADNI_CTX_PREFIX = "CTX_"
ADNI_SUVR_SUFFIX = "_SUVR"
HEMISPHERE_TO_ADNI = {"L": "LH", "R": "RH"}
HEMISPHERE_TO_NAME = {"L": "left", "R": "right"}


@dataclass
class RegionMappingResult:
    mapping_rows: list[dict[str, Any]]
    report_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def load_enigma_aparc_labels() -> list[str]:
    """Load ENIGMA aparc cortical structural-connectivity labels.

    ENIGMA's public API reads the same bundled CSV through
    ``load_sc(parcellation="aparc")``. Reading this tiny label file directly
    avoids loading the full connectivity matrix during the mapping step.
    """
    label_resource = (
        resources.files("enigmatoolbox")
        / "datasets"
        / "matrices"
        / "hcp_connectivity"
        / "strucLabels_ctx.csv"
    )
    with label_resource.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        labels = next(reader)
    return [label.strip() for label in labels if label.strip()]


def load_csv_header(path: str | Path) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        return next(reader)


def enigma_label_to_adni_column(label: str) -> tuple[str, str, str]:
    """Return ``(adni_column, hemisphere, aparc_region)`` for an ENIGMA label."""
    if "_" not in label:
        raise ValueError(f"Unexpected ENIGMA aparc label format: {label}")
    hemi, region = label.split("_", 1)
    if hemi not in HEMISPHERE_TO_ADNI:
        raise ValueError(f"Unexpected ENIGMA hemisphere prefix in label: {label}")
    adni_hemi = HEMISPHERE_TO_ADNI[hemi]
    adni_region = region.upper()
    return f"{ADNI_CTX_PREFIX}{adni_hemi}_{adni_region}{ADNI_SUVR_SUFFIX}", HEMISPHERE_TO_NAME[hemi], region


def build_adni_enigma_aparc_mapping(
    tau_observations_path: str | Path,
    enigma_labels: list[str] | None = None,
) -> RegionMappingResult:
    """Build and validate the ADNI tau column to ENIGMA aparc label mapping."""
    labels = enigma_labels or load_enigma_aparc_labels()
    tau_columns = load_csv_header(tau_observations_path)
    tau_column_set = set(tau_columns)
    adni_ctx_lr_columns = [
        col
        for col in tau_columns
        if (col.startswith("CTX_LH_") or col.startswith("CTX_RH_")) and col.endswith(ADNI_SUVR_SUFFIX)
    ]

    mapping_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    missing_adni_columns: list[str] = []

    for index, label in enumerate(labels):
        adni_column, hemisphere, aparc_region = enigma_label_to_adni_column(label)
        status = "matched" if adni_column in tau_column_set else "missing_adni_column"
        if status != "matched":
            missing_adni_columns.append(adni_column)
        row = {
            "enigma_index": index,
            "enigma_label": label,
            "hemisphere": hemisphere,
            "aparc_region": aparc_region,
            "adni_tau_column": adni_column,
            "status": status,
        }
        report_rows.append(row)
        if status == "matched":
            mapping_rows.append(row)

    duplicate_enigma_labels = sorted(label for label in set(labels) if labels.count(label) > 1)
    duplicate_adni_columns = sorted(
        col
        for col in {row["adni_tau_column"] for row in mapping_rows}
        if sum(1 for row in mapping_rows if row["adni_tau_column"] == col) > 1
    )
    unmapped_adni_lr_columns = sorted(set(adni_ctx_lr_columns) - {row["adni_tau_column"] for row in mapping_rows})

    summary = {
        "connectome_source": "enigma",
        "parcellation": "aparc",
        "enigma_label_count": len(labels),
        "adni_left_right_ctx_suvr_columns": len(adni_ctx_lr_columns),
        "matched_regions": len(mapping_rows),
        "missing_adni_columns": len(missing_adni_columns),
        "duplicate_enigma_labels": len(duplicate_enigma_labels),
        "duplicate_adni_columns": len(duplicate_adni_columns),
        "unmapped_adni_left_right_ctx_suvr_columns": len(unmapped_adni_lr_columns),
        "is_complete": (
            len(mapping_rows) == len(labels)
            and not missing_adni_columns
            and not duplicate_enigma_labels
            and not duplicate_adni_columns
        ),
        "missing_adni_column_names": missing_adni_columns,
        "duplicate_enigma_label_names": duplicate_enigma_labels,
        "duplicate_adni_column_names": duplicate_adni_columns,
        "unmapped_adni_left_right_ctx_suvr_column_names": unmapped_adni_lr_columns,
    }
    return RegionMappingResult(mapping_rows=mapping_rows, report_rows=report_rows, summary=summary)


def write_mapping_outputs(
    result: RegionMappingResult,
    config: dict[str, Any],
    project_root: str | Path,
) -> dict[str, Path]:
    region_config = config.get("region_mapping", {})
    output_config = config.get("outputs", {})

    mapping_path = resolve_project_path(region_config["mapping_file"], project_root)
    output_dir = resolve_project_path(config["paths"]["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / output_config.get("region_mapping_report", "region_mapping_report.csv")
    summary_path = output_dir / "region_mapping_summary.json"

    write_csv(mapping_path, result.mapping_rows)
    write_csv(report_path, result.report_rows)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {
        "mapping_file": mapping_path,
        "region_mapping_report": report_path,
        "region_mapping_summary": summary_path,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
