from __future__ import annotations

from pathlib import Path

import numpy as np

from bayesian_network_scm.atn_data import build_atn_rate_dataset
from bayesian_network_scm.constraints import CausalConstraints, CausalOrderingConstraints, default_variable_specs


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_atn_rate_dataset_has_expected_modalities_and_source_intervals() -> None:
    dataset = build_atn_rate_dataset(PROJECT_ROOT, max_date_distance_days=365)

    assert dataset.pair_count > 700
    assert dataset.report["rows"]["amyloid"] > 500
    assert dataset.report["rows"]["picsl_ashs"] > 500
    assert dataset.report["rows"]["all_main_modalities"] > 350
    assert dataset.target_time_years is not None

    required_targets = {
        "amyloid_rate:centiloids",
        "tau_rate:meta_temporal",
        "atrophy_rate:brain_volume",
        "ashs_rate:left_hippocampus",
        "mri_thickness_rate:L_entorhinal",
        "cognitive_rate:adas13",
    }
    assert required_targets.issubset(set(dataset.target_names))
    assert not any("bsi" in name.lower() for name in dataset.target_names)

    amyloid_idx = dataset.target_index("amyloid_rate:centiloids")
    tau_idx = dataset.target_index("tau_rate:meta_temporal")
    amyloid_intervals = dataset.target_time_years[:, amyloid_idx]
    tau_intervals = dataset.target_time_years[:, tau_idx]
    finite = np.isfinite(amyloid_intervals) & np.isfinite(tau_intervals)

    assert np.count_nonzero(finite) > 500
    assert np.nanmedian(np.abs(amyloid_intervals[finite] - tau_intervals[finite])) > 0.01


def test_atn_features_use_picsl_hippocampal_vulnerability_with_good_coverage() -> None:
    dataset = build_atn_rate_dataset(PROJECT_ROOT, max_date_distance_days=365)

    hippocampus = dataset.feature_matrix[:, dataset.feature_index("mri_hippocampus_total_volume")]
    vulnerability = dataset.feature_matrix[:, dataset.feature_index("mri_hippocampus_vulnerability")]
    interaction = dataset.feature_matrix[
        :, dataset.feature_index("interaction:amyloid_centiloids_x_mri_hippocampus_vulnerability")
    ]

    assert np.mean(np.isfinite(hippocampus)) > 0.80
    assert np.mean(np.isfinite(interaction)) > 0.70
    finite = np.isfinite(hippocampus) & np.isfinite(vulnerability)
    assert np.count_nonzero(finite) > 600
    assert np.allclose(vulnerability[finite], -hippocampus[finite])


def test_causal_ordering_constraints_keep_reverse_atn_edges_testable() -> None:
    names = [
        "amyloid_centiloids",
        "tau_meta_temporal",
        "mri_hippocampus_vulnerability",
        "adas13",
    ]
    targets = ["tau_rate:meta_temporal", "atrophy_rate:brain_volume", "cognitive_rate:adas13"]
    specs = default_variable_specs(names, targets)
    conservative = CausalConstraints(specs)
    exploratory = CausalOrderingConstraints(specs)

    assert not conservative.can_parent("mri_hippocampus_vulnerability", "tau_rate:meta_temporal")
    assert exploratory.can_parent("mri_hippocampus_vulnerability", "tau_rate:meta_temporal")
    assert exploratory.can_parent("tau_meta_temporal", "atrophy_rate:brain_volume")
    assert not exploratory.can_parent("adas13", "tau_rate:meta_temporal")
