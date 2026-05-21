from __future__ import annotations

import numpy as np

from bayesian_network_scm.constraints import CausalConstraints, default_variable_specs
from bayesian_network_scm.data import MultimodalPairDataset
from bayesian_network_scm.dynamic_scm import fit_dynamic_scm
from bayesian_network_scm.pseudotime import fit_pseudotime
from bayesian_network_scm.reporting import make_subject_split


def test_constraints_enforce_roots_sinks_and_pathology_direction() -> None:
    names = [
        "age_years",
        "apoe4_dose",
        "plasma_pt217",
        "amyloid_summary_suvr",
        "tau_meta_temporal",
        "mri_hippocampus_volume",
        "adas13",
    ]
    constraints = CausalConstraints(default_variable_specs(names, ["tau_rate:meta_temporal", "cognitive_rate:adas13"]))

    assert constraints.can_parent("age_years", "tau_rate:meta_temporal")
    assert constraints.can_parent("amyloid_summary_suvr", "tau_rate:meta_temporal")
    assert constraints.can_parent("plasma_pt217", "tau_rate:meta_temporal")
    assert not constraints.can_parent("adas13", "tau_rate:meta_temporal")
    assert not constraints.can_parent("tau_meta_temporal", "amyloid_summary_suvr")
    assert not constraints.can_parent("mri_hippocampus_volume", "apoe4_dose")


def test_pseudotime_is_oriented_and_explainable() -> None:
    rng = np.random.default_rng(7)
    burden = np.linspace(0.0, 1.0, 80)
    feature_names = ["amyloid_summary_suvr", "mri_hippocampus_volume", "adas13", "mmse"]
    features = np.column_stack(
        [
            1.0 + burden + rng.normal(0.0, 0.03, burden.size),
            5.0 - burden + rng.normal(0.0, 0.03, burden.size),
            5.0 + 20.0 * burden + rng.normal(0.0, 0.3, burden.size),
            30.0 - 6.0 * burden + rng.normal(0.0, 0.2, burden.size),
        ]
    )
    pt = fit_pseudotime(features, feature_names, np.arange(80), mode="global")
    z = pt.transform(features)

    assert np.corrcoef(z, burden)[0, 1] > 0.95
    assert pt.loading_rows()[0]["feature"] in set(feature_names)
    assert len(pt.subject_explanation(features, 0, top_k=2)) == 2


def test_dynamic_scm_learns_temporal_parent_signal() -> None:
    dataset = synthetic_dynamic_dataset()
    split = make_subject_split(dataset.metadata_rows, validation_fraction=0.2, test_fraction=0.2, random_seed=3)
    pt = fit_pseudotime(dataset.feature_matrix, dataset.feature_names, split.train_indices, mode="global")
    fit = fit_dynamic_scm(
        dataset,
        pt,
        split.train_indices,
        target_names=["tau_rate:meta_temporal"],
        max_parents_per_target=4,
        n_knots=1,
        spline_degree=0,
        ridge_alphas=(0.01, 0.1, 1.0),
        cv_folds=3,
        edge_effect_threshold=0.05,
    )
    target_fit = fit.target_fits[0]
    rows = fit.edge_effect_rows()
    pt217_rows = [row for row in rows if row["parent"] == "plasma_pt217" and row["target"] == "tau_rate:meta_temporal"]

    assert "plasma_pt217" in target_fit.parent_names
    assert pt217_rows
    assert pt217_rows[0]["max_abs_effect"] > 0.05
    predicted = fit.predict_rates(dataset)[:, dataset.target_index("tau_rate:meta_temporal")]
    observed = dataset.target_rates[:, dataset.target_index("tau_rate:meta_temporal")]
    assert np.corrcoef(predicted, observed)[0, 1] > 0.8


def synthetic_dynamic_dataset() -> MultimodalPairDataset:
    rng = np.random.default_rng(11)
    n = 90
    burden = np.linspace(0.0, 1.0, n)
    age = 65.0 + 12.0 * burden + rng.normal(0.0, 1.0, n)
    apoe = rng.binomial(2, 0.35, n).astype(float)
    pt217 = 0.5 + 1.5 * burden + rng.normal(0.0, 0.05, n)
    amyloid = 1.0 + burden + rng.normal(0.0, 0.05, n)
    tau0 = 1.0 + 0.6 * burden + rng.normal(0.0, 0.03, n)
    adas = 5.0 + 15.0 * burden + rng.normal(0.0, 0.2, n)
    features = np.column_stack([age, apoe, pt217, amyloid, tau0, adas])
    feature_names = ["age_years", "apoe4_dose", "plasma_pt217", "amyloid_summary_suvr", "tau_meta_temporal", "adas13"]
    time = rng.uniform(0.8, 1.2, n)
    rate = 0.25 * pt217 + 0.08 * amyloid - 0.02 * tau0 + rng.normal(0.0, 0.01, n)
    baseline = tau0.reshape(-1, 1)
    observed = baseline + time[:, None] * rate[:, None]
    metadata = [
        {
            "RID": str(idx),
            "dx_nearest_baseline": "CN" if idx < 30 else "MCI" if idx < 60 else "AD",
            "target_time_years": float(time[idx]),
        }
        for idx in range(n)
    ]
    target_names = ["tau_rate:meta_temporal"]
    return MultimodalPairDataset(
        metadata_rows=metadata,
        feature_names=feature_names,
        feature_matrix=features,
        target_names=target_names,
        target_baseline=baseline,
        target_observed=observed,
        target_rates=rate.reshape(-1, 1),
        time_years=time,
        variable_specs=default_variable_specs(feature_names, target_names),
        report={},
    )
