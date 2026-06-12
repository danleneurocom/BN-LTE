"""Biological graph constraints for BN-LTE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


LAYER_ORDER = {
    "root": 0,
    "fluid": 1,
    "pathology": 2,
    "neurodegeneration": 3,
    "clinical": 4,
}


@dataclass(frozen=True)
class VariableSpec:
    """Metadata needed to constrain a candidate causal variable."""

    name: str
    layer: str
    role: str = "intermediate"
    description: str = ""

    def __post_init__(self) -> None:
        if self.layer not in LAYER_ORDER:
            raise ValueError(f"Unknown variable layer for {self.name!r}: {self.layer!r}")


class CausalConstraints:
    """Root/sink/layer constraints for baseline-to-rate causal models."""

    def __init__(self, variable_specs: Iterable[VariableSpec]):
        self.specs = {spec.name: spec for spec in variable_specs}

    def spec(self, name: str) -> VariableSpec:
        try:
            return self.specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown variable in causal constraints: {name}") from exc

    def is_root(self, name: str) -> bool:
        spec = self.spec(name)
        return spec.layer == "root" or spec.role == "root"

    def is_sink(self, name: str) -> bool:
        spec = self.spec(name)
        return spec.layer == "clinical" or spec.role == "sink"

    def can_have_incoming(self, child: str) -> bool:
        return not self.is_root(child)

    def can_have_outgoing(self, parent: str) -> bool:
        return not self.is_sink(parent)

    def can_parent(self, parent: str, child: str) -> bool:
        """Return True when a directed baseline parent is biologically admissible."""

        if parent == child:
            return False
        if parent not in self.specs or child not in self.specs:
            return False
        if not self.can_have_outgoing(parent):
            return False
        if not self.can_have_incoming(child):
            return False

        parent_spec = self.specs[parent]
        child_spec = self.specs[child]
        parent_rank = LAYER_ORDER[parent_spec.layer]
        child_rank = LAYER_ORDER[child_spec.layer]

        if parent_rank < child_rank:
            return True
        if parent_rank > child_rank:
            return False

        return self._same_layer_allowed(parent, child, parent_spec.layer)

    def candidate_parents(self, child: str, feature_names: Iterable[str]) -> list[str]:
        return [name for name in feature_names if self.can_parent(name, child)]

    @staticmethod
    def _same_layer_allowed(parent: str, child: str, layer: str) -> bool:
        if layer == "root":
            return False
        if layer == "clinical":
            return False
        if layer == "fluid":
            return _fluid_order(parent) < _fluid_order(child)
        if layer == "pathology":
            if _is_amyloid(parent) and _is_tau(child):
                return True
            return False
        if layer == "neurodegeneration":
            return False
        return False

    def report(self) -> dict[str, object]:
        layer_counts: dict[str, int] = {}
        for spec in self.specs.values():
            layer_counts[spec.layer] = layer_counts.get(spec.layer, 0) + 1
        return {
            "variable_count": len(self.specs),
            "layer_counts": layer_counts,
            "root_nodes": sorted(name for name in self.specs if self.is_root(name)),
            "sink_nodes": sorted(name for name in self.specs if self.is_sink(name)),
        }


class CausalOrderingConstraints(CausalConstraints):
    """Exploratory constraints for baseline-to-future-rate event ordering.

    The default constraints encode a conservative A/T/N cascade and therefore
    prohibit neurodegeneration-to-pathology candidates. Event-ordering analyses
    need that direction to remain testable because the temporal direction is
    baseline value -> future rate, not simultaneous DAG structure.
    """

    def can_parent(self, parent: str, child: str) -> bool:
        if parent == child:
            return False
        if parent not in self.specs or child not in self.specs:
            return False
        if not self.can_have_outgoing(parent):
            return False
        if not self.can_have_incoming(child):
            return False
        parent_spec = self.specs[parent]
        child_spec = self.specs[child]
        if parent_spec.layer == "clinical":
            return False
        if child_spec.layer == "root":
            return False
        return True

    def report(self) -> dict[str, object]:
        report = super().report()
        report["constraint_mode"] = "causal_ordering_exploratory"
        report["note"] = "Allows A/T/N cross-direction baseline predictors for future-rate event-ordering tests."
        return report


def default_variable_specs(feature_names: Iterable[str], target_names: Iterable[str] = ()) -> list[VariableSpec]:
    """Infer conservative variable layers from names used by the data builder."""

    specs: list[VariableSpec] = []
    for name in list(feature_names) + list(target_names):
        specs.append(VariableSpec(name=name, layer=infer_layer(name), role=infer_role(name)))
    seen = set()
    deduped = []
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        deduped.append(spec)
    return deduped


def infer_layer(name: str) -> str:
    text = name.lower()
    if text in {"age_years", "sex_female", "education_years", "apoe4_dose"}:
        return "root"
    if any(token in text for token in ("plasma", "pt217", "nfl", "gfap", "ab42")):
        return "fluid"
    if any(token in text for token in ("amyloid", "tau")):
        return "pathology"
    if any(token in text for token in ("mri", "volume", "thickness", "hippocampus", "amygdala", "atrophy", "ashs", "ventricle")):
        return "neurodegeneration"
    if any(token in text for token in ("adas", "mmse", "ravlt", "avlt", "cdrsb", "diagnosis")):
        return "clinical"
    return "pathology"


def infer_role(name: str) -> str:
    layer = infer_layer(name)
    if layer == "root":
        return "root"
    if layer == "clinical":
        return "sink"
    return "intermediate"


def _is_amyloid(name: str) -> bool:
    text = name.lower()
    return "amyloid" in text or "ab42" in text


def _is_tau(name: str) -> bool:
    text = name.lower()
    return "tau" in text or "pt217" in text


def _fluid_order(name: str) -> int:
    text = name.lower()
    if "ab42" in text or "amyloid" in text:
        return 0
    if "pt217" in text or "ptau" in text:
        return 1
    if "gfap" in text:
        return 2
    if "nfl" in text:
        return 3
    return 4
