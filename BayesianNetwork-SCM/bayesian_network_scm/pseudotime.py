"""Explainable train-only disease pseudotime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .constraints import infer_layer


@dataclass
class PseudotimeModel:
    """Linear, train-only pseudotime with loadings and contributions."""

    feature_names: list[str]
    selected_feature_names: list[str]
    selected_indices: np.ndarray
    fill_values: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    component: np.ndarray
    score_lower: float
    score_upper: float
    explained_variance_ratio: float
    burden_correlation: float
    mode: str

    def transform(self, feature_matrix: np.ndarray, *, clip: bool = True) -> np.ndarray:
        selected = self._standardized_selected(feature_matrix)
        scores = np.einsum("ij,j->i", selected, self.component)
        denom = max(self.score_upper - self.score_lower, 1.0e-12)
        z = (scores - self.score_lower) / denom
        return np.clip(z, 0.0, 1.0) if clip else z

    def contributions(self, feature_matrix: np.ndarray) -> np.ndarray:
        selected = self._standardized_selected(feature_matrix)
        return selected * self.component[None, :]

    def subject_explanation(self, feature_matrix: np.ndarray, row_index: int, *, top_k: int = 5) -> list[dict[str, float | str]]:
        contributions = self.contributions(feature_matrix)[int(row_index)]
        order = np.argsort(-np.abs(contributions), kind="mergesort")[: int(top_k)]
        return [
            {
                "feature": self.selected_feature_names[int(index)],
                "contribution": float(contributions[int(index)]),
            }
            for index in order
        ]

    def loading_rows(self) -> list[dict[str, float | str]]:
        rows = [
            {
                "feature": name,
                "loading": float(loading),
                "abs_loading": abs(float(loading)),
                "layer": infer_layer(name),
            }
            for name, loading in zip(self.selected_feature_names, self.component, strict=True)
        ]
        return sorted(rows, key=lambda row: float(row["abs_loading"]), reverse=True)

    def report(self, feature_matrix: np.ndarray, metadata_rows: list[dict[str, object]] | None = None) -> dict[str, object]:
        z = self.transform(feature_matrix)
        report: dict[str, object] = {
            "mode": self.mode,
            "selected_feature_count": len(self.selected_feature_names),
            "selected_features": self.selected_feature_names,
            "explained_variance_ratio": self.explained_variance_ratio,
            "burden_correlation": self.burden_correlation,
            "score_lower": self.score_lower,
            "score_upper": self.score_upper,
            "z_min": float(np.min(z)) if z.size else float("nan"),
            "z_median": float(np.median(z)) if z.size else float("nan"),
            "z_max": float(np.max(z)) if z.size else float("nan"),
            "top_loadings": self.loading_rows()[:12],
        }
        if metadata_rows:
            report["diagnosis_ordering"] = diagnosis_ordering(z, metadata_rows)
        return report

    def _standardized_selected(self, feature_matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(feature_matrix, dtype=float)[:, self.selected_indices]
        filled = np.where(np.isfinite(values), values, self.fill_values[None, :])
        standardized = (filled - self.center[None, :]) / self.scale[None, :]
        return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)


def fit_pseudotime(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    train_indices: np.ndarray,
    *,
    mode: str = "tau_free",
    min_train_coverage: float = 0.5,
) -> PseudotimeModel:
    """Fit a transparent SVD pseudotime model on training rows only."""

    matrix = np.asarray(feature_matrix, dtype=float)
    train_indices = np.asarray(train_indices, dtype=int)
    if matrix.ndim != 2:
        raise ValueError("feature_matrix must be two-dimensional.")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names must match feature_matrix columns.")
    if train_indices.size < 3:
        raise ValueError("At least three training rows are required for pseudotime.")

    selected_indices = select_pseudotime_features(matrix, feature_names, train_indices, mode=mode, min_train_coverage=min_train_coverage)
    if selected_indices.size < 2:
        raise ValueError(f"Pseudotime mode {mode!r} selected fewer than two usable features.")

    train = matrix[train_indices[:, None], selected_indices[None, :]]
    fill_values = np.nanmedian(train, axis=0)
    global_fill = float(np.nanmedian(train)) if np.any(np.isfinite(train)) else 0.0
    fill_values = np.where(np.isfinite(fill_values), fill_values, global_fill)
    filled_train = np.where(np.isfinite(train), train, fill_values[None, :])
    center = np.mean(filled_train, axis=0)
    scale = np.std(filled_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    standardized_train = np.nan_to_num(
        (filled_train - center[None, :]) / scale[None, :],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    _, singular_values, vt = np.linalg.svd(standardized_train, full_matrices=False)
    component = np.asarray(vt[0], dtype=float)
    scores = np.einsum("ij,j->i", standardized_train, component)
    burden = disease_burden_score(filled_train, [feature_names[int(index)] for index in selected_indices])
    burden_corr = safe_correlation(scores, burden)
    if np.isfinite(burden_corr) and burden_corr < 0.0:
        component = -component
        scores = -scores
        burden_corr = -burden_corr

    lower = float(np.quantile(scores, 0.01))
    upper = float(np.quantile(scores, 0.99))
    if not np.isfinite(upper - lower) or upper <= lower:
        lower = float(np.min(scores))
        upper = float(np.max(scores))
    if not np.isfinite(upper - lower) or upper <= lower:
        upper = lower + 1.0

    variance = singular_values**2
    explained = float(variance[0] / np.sum(variance)) if np.sum(variance) > 0.0 else 0.0
    return PseudotimeModel(
        feature_names=list(feature_names),
        selected_feature_names=[feature_names[int(index)] for index in selected_indices],
        selected_indices=selected_indices,
        fill_values=np.asarray(fill_values, dtype=float),
        center=np.asarray(center, dtype=float),
        scale=np.asarray(scale, dtype=float),
        component=np.asarray(component, dtype=float),
        score_lower=lower,
        score_upper=upper,
        explained_variance_ratio=explained,
        burden_correlation=float(burden_corr),
        mode=mode,
    )


def select_pseudotime_features(
    matrix: np.ndarray,
    feature_names: list[str],
    train_indices: np.ndarray,
    *,
    mode: str,
    min_train_coverage: float,
) -> np.ndarray:
    selected = []
    train = matrix[np.asarray(train_indices, dtype=int)]
    for idx, name in enumerate(feature_names):
        if not allowed_for_mode(name, mode):
            continue
        column = train[:, idx]
        finite = np.isfinite(column)
        if float(np.mean(finite)) < float(min_train_coverage):
            continue
        if not np.any(finite) or np.nanstd(column[finite]) <= 1.0e-12:
            continue
        selected.append(idx)
    return np.asarray(selected, dtype=int)


def allowed_for_mode(name: str, mode: str) -> bool:
    text = name.lower()
    if mode == "global":
        return True
    if mode == "tau_free":
        return "tau" not in text and "pt217" not in text and "ptau" not in text
    if mode == "pt217_free":
        return "pt217" not in text and "ptau" not in text
    if mode == "clinical_free":
        return infer_layer(name) != "clinical"
    raise ValueError(f"Unknown pseudotime mode: {mode}")


def disease_burden_score(values: np.ndarray, names: Iterable[str]) -> np.ndarray:
    """Return an orientation-only burden proxy from filled unstandardized values."""

    names = list(names)
    standardized = standardize_columns(values)
    signs = np.asarray([burden_sign(name) for name in names], dtype=float)
    usable = signs != 0.0
    if np.any(usable):
        numerator = np.einsum("ij,j->i", standardized[:, usable], signs[usable])
        return numerator / max(float(np.sum(np.abs(signs[usable]))), 1.0)
    return np.mean(standardized, axis=1)


def burden_sign(name: str) -> float:
    text = name.lower()
    if any(token in text for token in ("tau", "pt217", "amyloid", "centiloid", "nfl", "gfap", "adas", "cdrsb")):
        return 1.0
    if any(token in text for token in ("mmse", "ravlt", "avlt", "volume", "thickness", "ab42_ab40")):
        return -1.0
    return 0.0


def standardize_columns(values: np.ndarray) -> np.ndarray:
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    return np.nan_to_num((values - center[None, :]) / scale[None, :], nan=0.0, posinf=0.0, neginf=0.0)


def diagnosis_ordering(z: np.ndarray, metadata_rows: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = {}
    for value, row in zip(z, metadata_rows, strict=True):
        label = str(row.get("dx_nearest_baseline", "") or "unknown")
        groups.setdefault(label, []).append(float(value))
    return {
        label: {
            "n": len(values),
            "mean_z": float(np.mean(values)),
            "median_z": float(np.median(values)),
        }
        for label, values in sorted(groups.items())
        if values
    }


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
