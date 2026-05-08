from __future__ import annotations

import unittest

from spread_toolbox.forecasting import (
    compute_aggregate_metrics,
    make_subject_train_validation_test_split,
    split_labels_by_index,
)


class ForecastingSplitTests(unittest.TestCase):
    def test_train_validation_test_split_keeps_subjects_disjoint(self) -> None:
        pairs = [{"RID": str(rid)} for rid in range(10) for _ in range(2)]
        split = make_subject_train_validation_test_split(
            pairs,
            validation_fraction=0.2,
            test_fraction=0.2,
            random_seed=7,
        )
        self.assertEqual(len(split.train_rids), 6)
        self.assertEqual(len(split.validation_rids), 2)
        self.assertEqual(len(split.test_rids), 2)

        labels = split_labels_by_index(split)
        labels_by_rid: dict[str, set[str]] = {}
        for index, pair in enumerate(pairs):
            labels_by_rid.setdefault(pair["RID"], set()).add(labels[index])

        self.assertTrue(all(len(labels_for_rid) == 1 for labels_for_rid in labels_by_rid.values()))
        self.assertEqual(set(split.train_rids).isdisjoint(split.validation_rids), True)
        self.assertEqual(set(split.train_rids).isdisjoint(split.test_rids), True)
        self.assertEqual(set(split.validation_rids).isdisjoint(split.test_rids), True)

    def test_aggregate_metrics_includes_validation_split(self) -> None:
        pair_metrics = [
            metric_row("m", "train", 1.0),
            metric_row("m", "validation", 2.0),
            metric_row("m", "test", 3.0),
        ]
        aggregate = compute_aggregate_metrics(pair_metrics)
        mae_splits = [row["split"] for row in aggregate if row["metric"] == "mae"]
        self.assertEqual(mae_splits, ["train", "validation", "test", "all"])

    def test_aggregate_metrics_groups_by_model(self) -> None:
        pair_metrics = [
            metric_row("a", "validation", 1.0),
            metric_row("b", "validation", 3.0),
        ]
        aggregate = compute_aggregate_metrics(pair_metrics)
        mae_rows = [row for row in aggregate if row["split"] == "validation" and row["metric"] == "mae"]
        self.assertEqual([row["model"] for row in mae_rows], ["a", "b"])
        self.assertEqual([row["median"] for row in mae_rows], [1.0, 3.0])


def metric_row(model: str, split: str, value: float) -> dict[str, float | str]:
    return {
        "model": model,
        "split": split,
        "mae": value,
        "rmse": value,
        "subject_spearman": value,
        "subject_pearson": value,
        "delta_spearman": value,
        "delta_pearson": value,
        "top5_overlap": value,
        "top10_overlap": value,
    }


if __name__ == "__main__":
    unittest.main()
