from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from spread_toolbox.region_mapping import (
    build_adni_enigma_aparc_mapping,
    enigma_label_to_adni_column,
)


class RegionMappingTests(unittest.TestCase):
    def test_enigma_label_to_adni_column(self) -> None:
        self.assertEqual(
            enigma_label_to_adni_column("L_entorhinal"),
            ("CTX_LH_ENTORHINAL_SUVR", "left", "entorhinal"),
        )
        self.assertEqual(
            enigma_label_to_adni_column("R_rostralanteriorcingulate"),
            ("CTX_RH_ROSTRALANTERIORCINGULATE_SUVR", "right", "rostralanteriorcingulate"),
        )

    def test_build_mapping_from_toy_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cohort_tau_observations.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "RID",
                        "CTX_LH_ENTORHINAL_SUVR",
                        "CTX_RH_ENTORHINAL_SUVR",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "RID": "1",
                        "CTX_LH_ENTORHINAL_SUVR": "1.1",
                        "CTX_RH_ENTORHINAL_SUVR": "1.2",
                    }
                )

            result = build_adni_enigma_aparc_mapping(path, ["L_entorhinal", "R_entorhinal"])

            self.assertTrue(result.summary["is_complete"])
            self.assertEqual(result.summary["matched_regions"], 2)
            self.assertEqual(result.mapping_rows[0]["enigma_index"], 0)
            self.assertEqual(result.mapping_rows[1]["adni_tau_column"], "CTX_RH_ENTORHINAL_SUVR")


if __name__ == "__main__":
    unittest.main()
