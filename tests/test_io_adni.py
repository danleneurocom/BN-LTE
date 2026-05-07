from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from spread_toolbox.io_adni import build_longitudinal_tau_cohort, parse_date


class AdniCohortTests(unittest.TestCase):
    def test_parse_date_accepts_adni_formats(self) -> None:
        self.assertEqual(str(parse_date("2020-01-02")), "2020-01-02")
        self.assertEqual(str(parse_date("01/02/2020")), "2020-01-02")
        self.assertEqual(str(parse_date("20200102")), "2020-01-02")
        self.assertIsNone(parse_date(""))

    def test_build_longitudinal_tau_cohort_from_toy_study_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adni_dir = root / "ADNI"
            self._write_csv(
                adni_dir / "PET_Image_Analysis" / "UCBERKELEY_TAU_6MM.csv",
                [
                    {
                        "RID": "1",
                        "PTID": "001_S_0001",
                        "TRACER": "FTP",
                        "SCANDATE": "2020-01-01",
                        "qc_flag": "2",
                        "LONIUID": "I100",
                        "META_TEMPORAL_SUVR": "1.0",
                        "CTX_LH_ENTORHINAL_SUVR": "1.1",
                    },
                    {
                        "RID": "1",
                        "PTID": "001_S_0001",
                        "TRACER": "FTP",
                        "SCANDATE": "2021-01-01",
                        "qc_flag": "2",
                        "LONIUID": "I101",
                        "META_TEMPORAL_SUVR": "1.2",
                        "CTX_LH_ENTORHINAL_SUVR": "1.3",
                    },
                    {
                        "RID": "1",
                        "PTID": "001_S_0001",
                        "TRACER": "FTP",
                        "SCANDATE": "2022-01-01",
                        "qc_flag": "2",
                        "LONIUID": "I102",
                        "META_TEMPORAL_SUVR": "1.5",
                        "CTX_LH_ENTORHINAL_SUVR": "1.6",
                    },
                    {
                        "RID": "2",
                        "PTID": "001_S_0002",
                        "TRACER": "FTP",
                        "SCANDATE": "2020-01-01",
                        "qc_flag": "0",
                        "LONIUID": "I200",
                        "META_TEMPORAL_SUVR": "1.0",
                        "CTX_LH_ENTORHINAL_SUVR": "1.1",
                    },
                ],
            )
            self._write_csv(adni_dir / "PET_Image_Acquisition" / "TAUMETA.csv", [])
            self._write_csv(adni_dir / "PET_Image_Quality" / "TAUQC.csv", [])
            self._write_csv(
                adni_dir / "Enrollment" / "ROSTER.csv",
                [{"RID": "1", "PTID": "001_S_0001"}],
            )
            self._write_csv(
                adni_dir / "PET_Image_Analysis" / "UCBERKELEY_AMY_6MM.csv",
                [{"RID": "1", "SCANDATE": "2020-02-01", "qc_flag": "2"}],
            )
            self._write_csv(
                adni_dir / "Diagnosis" / "DXSUM.csv",
                [{"RID": "1", "EXAMDATE": "2020-01-15", "DIAGNOSIS": "2"}],
            )
            self._write_csv(
                adni_dir / "Subject_Demographics" / "PTDEMOG.csv",
                [{"RID": "1", "PTGENDER": "2", "PTEDUCAT": "16"}],
            )
            self._write_csv(
                adni_dir / "Genetic_APOE" / "APOERES.csv",
                [{"RID": "1", "GENOTYPE": "3/4"}],
            )

            config = {
                "paths": {"adni_dir": "ADNI", "output_dir": "output"},
                "adni_files": {
                    "enrollment_roster": "Enrollment/ROSTER.csv",
                    "tau_analysis": "PET_Image_Analysis/UCBERKELEY_TAU_6MM.csv",
                    "tau_metadata": "PET_Image_Acquisition/TAUMETA.csv",
                    "tau_qc": "PET_Image_Quality/TAUQC.csv",
                    "amyloid_analysis": "PET_Image_Analysis/UCBERKELEY_AMY_6MM.csv",
                    "diagnosis": "Diagnosis/DXSUM.csv",
                    "demographics": "Subject_Demographics/PTDEMOG.csv",
                    "apoe": "Genetic_APOE/APOERES.csv",
                },
                "cohort": {
                    "min_tau_timepoints": 2,
                    "require_same_tracer": True,
                    "tau_pass_values": ["2"],
                    "allow_partial_tau_qc": False,
                },
            }

            result = build_longitudinal_tau_cohort(config, root)

            self.assertEqual(len(result.cohort_rows), 1)
            self.assertEqual(len(result.forecast_pair_rows), 2)
            self.assertEqual(len(result.tau_observation_rows), 3)
            self.assertEqual(result.row_counts["tau_failed_qc"], 1)
            self.assertEqual(result.cohort_rows[0]["dx_nearest_baseline"], "MCI")
            self.assertEqual(result.cohort_rows[0]["apoe4_dose"], "1")

    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
