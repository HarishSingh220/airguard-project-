"""Unit tests for CPCB parsing safeguards; no network or database access."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

INGESTION_DIR = Path(__file__).resolve().parents[1] / "ingestion"
sys.path.insert(0, str(INGESTION_DIR))

import fetch_cpcb  # noqa: E402


class FetchCPCBTests(unittest.TestCase):
    def setUp(self):
        self.station_lookup = {
            "Anand Vihar, Delhi - DPCC": "Anand Vihar",
        }

    def test_empty_response_has_upsert_key_columns(self):
        frame = fetch_cpcb.build_dataframe([], self.station_lookup)
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), fetch_cpcb.EXPECTED_COLUMNS)
        for column in fetch_cpcb.PRIMARY_KEY:
            self.assertIn(column, frame.columns)

    def test_valid_pollutant_average_is_mapped(self):
        records = [{
            "station": "Anand Vihar, Delhi - DPCC",
            "pollutant_id": "PM2.5",
            "last_update": "12-11-2026 10:30:00",
            "pollutant_avg": "12.5",
            "pollutant_min": "10",
            "pollutant_max": "15",
            "pollutant_unit": "ug/m3",
        }]
        frame = fetch_cpcb.build_dataframe(records, self.station_lookup)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "station"], "Anand Vihar")
        self.assertEqual(frame.loc[0, "pollutant"], "PM25")
        self.assertEqual(frame.loc[0, "value_avg"], 12.5)

    def test_non_finite_values_become_null(self):
        records = [{
            "station": "Anand Vihar, Delhi - DPCC",
            "pollutant_id": "PM10",
            "last_update": "12-11-2026 10:30:00",
            "pollutant_avg": "inf",
            "pollutant_min": "NaN",
            "pollutant_max": "-inf",
        }]
        frame = fetch_cpcb.build_dataframe(records, self.station_lookup)
        self.assertEqual(len(frame), 1)
        for column in ("value_avg", "value_min", "value_max"):
            self.assertTrue(pd.isna(frame.loc[0, column]))

    def test_duplicate_database_keys_can_be_collapsed(self):
        record = {
            "station": "Anand Vihar, Delhi - DPCC",
            "pollutant_id": "PM2.5",
            "last_update": "12-11-2026 10:30:00",
            "pollutant_avg": "12.5",
        }
        frame = fetch_cpcb.build_dataframe([record, {**record, "pollutant_avg": "13"}], self.station_lookup)
        self.assertEqual(len(frame), 2)
        deduplicated = frame.drop_duplicates(subset=fetch_cpcb.PRIMARY_KEY, keep="last")
        self.assertEqual(len(deduplicated), 1)
        self.assertEqual(deduplicated.iloc[0]["value_avg"], 13.0)

    def test_diagnostic_omits_unrelated_response_fields(self):
        diagnostic = fetch_cpcb._record_diagnostic([{
            "station": "Anand Vihar, Delhi - DPCC",
            "pollutant_id": "PM2.5",
            "last_update": "12-11-2026 10:30:00",
            "pollutant_avg": "NA",
            "api_secret": "must-not-be-logged",
        }])
        self.assertEqual(set(diagnostic), {"record_keys", "sample_pollutant_fields"})
        self.assertNotIn("must-not-be-logged", str(diagnostic))


if __name__ == "__main__":
    unittest.main()
