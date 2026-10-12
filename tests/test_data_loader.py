"""
test_data_loader.py
-------------------
Unit tests for data_loader.py using Python's standard unittest module.
"""

import os
import unittest
import pandas as pd
from data_loader import load_dataset, get_dataset_overview


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.sample_csv = os.path.join("datasets", "sample_data.csv")
        self.sample_xlsx = os.path.join("datasets", "sample_data.xlsx")

    def test_load_valid_csv(self):
        """Test loading a real valid CSV file."""
        df, err = load_dataset(self.sample_csv)
        self.assertIsNone(err, "Error should be None for valid CSV")
        self.assertIsNotNone(df, "DataFrame should not be None")
        self.assertEqual(len(df), 15, "Sample dataset should contain 15 rows")
        self.assertIn("Salary", df.columns, "Columns must contain 'Salary'")

    def test_load_valid_excel(self):
        """Test loading a valid Excel (.xlsx) file."""
        df, err = load_dataset(self.sample_xlsx)
        self.assertIsNone(err, "Error should be None for valid Excel file")
        self.assertIsNotNone(df, "DataFrame should not be None")
        self.assertEqual(len(df), 15)
        self.assertIn("Department", df.columns)

    def test_load_nonexistent_file(self):
        """Test handling of a non-existent file path."""
        df, err = load_dataset("datasets/non_existent_file.csv")
        self.assertIsNone(df)
        self.assertIn("File not found", err)

    def test_load_unsupported_format(self):
        """Test rejection of unsupported file types (e.g. .txt)."""
        df, err = load_dataset("README.md")
        self.assertIsNone(df)
        self.assertIn("Unsupported format", err)

    def test_get_dataset_overview(self):
        """Test dataset metadata extraction."""
        df, _ = load_dataset(self.sample_csv)
        overview = get_dataset_overview(df, dataset_name="sample_data.csv")
        self.assertEqual(overview["total_rows"], 15)
        self.assertEqual(overview["total_columns"], 9)
        self.assertEqual(len(overview["head_preview"]), 5)
        self.assertIn("Performance_Score", overview["missing_values"])


if __name__ == "__main__":
    unittest.main()
