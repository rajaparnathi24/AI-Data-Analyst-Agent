"""
test_data_profiler.py
---------------------
Unit tests for data_profiler.py verifying descriptive statistics,
column typing, duplicate detection, and missing value calculations.
"""

import os
import unittest
import pandas as pd
from data_loader import load_dataset
from data_profiler import (
    get_column_types,
    get_missing_summary,
    get_duplicate_summary,
    get_numerical_statistics,
    get_categorical_statistics,
    generate_full_profile,
)


class TestDataProfiler(unittest.TestCase):

    def setUp(self):
        sample_csv = os.path.join("datasets", "sample_data.csv")
        self.df, _ = load_dataset(sample_csv)

    def test_column_types(self):
        """Test correct separation of numerical and categorical columns."""
        types = get_column_types(self.df)
        self.assertIn("Salary", types["numerical"])
        self.assertIn("Department", types["categorical"])
        self.assertEqual(len(types["numerical"]), 5)
        self.assertEqual(len(types["categorical"]), 4)

    def test_missing_summary(self):
        """Test missing values detection."""
        missing_df = get_missing_summary(self.df)
        perf_missing = missing_df.loc[missing_df["Column"] == "Performance_Score", "Missing Count"].values[0]
        self.assertEqual(perf_missing, 1)

    def test_duplicate_summary(self):
        """Test duplicate detection."""
        dup_info = get_duplicate_summary(self.df)
        self.assertEqual(dup_info["duplicate_count"], 0)
        self.assertEqual(dup_info["duplicate_percentage"], 0.0)

    def test_numerical_statistics(self):
        """Test numerical calculations against expected ground truth."""
        num_stats = get_numerical_statistics(self.df)
        salary_row = num_stats.loc[num_stats["Column"] == "Salary"].iloc[0]
        self.assertEqual(salary_row["Min"], 46000.0)
        self.assertEqual(salary_row["Max"], 92000.0)
        self.assertAlmostEqual(salary_row["Mean"], 67333.33, places=1)
        self.assertEqual(salary_row["Median"], 68000.0)

    def test_categorical_statistics(self):
        """Test categorical cardinality and mode detection."""
        cat_stats = get_categorical_statistics(self.df)
        dept_row = cat_stats.loc[cat_stats["Column"] == "Department"].iloc[0]
        self.assertEqual(dept_row["Unique Count"], 5)
        self.assertEqual(dept_row["Top Category"], "Engineering")
        self.assertEqual(dept_row["Frequency"], 6)

    def test_generate_full_profile(self):
        """Test aggregate profile generation."""
        profile = generate_full_profile(self.df)
        self.assertEqual(profile["total_rows"], 15)
        self.assertEqual(profile["total_columns"], 9)
        self.assertEqual(profile["total_missing_cells"], 1)


if __name__ == "__main__":
    unittest.main()
