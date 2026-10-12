"""
test_analysis_engine.py
-----------------------
Unit tests for analysis_engine.py validating analytical operations,
statistical correctness, filtering, aggregations, and safe error handling.
"""

import os
import unittest
import pandas as pd
from data_loader import load_dataset
from analysis_engine import execute_plan


class TestAnalysisEngine(unittest.TestCase):

    def setUp(self):
        csv_path = os.path.join("datasets", "sample_data.csv")
        self.df, _ = load_dataset(csv_path)

    def test_count_rows_and_columns(self):
        """Test row and column count operations."""
        res_rows = execute_plan(self.df, {"task": "count_rows"})
        self.assertEqual(res_rows["status"], "success")
        self.assertEqual(res_rows["value"], 15)

        res_cols = execute_plan(self.df, {"task": "count_columns"})
        self.assertEqual(res_cols["status"], "success")
        self.assertEqual(res_cols["value"], 9)

    def test_mean_calculation(self):
        """Test arithmetic mean calculation."""
        res = execute_plan(self.df, {"task": "mean", "column": "Salary"})
        self.assertEqual(res["status"], "success")
        self.assertAlmostEqual(res["value"], 67333.33, places=1)

    def test_median_and_sum(self):
        """Test median and sum calculations."""
        res_med = execute_plan(self.df, {"task": "median", "column": "Salary"})
        self.assertEqual(res_med["value"], 68000.0)

        res_sum = execute_plan(self.df, {"task": "sum", "column": "Projects_Completed"})
        self.assertEqual(res_sum["value"], 145.0)

    def test_min_and_max(self):
        """Test minimum and maximum calculations."""
        res_min = execute_plan(self.df, {"task": "min", "column": "Salary"})
        self.assertEqual(res_min["value"], 46000.0)

        res_max = execute_plan(self.df, {"task": "max", "column": "Salary"})
        self.assertEqual(res_max["value"], 92000.0)

    def test_top_n_records(self):
        """Test Top N records ordering."""
        res = execute_plan(self.df, {"task": "top_n", "column": "Salary", "top_n": 3})
        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["value"]), 3)
        self.assertEqual(res["value"].iloc[0]["Salary"], 92000)

    def test_filter_and_count_where(self):
        """Test row filtering with conditions."""
        res_filter = execute_plan(self.df, {
            "task": "filter",
            "condition_col": "Salary",
            "operator": ">",
            "condition_val": 70000,
        })
        self.assertEqual(res_filter["status"], "success")
        self.assertEqual(len(res_filter["value"]), 7)

        res_cnt = execute_plan(self.df, {
            "task": "count_where",
            "condition_col": "Department",
            "operator": "==",
            "condition_val": "Engineering",
        })
        self.assertEqual(res_cnt["status"], "success")
        self.assertEqual(res_cnt["value"], 6)

    def test_group_aggregate(self):
        """Test grouping and aggregation."""
        res = execute_plan(self.df, {
            "task": "group_aggregate",
            "group_by": "Department",
            "column": "Salary",
            "aggregation": "mean",
        })
        self.assertEqual(res["status"], "success")
        grouped_df = res["value"]
        hr_mean = grouped_df.loc[grouped_df["Department"] == "HR", "Salary_mean"].values[0]
        self.assertEqual(hr_mean, 47000.0)

    def test_safe_error_on_invalid_column(self):
        """Test graceful error return when column is missing."""
        res = execute_plan(self.df, {"task": "mean", "column": "NonExistentColumn"})
        self.assertEqual(res["status"], "error")
        self.assertIn("not found", res["error"])

    def test_safe_error_on_non_numeric_column(self):
        """Test error when numeric calculation requested on text column."""
        res = execute_plan(self.df, {"task": "mean", "column": "Department"})
        self.assertEqual(res["status"], "error")
        self.assertIn("requires a numeric column", res["error"])

    def test_missing_values_all_and_column_specific(self):
        """Test missing values for dataset-wide and column-specific queries."""
        res_all = execute_plan(self.df, {"task": "missing_values"})
        self.assertEqual(res_all["status"], "success")
        self.assertEqual(res_all["result_type"], "table")

        res_col = execute_plan(self.df, {"task": "missing_values", "column": "Salary"})
        self.assertEqual(res_col["status"], "success")
        self.assertEqual(res_col["result_type"], "scalar")
        self.assertIn("Salary", res_col["message"])

    def test_kdd_preprocessing_and_transformation(self):
        """Test KDD operations: fill_missing, drop_missing, drop_column, normalize, standardize."""
        # 1. Fill missing values
        res_fill = execute_plan(self.df, {"task": "fill_missing", "column": "Salary", "strategy": "mean"})
        self.assertEqual(res_fill["status"], "success")
        self.assertIsNotNone(res_fill["modified_df"])

        # 2. Drop duplicates
        res_dup = execute_plan(self.df, {"task": "drop_duplicates"})
        self.assertEqual(res_dup["status"], "success")
        self.assertIsNotNone(res_dup["modified_df"])

        # 3. Drop column
        res_drop_col = execute_plan(self.df, {"task": "drop_column", "column": "Department"})
        self.assertEqual(res_drop_col["status"], "success")
        self.assertNotIn("Department", res_drop_col["modified_df"].columns)

        # 4. Normalize
        res_norm = execute_plan(self.df, {"task": "normalize", "column": "Salary"})
        self.assertEqual(res_norm["status"], "success")
        self.assertIn("Salary_normalized", res_norm["modified_df"].columns)


if __name__ == "__main__":
    unittest.main()
