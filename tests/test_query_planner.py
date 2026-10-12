"""
test_query_planner.py
---------------------
Unit tests for query_planner.py verifying plan validation, schema checking,
and natural language query parsing.
"""

import os
import unittest
import pandas as pd
from data_loader import load_dataset
from query_planner import plan_query, validate_plan


class TestQueryPlanner(unittest.TestCase):

    def setUp(self):
        sample_path = os.path.join("datasets", "sample_data.csv")
        self.df, _ = load_dataset(sample_path)

    def test_validate_valid_mean_plan(self):
        """Test validation of a valid mean plan."""
        plan = {"task": "mean", "column": "Salary"}
        valid_plan, err = validate_plan(plan, self.df)
        self.assertIsNone(err)
        self.assertEqual(valid_plan["column"], "Salary")

    def test_validate_unsupported_task(self):
        """Test rejection of unsupported tasks."""
        plan = {"task": "hack_system", "column": "Salary"}
        _, err = validate_plan(plan, self.df)
        self.assertIn("Unsupported task", err)

    def test_validate_invalid_column(self):
        """Test rejection when column does not exist."""
        plan = {"task": "mean", "column": "Bonus"}
        _, err = validate_plan(plan, self.df)
        self.assertIn("does not exist", err)

    def test_validate_non_numeric_mean(self):
        """Test rejection of numeric operation on string column."""
        plan = {"task": "mean", "column": "Department"}
        _, err = validate_plan(plan, self.df)
        self.assertIn("requires a numeric column", err)

    def test_plan_query_questions(self):
        """Test end-to-end question planning for various natural queries."""
        p1, e1 = plan_query("What is the average salary?", self.df)
        self.assertIsNone(e1)
        self.assertEqual(p1["task"], "mean")
        self.assertEqual(p1["column"], "Salary")

        p2, e2 = plan_query("How many rows are there?", self.df)
        self.assertIsNone(e2)
        self.assertEqual(p2["task"], "count_rows")

        p3, e3 = plan_query("Show average salary by department", self.df)
        self.assertIsNone(e3)
        self.assertEqual(p3["task"], "group_aggregate")
        self.assertEqual(p3["group_by"], "Department")

    def test_plan_query_column_specific_missing_values(self):
        """Test question planning for column-specific missing values."""
        plan, err = plan_query("how many total missing values in salary column", self.df)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "missing_values")
        self.assertEqual(plan["column"], "Salary")

    def test_plan_query_kdd_cleaning_and_transformation(self):
        """Test question planning for KDD data cleaning operations."""
        p1, e1 = plan_query("fill missing values in salary by mean", self.df)
        self.assertIsNone(e1)
        self.assertEqual(p1["task"], "fill_missing")
        self.assertEqual(p1["column"], "Salary")
        self.assertEqual(p1["strategy"], "mean")

        p2, e2 = plan_query("remove duplicate rows", self.df)
        self.assertIsNone(e2)
        self.assertEqual(p2["task"], "drop_duplicates")

        p3, e3 = plan_query("drop column department", self.df)
        self.assertIsNone(e3)
        self.assertEqual(p3["task"], "drop_column")
        self.assertEqual(p3["column"], "Department")


if __name__ == "__main__":
    unittest.main()
