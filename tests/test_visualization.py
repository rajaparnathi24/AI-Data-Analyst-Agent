"""
test_visualization.py
---------------------
Unit tests for visualization.py verifying chart generation,
matplotlib Figure returns, and graceful error handling on bad inputs.
"""

import os
import unittest
import matplotlib.pyplot as plt
from data_loader import load_dataset
from analysis_engine import execute_plan
from visualization import (
    create_bar_chart,
    create_histogram,
    create_line_chart,
    create_scatter_plot,
    create_auto_chart,
)


class TestVisualization(unittest.TestCase):

    def setUp(self):
        sample_path = os.path.join("datasets", "sample_data.csv")
        self.df, _ = load_dataset(sample_path)

    def tearDown(self):
        # Close any open figures to avoid memory accumulation
        plt.close("all")

    def test_create_bar_chart(self):
        """Test bar chart generation for grouped data."""
        plan = {"task": "group_aggregate", "group_by": "Department", "column": "Salary", "aggregation": "mean"}
        res = execute_plan(self.df, plan)
        fig, err = create_bar_chart(res["value"], x_col="Department", y_col="Salary_mean")
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    def test_create_histogram(self):
        """Test histogram generation for numerical distribution."""
        fig, err = create_histogram(self.df, col="Salary", bins=5)
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    def test_create_line_chart(self):
        """Test line chart generation."""
        fig, err = create_line_chart(self.df, x_col="Experience_Years", y_col="Salary")
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    def test_create_scatter_plot(self):
        """Test scatter plot generation."""
        fig, err = create_scatter_plot(self.df, x_col="Experience_Years", y_col="Salary")
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    def test_auto_chart_on_group_aggregate(self):
        """Test smart auto chart generator."""
        plan = {"task": "group_aggregate", "group_by": "Department", "column": "Salary", "aggregation": "mean"}
        res = execute_plan(self.df, plan)
        fig, err = create_auto_chart(res, self.df)
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    def test_error_handling_invalid_column(self):
        """Test graceful error message when non-existent column is passed."""
        fig, err = create_histogram(self.df, col="GhostColumn")
        self.assertIsNone(fig)
        self.assertIn("not available", err)


if __name__ == "__main__":
    unittest.main()
