"""
test_chart_generation.py
-------------------------
Unit tests verifying chart intent detection, dynamic column resolution,
chart type selection, conversational follow-ups, and invalid column suggestions.
"""

import os
import unittest
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_dataset
from query_planner import plan_query
from analysis_engine import execute_plan
from visualization import create_auto_chart


class TestChartGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        titanic_path = "Titanic-Dataset.csv"
        if not os.path.exists(titanic_path):
            titanic_path = os.path.join("datasets", "sample_data.csv")
        cls.df, _ = load_dataset(titanic_path)

    def tearDown(self):
        plt.close("all")

    def test_pie_chart_intent(self):
        """Test: 'create piechart for Embarked column' produces a pie chart."""
        q = "create piechart for Embarked column"
        plan, err = plan_query(q, self.df)
        self.assertIsNone(err, f"Plan failed: {err}")
        self.assertEqual(plan["task"], "pie_chart")
        self.assertEqual(plan["column"], "Embarked")

        result = execute_plan(self.df, plan)
        self.assertEqual(result["status"], "success")

        fig, chart_err = create_auto_chart(result, self.df)
        self.assertIsNone(chart_err)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_bar_chart_intent(self):
        """Test: 'show bar chart of Sex' produces a bar chart."""
        q = "show bar chart of Sex"
        plan, err = plan_query(q, self.df)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "bar_chart")
        self.assertEqual(plan["column"], "Sex")

        result = execute_plan(self.df, plan)
        self.assertEqual(result["status"], "success")

        fig, chart_err = create_auto_chart(result, self.df)
        self.assertIsNone(chart_err)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_histogram_intent(self):
        """Test: 'plot histogram for Age' produces a histogram."""
        q = "plot histogram for Age"
        plan, err = plan_query(q, self.df)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "histogram")
        self.assertEqual(plan["column"], "Age")

        result = execute_plan(self.df, plan)
        self.assertEqual(result["status"], "success")

        fig, chart_err = create_auto_chart(result, self.df)
        self.assertIsNone(chart_err)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_line_chart_intent(self):
        """Test: 'create line graph for Fare' produces a line chart."""
        q = "create line graph for Fare"
        plan, err = plan_query(q, self.df)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "line_chart")
        self.assertEqual(plan["column"], "Fare")

        result = execute_plan(self.df, plan)
        self.assertEqual(result["status"], "success")

        fig, chart_err = create_auto_chart(result, self.df)
        self.assertIsNone(chart_err)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_scatter_plot_intent(self):
        """Test: 'plot Age vs Fare' produces a scatter plot."""
        q = "plot Age vs Fare"
        plan, err = plan_query(q, self.df)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "scatter_plot")
        self.assertEqual(plan.get("x_col"), "Age")
        self.assertEqual(plan.get("y_col"), "Fare")

        result = execute_plan(self.df, plan)
        self.assertEqual(result["status"], "success")

        fig, chart_err = create_auto_chart(result, self.df)
        self.assertIsNone(chart_err)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_boxplot_intent(self):
        """Test: 'show boxplot for Age' produces a box plot."""
        q = "show boxplot for Age"
        plan, err = plan_query(q, self.df)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "box_plot")
        self.assertEqual(plan["column"], "Age")

        result = execute_plan(self.df, plan)
        self.assertEqual(result["status"], "success")

        fig, chart_err = create_auto_chart(result, self.df)
        self.assertIsNone(chart_err)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_count_plot_intent(self):
        """Test: 'show count plot of Embarked' produces a count plot."""
        q = "show count plot of Embarked"
        plan, err = plan_query(q, self.df)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "count_plot")
        self.assertEqual(plan["column"], "Embarked")

        result = execute_plan(self.df, plan)
        self.assertEqual(result["status"], "success")

        fig, chart_err = create_auto_chart(result, self.df)
        self.assertIsNone(chart_err)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_heatmap_intent(self):
        """Test: 'create correlation heatmap' produces a heatmap."""
        q = "create correlation heatmap"
        plan, err = plan_query(q, self.df)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "heatmap")

        result = execute_plan(self.df, plan)
        self.assertEqual(result["status"], "success")

        fig, chart_err = create_auto_chart(result, self.df)
        self.assertIsNone(chart_err)
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)

    def test_conversational_followup_change_chart_type(self):
        """
        Turn 1: 'plot histogram for Age'
        Turn 2: 'Now show it as a bar chart' -> should reuse Age from Turn 1!
        """
        history = [
            {
                "user": "plot histogram for Age",
                "task": "histogram",
                "column": "Age",
                "analyzed_column": "Age",
            }
        ]
        q = "Now show it as a bar chart"
        plan, err = plan_query(q, self.df, conversation_history=history)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "bar_chart")
        self.assertEqual(plan["column"], "Age")

    def test_conversational_followup_change_column(self):
        """
        Turn 1: 'create piechart for Embarked column'
        Turn 2: 'Create a pie chart for Sex instead' -> changes target to Sex!
        """
        history = [
            {
                "user": "create piechart for Embarked column",
                "task": "pie_chart",
                "column": "Embarked",
                "analyzed_column": "Embarked",
            }
        ]
        q = "Create a pie chart for Sex instead"
        plan, err = plan_query(q, self.df, conversation_history=history)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "pie_chart")
        self.assertEqual(plan["column"], "Sex")

    def test_conversational_followup_scatter_against(self):
        """Test: 'Plot Age against Fare' produces scatter plot with both columns."""
        q = "Plot Age against Fare"
        plan, err = plan_query(q, self.df)
        self.assertIsNone(err)
        self.assertEqual(plan["task"], "scatter_plot")
        self.assertEqual(plan.get("x_col"), "Age")
        self.assertEqual(plan.get("y_col"), "Fare")

    def test_invalid_column_suggests_close_matches(self):
        """Test: If user requests non-existent column, returns helpful suggestions instead of generic error."""
        q = "create piechart for FooBarGhost column"
        plan, err = plan_query(q, self.df)
        self.assertIsNone(plan)
        self.assertIsNotNone(err)
        self.assertTrue("not found" in err.lower() or "did you mean" in err.lower())
        self.assertIn("Available columns:", err)


if __name__ == "__main__":
    unittest.main()
