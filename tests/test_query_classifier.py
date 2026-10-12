"""
test_query_classifier.py
------------------------
Unit tests for the supervised ML Query Classifier (TF-IDF + Logistic Regression).
Verifies model convergence, evaluation metric generation, and inference predictions.
"""

import unittest
from query_classifier import QueryClassifierModel, classify_user_query


class TestQueryClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = QueryClassifierModel()
        cls.metrics = cls.model.train_and_evaluate(test_size=0.25, random_state=42)

    def test_metrics_structure(self):
        """Test evaluation metrics keys and types."""
        self.assertIn("accuracy", self.metrics)
        self.assertIn("macro_f1", self.metrics)
        self.assertIn("weighted_f1", self.metrics)
        self.assertIn("classification_report_text", self.metrics)
        self.assertIn("confusion_matrix", self.metrics)
        self.assertEqual(self.metrics["total_classes"], 11)

    def test_accuracy_threshold(self):
        """Ensure model achieves reasonable accuracy (>70%) on held-out test split."""
        self.assertGreater(self.metrics["accuracy"], 0.70)

    def test_prediction_aggregation(self):
        """Test prediction on aggregation query."""
        label, conf = self.model.predict("What is the average salary?")
        self.assertEqual(label, "aggregation")
        self.assertGreater(conf, 0.3)

    def test_prediction_visualization(self):
        """Test prediction on visualization query."""
        label, conf = self.model.predict("Plot a histogram of salaries")
        self.assertEqual(label, "visualization")
        self.assertGreater(conf, 0.4)

    def test_prediction_data_quality(self):
        """Test prediction on data quality / missing value query."""
        label, conf = self.model.predict("Are there any missing values?")
        self.assertEqual(label, "data_quality")


if __name__ == "__main__":
    unittest.main()
