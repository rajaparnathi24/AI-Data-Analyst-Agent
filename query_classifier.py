"""
query_classifier.py
-------------------
Supervised Machine Learning module for natural-language query intent classification.

ML Pipeline:
User Query -> Text Preprocessing -> TF-IDF Vectorization -> Logistic Regression -> Query Intent

Target Classes (11 Categories):
- aggregation, maximum, minimum, counting, filtering,
- ranking, group_comparison, trend_analysis, visualization,
- data_quality, correlation

College Viva Key Points:
1. TF-IDF (Term Frequency - Inverse Document Frequency): Converts words into numerical features,
   weighting words that are uniquely informative for a category higher than ubiquitous common words.
2. Logistic Regression: Linear model that calculates the log-odds (probability) of a query
   belonging to each of the 11 classes using the Softmax / Multinomial function.
3. Stratified Split: Ensures every query category is proportionately represented in both train and test splits.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# -------------------------------------------------------------
# 1. Labeled Training Dataset for Query Classification
# -------------------------------------------------------------

SAMPLE_QUERIES_DATASET = [
    # aggregation (mean, median, sum)
    ("What is the average salary?", "aggregation"),
    ("Calculate the mean score of students", "aggregation"),
    ("Find the total sum of all projects", "aggregation"),
    ("What is the average experience in the company?", "aggregation"),
    ("Calculate median salary of employees", "aggregation"),
    ("Compute overall average revenue", "aggregation"),
    ("Find the total expenses across all records", "aggregation"),
    ("What is the average age of staff?", "aggregation"),
    ("Calculate total sales amount", "aggregation"),
    ("Find the mean performance rating", "aggregation"),

    # maximum
    ("Who has the highest marks?", "maximum"),
    ("What is the maximum salary?", "maximum"),
    ("Find the greatest project score", "maximum"),
    ("Which employee has the highest experience?", "maximum"),
    ("Find the maximum performance rating", "maximum"),
    ("Show the highest value in revenue", "maximum"),
    ("Who has the highest salary in the team?", "maximum"),
    ("Find the peak sales record", "maximum"),
    ("What is the maximum bonus awarded?", "maximum"),
    ("Identify the maximum value", "maximum"),

    # minimum
    ("What is the lowest salary?", "minimum"),
    ("Find the minimum marks among students", "minimum"),
    ("Who has the lowest rating?", "minimum"),
    ("What is the minimum experience required?", "minimum"),
    ("Find the smallest project count", "minimum"),
    ("Show the minimum value in expenses", "minimum"),
    ("Who has the lowest score?", "minimum"),
    ("Find the minimum salary offered", "minimum"),
    ("What is the smallest number of completed projects?", "minimum"),
    ("Identify the lowest earner", "minimum"),

    # counting
    ("How many rows are in this dataset?", "counting"),
    ("Count the total number of records", "counting"),
    ("How many employees are there?", "counting"),
    ("What is the row count?", "counting"),
    ("How many columns are present?", "counting"),
    ("Count the number of items", "counting"),
    ("How many entries exist?", "counting"),
    ("What is the total count of staff?", "counting"),
    ("Count how many records are stored", "counting"),
    ("Give me the count of rows", "counting"),

    # filtering
    ("Show employees with salary greater than 50000", "filtering"),
    ("Find all records from the IT department", "filtering"),
    ("Filter rows where experience is less than 3 years", "filtering"),
    ("Show workers who joined after 2022", "filtering"),
    ("List all people with performance score above 4", "filtering"),
    ("Find entries where status is active", "filtering"),
    ("Filter records with salary above 70000", "filtering"),
    ("Show employees where role is engineer", "filtering"),
    ("Find all staff where experience is greater than 5", "filtering"),
    ("Filter rows where department is sales", "filtering"),

    # ranking
    ("Show the top 5 students", "ranking"),
    ("Find the top 3 highest paid employees", "ranking"),
    ("List the bottom 5 performers", "ranking"),
    ("Display the top 10 products by sales", "ranking"),
    ("Show lowest 3 salaries in the company", "ranking"),
    ("Who are the top 5 earners?", "ranking"),
    ("Show the top 3 records", "ranking"),
    ("Find bottom 5 students by marks", "ranking"),
    ("Rank the top 10 employees by projects", "ranking"),
    ("Display bottom 3 values", "ranking"),

    # group_comparison
    ("Compare average salary across departments", "group_comparison"),
    ("Show average sales by region", "group_comparison"),
    ("Calculate mean performance by role", "group_comparison"),
    ("Compare project counts per department", "group_comparison"),
    ("Show total revenue grouped by product category", "group_comparison"),
    ("Group by department and find average experience", "group_comparison"),
    ("Compare scores across all branches", "group_comparison"),
    ("Show breakdown of salaries by department", "group_comparison"),
    ("Group by category and calculate total count", "group_comparison"),
    ("Compare average experience per job role", "group_comparison"),

    # trend_analysis
    ("Show monthly sales trends over time", "trend_analysis"),
    ("Display hiring trend across years", "trend_analysis"),
    ("How has salary changed over the years?", "trend_analysis"),
    ("Show performance trends over join dates", "trend_analysis"),
    ("Display the trend of expenses by month", "trend_analysis"),
    ("Track quarterly growth trends", "trend_analysis"),
    ("Show trend of projects completed each year", "trend_analysis"),
    ("How do sales trend over the quarters?", "trend_analysis"),
    ("Plot historical trends over time", "trend_analysis"),
    ("Analyze growth trends across dates", "trend_analysis"),

    # visualization
    ("Plot a bar chart of sales by product", "visualization"),
    ("Create a histogram of salaries", "visualization"),
    ("Generate a scatter plot of experience vs salary", "visualization"),
    ("Draw a line chart of monthly revenue", "visualization"),
    ("Plot a chart comparing departments", "visualization"),
    ("Show a graph of performance distribution", "visualization"),
    ("Create a visual plot of project counts", "visualization"),
    ("Generate a bar chart of revenue", "visualization"),
    ("Display a histogram showing salary spread", "visualization"),
    ("Plot a scatter diagram", "visualization"),
    ("Create piechart for Embarked column", "visualization"),
    ("Create a pie chart for Sex", "visualization"),
    ("Show bar chart of Sex", "visualization"),
    ("Plot histogram for Age", "visualization"),
    ("Create line graph for Fare", "visualization"),
    ("Plot Age vs Fare", "visualization"),
    ("Show boxplot for Age", "visualization"),
    ("Show count plot of Embarked", "visualization"),
    ("Create correlation heatmap", "visualization"),
    ("Draw a box plot of salary", "visualization"),

    # data_quality
    ("Are there any missing values in this dataset?", "data_quality"),
    ("Identify null values in each column", "data_quality"),
    ("Check for duplicate rows", "data_quality"),
    ("How many empty cells are there?", "data_quality"),
    ("Show data quality and missing records", "data_quality"),
    ("Find duplicate records in the data", "data_quality"),
    ("Check missing value percentages", "data_quality"),
    ("Are there duplicate rows in this table?", "data_quality"),
    ("Identify all NaN values", "data_quality"),
    ("Report missing data count", "data_quality"),

    # correlation
    ("What is the correlation between experience and salary?", "correlation"),
    ("Show correlation between numerical columns", "correlation"),
    ("Is performance score correlated with salary?", "correlation"),
    ("Generate a correlation matrix", "correlation"),
    ("Check correlation between projects and experience", "correlation"),
    ("Find relationships between numerical variables", "correlation"),
    ("Compute correlation coefficients", "correlation"),
    ("Are experience and salary strongly correlated?", "correlation"),
    ("Show pairwise correlation for numbers", "correlation"),
    ("Calculate the correlation matrix table", "correlation"),
]


def preprocess_text(text: str) -> str:
    """Cleans and standardizes raw text."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)      # normalize extra spaces
    return text


class QueryClassifierModel:
    """Wrapper managing training, evaluation, and inference of the ML intent classifier."""

    def __init__(self):
        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    ngram_range=(1, 2),  # captures unigrams ("salary") & bigrams ("average salary")
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    random_state=42,
                    max_iter=300,
                    C=5.0,
                    solver="lbfgs",
                ),
            ),
        ])
        self.is_trained = False
        self.classes_ = []

    def train_and_evaluate(self, test_size: float = 0.25, random_state: int = 42) -> dict:
        """
        Trains the TF-IDF + Logistic Regression pipeline using Stratified Train/Test Split.

        Returns:
            dict: Evaluation metrics (accuracy, precision, recall, f1, classification report).
        """
        df_data = pd.DataFrame(SAMPLE_QUERIES_DATASET, columns=["query", "label"])
        X = df_data["query"]
        y = df_data["label"]

        # Stratified train/test split ensures every category appears in both train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        # Train the ML pipeline
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        self.classes_ = list(self.pipeline.classes_)

        # Predict on held-out test set
        y_pred = self.pipeline.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, zero_division=0)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        conf_mat = confusion_matrix(y_test, y_pred)

        return {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_classes": len(self.classes_),
            "accuracy": round(accuracy, 4),
            "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
            "classification_report_text": report,
            "confusion_matrix": conf_mat,
            "classes": self.classes_,
        }

    def predict(self, query: str) -> tuple:
        """
        Predicts the intent category and probability confidence score.

        Returns:
            tuple: (predicted_class, confidence_score_float)
        """
        if not self.is_trained:
            self.train_and_evaluate()

        cleaned_q = preprocess_text(query)
        pred_label = self.pipeline.predict([cleaned_q])[0]
        probs = self.pipeline.predict_proba([cleaned_q])[0]
        confidence = float(np.max(probs))

        return pred_label, round(confidence, 3)


# Singleton instance for quick module access
_classifier_instance = None


def get_trained_classifier() -> QueryClassifierModel:
    global _classifier_instance
    if _classifier_instance is None or not _classifier_instance.is_trained:
        _classifier_instance = QueryClassifierModel()
        _classifier_instance.train_and_evaluate()
    return _classifier_instance


def classify_user_query(query: str) -> tuple:
    """Convenience function returning (category, confidence)."""
    model = get_trained_classifier()
    return model.predict(query)


# Standalone demonstration and ML viva evaluation script
if __name__ == "__main__":
    print("=" * 70)
    print("MACHINE LEARNING QUERY CLASSIFIER (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 70)

    model = QueryClassifierModel()
    metrics = model.train_and_evaluate(test_size=0.25, random_state=42)

    print(f"Total Categories:  {metrics['total_classes']}")
    print(f"Training Samples:  {metrics['train_samples']}")
    print(f"Testing Samples:   {metrics['test_samples']}")
    print(f"Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")

    print("\n--- DETAILED CLASSIFICATION REPORT ---")
    print(metrics["classification_report_text"])

    print("\n--- SAMPLE INFERENCE PREDICTIONS ---")
    demo_queries = [
        "What is the average mathematics score?",
        "Find the top 5 students in science",
        "Plot a histogram of salaries",
        "Are there any missing values?",
        "Compare average salaries across departments",
    ]

    for q in demo_queries:
        cat, conf = model.predict(q)
        print(f"Query: '{q}'")
        print(f"  -> Predicted Intent: [{cat}] (Confidence: {conf * 100:.1f}%)\n")

    print("SUCCESS: query_classifier.py trained and evaluated cleanly!")
