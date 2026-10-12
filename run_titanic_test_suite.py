"""
run_titanic_test_suite.py
-------------------------
Runs all 110 questions from sections A through K against Titanic-Dataset.csv.
Evaluates query planning, execution, and detects any failure or unhandled queries.
"""

import os
import sys
import pandas as pd
from data_loader import load_dataset
from query_planner import plan_query
from analysis_engine import execute_plan
from data_profiler import (
    get_missing_summary,
    get_duplicate_summary,
    get_numerical_statistics,
    get_categorical_statistics,
    detect_identifier_columns,
    detect_constant_columns,
    get_memory_usage,
    generate_full_profile,
    get_column_types,
)
from visualization import create_auto_chart

TASKS = {
    "A": [
        ("A1", "Give me a complete overview of the Titanic dataset."),
        ("A2", "How many rows and columns are present?"),
        ("A3", "Show the first 5 records."),
        ("A4", "Show the last 10 records."),
        ("A5", "Display 20 random passengers."),
        ("A6", "List all column names and their data types."),
        ("A7", "Which columns are numerical, categorical, text, and identifiers?"),
        ("A8", "Show the dataset shape and memory usage."),
        ("A9", "Generate a complete dataset profiling report."),
        ("A10", "Identify constant columns and columns with unique values."),
    ],
    "B": [
        ("B1", "Show missing values in every column."),
        ("B2", "Calculate missing-value percentages for all columns."),
        ("B3", "Which column has the highest percentage of missing values?"),
        ("B4", "Show columns with no missing values."),
        ("B5", "Count duplicate rows."),
        ("B6", "Display duplicate records."),
        ("B7", "Check for empty strings and whitespace-only values."),
        ("B8", "Identify inconsistent data types."),
        ("B9", "Check whether PassengerId contains unique values."),
        ("B10", "Generate a complete data quality report."),
    ],
    "C": [
        ("C1", "Calculate mean, median, and standard deviation for Age and Fare."),
        ("C2", "Show minimum, maximum, and quartiles for all numerical columns."),
        ("C3", "Show the frequency distribution of Sex."),
        ("C4", "Count passengers in each Pclass."),
        ("C5", "Show unique values and counts for Embarked."),
        ("C6", "Show the 10 most frequent Ticket values."),
        ("C7", "Calculate the number of unique Cabin values."),
        ("C8", "Show the distribution of Survived."),
        ("C9", "Calculate skewness for Age and Fare."),
        ("C10", "Find numerical columns with zero variance."),
    ],
    "D": [
        ("D1", "How many passengers survived?"),
        ("D2", "How many passengers did not survive?"),
        ("D3", "Calculate the overall survival rate."),
        ("D4", "Compare survival rates by Sex."),
        ("D5", "Compare survival rates by Pclass."),
        ("D6", "Compare survival rates by Embarked."),
        ("D7", "Calculate survival rates for different age groups."),
        ("D8", "Compare survival rates for passengers travelling alone versus with family."),
        ("D9", "Compare average Age for survivors and non-survivors."),
        ("D10", "Summarize the strongest observed survival patterns."),
    ],
    "E": [
        ("E1", "Show only PassengerId, Name, Sex, Age, and Fare."),
        ("E2", "Show passengers older than 40."),
        ("E3", "Filter female passengers who survived."),
        ("E4", "Show male passengers who did not survive."),
        ("E5", "Show first-class passengers."),
        ("E6", "Show passengers with Fare greater than 100."),
        ("E7", "Filter passengers aged between 18 and 35."),
        ("E8", "Show passengers with missing Age."),
        ("E9", "Show passengers with missing Cabin."),
        ("E10", "Show passengers who survived and paid more than 50."),
        ("E11", "Show passengers in Pclass 1 who were older than 50."),
        ("E12", "Select the top 20 passengers by Fare."),
        ("E13", "Sort passengers by Age in ascending order."),
        ("E14", "Show a random sample of 50 passengers."),
    ],
    "F": [
        ("F1", "Fill missing Age values with the median."),
        ("F2", "Fill missing Age values with the mean."),
        ("F3", "Fill missing Embarked values with the mode."),
        ("F4", "Fill missing Cabin values with the text Unknown."),
        ("F5", "Remove rows with missing Age."),
        ("F6", "Remove columns with more than 50% missing values."),
        ("F7", "Remove duplicate rows."),
        ("F8", "Trim whitespace from all text columns."),
        ("F9", "Convert Age and Fare to numeric types."),
        ("F10", "Show a before-and-after missing-value report."),
        ("F11", "Detect outliers in Fare using the IQR method."),
        ("F12", "Cap Fare outliers using the IQR method."),
        ("F13", "Undo the last data-cleaning operation."),
        ("F14", "Reset the dataset to its original state."),
    ],
    "G": [
        ("G1", "Normalize Age between 0 and 1 in a new column."),
        ("G2", "Standardize Fare using Z-score in a new column."),
        ("G3", "Create an Age_Group column with child, adult, and senior groups."),
        ("G4", "Create FamilySize by adding SibSp and Parch and including the passenger."),
        ("G5", "Create IsAlone based on FamilySize."),
        ("G6", "Create a Fare_Category column."),
        ("G7", "Create a column indicating whether Cabin is missing."),
        ("G8", "Extract a new feature from the Ticket column if appropriate."),
        ("G9", "One-hot encode Sex and Embarked."),
        ("G10", "Create a pivot table of survival rate by Sex and Pclass."),
        ("G11", "Group by Pclass and calculate average Fare and Age."),
        ("G12", "Download the transformed dataset."),
    ],
    "H": [
        ("H1", "Calculate average Fare by Pclass."),
        ("H2", "Calculate median Age by Sex."),
        ("H3", "Count passengers by Sex and Pclass."),
        ("H4", "Create a crosstab of Sex versus Survived."),
        ("H5", "Create a crosstab of Pclass versus Survived."),
        ("H6", "Calculate survival percentages for each group."),
        ("H7", "Compare mean and median Fare."),
        ("H8", "Calculate correlation between Age and Fare."),
        ("H9", "Show the correlation matrix for numerical columns."),
        ("H10", "Find the strongest positive and negative correlations."),
        ("H11", "Analyze whether Fare differs between survivors and non-survivors."),
        ("H12", "Explain why correlation does not prove causation."),
    ],
    "I": [
        ("I1", "Create a bar chart of passenger counts by Sex."),
        ("I2", "Create a bar chart of survival counts."),
        ("I3", "Plot survival rate by Pclass."),
        ("I4", "Create a histogram of Age."),
        ("I5", "Create a histogram of Fare."),
        ("I6", "Create a box plot of Fare by Pclass."),
        ("I7", "Create a box plot of Age by Survived."),
        ("I8", "Create a scatter plot of Age versus Fare."),
        ("I9", "Create a correlation heatmap."),
        ("I10", "Create a stacked bar chart of Sex and Survived."),
        ("I11", "Plot passenger counts by Embarked."),
        ("I12", "Visualize missing values by column."),
        ("I13", "Plot the distribution of Age_normalized if available."),
        ("I14", "Create a chart comparing average Fare for survivors and non-survivors."),
    ],
    "J": [
        ("J1", "Cluster passengers into three groups using K-Means."),
        ("J2", "Apply DBSCAN and identify noise points."),
        ("J3", "Perform PCA on suitable numerical features."),
        ("J4", "Train a decision tree to predict Survived."),
        ("J5", "Train logistic regression to predict Survived."),
        ("J6", "Train a random forest classifier for Survived."),
        ("J7", "Compare classification models using the same test split."),
        ("J8", "Show the confusion matrix for the best model."),
        ("J9", "Calculate accuracy, precision, recall, and F1-score."),
        ("J10", "Calculate ROC-AUC when appropriate."),
        ("J11", "Show feature importance for a trained tree-based model."),
        ("J12", "Use cross-validation to evaluate the classifier."),
        ("J13", "Explain class imbalance in the Survived column."),
        ("J14", "Identify potential data leakage in the features."),
        ("J15", "Explain the limitations of predicting survival from this dataset."),
    ],
    "K": [
        ("K1", "Summarize the most important findings in the dataset."),
        ("K2", "What factors are associated with survival?"),
        ("K3", "Which passenger class had the highest survival rate?"),
        ("K4", "Explain the survival chart in simple language."),
        ("K5", "Which columns have the most missing values?"),
        ("K6", "What should be cleaned before machine learning?"),
        ("K7", "Recommend suitable visualizations for this dataset."),
        ("K8", "Which features should be excluded from a predictive model and why?"),
        ("K9", "Compare the results before and after normalization."),
        ("K10", "Give me a final KDD analysis report."),
    ],
}


def test_single_task(code, question, df):
    # Try plan_query
    plan, err = plan_query(question, df)
    if err:
        return {"code": code, "status": "FAIL_PLAN", "err": err, "plan": None}
    
    # Try execute_plan
    res = execute_plan(df, plan)
    if res.get("status") == "error":
        return {"code": code, "status": "FAIL_EXEC", "err": res.get("message") or res.get("error"), "plan": plan}
    
    return {"code": code, "status": "PASS", "task": plan.get("task"), "plan": plan}


def main():
    csv_path = "Titanic-Dataset.csv"
    df, err = load_dataset(csv_path)
    if err:
        print(f"Error loading {csv_path}: {err}", flush=True)
        sys.exit(1)
    print(f"Loaded Titanic dataset: {df.shape[0]} rows, {df.shape[1]} columns.\n", flush=True)

    summary = {}
    for section, tasks in TASKS.items():
        print(f"=== Testing Section {section} ({len(tasks)} queries) ===", flush=True)
        section_results = []
        for code, q in tasks:
            r = test_single_task(code, q, df)
            status = r["status"]
            if status == "PASS":
                print(f"  [{code}] PASS: task={r['task']}", flush=True)
            else:
                print(f"  [{code}] {status}: '{q}' -> {r['err']}", flush=True)
            section_results.append(r)
        summary[section] = section_results
        print(flush=True)

    total_tasks = sum(len(t) for t in TASKS.values())
    total_passed = sum(1 for res_list in summary.values() for r in res_list if r["status"] == "PASS")
    print(f"TOTAL: {total_passed}/{total_tasks} passed.", flush=True)


if __name__ == "__main__":
    main()
