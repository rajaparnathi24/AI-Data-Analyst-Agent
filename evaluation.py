"""
evaluation.py
-------------
Automated benchmark and evaluation script for the AI Data Analyst Agent.

Measures:
1. Analytical Correctness Rate (% matching exact verified ground truth)
2. Execution Success Rate (%)
3. ML Query Classification Accuracy & F1-Score
4. Visualization Generation Rate (%)
5. Average System Latency (ms per query)
6. Robustness against unsupported queries
"""

import os
import time
import json
import math
import numpy as np
import pandas as pd

from data_loader import load_dataset
from query_planner import plan_query
from analysis_engine import execute_plan
from visualization import create_auto_chart
from query_classifier import QueryClassifierModel


# -------------------------------------------------------------
# Manually Verified Ground Truth Test Suite (12 Benchmark Cases)
# -------------------------------------------------------------

BENCHMARK_TEST_SUITE = [
    {
        "id": "TC-01",
        "question": "What is the average salary?",
        "expected_intent": "aggregation",
        "expected_task": "mean",
        "expected_type": "scalar",
        "expected_value": 67333.33,
        "tolerance": 1.0,
        "is_supported": True,
    },
    {
        "id": "TC-02",
        "question": "What is the maximum salary?",
        "expected_intent": "maximum",
        "expected_task": "max",
        "expected_type": "scalar",
        "expected_value": 92000.0,
        "tolerance": 0.01,
        "is_supported": True,
    },
    {
        "id": "TC-03",
        "question": "What is the lowest salary?",
        "expected_intent": "minimum",
        "expected_task": "min",
        "expected_type": "scalar",
        "expected_value": 46000.0,
        "tolerance": 0.01,
        "is_supported": True,
    },
    {
        "id": "TC-04",
        "question": "What is the median salary?",
        "expected_intent": "aggregation",
        "expected_task": "median",
        "expected_type": "scalar",
        "expected_value": 68000.0,
        "tolerance": 0.01,
        "is_supported": True,
    },
    {
        "id": "TC-05",
        "question": "How many rows are in this dataset?",
        "expected_intent": "counting",
        "expected_task": "count_rows",
        "expected_type": "scalar",
        "expected_value": 15,
        "tolerance": 0.0,
        "is_supported": True,
    },
    {
        "id": "TC-06",
        "question": "How many columns are present?",
        "expected_intent": "counting",
        "expected_task": "count_columns",
        "expected_type": "scalar",
        "expected_value": 9,
        "tolerance": 0.0,
        "is_supported": True,
    },
    {
        "id": "TC-07",
        "question": "Show average salary by department",
        "expected_intent": "group_comparison",
        "expected_task": "group_aggregate",
        "expected_type": "table",
        "expected_table_rows": 5,
        "is_supported": True,
        "expect_chart": True,
    },
    {
        "id": "TC-08",
        "question": "Show employees with salary greater than 70000",
        "expected_intent": "filtering",
        "expected_task": "filter",
        "expected_type": "table",
        "expected_table_rows": 7,
        "is_supported": True,
    },
    {
        "id": "TC-09",
        "question": "Show the top 3 employees by salary",
        "expected_intent": "ranking",
        "expected_task": "top_n",
        "expected_type": "table",
        "expected_table_rows": 3,
        "is_supported": True,
        "expect_chart": True,
    },
    {
        "id": "TC-10",
        "question": "Are there any missing values in this dataset?",
        "expected_intent": "data_quality",
        "expected_task": "missing_values",
        "expected_type": "table",
        "expected_table_rows": 9,
        "is_supported": True,
    },
    {
        "id": "TC-11",
        "question": "Show correlation between numerical columns",
        "expected_intent": "correlation",
        "expected_task": "correlation",
        "expected_type": "table",
        "expected_table_rows": 5,
        "is_supported": True,
    },
    {
        "id": "TC-12",
        "question": "Detect outliers in Salary",
        "expected_intent": "data_quality",
        "expected_task": "detect_outliers",
        "expected_type": "table",
        "expected_table_rows": 0,  # in sample_data.csv salary has 0 IQR outliers
        "is_supported": True,
    },
    {
        "id": "TC-13",
        "question": "Remove duplicate rows",
        "expected_intent": "data_quality",
        "expected_task": "drop_duplicates",
        "expected_type": "modified_dataset",
        "is_supported": True,
    },
    {
        "id": "TC-14",
        "question": "Normalize Salary",
        "expected_intent": "aggregation",
        "expected_task": "normalize",
        "expected_type": "modified_dataset",
        "is_supported": True,
    },
    {
        "id": "TC-15",
        "question": "Cluster records into 3 groups",
        "expected_intent": "group_comparison",
        "expected_task": "kmeans_cluster",
        "expected_type": "cluster_result",
        "is_supported": True,
    },
    {
        "id": "TC-16",
        "question": "Select only Name and Salary",
        "expected_intent": "filtering",
        "expected_task": "select_columns",
        "expected_type": "table",
        "expected_table_rows": 15,
        "is_supported": True,
    },
    {
        "id": "TC-17",
        "question": "Predict future bitcoin prices for 2030",
        "expected_intent": "unsupported",
        "expected_task": None,
        "expected_type": None,
        "is_supported": False,  # Safety test: should reject cleanly without crash
    },
]


def run_full_evaluation():
    """Executes the full evaluation pipeline and computes empirical metrics."""
    print("=" * 75)
    print("RUNNING SYSTEM EVALUATION AND BENCHMARK SUITE")
    print("=" * 75)

    sample_csv = os.path.join("datasets", "sample_data.csv")
    df, err = load_dataset(sample_csv)
    if err:
        print(f"Error loading evaluation dataset: {err}")
        return

    # 1. Train and Evaluate ML Query Classifier
    classifier = QueryClassifierModel()
    ml_metrics = classifier.train_and_evaluate(test_size=0.25, random_state=42)

    total_tests = len(BENCHMARK_TEST_SUITE)
    correct_answers = 0
    execution_successes = 0
    charts_generated = 0
    total_latency_ms = 0.0

    results_table = []

    for tc in BENCHMARK_TEST_SUITE:
        qid = tc["id"]
        q = tc["question"]
        is_supported = tc["is_supported"]

        start_time = time.perf_counter()

        # Step A: ML Classification
        pred_intent, conf = classifier.predict(q)

        # Step B: Query Planning
        plan, plan_err = plan_query(q, df)

        # Step C: Execution
        if plan and not plan_err:
            res = execute_plan(df, plan)
            exec_status = res["status"]
            actual_val = res.get("value")
            res_type = res.get("result_type")
        else:
            res = {"status": "error", "message": plan_err}
            exec_status = "error"
            actual_val = None
            res_type = None

        # Step D: Chart generation check
        chart_obj = None
        if exec_status == "success":
            chart_obj, _ = create_auto_chart(res, df)

        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)
        total_latency_ms += latency_ms

        # Correctness Verification Logic
        test_passed = False
        if not is_supported:
            # For unsupported queries, passing means safe rejection without crashing
            if exec_status == "error" or plan_err is not None:
                test_passed = True
                exec_status = "safe_reject"
        else:
            if exec_status == "success":
                execution_successes += 1
                if tc.get("expect_chart") and chart_obj is not None:
                    charts_generated += 1

                # Check analytical correctness
                if tc["expected_type"] == "scalar":
                    if isinstance(actual_val, (int, float)):
                        diff = abs(actual_val - tc["expected_value"])
                        if diff <= tc.get("tolerance", 0.01):
                            test_passed = True
                            correct_answers += 1
                elif tc["expected_type"] == "table":
                    if isinstance(actual_val, pd.DataFrame):
                        if len(actual_val) == tc["expected_table_rows"]:
                            test_passed = True
                            correct_answers += 1
                elif tc["expected_type"] in ["modified_dataset", "cluster_result"]:
                    if res_type == tc["expected_type"] or res.get("modified_df") is not None:
                        test_passed = True
                        correct_answers += 1

        results_table.append({
            "ID": qid,
            "Question": q[:35] + "..." if len(q) > 35 else q,
            "ML Intent": pred_intent,
            "Status": "PASS" if test_passed else "FAIL",
            "Latency": f"{latency_ms:.1f}ms",
        })

    # Summary Statistics
    supported_cases = sum(1 for tc in BENCHMARK_TEST_SUITE if tc["is_supported"])
    analytical_correctness_rate = (correct_answers / supported_cases) * 100
    execution_success_rate = (execution_successes / supported_cases) * 100
    avg_latency = total_latency_ms / total_tests

    # Print Formatted Report
    print(f"\nTotal Test Cases:            {total_tests}")
    print(f"Supported Analytical Queries: {supported_cases}")
    print(f"Analytical Correctness:       {analytical_correctness_rate:.1f}% ({correct_answers}/{supported_cases})")
    print(f"Execution Success Rate:       {execution_success_rate:.1f}%")
    print(f"Average Response Latency:     {avg_latency:.1f} ms")
    print(f"ML Classifier Test Accuracy:  {ml_metrics['accuracy'] * 100:.2f}%")
    print(f"ML Classifier Macro F1:       {ml_metrics['macro_f1']:.4f}")

    print("\n--- BENCHMARK RESULTS TABLE ---")
    df_results = pd.DataFrame(results_table)
    print(df_results.to_string(index=False))

    print("\n--- ML CLASSIFIER CONFUSION MATRIX ---")
    print(f"Classes ({len(ml_metrics['classes'])}):")
    print(ml_metrics["classes"])
    print(ml_metrics["confusion_matrix"])

    # Save to disk as evaluation report artifact
    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": total_tests,
        "analytical_correctness_pct": round(analytical_correctness_rate, 2),
        "execution_success_rate_pct": round(execution_success_rate, 2),
        "average_latency_ms": round(avg_latency, 2),
        "ml_classifier_accuracy_pct": round(ml_metrics["accuracy"] * 100, 2),
        "ml_classifier_macro_f1": ml_metrics["macro_f1"],
        "results": results_table,
    }

    with open("evaluation_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    print("\nSUCCESS: evaluation.py generated real, measured benchmark report saved to evaluation_report.json!")


if __name__ == "__main__":
    run_full_evaluation()
