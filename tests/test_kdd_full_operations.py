"""
test_kdd_full_operations.py
---------------------------
Exhaustive automated test suite verifying all 5 KDD stages across different data types:
1. Selection (select_columns, sample_rows, slice_rows, select_by_dtype, identify_id_columns, identify_target, multi-condition filter)
2. Preprocessing & Cleaning (trim_whitespace, text_case, standardize_categories, to_numeric, to_datetime, handle_infinite, detect_outliers, cap_outliers, remove_outliers, cleaning_summary, ffill/bfill)
3. Transformation (robust_scale, log_transform, bin_column, one_hot_encode, extract_date_part, create_derived_column, create_conditional_column, pivot_table, crosstab)
4. Data Mining (Spearman correlation, K-Means, DBSCAN, Decision Tree, Logistic Regression, Linear Regression, PCA)
5. Presentation & Visualizations (auto-charts for clustering, confusion matrices, actual vs predicted, PCA)
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_engine import execute_plan
from query_planner import plan_query, validate_plan
from visualization import (
    create_box_plot,
    create_pie_chart,
    create_time_series_chart,
    create_cluster_scatter,
    create_confusion_matrix_plot,
    create_actual_vs_predicted_plot,
    create_pca_scatter,
    create_auto_chart,
)


class TestKDDFullOperations(unittest.TestCase):

    def setUp(self):
        # 1. Deterministic fixture: Multi-type tabular dataset
        self.df = pd.DataFrame({
            "Emp_ID": [101, 102, 103, 104, 105, 106, 107, 108],
            "Name": [" Alice ", "Bob", " Charlie", "David ", "Eve", "Frank ", "Grace", "Heidi"],
            "Department": ["IT", "IT", "HR", "HR", "Sales", "Sales", "IT", "Sales"],
            "Gender": ["female", "MALE", "Female", "male", "female", "Male", "Female", "female"],
            "Salary": [75000.0, 82000.0, 50000.0, 52000.0, 61000.0, 63000.0, 150000.0, 60000.0],  # 150000 is outlier
            "Rating": [4.5, 3.8, 3.2, 4.0, 4.8, 3.5, 4.2, 3.9],
            "JoinDate": ["2020-01-15", "2021-03-20", "2019-06-01", "2022-11-10", "2020-08-25", "2023-02-14", "2018-04-12", "2021-07-30"],
            "NumericStr": ["10", "20", "invalid", "40", "50", "60", "70", "80"],
            "MissingCol": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0],
            "Survived": [1, 0, 1, 0, 1, 0, 1, 1],  # Binary classification target
        })

    def tearDown(self):
        plt.close("all")

    # =========================================================
    # STAGE 1: SELECTION TESTS
    # =========================================================

    def test_select_columns(self):
        res = execute_plan(self.df, {"task": "select_columns", "columns": ["Name", "Salary"]})
        self.assertEqual(res["status"], "success")
        self.assertEqual(list(res["value"].columns), ["Name", "Salary"])
        self.assertEqual(len(res["value"]), 8)

    def test_sample_rows(self):
        res = execute_plan(self.df, {"task": "sample_rows", "n": 4})
        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["value"]), 4)

    def test_slice_rows(self):
        res = execute_plan(self.df, {"task": "slice_rows", "start": 2, "end": 5})
        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["value"]), 3)

    def test_select_by_dtype(self):
        res_num = execute_plan(self.df, {"task": "select_by_dtype", "dtype": "numeric"})
        self.assertEqual(res_num["status"], "success")
        self.assertIn("Salary", res_num["value"].columns)
        self.assertNotIn("Name", res_num["value"].columns)

        res_cat = execute_plan(self.df, {"task": "select_by_dtype", "dtype": "categorical"})
        self.assertEqual(res_cat["status"], "success")
        self.assertIn("Name", res_cat["value"].columns)

    def test_identify_id_columns(self):
        res = execute_plan(self.df, {"task": "identify_id_columns"})
        self.assertEqual(res["status"], "success")
        self.assertIn("Emp_ID", res["value"]["Potential_ID_Columns"].values)

    def test_identify_target(self):
        res = execute_plan(self.df, {"task": "identify_target"})
        self.assertEqual(res["status"], "success")
        self.assertIn("Survived", res["value"]["Candidate_Target_Columns"].values)

    def test_multi_condition_filter(self):
        plan = {
            "task": "filter",
            "conditions": [
                {"col": "Department", "op": "==", "val": "IT"},
                {"col": "Salary", "op": ">", "val": 80000},
            ],
            "logic": "and",
        }
        res = execute_plan(self.df, plan)
        self.assertEqual(res["status"], "success")
        # 102 and 107 match
        self.assertEqual(len(res["value"]), 2)

    # =========================================================
    # STAGE 2: PREPROCESSING & CLEANING TESTS
    # =========================================================

    def test_ffill_and_bfill(self):
        res_f = execute_plan(self.df, {"task": "fill_missing", "column": "MissingCol", "strategy": "ffill"})
        self.assertEqual(res_f["status"], "success")
        self.assertEqual(res_f["modified_df"]["MissingCol"].isnull().sum(), 0)

        res_b = execute_plan(self.df, {"task": "fill_missing", "column": "MissingCol", "strategy": "bfill"})
        self.assertEqual(res_b["status"], "success")
        self.assertEqual(res_b["modified_df"]["MissingCol"].isnull().sum(), 0)

    def test_trim_whitespace(self):
        res = execute_plan(self.df, {"task": "trim_whitespace"})
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["modified_df"]["Name"].iloc[0], "Alice")
        self.assertEqual(res["modified_df"]["Name"].iloc[2], "Charlie")

    def test_text_case(self):
        res = execute_plan(self.df, {"task": "text_case", "column": "Gender", "case": "upper"})
        self.assertEqual(res["status"], "success")
        self.assertTrue((res["modified_df"]["Gender"].str.isupper()).all())

    def test_standardize_categories(self):
        res = execute_plan(self.df, {"task": "standardize_categories", "column": "Gender"})
        self.assertEqual(res["status"], "success")
        unique_genders = set(res["modified_df"]["Gender"].unique())
        self.assertEqual(unique_genders, {"Female", "Male"})

    def test_to_numeric_with_coercion_report(self):
        res = execute_plan(self.df, {"task": "to_numeric", "column": "NumericStr"})
        self.assertEqual(res["status"], "success")
        self.assertTrue(pd.api.types.is_numeric_dtype(res["modified_df"]["NumericStr"]))
        self.assertEqual(res["modified_df"]["NumericStr"].isnull().sum(), 1)  # 'invalid' coerced to NaN

    def test_to_datetime_conversion(self):
        res = execute_plan(self.df, {"task": "to_datetime", "column": "JoinDate"})
        self.assertEqual(res["status"], "success")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(res["modified_df"]["JoinDate"]))

    def test_handle_infinite_values(self):
        inf_df = self.df.copy()
        inf_df.loc[0, "Salary"] = np.inf
        res = execute_plan(inf_df, {"task": "handle_infinite"})
        self.assertEqual(res["status"], "success")
        self.assertFalse(np.isinf(res["modified_df"]["Salary"]).any())

    def test_outlier_detection_and_capping(self):
        res_detect = execute_plan(self.df, {"task": "detect_outliers", "column": "Salary"})
        self.assertEqual(res_detect["status"], "success")
        self.assertGreaterEqual(len(res_detect["value"]), 1)  # 150000 is detected

        res_cap = execute_plan(self.df, {"task": "cap_outliers", "column": "Salary"})
        self.assertEqual(res_cap["status"], "success")
        self.assertLess(res_cap["modified_df"]["Salary"].max(), 150000.0)

        res_rem = execute_plan(self.df, {"task": "remove_outliers", "column": "Salary"})
        self.assertEqual(res_rem["status"], "success")
        self.assertEqual(len(res_rem["modified_df"]), 7)

    def test_cleaning_summary(self):
        res = execute_plan(self.df, {"task": "cleaning_summary"})
        self.assertEqual(res["status"], "success")
        self.assertIn("Total Rows", res["value"]["Metric"].values)

    # =========================================================
    # STAGE 3: TRANSFORMATION TESTS
    # =========================================================

    def test_robust_scaling(self):
        res = execute_plan(self.df, {"task": "robust_scale", "column": "Salary"})
        self.assertEqual(res["status"], "success")
        self.assertIn("Salary_robust", res["modified_df"].columns)

    def test_log_transformation(self):
        res = execute_plan(self.df, {"task": "log_transform", "column": "Salary"})
        self.assertEqual(res["status"], "success")
        self.assertIn("Salary_log", res["modified_df"].columns)

    def test_binning(self):
        res_eq = execute_plan(self.df, {"task": "bin_column", "column": "Rating", "bins": 3, "bin_method": "equal_width"})
        self.assertEqual(res_eq["status"], "success")
        self.assertIn("Rating_binned", res_eq["modified_df"].columns)

        res_q = execute_plan(self.df, {"task": "bin_column", "column": "Salary", "bins": 3, "bin_method": "quantile"})
        self.assertEqual(res_q["status"], "success")
        self.assertIn("Salary_binned", res_q["modified_df"].columns)

    def test_one_hot_encoding(self):
        res = execute_plan(self.df, {"task": "one_hot_encode", "column": "Department"})
        self.assertEqual(res["status"], "success")
        self.assertIn("Department_IT", res["modified_df"].columns)
        self.assertIn("Department_HR", res["modified_df"].columns)

    def test_extract_date_part(self):
        # Convert JoinDate first
        df_dt = self.df.copy()
        df_dt["JoinDate"] = pd.to_datetime(df_dt["JoinDate"])
        res = execute_plan(df_dt, {"task": "extract_date_part", "column": "JoinDate", "part": "year"})
        self.assertEqual(res["status"], "success")
        self.assertIn("JoinDate_year", res["modified_df"].columns)
        self.assertEqual(res["modified_df"]["JoinDate_year"].iloc[0], 2020)

    def test_derived_and_conditional_column(self):
        res_derived = execute_plan(self.df, {
            "task": "create_derived_column",
            "col1": "Salary",
            "col2": "Rating",
            "operator": "*",
            "new_column_name": "ScoreWeightedSalary",
        })
        self.assertEqual(res_derived["status"], "success")
        self.assertIn("ScoreWeightedSalary", res_derived["modified_df"].columns)

        res_cond = execute_plan(self.df, {
            "task": "create_conditional_column",
            "condition_col": "Salary",
            "operator": ">",
            "condition_val": 70000,
            "new_column_name": "HighSalary",
        })
        self.assertEqual(res_cond["status"], "success")
        self.assertIn("HighSalary", res_cond["modified_df"].columns)

    def test_pivot_and_crosstab(self):
        res_piv = execute_plan(self.df, {
            "task": "pivot_table",
            "index": "Department",
            "pivot_columns": "Gender",
            "values": "Salary",
        })
        self.assertEqual(res_piv["status"], "success")
        self.assertIsInstance(res_piv["value"], pd.DataFrame)

        res_ct = execute_plan(self.df, {
            "task": "crosstab",
            "column1": "Department",
            "column2": "Gender",
        })
        self.assertEqual(res_ct["status"], "success")
        self.assertIsInstance(res_ct["value"], pd.DataFrame)

    # =========================================================
    # STAGE 4: DATA MINING TESTS
    # =========================================================

    def test_spearman_correlation(self):
        res = execute_plan(self.df, {"task": "correlation", "method": "spearman"})
        self.assertEqual(res["status"], "success")
        self.assertIn("Variable", res["value"].columns)

    def test_kmeans_clustering(self):
        res = execute_plan(self.df, {"task": "kmeans_cluster", "n_clusters": 3})
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["result_type"], "cluster_result")
        self.assertIn("Cluster", res["modified_df"].columns)
        self.assertEqual(len(res["modified_df"]["Cluster"].unique()), 3)

    def test_dbscan_clustering(self):
        res = execute_plan(self.df, {"task": "dbscan_cluster", "eps": 1.5, "min_samples": 2})
        self.assertEqual(res["status"], "success")
        self.assertIn("Cluster", res["modified_df"].columns)

    def test_train_classification(self):
        res_tree = execute_plan(self.df, {"task": "train_classification", "target": "Survived", "model": "decision_tree"})
        self.assertEqual(res_tree["status"], "success")
        self.assertEqual(res_tree["result_type"], "ml_model")
        val = res_tree["value"]
        self.assertIn("accuracy", val)
        self.assertIn("confusion_matrix", val)
        self.assertIn("f1_score", val)

        res_log = execute_plan(self.df, {"task": "train_classification", "target": "Survived", "model": "logistic_regression"})
        self.assertEqual(res_log["status"], "success")

    def test_train_regression(self):
        res_reg = execute_plan(self.df, {"task": "train_regression", "target": "Salary"})
        self.assertEqual(res_reg["status"], "success")
        val = res_reg["value"]
        self.assertIn("mae", val)
        self.assertIn("rmse", val)
        self.assertIn("r2", val)
        self.assertIn("actual_vs_predicted", val)

    def test_pca_reduction(self):
        res = execute_plan(self.df, {"task": "pca_reduction"})
        self.assertEqual(res["status"], "success")
        self.assertIn("PCA1", res["value"].columns)
        self.assertIn("PCA2", res["value"].columns)

    # =========================================================
    # STAGE 5: VISUALIZATION RENDERING TESTS
    # =========================================================

    def test_box_plot_generation(self):
        fig, err = create_box_plot(self.df, col="Salary", by_col="Department")
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    def test_pie_chart_generation(self):
        fig, err = create_pie_chart(self.df, col="Department")
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    def test_time_series_generation(self):
        fig, err = create_time_series_chart(self.df, date_col="JoinDate", val_col="Salary")
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    def test_cluster_scatter_generation(self):
        clustered_df = self.df.copy()
        clustered_df["Cluster"] = [0, 1, 0, 1, 2, 0, 1, 2]
        fig, err = create_cluster_scatter(clustered_df, x_col="Rating", y_col="Salary")
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    def test_confusion_matrix_plot(self):
        cm = [[4, 1], [0, 5]]
        fig, err = create_confusion_matrix_plot(cm, labels=["Class 0", "Class 1"])
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    def test_actual_vs_predicted_plot(self):
        y_true = [50000, 60000, 70000, 80000]
        y_pred = [52000, 58000, 71000, 79000]
        fig, err = create_actual_vs_predicted_plot(y_true, y_pred)
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    def test_pca_scatter_plot(self):
        pca_df = pd.DataFrame({"PCA1": [-1.2, 0.5, 1.4], "PCA2": [0.3, -0.8, 0.5]})
        fig, err = create_pca_scatter(pca_df)
        self.assertIsNone(err)
        self.assertIsNotNone(fig)

    # =========================================================
    # NATURAL LANGUAGE QUERY PLANNER TESTS FOR NEW OPERATIONS
    # =========================================================

    def test_nl_planning_new_operations(self):
        p_sel, _ = plan_query("select only Name and Salary", self.df)
        self.assertIsNotNone(p_sel)
        self.assertEqual(p_sel["task"], "select_columns")

        p_sample, _ = plan_query("give me a sample of 3 records", self.df)
        self.assertEqual(p_sample["task"], "sample_rows")

        p_outliers, _ = plan_query("detect outliers in salary", self.df)
        self.assertEqual(p_outliers["task"], "detect_outliers")

        p_cap, _ = plan_query("cap outliers in salary", self.df)
        self.assertEqual(p_cap["task"], "cap_outliers")

        p_trim, _ = plan_query("trim whitespace from text", self.df)
        self.assertEqual(p_trim["task"], "trim_whitespace")

        p_robust, _ = plan_query("apply robust scaling to salary", self.df)
        self.assertEqual(p_robust["task"], "robust_scale")

        p_bin, _ = plan_query("bin salary into 3 bins", self.df)
        self.assertEqual(p_bin["task"], "bin_column")

        p_kmeans, _ = plan_query("cluster records into 3 groups", self.df)
        self.assertEqual(p_kmeans["task"], "kmeans_cluster")

        p_pca, _ = plan_query("perform pca dimensionality reduction", self.df)
        self.assertEqual(p_pca["task"], "pca_reduction")


if __name__ == "__main__":
    unittest.main()
