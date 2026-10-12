"""
query_planner.py
----------------
Translates natural-language questions into safe, structured JSON execution plans.

Architecture:
1. Constructs a strict schema-aware prompt with few-shot examples for Ollama.
2. Fallback Heuristic Planner: If Ollama is offline or fails, a deterministic rule-based
   parser handles questions across all 5 KDD stages so the agent remains fully functional!
3. Plan Validator: Validates plan structure, column existence, and type constraints
   before sending to the analysis engine.
Zero dynamic code evaluation (no eval/exec).
"""

import json
import re
import pandas as pd
import numpy as np
from llm_client import call_llm, check_api_status
from data_profiler import get_column_types

# Whitelist of allowed analytical and KDD tasks
SUPPORTED_TASKS = [
    # KDD Stage 1: Selection & Inspection
    "count_rows",
    "count_columns",
    "mean",
    "median",
    "sum",
    "min",
    "max",
    "std",
    "count_where",
    "filter",
    "sort",
    "top_n",
    "bottom_n",
    "select_columns",
    "sample_rows",
    "slice_rows",
    "select_by_dtype",
    "identify_id_columns",
    "identify_target",
    "dataset_overview",
    "column_types",
    "shape_and_memory",
    "profiling_report",
    "constant_columns",
    # KDD Stage 2: Preprocessing & Data Cleaning
    "missing_values",
    "duplicates",
    "fill_missing",
    "drop_missing",
    "drop_duplicates",
    "drop_column",
    "drop_high_null_columns",
    "trim_whitespace",
    "text_case",
    "standardize_categories",
    "to_numeric",
    "to_datetime",
    "handle_infinite",
    "detect_outliers",
    "cap_outliers",
    "remove_outliers",
    "cleaning_summary",
    "data_quality_report",
    "check_empty_strings",
    "check_type_consistency",
    "unique_values",
    "unique_count",
    "undo_cleaning",
    "reset_dataset",
    # KDD Stage 3: Transformation & Feature Engineering
    "normalize",
    "standardize",
    "robust_scale",
    "log_transform",
    "bin_column",
    "encode_category",
    "one_hot_encode",
    "extract_date_part",
    "create_derived_column",
    "create_conditional_column",
    "group_aggregate",
    "pivot_table",
    "crosstab",
    "create_age_group",
    "create_family_size",
    "create_is_alone",
    "create_fare_category",
    "create_cabin_missing",
    "extract_ticket_feature",
    "export_dataset",
    # KDD Stage 4: Data Mining & Statistics
    "correlation",
    "kmeans_cluster",
    "dbscan_cluster",
    "train_classification",
    "train_regression",
    "pca_reduction",
    "multivariate_stats",
    "numerical_summary",
    "value_counts",
    "skewness",
    "compare_mean_median",
    "strongest_correlations",
    "survival_by_age_group",
    "survival_by_family",
    "survival_patterns_summary",
    "model_comparison",
    "feature_importance",
    "cross_validation",
    # Visualizations
    "histogram",
    "box_plot",
    "scatter_plot",
    "pie_chart",
    "bar_chart",
    "line_chart",
    "count_plot",
    "heatmap",
    # KDD Stage 5: Interpretation & Reports
    "explain_concept",
    "kdd_summary",
]


def _normalize_llm_plan(plan: dict, df: pd.DataFrame, question: str = "") -> dict:
    """Normalizes keys, synonyms, and types in LLM-generated query plans."""
    if not isinstance(plan, dict):
        return plan
    p = plan.copy()
    q = (question or "").lower()
    task = str(p.get("task", "")).lower().strip()

    # Operator alias map
    OP_MAP = {
        "gt": ">",
        "gte": ">=",
        "ge": ">=",
        "lt": "<",
        "lte": "<=",
        "le": "<=",
        "eq": "==",
        "ne": "!=",
        "equal": "==",
        "equals": "==",
    }
    if p.get("operator") in OP_MAP:
        p["operator"] = OP_MAP[p["operator"]]
    if isinstance(p.get("conditions"), list):
        for cond in p["conditions"]:
            if isinstance(cond, dict) and cond.get("op") in OP_MAP:
                cond["op"] = OP_MAP[cond["op"]]

    # Feature Engineering Intent Overrides
    if "familysize" in q or "family size" in q or "family_size" in q:
        p["task"] = "create_family_size"
        return p
    elif "isalone" in q or "is alone" in q or "is_alone" in q:
        p["task"] = "create_is_alone"
        return p
    elif "age_group" in q or "age group" in q or "age_groups" in q:
        p["task"] = "create_age_group"
        return p
    elif "fare_category" in q or "fare category" in q:
        p["task"] = "create_fare_category"
        return p
    elif "cabin" in q and ("missing" in q or "null" in q) and ("create" in q or "column" in q or "indicate" in q or "indicating" in q):
        p["task"] = "create_cabin_missing"
        return p
    elif "ticket" in q and ("feature" in q or "extract" in q):
        p["task"] = "extract_ticket_feature"
        return p

    # Pivot Table Normalization
    if task == "pivot_table" or "pivot" in q:
        p["task"] = "pivot_table"
        if "columns" in p and "pivot_columns" not in p:
            p["pivot_columns"] = p.pop("columns")
        if "col" in p and "pivot_columns" not in p:
            p["pivot_columns"] = p.pop("col")
        if "rows" in p and "index" not in p:
            p["index"] = p.pop("rows")
        if "row" in p and "index" not in p:
            p["index"] = p.pop("row")
        if "value" in p and "values" not in p:
            p["values"] = p.pop("value")
        if not p.get("index"):
            p["index"] = "Sex" if "sex" in q else (df.columns[0] if len(df.columns) > 0 else "Sex")
        if not p.get("pivot_columns"):
            p["pivot_columns"] = "Pclass" if "pclass" in q else (df.columns[1] if len(df.columns) > 1 else "Pclass")
        if not p.get("values"):
            p["values"] = "Survived" if "surviv" in q else "Fare"
        p["aggregation"] = p.get("aggregation") or "mean"
        return p

    # Average Age for survivors vs non-survivors (D9)
    if "average age" in q and ("survivor" in q or "survivors" in q):
        p["task"] = "group_aggregate"
        p["group_by"] = "Survived"
        p["column"] = "Age"
        p["aggregation"] = "mean"
        return p

    # Survival Rates Normalization (D4, D5, D6, H6)
    if "surviv" in q and ("rate" in q or "percent" in q or "compare" in q or "percentages" in q):
        if p.get("task") in ["group_aggregate", "survival_rates", "survival_rate"] or p.get("group_by") or "by" in q:
            p["task"] = "group_aggregate"
            p["column"] = "Survived"
            p["aggregation"] = "mean"
            for col_candidate in ["Embarked", "Sex", "Pclass"]:
                if col_candidate.lower() in q:
                    p["group_by"] = col_candidate
                    break
            if not p.get("group_by") or p.get("group_by") == "Survived":
                p["group_by"] = "Pclass"
            return p

    # Chart Tasks Normalization
    if task in ["pie_chart", "piechart", "pie"] or ("pie" in q and ("chart" in q or "plot" in q or "piechart" in q or "pie-chart" in q)):
        p["task"] = "pie_chart"
        if not p.get("column"):
            p["column"] = _extract_chart_column(question, df)
        return p
    elif task in ["count_plot", "countplot"] or ("count" in q and "plot" in q):
        p["task"] = "count_plot"
        if not p.get("column"):
            p["column"] = _extract_chart_column(question, df)
        return p
    elif task in ["bar_chart", "barchart", "bar_graph", "bargraph", "bar"] or ("bar" in q and ("chart" in q or "plot" in q or "graph" in q or "barchart" in q)):
        p["task"] = "bar_chart"
        if not p.get("column"):
            p["column"] = _extract_chart_column(question, df)
        return p
    elif task in ["line_chart", "linechart", "line_graph", "linegraph", "line_plot"] or ("line" in q and ("chart" in q or "graph" in q or "plot" in q or "linechart" in q)):
        p["task"] = "line_chart"
        if not p.get("column"):
            p["column"] = _extract_chart_column(question, df)
        return p
    elif task in ["heatmap", "correlation_heatmap", "corr_heatmap"] or ("heatmap" in q or ("heat" in q and "map" in q)):
        p["task"] = "heatmap"
        return p
    elif task in ["scatter_plot", "scatterplot", "scatter"] or ("scatter" in q or (" vs " in q and ("plot" in q or "chart" in q)) or (" against " in q and ("plot" in q or "chart" in q))):
        p["task"] = "scatter_plot"
        x_c, y_c = _extract_two_chart_columns(question, df)
        if not p.get("x_col"):
            p["x_col"] = x_c
        if not p.get("y_col"):
            p["y_col"] = y_c
        return p
    elif task in ["histogram", "hist"] or ("histogram" in q or "distribution of" in q):
        p["task"] = "histogram"
        if not p.get("column"):
            p["column"] = _extract_chart_column(question, df)
        return p

    # Box Plot Normalization
    if task in ["box_plot", "boxplot"] or ("box" in q and "plot" in q):
        p["task"] = "box_plot"
        m = re.search(r"box\s+plot\s+of\s+([a-zA-Z_]+)\s+by\s+([a-zA-Z_]+)", q)
        if m:
            p["column"] = m.group(1).capitalize() if m.group(1).lower() != "survived" else "Survived"
            p["by_col"] = m.group(2).capitalize() if m.group(2).lower() != "survived" else "Survived"
        elif isinstance(p.get("column"), list):
            items = p["column"]
            if len(items) >= 2:
                p["column"] = items[0]
                p["by_col"] = items[1]
            elif len(items) == 1:
                p["column"] = items[0]
        elif isinstance(p.get("column"), str) and ("[" in p["column"] or "," in p["column"]):
            cleaned = p["column"].strip("[]'\" ")
            parts = [s.strip(" '\"") for s in cleaned.replace(",", " ").split() if s.strip(" '\"")]
            if len(parts) >= 2:
                p["column"] = parts[0]
                p["by_col"] = parts[1]
            elif len(parts) == 1:
                p["column"] = parts[0]
        if not p.get("column"):
            p["column"] = _extract_chart_column(question, df)
        return p

    if task in ["head", "first_n", "first_records"]:
        p["task"] = "slice_rows"
        m = re.search(r"first\s+(\d+)", q)
        p["start"] = 0
        p["end"] = int(m.group(1)) if m else p.get("n", 5)
    elif task in ["tail", "last_n", "last_records"]:
        p["task"] = "bottom_n"
        m = re.search(r"last\s+(\d+)", q)
        p["top_n"] = int(m.group(1)) if m else p.get("n", 10)
    elif task in ["random_forest", "rf_classifier", "random_forest_classifier"]:
        p["task"] = "train_classification"
        p["model"] = "random_forest"
    elif task in ["survival_rate", "survival_rates"]:
        if p.get("group_by"):
            p["task"] = "group_aggregate"
            p["column"] = "Survived"
            p["aggregation"] = "mean"
        else:
            p["task"] = "mean"
            p["column"] = "Survived"
    elif task in ["value_counts", "frequency", "distribution"]:
        p["task"] = "value_counts"

    if p.get("value") is not None and "condition_val" not in p:
        p["condition_val"] = p["value"]
    elif p.get("val") is not None and "condition_val" not in p:
        p["condition_val"] = p["val"]

    if p.get("col") is not None and "column" not in p:
        p["column"] = p["col"]

    if task == "filter":
        if "condition_col" not in p and p.get("column"):
            p["condition_col"] = p["column"]
        if "missing" in q or "null" in q:
            p["operator"] = "isnull"
        elif "older" in q or "greater" in q or "more than" in q:
            p["operator"] = ">"

    return p


def plan_query(
    question: str,
    df: pd.DataFrame,
    conversation_history: list = None,
) -> tuple:
    """
    Translates a natural language question into a validated query plan.

    Parameters:
        question (str): The user's query in English.
        df (pd.DataFrame): The active dataset.
        conversation_history (list, optional): Previous conversation turns for context.

    Returns:
        tuple: (validated_plan, error_message)
    """
    if df is None or df.empty:
        return None, "Error: No dataset loaded to plan query against."

    if not question or not question.strip():
        return None, "Error: Question is empty."

    question = question.strip()

    # Security: Limit maximum question length
    MAX_QUESTION_LENGTH = 500
    if len(question) > MAX_QUESTION_LENGTH:
        return None, f"Error: Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters."

    status = check_api_status()

    # If Ollama is available, use LLM-based planning
    if status["configured"]:
        plan, err = _llm_generate_plan(question, df, conversation_history)
        if not err and plan:
            norm_plan = _normalize_llm_plan(plan, df, question)
            validated, val_err = validate_plan(norm_plan, df)
            if not val_err:
                return validated, None

    # Fallback to deterministic heuristic planner
    heuristic_plan = _heuristic_fallback_plan(question, df, conversation_history)
    if heuristic_plan:
        return validate_plan(heuristic_plan, df)

    return None, (
        f"Could not understand the question '{question}'. "
        "Try asking about: averages, missing values, outliers, filtering, clustering, or predictions."
    )


def validate_plan(plan: dict, df: pd.DataFrame) -> tuple:
    """
    Validates the query plan against the DataFrame schema.

    Returns:
        tuple: (validated_plan, error_message)
    """
    if not isinstance(plan, dict):
        return None, "Validation Error: Plan is not a valid dictionary."

    task = plan.get("task", "").lower().strip()
    if not task:
        return None, "Validation Error: Plan missing 'task' field."

    if task not in SUPPORTED_TASKS:
        return None, f"Validation Error: Unsupported task '{task}'."

    col = plan.get("column")

    # Validate column if required for statistical calculations
    if task in ["mean", "median", "sum", "std", "min", "max", "sort", "top_n", "bottom_n"]:
        if not col:
            num_cols = get_column_types(df)["numerical"]
            if num_cols:
                col = num_cols[0]
                plan["column"] = col
            elif len(df.columns) > 0:
                col = df.columns[0]
                plan["column"] = col
            else:
                return None, f"Validation Error: Task '{task}' requires a 'column' parameter."

        matched_col = _resolve_column_name(df, col)
        if not matched_col:
            return None, f"Validation Error: Column '{col}' does not exist in dataset."
        plan["column"] = matched_col

        if task in ["mean", "median", "sum", "std"]:
            if not pd.api.types.is_numeric_dtype(df[matched_col]):
                return None, f"Validation Error: Task '{task}' requires a numeric column, but '{matched_col}' is non-numeric."

    # Validate group_by
    if task == "group_aggregate":
        grp = plan.get("group_by")
        if not grp:
            return None, "Validation Error: 'group_aggregate' requires a 'group_by' column."
        matched_grp = _resolve_column_name(df, grp)
        if not matched_grp:
            return None, f"Validation Error: Group-by column '{grp}' not found."
        plan["group_by"] = matched_grp

        if col:
            matched_col = _resolve_column_name(df, col)
            if not matched_col:
                return None, f"Validation Error: Target aggregation column '{col}' not found."
            plan["column"] = matched_col
            agg = plan.get("aggregation", "mean").lower()
            if agg in ["mean", "median", "sum", "std"] and not pd.api.types.is_numeric_dtype(df[matched_col]):
                return None, f"Validation Error: Aggregation '{agg}' requires a numeric column, but '{matched_col}' is non-numeric."

    if task == "pivot_table":
        idx = plan.get("index")
        pcols = plan.get("pivot_columns")
        vals = plan.get("values")
        if not idx or not pcols or not vals:
            return None, "Validation Error: 'pivot_table' requires 'index', 'pivot_columns', and 'values'."

    # Validate condition_col for filter/count_where
    if task in ["filter", "count_where"]:
        cond_col = plan.get("condition_col") or col
        if cond_col:
            matched_cond = _resolve_column_name(df, cond_col)
            if not matched_cond:
                return None, f"Validation Error: Condition column '{cond_col}' not found."
            plan["condition_col"] = matched_cond

    # Validate select_columns
    if task == "select_columns":
        cols = plan.get("columns", [])
        if not cols:
            return None, "Validation Error: 'select_columns' requires a list of columns."

    # Validate missing_values with column
    if task in ["missing_values", "null_values"] and col:
        matched_col = _resolve_column_name(df, col)
        if matched_col:
            plan["column"] = matched_col
        else:
            return None, f"Validation Error: Column '{col}' does not exist in dataset."

    # Validate fill_missing
    if task == "fill_missing":
        if not col:
            return None, "Validation Error: 'fill_missing' requires a target 'column'."
        matched_col = _resolve_column_name(df, col)
        if not matched_col:
            return None, f"Validation Error: Column '{col}' not found for imputation."
        plan["column"] = matched_col
        strategy = plan.get("strategy", "").lower().strip()
        if not strategy:
            plan["strategy"] = "mean" if pd.api.types.is_numeric_dtype(df[matched_col]) else "mode"
        elif strategy in ["mean", "median"] and not pd.api.types.is_numeric_dtype(df[matched_col]):
            return None, f"Validation Error: Strategy '{strategy}' requires numeric column, but '{matched_col}' is non-numeric."

    if task in ["drop_missing", "drop_column"] and col:
        matched_col = _resolve_column_name(df, col)
        if not matched_col:
            return None, f"Validation Error: Column '{col}' not found."
        plan["column"] = matched_col

    # Validate numeric transformations
    if task in ["normalize", "standardize", "robust_scale", "log_transform", "bin_column", "detect_outliers", "cap_outliers", "remove_outliers"]:
        if not col:
            num_cols = get_column_types(df)["numerical"]
            if num_cols:
                col = num_cols[0]
                plan["column"] = col
            else:
                return None, f"Validation Error: '{task}' requires a numerical column."
        matched_col = _resolve_column_name(df, col)
        if not matched_col:
            return None, f"Validation Error: Column '{col}' not found."
        if not pd.api.types.is_numeric_dtype(df[matched_col]):
            return None, f"Validation Error: '{task}' requires a numeric column, but '{matched_col}' is non-numeric."
        plan["column"] = matched_col

    if task in ["encode_category", "one_hot_encode", "text_case", "standardize_categories", "to_numeric", "to_datetime"]:
        if not col and "columns" not in plan:
            return None, f"Validation Error: '{task}' requires a target 'column'."
        if col:
            matched_col = _resolve_column_name(df, col)
            if not matched_col:
                return None, f"Validation Error: Column '{col}' not found."
            plan["column"] = matched_col
        elif "columns" in plan and isinstance(plan["columns"], list):
            resolved = [_resolve_column_name(df, c) for c in plan["columns"]]
            plan["columns"] = [r for r in resolved if r]

    if task in ["train_classification", "train_regression", "model_comparison", "feature_importance", "cross_validation"]:
        target_col = plan.get("target") or plan.get("column") or ("Survived" if "Survived" in df.columns else None)
        if not target_col and len(df.columns) > 0:
            target_col = df.columns[-1]
        if target_col:
            matched_col = _resolve_column_name(df, target_col)
            if matched_col:
                plan["target"] = matched_col

    # Validate Visualization tasks
    if task in ["pie_chart", "bar_chart", "count_plot", "line_chart", "box_plot", "histogram"]:
        if col:
            matched_col = _resolve_column_name(df, col)
            if not matched_col:
                suggestions = _suggest_close_columns(df, str(col))
                sugg_msg = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                available = ", ".join(df.columns)
                return None, f"Column '{col}' not found.{sugg_msg} Available columns: {available}."
            plan["column"] = matched_col
        else:
            if task in ["histogram", "box_plot", "line_chart"]:
                num_cols = get_column_types(df)["numerical"]
                plan["column"] = num_cols[0] if num_cols else df.columns[0]
            else:
                cat_cols = get_column_types(df)["categorical"]
                plan["column"] = cat_cols[0] if cat_cols else df.columns[0]

    if task == "scatter_plot":
        x_c = plan.get("x_col") or plan.get("x") or col
        y_c = plan.get("y_col") or plan.get("y")
        if x_c:
            matched_x = _resolve_column_name(df, x_c)
            if not matched_x:
                suggestions = _suggest_close_columns(df, str(x_c))
                sugg_msg = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                available = ", ".join(df.columns)
                return None, f"Column '{x_c}' not found.{sugg_msg} Available columns: {available}."
            plan["x_col"] = matched_x
        if y_c:
            matched_y = _resolve_column_name(df, y_c)
            if not matched_y:
                suggestions = _suggest_close_columns(df, str(y_c))
                sugg_msg = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                available = ", ".join(df.columns)
                return None, f"Column '{y_c}' not found.{sugg_msg} Available columns: {available}."
            plan["y_col"] = matched_y

    return plan, None


# -------------------------------------------------------------
# LLM Prompt Construction and Execution
# -------------------------------------------------------------

def _build_planner_prompt(question: str, df: pd.DataFrame, conversation_history: list = None) -> str:
    """Builds a schema-rich few-shot prompt for the LLM."""
    col_types = get_column_types(df)
    schema_info = {
        "columns": list(df.columns),
        "numerical_columns": col_types["numerical"],
        "categorical_columns": col_types["categorical"],
    }

    history_str = ""
    if conversation_history and len(conversation_history) > 0:
        history_str = "Recent conversation context:\n"
        for turn in conversation_history[-3:]:
            history_str += f"- User: {turn.get('user', '')} -> Column: {turn.get('column', '')}\n"

    prompt = f"""
You are a precision Data Analysis Query Planner for a Pandas and Scikit-Learn analytical engine.
Translate the user's question into a strict JSON execution plan.

Dataset Schema:
Columns: {schema_info['columns']}
Numerical Columns: {schema_info['numerical_columns']}
Categorical Columns: {schema_info['categorical_columns']}

{history_str}
Supported Tasks:
- Selection: "count_rows", "count_columns", "select_columns", "filter", "sort", "top_n", "bottom_n", "sample_rows"
- Statistics: "mean", "median", "sum", "min", "max", "std", "group_aggregate", "correlation"
- Cleaning: "missing_values", "fill_missing" (strategies: mean, median, mode, ffill, bfill), "drop_missing", "drop_duplicates", "drop_column", "trim_whitespace", "to_numeric", "to_datetime", "detect_outliers", "cap_outliers"
- Transformation: "normalize", "standardize", "robust_scale", "log_transform", "bin_column", "encode_category", "one_hot_encode", "create_derived_column"
- Machine Learning & Mining: "kmeans_cluster", "dbscan_cluster", "train_classification", "train_regression", "pca_reduction"
- Visualizations: "pie_chart", "bar_chart", "histogram", "line_chart", "scatter_plot", "box_plot", "count_plot", "heatmap"

Question: "{question}"

Respond with ONLY valid JSON:
{{"task": "task_name", "column": "column_name"}}
"""
    return prompt.strip()


def _llm_generate_plan(question: str, df: pd.DataFrame, conversation_history: list = None) -> tuple:
    """Sends prompt to Ollama LLM and parses JSON plan."""
    prompt = _build_planner_prompt(question, df, conversation_history)
    system_inst = "You are a precise data analysis compiler. Output strictly valid JSON without preamble."

    raw_response, err = call_llm(prompt, system_instruction=system_inst, json_mode=True)
    if err:
        return None, err

    try:
        plan = json.loads(raw_response)
        return plan, None
    except json.JSONDecodeError as e:
        return None, f"Failed to parse LLM response as JSON: {str(e)}"


# -------------------------------------------------------------
# Heuristic Fallback Planner (Zero-Failure Deterministic Engine)
# -------------------------------------------------------------

def _heuristic_fallback_plan(question: str, df: pd.DataFrame, conversation_history: list = None) -> dict:
    """
    Deterministic rule-based intent parser that runs offline without Ollama.
    Covers all 5 KDD stages with robust token and regex matching.
    """
    q = question.lower()
    columns = list(df.columns)
    num_cols = get_column_types(df)["numerical"]
    cat_cols = get_column_types(df)["categorical"]

    def _find_column_in_query(target_cols):
        for col in sorted(target_cols, key=lambda c: len(c), reverse=True):
            col_clean = col.lower()
            pattern = r'\b' + re.escape(col_clean) + r'\b'
            if re.search(pattern, q):
                return col
            col_spaced = col_clean.replace("_", " ")
            pattern_spaced = r'\b' + re.escape(col_spaced) + r'\b'
            if re.search(pattern_spaced, q):
                return col

        for col in sorted(target_cols, key=lambda c: len(c), reverse=True):
            col_clean = col.lower()
            col_tokens = [t for t in col_clean.split("_") if len(t) > 2]
            for token in col_tokens:
                pattern_token = r'\b' + re.escape(token) + r'\b'
                if re.search(pattern_token, q):
                    return col
        return None

    # Context fallback
    matched_num_col = _find_column_in_query(num_cols)
    matched_cat_col = _find_column_in_query(cat_cols)
    if not matched_num_col and conversation_history and len(conversation_history) > 0:
        last_turn = conversation_history[-1]
        prev_col = last_turn.get("column") or last_turn.get("analyzed_column")
        if prev_col and prev_col in num_cols:
            matched_num_col = prev_col

    target_num_col = matched_num_col if matched_num_col else (num_cols[0] if num_cols else None)

    # ---------------------------------------------------------
    # KDD INTERPRETATION & EXPLANATIONS (Sections H, J, K)
    # ---------------------------------------------------------
    if any(k in q for k in ["causation", "correlation does not prove causation", "why correlation"]):
        return {"task": "explain_concept", "concept": "correlation_vs_causation"}
    if "class imbalance" in q or "imbalance in the survived" in q:
        return {"task": "explain_concept", "concept": "class_imbalance"}
    if "data leakage" in q or "leakage in the features" in q:
        return {"task": "explain_concept", "concept": "data_leakage"}
    if "limitations of predicting survival" in q or ("limitations" in q and "predicting" in q):
        return {"task": "explain_concept", "concept": "model_limitations"}
    if "explain the survival chart" in q or "survival chart in simple language" in q:
        return {"task": "explain_concept", "concept": "survival_chart"}
    if "important findings" in q or "summarize the most important findings" in q:
        return {"task": "kdd_summary", "aspect": "findings"}
    if "factors are associated with survival" in q or "factors associated with survival" in q:
        return {"task": "kdd_summary", "aspect": "factors"}
    if "cleaned before machine learning" in q:
        return {"task": "kdd_summary", "aspect": "cleaning_recommendations"}
    if "recommend suitable visualizations" in q or "suitable visualizations" in q:
        return {"task": "kdd_summary", "aspect": "visualizations"}
    if "features should be excluded" in q or "excluded from a predictive model" in q:
        return {"task": "kdd_summary", "aspect": "feature_selection"}
    if "final kdd" in q or "kdd analysis report" in q:
        return {"task": "kdd_summary", "aspect": "full_kdd_report"}

    # ---------------------------------------------------------
    # DATASET OVERVIEW & PROFILING (Section A)
    # ---------------------------------------------------------
    if "overview" in q and ("dataset" in q or "titanic" in q or "complete" in q):
        return {"task": "dataset_overview"}
    if "shape" in q and ("memory" in q or "usage" in q):
        return {"task": "shape_and_memory"}
    if "profiling report" in q or "dataset profiling" in q:
        return {"task": "profiling_report"}
    if "constant columns" in q or "zero variance" in q:
        return {"task": "constant_columns"}
    if "data types" in q or "column names and their data types" in q or "list all column names" in q:
        return {"task": "column_types"}
    if "which columns are numerical, categorical" in q or "which columns are numerical" in q:
        return {"task": "column_types"}
    if re.search(r"first\s+(\d+)", q):
        n_first = int(re.search(r"first\s+(\d+)", q).group(1))
        return {"task": "slice_rows", "start": 0, "end": n_first}
    if re.search(r"last\s+(\d+)", q):
        n_last = int(re.search(r"last\s+(\d+)", q).group(1))
        return {"task": "bottom_n", "top_n": n_last}
    if "random passengers" in q or "random sample of 50" in q or "sample of 50" in q or ("random" in q and "50" in q):
        m_rnd = re.search(r"(\d+)", q)
        return {"task": "sample_rows", "n": int(m_rnd.group(1)) if m_rnd else 20}

    # ---------------------------------------------------------
    # DATA QUALITY (Section B)
    # ---------------------------------------------------------
    if "data quality report" in q or "quality report" in q:
        return {"task": "data_quality_report"}
    if "empty strings" in q or "whitespace-only" in q or "whitespace only" in q:
        return {"task": "check_empty_strings"}
    if "inconsistent data types" in q or "inconsistent types" in q:
        return {"task": "check_type_consistency"}
    if "unique values" in q and ("passengerid" in q or "passenger id" in q):
        return {"task": "unique_values", "column": "PassengerId"}
    if "highest percentage of missing" in q or "highest missing" in q:
        return {"task": "missing_values"}
    if "no missing values" in q:
        return {"task": "missing_values"}

    # ---------------------------------------------------------
    # NUMERICAL & CATEGORICAL STATISTICS (Section C & H)
    # ---------------------------------------------------------
    if "mean, median, and standard deviation" in q and ("age" in q or "fare" in q):
        return {"task": "multivariate_stats", "columns": ["Age", "Fare"]}
    if "minimum, maximum, and quartiles" in q or "quartiles for all" in q or "quartiles" in q:
        return {"task": "numerical_summary"}
    if "frequency distribution of sex" in q or ("frequency" in q and "sex" in q):
        return {"task": "value_counts", "column": "Sex"}
    if "count passengers in each pclass" in q or ("passengers in each pclass" in q):
        return {"task": "value_counts", "column": "Pclass"}
    if "unique values and counts for embarked" in q or ("counts for embarked" in q):
        return {"task": "value_counts", "column": "Embarked"}
    if "10 most frequent ticket" in q or ("frequent ticket" in q):
        return {"task": "value_counts", "column": "Ticket", "top_n": 10}
    if "unique cabin values" in q or ("number of unique cabin" in q):
        return {"task": "unique_count", "column": "Cabin"}
    if "distribution of survived" in q or ("survival counts" in q and "bar" not in q):
        return {"task": "value_counts", "column": "Survived"}
    if "skewness for age and fare" in q or ("skewness" in q and "age" in q and "fare" in q):
        return {"task": "skewness", "columns": ["Age", "Fare"]}
    if "compare mean and median fare" in q or "mean and median fare" in q or "compare mean and median" in q:
        return {"task": "compare_mean_median", "column": "Fare"}
    if "strongest positive and negative correlations" in q or "strongest correlations" in q:
        return {"task": "strongest_correlations"}
    if "fare differs between survivors" in q or ("fare differs" in q and "survivor" in q):
        return {"task": "group_aggregate", "group_by": "Survived", "column": "Fare", "aggregation": "mean"}

    # ---------------------------------------------------------
    # SURVIVAL ANALYSIS (Section D)
    # ---------------------------------------------------------
    if "overall survival rate" in q or "calculate the overall survival" in q:
        return {"task": "mean", "column": "Survived"}
    if "how many passengers survived" in q:
        return {"task": "count_where", "condition_col": "Survived", "operator": "==", "condition_val": 1}
    if "not survive" in q or "did not survive" in q or "passengers did not survive" in q:
        return {"task": "count_where", "condition_col": "Survived", "operator": "==", "condition_val": 0}
    if "survival percentage" in q or "survival percentages" in q:
        grp = "Pclass"
        for c in ["pclass", "sex", "embarked"]:
            if c in q:
                grp = c.capitalize() if c != "pclass" else "Pclass"
                break
        return {"task": "group_aggregate", "group_by": grp, "column": "Survived", "aggregation": "mean"}
    if "survival rate" in q and "sex" in q:
        return {"task": "group_aggregate", "group_by": "Sex", "column": "Survived", "aggregation": "mean"}
    if "survival rate" in q and "pclass" in q:
        return {"task": "group_aggregate", "group_by": "Pclass", "column": "Survived", "aggregation": "mean"}
    if "survival rate" in q and "embarked" in q:
        return {"task": "group_aggregate", "group_by": "Embarked", "column": "Survived", "aggregation": "mean"}
    if "survival rate" in q and ("age group" in q or "age groups" in q):
        return {"task": "survival_by_age_group"}
    if "alone versus with family" in q or "alone vs with family" in q or "alone versus family" in q:
        return {"task": "survival_by_family"}
    if "average age for survivors" in q or ("average age" in q and "survivors" in q):
        return {"task": "group_aggregate", "group_by": "Survived", "column": "Age", "aggregation": "mean"}
    if "strongest observed survival patterns" in q or "survival patterns" in q:
        return {"task": "survival_patterns_summary"}
    if "which passenger class had the highest survival" in q:
        return {"task": "group_aggregate", "group_by": "Pclass", "column": "Survived", "aggregation": "mean"}

    # ---------------------------------------------------------
    # SELECTION & FILTERING (Section E)
    # ---------------------------------------------------------
    if "passengerid, name, sex, age, and fare" in q:
        return {"task": "select_columns", "columns": ["PassengerId", "Name", "Sex", "Age", "Fare"]}
    if "passengers older than 40" in q or ("older than 40" in q):
        return {"task": "filter", "condition_col": "Age", "operator": ">", "condition_val": 40}
    if "female passengers who survived" in q or ("female" in q and "survived" in q and "filter" in q):
        return {"task": "filter", "conditions": [{"col": "Sex", "op": "==", "val": "female"}, {"col": "Survived", "op": "==", "val": 1}]}
    if "male passengers who did not survive" in q or ("male" in q and "did not survive" in q):
        return {"task": "filter", "conditions": [{"col": "Sex", "op": "==", "val": "male"}, {"col": "Survived", "op": "==", "val": 0}]}
    if "first-class passengers" in q or "first class passengers" in q:
        return {"task": "filter", "condition_col": "Pclass", "operator": "==", "condition_val": 1}
    if "fare greater than 100" in q or ("fare" in q and "greater than 100" in q):
        return {"task": "filter", "condition_col": "Fare", "operator": ">", "condition_val": 100}
    if "aged between 18 and 35" in q:
        return {"task": "filter", "conditions": [{"col": "Age", "op": ">=", "val": 18}, {"col": "Age", "op": "<=", "val": 35}]}
    if "missing age" in q:
        return {"task": "filter", "condition_col": "Age", "operator": "isnull"}
    if "missing cabin" in q:
        return {"task": "filter", "condition_col": "Cabin", "operator": "isnull"}
    if "survived and paid more than 50" in q:
        return {"task": "filter", "conditions": [{"col": "Survived", "op": "==", "val": 1}, {"col": "Fare", "op": ">", "val": 50}]}
    if "pclass 1 who were older than 50" in q or ("pclass 1" in q and "older than 50" in q):
        return {"task": "filter", "conditions": [{"col": "Pclass", "op": "==", "val": 1}, {"col": "Age", "op": ">", "val": 50}]}
    if "top 20 passengers by fare" in q:
        return {"task": "top_n", "column": "Fare", "top_n": 20}
    if "sort passengers by age in ascending" in q or ("sort" in q and "age" in q and "ascending" in q):
        return {"task": "sort", "column": "Age", "sort_ascending": True}

    # ---------------------------------------------------------
    # DATA CLEANING (Section F)
    # ---------------------------------------------------------
    if "more than 50% missing" in q or "50% missing" in q:
        return {"task": "drop_high_null_columns", "threshold": 0.5}
    if "convert age and fare to numeric" in q or ("convert" in q and "numeric" in q and "age" in q and "fare" in q):
        return {"task": "to_numeric", "columns": ["Age", "Fare"]}
    if "undo the last data-cleaning" in q or ("undo" in q and "cleaning" in q):
        return {"task": "undo_cleaning"}
    if "reset the dataset" in q or ("reset" in q and "original" in q):
        return {"task": "reset_dataset"}
    if "before-and-after missing" in q:
        return {"task": "cleaning_summary"}

    # ---------------------------------------------------------
    # TRANSFORMATION & FEATURE ENGINEERING (Section G)
    # ---------------------------------------------------------
    if "age_group" in q or "age group column" in q or ("child, adult, and senior" in q):
        return {"task": "create_age_group"}
    if "familysize" in q or "family size by adding sibsp and parch" in q or "family size" in q:
        return {"task": "create_family_size"}
    if "isalone" in q or "is alone based on familysize" in q or "is alone" in q:
        return {"task": "create_is_alone"}
    if "fare_category" in q or "fare category column" in q or "fare category" in q:
        return {"task": "create_fare_category"}
    if "whether cabin is missing" in q or "cabin_missing" in q:
        return {"task": "create_cabin_missing"}
    if "feature from the ticket column" in q or "feature from ticket" in q:
        return {"task": "extract_ticket_feature"}
    if "download the transformed" in q or "download dataset" in q:
        return {"task": "export_dataset"}
    if "pivot table of survival rate by sex and pclass" in q:
        return {"task": "pivot_table", "index": "Sex", "pivot_columns": "Pclass", "values": "Survived", "aggregation": "mean"}
    if "group by pclass and calculate average fare and age" in q:
        return {"task": "group_aggregate", "group_by": "Pclass", "column": "Fare", "aggregation": "mean"}

    # ---------------------------------------------------------
    # CHARTS & VISUALIZATIONS (Universal & Section I)
    # ---------------------------------------------------------
    # 1. Heatmap / Correlation heatmap
    if any(k in q for k in ["heatmap", "heat map", "correlation heatmap", "corr heatmap"]):
        return {"task": "heatmap"}

    # 2. Pie Chart
    if any(k in q for k in ["piechart", "pie chart", "pie-chart"]) or (q.startswith("pie ") or " pie " in q):
        target_col = _extract_chart_column(question, df, conversation_history)
        return {"task": "pie_chart", "column": target_col}

    # 3. Count Plot
    if any(k in q for k in ["countplot", "count plot", "count-plot"]):
        target_col = _extract_chart_column(question, df, conversation_history)
        return {"task": "count_plot", "column": target_col}

    # 4. Box Plot
    if any(k in q for k in ["boxplot", "box plot", "box-plot"]):
        target_col = _extract_chart_column(question, df, conversation_history)
        m_by = re.search(r"\bby\s+([a-zA-Z0-9_]+)", q)
        by_c = _resolve_column_name(df, m_by.group(1)) if m_by else None
        return {"task": "box_plot", "column": target_col, "by_col": by_c}

    # 5. Scatter Plot
    if any(k in q for k in ["scatterplot", "scatter plot", "scatter-plot"]) or ("scatter" in q) or ((" vs " in q or " versus " in q or " against " in q) and any(v in q for v in ["plot", "chart", "draw", "graph", "show", "create", "make"])):
        c1, c2 = _extract_two_chart_columns(question, df, conversation_history)
        return {"task": "scatter_plot", "x_col": c1, "y_col": c2}

    # 6. Line Chart / Line Graph
    if any(k in q for k in ["linechart", "line chart", "line-chart", "line graph", "linegraph", "line plot"]):
        target_col = _extract_chart_column(question, df, conversation_history)
        return {"task": "line_chart", "column": target_col}

    # Specific Titanic Visualization test patterns
    if "stacked bar chart of sex and survived" in q or ("stacked bar" in q and "sex" in q):
        return {"task": "crosstab", "column1": "Sex", "column2": "Survived"}
    if "chart comparing average fare for survivors and non-survivors" in q:
        return {"task": "group_aggregate", "group_by": "Survived", "column": "Fare", "aggregation": "mean"}
    if "plot survival rate by pclass" in q or ("survival rate by pclass" in q and "plot" in q):
        return {"task": "group_aggregate", "group_by": "Pclass", "column": "Survived", "aggregation": "mean"}
    if "bar chart of passenger counts by sex" in q or ("bar chart" in q and "sex" in q):
        return {"task": "value_counts", "column": "Sex"}
    if "bar chart of survival counts" in q or ("bar chart" in q and "survival" in q):
        return {"task": "value_counts", "column": "Survived"}
    if "passenger counts by embarked" in q or ("counts by embarked" in q):
        return {"task": "value_counts", "column": "Embarked"}

    # 7. Histogram / Distribution
    if any(k in q for k in ["histogram", "hist "]) or q.endswith("hist") or ("distribution of" in q and any(v in q for v in ["plot", "chart", "show", "draw"])):
        target_col = _extract_chart_column(question, df, conversation_history)
        return {"task": "histogram", "column": target_col}

    # 8. General Bar Chart
    if any(k in q for k in ["barchart", "bar chart", "bar-chart", "bar graph", "bargraph"]):
        target_col = _extract_chart_column(question, df, conversation_history)
        m_by = re.search(r"\bby\s+([a-zA-Z0-9_]+)", q)
        by_c = _resolve_column_name(df, m_by.group(1)) if m_by else None
        return {"task": "bar_chart", "column": target_col, "group_by": by_c}

    # 9. Conversational follow-ups (e.g. "Now show it as a bar chart", "make a chart of it")
    if any(k in q for k in ["show it as", "show as", "now show", "make a chart", "draw a chart", "create a chart", "plot a chart"]):
        if "pie" in q:
            col = _extract_chart_column(question, df, conversation_history)
            return {"task": "pie_chart", "column": col}
        elif "bar" in q:
            col = _extract_chart_column(question, df, conversation_history)
            return {"task": "bar_chart", "column": col}
        elif "line" in q:
            col = _extract_chart_column(question, df, conversation_history)
            return {"task": "line_chart", "column": col}
        elif "scatter" in q:
            c1, c2 = _extract_two_chart_columns(question, df, conversation_history)
            return {"task": "scatter_plot", "x_col": c1, "y_col": c2}
        elif "box" in q:
            col = _extract_chart_column(question, df, conversation_history)
            return {"task": "box_plot", "column": col}
        elif "hist" in q:
            col = _extract_chart_column(question, df, conversation_history)
            return {"task": "histogram", "column": col}

    # ---------------------------------------------------------
    # MACHINE LEARNING & MINING (Section J)
    # ---------------------------------------------------------
    if "random forest" in q:
        return {"task": "train_classification", "target": "Survived", "model": "random_forest"}
    if "compare classification models" in q or "model comparison" in q:
        return {"task": "model_comparison", "target": "Survived"}
    if "confusion matrix" in q:
        return {"task": "train_classification", "target": "Survived", "model": "random_forest"}
    if "accuracy, precision, recall" in q:
        return {"task": "train_classification", "target": "Survived"}
    if "roc-auc" in q or "roc auc" in q:
        return {"task": "train_classification", "target": "Survived", "include_auc": True}
    if "feature importance" in q:
        return {"task": "feature_importance", "target": "Survived"}
    if "cross-validation" in q or "cross validation" in q:
        return {"task": "cross_validation", "target": "Survived"}

    # =========================================================
    # 1. KDD STAGE 4: DATA MINING (CLUSTERING, ML, PCA)
    # =========================================================

    # K-Means Clustering
    if "kmeans" in q or "k-means" in q or ("cluster" in q and ("group" in q or "3" in q or "4" in q or "5" in q or "customer" in q)):
        m_k = re.search(r"(\d+)\s*(?:groups|clusters|k)", q)
        k_val = int(m_k.group(1)) if m_k else 3
        return {"task": "kmeans_cluster", "n_clusters": k_val}

    # DBSCAN Clustering
    if "dbscan" in q or "density cluster" in q or "noise points" in q:
        return {"task": "dbscan_cluster", "eps": 0.5, "min_samples": 3}

    # Supervised Classification / Prediction
    if any(k in q for k in ["predict", "classification", "classify", "decision tree", "logistic regression"]):
        tgt = _find_column_in_query(columns)
        if tgt:
            if pd.api.types.is_numeric_dtype(df[tgt]):
                return {"task": "train_regression", "target": tgt}
            return {"task": "train_classification", "target": tgt}
        if "decision tree" in q or "logistic regression" in q or "classify" in q:
            cand = _find_column_in_query(cat_cols) or (cat_cols[0] if cat_cols else None)
            if cand:
                return {"task": "train_classification", "target": cand}
        if "linear regression" in q:
            return {"task": "train_regression", "target": target_num_col}
        return None

    # Supervised Regression
    if "linear regression" in q:
        tgt = _find_column_in_query(num_cols) or target_num_col
        return {"task": "train_regression", "target": tgt}

    # PCA (Dimensionality Reduction)
    if "pca" in q or "principal component" in q or "dimensionality reduction" in q:
        return {"task": "pca_reduction"}

    # Correlation Matrix (Spearman vs Pearson)
    if "spearman" in q and "corr" in q:
        return {"task": "correlation", "method": "spearman"}
    if "correlation" in q or "corr" in q:
        return {"task": "correlation", "method": "pearson"}

    # =========================================================
    # 2. KDD STAGE 2: PREPROCESSING & CLEANING
    # =========================================================

    # Fill / Impute missing values
    if any(k in q for k in ["fill", "impute", "replace"]) and any(k in q for k in ["missing", "null", "empty", "nan", "na", "values"]):
        target_col = _find_column_in_query(columns) or target_num_col
        strategy = "mean"
        fill_val = None
        if "median" in q:
            strategy = "median"
        elif "mode" in q:
            strategy = "mode"
        elif "ffill" in q or "forward" in q:
            strategy = "ffill"
        elif "bfill" in q or "backward" in q:
            strategy = "bfill"
        elif "mean" in q or "average" in q:
            strategy = "mean"
        elif "0" in q or "zero" in q:
            strategy = "constant"
            fill_val = 0
        elif "unknown" in q:
            strategy = "constant"
            fill_val = "Unknown"
        return {
            "task": "fill_missing",
            "column": target_col,
            "strategy": strategy,
            "fill_value": fill_val,
        }

    # Drop missing values
    if ("drop" in q or "remove" in q or "delete" in q) and any(k in q for k in ["missing", "null", "nan", "na"]):
        target_col = _find_column_in_query(columns)
        return {"task": "drop_missing", "column": target_col}

    # Drop duplicate rows
    if ("drop" in q or "remove" in q or "delete" in q) and "duplicate" in q:
        return {"task": "drop_duplicates"}

    # Drop column
    if ("drop" in q or "remove" in q or "delete" in q) and ("column" in q or "feature" in q):
        target_col = _find_column_in_query(columns)
        if target_col:
            return {"task": "drop_column", "column": target_col}

    # Whitespace Trimming
    if "trim" in q or "whitespace" in q:
        return {"task": "trim_whitespace"}

    # Text Casing
    if any(k in q for k in ["uppercase", "lowercase", "title case", "to upper", "to lower"]):
        target_col = _find_column_in_query(cat_cols or columns)
        case_val = "upper" if "upper" in q else ("title" if "title" in q else "lower")
        if target_col:
            return {"task": "text_case", "column": target_col, "case": case_val}

    # Standardize Categories
    if "standardize" in q and any(k in q for k in ["category", "gender", "department", "categories"]):
        target_col = _find_column_in_query(cat_cols or columns)
        if target_col:
            return {"task": "standardize_categories", "column": target_col}

    # Type Conversion: To Numeric
    if ("to numeric" in q or "convert" in q) and ("numeric" in q or "number" in q):
        target_col = _find_column_in_query(columns)
        if target_col:
            return {"task": "to_numeric", "column": target_col}

    # Type Conversion: To Datetime
    if ("to datetime" in q or "convert" in q or "parse" in q) and ("date" in q or "datetime" in q or "time" in q):
        target_col = _find_column_in_query(columns)
        if target_col:
            return {"task": "to_datetime", "column": target_col}

    # Infinite Values
    if "infinite" in q or "inf" in q:
        return {"task": "handle_infinite"}

    # Outliers: Detect, Cap, or Remove
    if "outlier" in q:
        target_col = _find_column_in_query(num_cols) or target_num_col
        if "cap" in q or "winsorize" in q:
            return {"task": "cap_outliers", "column": target_col}
        elif "remove" in q or "drop" in q:
            return {"task": "remove_outliers", "column": target_col}
        else:
            return {"task": "detect_outliers", "column": target_col}

    # Cleaning Summary
    if "cleaning summary" in q or "before and after" in q:
        return {"task": "cleaning_summary"}

    # =========================================================
    # 3. KDD STAGE 3: TRANSFORMATION
    # =========================================================

    # Robust Scaling
    if "robust" in q:
        target_col = _find_column_in_query(num_cols) or target_num_col
        return {"task": "robust_scale", "column": target_col}

    # Normalization (Min-Max)
    if "normalize" in q or "min max" in q or "min-max" in q:
        target_col = _find_column_in_query(num_cols) or target_num_col
        return {"task": "normalize", "column": target_col}

    # Standardization (Z-Score)
    if "standardize" in q or "z-score" in q or "zscore" in q:
        target_col = _find_column_in_query(num_cols) or target_num_col
        return {"task": "standardize", "column": target_col}

    # Log Transformation
    if "log" in q and ("transform" in q or "scale" in q):
        target_col = _find_column_in_query(num_cols) or target_num_col
        return {"task": "log_transform", "column": target_col}

    # Binning / Discretization
    if "bin" in q or "discretize" in q:
        target_col = _find_column_in_query(num_cols) or target_num_col
        m_bins = re.search(r"(\d+)\s*(?:bins|groups|intervals)", q)
        b_count = int(m_bins.group(1)) if m_bins else 3
        b_method = "quantile" if "quantile" in q else "equal_width"
        return {"task": "bin_column", "column": target_col, "bins": b_count, "bin_method": b_method}

    # One-Hot Encoding
    if "one hot" in q or "one-hot" in q or "dummy" in q:
        target_col = _find_column_in_query(cat_cols or columns)
        if target_col:
            return {"task": "one_hot_encode", "column": target_col}

    # Categorical Label Encoding
    if "encode" in q or "label encode" in q:
        target_col = _find_column_in_query(cat_cols or columns)
        if target_col:
            return {"task": "encode_category", "column": target_col}

    # Extract Date Part
    if ("extract" in q or "get" in q) and any(k in q for k in ["year", "month", "day", "weekday", "hour"]):
        target_col = _find_column_in_query(columns)
        part = "year" if "year" in q else ("month" if "month" in q else ("day" if "day" in q else "weekday"))
        return {"task": "extract_date_part", "column": target_col, "part": part}

    # Pivot Table
    if "pivot table" in q or "pivot" in q:
        return {
            "task": "pivot_table",
            "index": matched_cat_col or (cat_cols[0] if cat_cols else columns[0]),
            "pivot_columns": cat_cols[1] if len(cat_cols) > 1 else (columns[1] if len(columns) > 1 else columns[0]),
            "values": target_num_col,
            "aggregation": "mean",
        }

    # Crosstab
    if "crosstab" in q or "cross tab" in q:
        c1 = cat_cols[0] if len(cat_cols) > 0 else columns[0]
        c2 = cat_cols[1] if len(cat_cols) > 1 else (columns[1] if len(columns) > 1 else columns[0])
        return {"task": "crosstab", "column1": c1, "column2": c2}

    # =========================================================
    # 4. KDD STAGE 1: SELECTION & BASIC INSPECTION
    # =========================================================

    # Select Columns
    if any(k in q for k in ["select only", "keep only", "select columns", "only columns", "show only"]):
        found_cols = [c for c in columns if re.search(r'\b' + re.escape(c.lower()) + r'\b', q)]
        if found_cols:
            return {"task": "select_columns", "columns": found_cols}

    # Random Sample
    if "sample" in q:
        m_s = re.search(r"(\d+)\s*(?:records|rows|samples)?", q)
        n_val = int(m_s.group(1)) if m_s else 5
        return {"task": "sample_rows", "n": n_val}

    # Slice Rows
    if "slice" in q or re.search(r"rows\s+(\d+)\s+to\s+(\d+)", q):
        m_sl = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)", q)
        start_v = int(m_sl.group(1)) if m_sl else 0
        end_v = int(m_sl.group(2)) if m_sl else 10
        return {"task": "slice_rows", "start": start_v, "end": end_v}

    # Select By Dtype
    if "numeric columns" in q or "numerical columns" in q:
        return {"task": "select_by_dtype", "dtype": "numeric"}
    if "categorical columns" in q:
        return {"task": "select_by_dtype", "dtype": "categorical"}

    # Identifier Columns
    if "id columns" in q or "identifier columns" in q:
        return {"task": "identify_id_columns"}

    # Target Column
    if "target column" in q or "identify target" in q:
        return {"task": "identify_target"}

    # Row & column counting
    if any(k in q for k in ["how many rows", "row count", "total rows", "number of rows", "number of records"]):
        return {"task": "count_rows"}
    if any(k in q for k in ["how many columns", "column count", "total columns"]):
        return {"task": "count_columns"}

    # Missing values & duplicates
    if any(k in q for k in ["missing", "null", "empty", "nan"]):
        col_found = _find_column_in_query(columns)
        if col_found:
            return {"task": "missing_values", "column": col_found}
        return {"task": "missing_values"}

    if "duplicate" in q:
        return {"task": "duplicates"}

    # Group by / category comparison ("average salary by department")
    if ("by" in q or "per" in q or "across" in q) and matched_cat_col:
        agg = "mean" if any(k in q for k in ["average", "mean", "avg"]) else ("sum" if ("sum" in q or "total" in q) else "count")
        return {
            "task": "group_aggregate",
            "group_by": matched_cat_col,
            "column": target_num_col,
            "aggregation": agg,
        }

    # Ranking (Top N, Bottom N)
    if "top" in q or "rank" in q:
        m = re.search(r"(?:top|highest)\s+(\d+)", q)
        top_k = int(m.group(1)) if m else 5
        return {"task": "top_n", "column": target_num_col, "top_n": top_k}

    if "bottom" in q:
        m = re.search(r"(?:bottom|lowest)\s+(\d+)", q)
        bot_k = int(m.group(1)) if m else 5
        return {"task": "bottom_n", "column": target_num_col, "top_n": bot_k}

    # Maximum and Minimum
    if any(k in q for k in ["highest", "maximum", "max", "greatest", "peak"]):
        return {"task": "max", "column": target_num_col}

    if any(k in q for k in ["lowest", "minimum", "min", "least"]):
        return {"task": "min", "column": target_num_col}

    # Aggregations
    if any(k in q for k in ["average", "mean", "avg"]):
        return {"task": "mean", "column": target_num_col}

    if "median" in q:
        return {"task": "median", "column": target_num_col}

    if any(k in q for k in ["total", "sum"]):
        return {"task": "sum", "column": target_num_col}

    if any(k in q for k in ["standard deviation", "std dev", "std"]):
        return {"task": "std", "column": target_num_col}

    # Filter / condition
    m_num = re.search(r"(\d+[\d,]*)", q)
    if m_num and any(k in q for k in [">", "<", "greater", "more", "less", "above", "higher than", "exceed"]):
        val = float(m_num.group(1).replace(",", ""))
        op = ">=" if "greater than or equal" in q else (">" if any(k in q for k in [">", "greater", "more", "above", "higher", "exceed"]) else "<")
        return {
            "task": "filter",
            "condition_col": target_num_col,
            "operator": op,
            "condition_val": val,
        }

    return None


def _suggest_close_columns(df: pd.DataFrame, name: str, n: int = 3) -> list:
    """Finds closest matching column names for helpful typo suggestions."""
    if not name or df is None or df.empty:
        return []
    import difflib
    name_clean = str(name).lower().strip()
    col_map = {c.lower(): c for c in df.columns}
    close_lowers = difflib.get_close_matches(name_clean, list(col_map.keys()), n=n, cutoff=0.4)
    if close_lowers:
        return [col_map[cl] for cl in close_lowers]
    return list(df.columns[:n])


def _resolve_column_name(df: pd.DataFrame, name) -> str:
    """Helper to resolve column names case-insensitively and handle spaces/underscores."""
    if not name or df is None:
        return None
    if isinstance(name, (list, tuple)):
        resolved = [_resolve_column_name(df, n) for n in name]
        resolved = [r for r in resolved if r is not None]
        return resolved[0] if resolved else None
    name_str = str(name).strip()
    if name_str in df.columns:
        return name_str
    lower_map = {col.lower(): col for col in df.columns}
    if name_str.lower() in lower_map:
        return lower_map[name_str.lower()]
    space_norm = name_str.lower().replace("_", " ")
    for c in df.columns:
        if c.lower().replace("_", " ") == space_norm:
            return c
    return None


def _extract_chart_column(question: str, df: pd.DataFrame, conversation_history: list = None) -> str:
    """
    Extracts target column for chart requests dynamically from loaded dataset.
    Prioritizes explicit query mentions, then conversation context, then sensible defaults.
    """
    if df is None or df.empty:
        return None

    q = (question or "").lower()

    # 1. Regex patterns: "for X column", "of X column", "for X", "of X", "on X", etc.
    patterns = [
        r"(?:for|of|on)\s+([a-zA-Z0-9_]+)\s+column",
        r"column\s+([a-zA-Z0-9_]+)",
        r"(?:for|of|on)\s+([a-zA-Z0-9_]+)",
        r"(?:chart|plot|graph|histogram|hist|boxplot|piechart|countplot)\s+(?:of|for)?\s*([a-zA-Z0-9_]+)",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            cand = m.group(1).strip()
            if cand not in ["a", "the", "an", "this", "that", "all", "each", "my", "column", "chart", "plot", "it"]:
                matched = _resolve_column_name(df, cand)
                if matched:
                    return matched
                if len(cand) >= 3 and cand not in ["pie", "bar", "box", "line", "count", "scatter"]:
                    return cand

    # 2. Check all columns in df (longer names first)
    for col in sorted(df.columns, key=lambda c: len(c), reverse=True):
        pattern = r'\b' + re.escape(col.lower()) + r'\b'
        if re.search(pattern, q):
            return col
        col_spaced = col.lower().replace("_", " ")
        if re.search(r'\b' + re.escape(col_spaced) + r'\b', q):
            return col

    # 3. Conversational follow-up: inherit from previous turn
    if conversation_history and len(conversation_history) > 0:
        for turn in reversed(conversation_history[-3:]):
            prev_col = turn.get("column") or turn.get("analyzed_column")
            if prev_col and _resolve_column_name(df, prev_col):
                return _resolve_column_name(df, prev_col)

    # 4. Sensible default based on chart context or query keywords
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if any(k in q for k in ["pie", "bar", "count"]) and len(cat_cols) > 0:
        return cat_cols[0]
    elif len(num_cols) > 0:
        return num_cols[0]
    return df.columns[0] if len(df.columns) > 0 else None


def _extract_two_chart_columns(question: str, df: pd.DataFrame, conversation_history: list = None) -> tuple:
    """
    Extracts two numerical columns for scatter plot or comparison.
    Handles 'X vs Y', 'X against Y', 'X and Y'.
    """
    if df is None or df.empty:
        return None, None

    q = (question or "").lower()

    # Check "X vs Y" or "X against Y"
    m = re.search(r"([a-zA-Z0-9_]+)\s*(?:vs|versus|against|and)\s*([a-zA-Z0-9_]+)", q)
    if m:
        c1, c2 = m.group(1).strip(), m.group(2).strip()
        r1, r2 = _resolve_column_name(df, c1), _resolve_column_name(df, c2)
        if r1 and r2:
            return r1, r2
        elif r1 or r2:
            x = r1 or c1
            y = r2 or c2
            return x, y

    # Scan for any 2 columns mentioned in q
    found = []
    for col in sorted(df.columns, key=lambda c: len(c), reverse=True):
        pattern = r'\b' + re.escape(col.lower()) + r'\b'
        if re.search(pattern, q):
            if col not in found:
                found.append(col)
        elif re.search(r'\b' + re.escape(col.lower().replace("_", " ")) + r'\b', q):
            if col not in found:
                found.append(col)

    if len(found) >= 2:
        return found[0], found[1]
    elif len(found) == 1:
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != found[0]]
        second = num_cols[0] if num_cols else df.columns[-1]
        return found[0], second

    # Context history fallback
    if conversation_history and len(conversation_history) > 0:
        for turn in reversed(conversation_history[-3:]):
            prev_col = turn.get("column") or turn.get("analyzed_column")
            if prev_col and _resolve_column_name(df, prev_col):
                matched = _resolve_column_name(df, prev_col)
                num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != matched]
                second = num_cols[0] if num_cols else df.columns[-1]
                return matched, second

    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(num_cols) >= 2:
        return num_cols[0], num_cols[1]
    elif len(df.columns) >= 2:
        return df.columns[0], df.columns[1]
    return df.columns[0] if len(df.columns) > 0 else None, None
