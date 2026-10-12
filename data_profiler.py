"""
data_profiler.py
----------------
Module responsible for computing exploratory data analysis (EDA) and profiling summaries:
- Identifies numerical vs. categorical vs. datetime columns
- Computes missing values and duplicate rows
- Calculates descriptive statistics: mean, median, min, max, std
- Summarizes categorical columns (cardinality, most frequent values)
- Formats insights in beginner-friendly, clean formats
"""

import pandas as pd
import numpy as np


def get_column_types(df: pd.DataFrame) -> dict:
    """
    Classifies DataFrame columns into numerical, categorical, and datetime types.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: Lists of column names categorized by type.
    """
    if df is None or df.empty:
        return {"numerical": [], "categorical": [], "datetime": []}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    return {
        "numerical": num_cols,
        "categorical": cat_cols,
        "datetime": date_cols,
    }


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates missing (NaN/null) values count and percentage for each column.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Table with columns ['Column', 'Missing Count', 'Missing Percentage (%)'].
    """
    if df is None or df.empty:
        return pd.DataFrame()

    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / total_rows * 100).round(2)

    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_counts.values,
        "Missing Percentage (%)": missing_percentages.values,
    })

    # Sort so columns with highest missing values appear on top
    summary_df = summary_df.sort_values(by="Missing Count", ascending=False).reset_index(drop=True)
    return summary_df


def get_duplicate_summary(df: pd.DataFrame) -> dict:
    """
    Calculates total duplicate rows and their percentage in the dataset.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: {'duplicate_count': int, 'duplicate_percentage': float}
    """
    if df is None or df.empty:
        return {"duplicate_count": 0, "duplicate_percentage": 0.0}

    total_rows = len(df)
    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
    }


def get_numerical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes key descriptive statistics for all numerical columns:
    - Count: Number of non-null records
    - Mean: Mathematical average
    - Std Dev: How spread out the values are from the average
    - Min: Smallest value
    - Median (50%): Middle value when sorted
    - Max: Largest value
    - Skewness: Asymmetry of the distribution
    - Kurtosis: Tailedness of the distribution
    - Outliers (IQR): Count of values beyond 1.5 * IQR

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Summary table of numerical statistics.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns

    if len(num_cols) == 0:
        return pd.DataFrame(columns=["Column", "Count", "Mean", "Std Dev", "Min", "Median", "Max", "Skewness", "Kurtosis", "Outliers (IQR)"])

    records = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        # IQR Outliers
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

        skew_val = round(float(series.skew()), 2) if len(series) > 2 else 0.0
        kurt_val = round(float(series.kurtosis()), 2) if len(series) > 3 else 0.0

        records.append({
            "Column": col,
            "Count": int(series.count()),
            "Mean": round(float(series.mean()), 2),
            "Std Dev": round(float(series.std()), 2) if len(series) > 1 else 0.0,
            "Min": round(float(series.min()), 2),
            "Median": round(float(series.median()), 2),
            "Max": round(float(series.max()), 2),
            "Skewness": skew_val,
            "Kurtosis": kurt_val,
            "Outliers (IQR)": outlier_count,
        })

    return pd.DataFrame(records)


def get_categorical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary metrics for categorical columns:
    - Unique Count: Number of distinct categories
    - Top Category: The most frequently occurring value (mode)
    - Frequency: How many times the top category appears

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Summary table of categorical statistics.
    """
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns

    if len(cat_cols) == 0:
        return pd.DataFrame(columns=["Column", "Unique Count", "Top Category", "Frequency"])

    records = []
    for col in cat_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        unique_cnt = int(series.nunique())
        mode_val = series.mode()[0] if not series.empty else "N/A"
        freq = int((series == mode_val).sum()) if not series.empty else 0

        records.append({
            "Column": col,
            "Unique Count": unique_cnt,
            "Top Category": str(mode_val),
            "Frequency": freq,
        })

    return pd.DataFrame(records)


def detect_identifier_columns(df: pd.DataFrame) -> list:
    """Detects potential primary key, ID, or high-uniqueness identifier columns."""
    if df is None or df.empty:
        return []
    id_cols = []
    total = len(df)
    for col in df.columns:
        col_lower = str(col).lower()
        unique_cnt = df[col].nunique(dropna=True)
        # Check by name or 100% uniqueness for non-empty tables
        if any(kw in col_lower for kw in ["id", "uuid", "ssn", "passport", "code", "key"]):
            id_cols.append(col)
        elif total > 1 and unique_cnt == total:
            id_cols.append(col)
    return list(dict.fromkeys(id_cols))


def detect_constant_columns(df: pd.DataFrame) -> list:
    """Detects columns with zero variance or single unique value."""
    if df is None or df.empty:
        return []
    return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]


def get_memory_usage(df: pd.DataFrame) -> str:
    """Returns total deep memory usage formatted in KB or MB."""
    if df is None or df.empty:
        return "0 KB"
    bytes_used = df.memory_usage(deep=True).sum()
    if bytes_used >= 1024 * 1024:
        return f"{bytes_used / (1024 * 1024):.2f} MB"
    return f"{bytes_used / 1024:.2f} KB"


def get_text_length_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes string length statistics for free-text columns."""
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    records = []
    for col in text_cols:
        lengths = df[col].dropna().astype(str).str.len()
        if len(lengths) > 0:
            records.append({
                "Column": col,
                "Avg Length": round(float(lengths.mean()), 1),
                "Min Length": int(lengths.min()),
                "Max Length": int(lengths.max()),
            })
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["Column", "Avg Length", "Min Length", "Max Length"])


def get_date_range_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Computes date range and duration for datetime columns."""
    date_cols = df.select_dtypes(include=["datetime"]).columns
    records = []
    for col in date_cols:
        series = df[col].dropna()
        if not series.empty:
            min_date = series.min()
            max_date = series.max()
            records.append({
                "Column": col,
                "Start Date": str(min_date),
                "End Date": str(max_date),
                "Days Range": (max_date - min_date).days if hasattr(max_date - min_date, "days") else "N/A",
            })
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["Column", "Start Date", "End Date", "Days Range"])


def generate_full_profile(df: pd.DataFrame) -> dict:
    """
    Aggregates all profiling components into a unified profile dictionary.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: Complete profiling report containing DataFrames and metrics.
    """
    if df is None or df.empty:
        return {}

    col_types = get_column_types(df)
    missing_df = get_missing_summary(df)
    duplicate_info = get_duplicate_summary(df)
    num_stats = get_numerical_statistics(df)
    cat_stats = get_categorical_statistics(df)
    id_cols = detect_identifier_columns(df)
    const_cols = detect_constant_columns(df)
    mem_usage = get_memory_usage(df)
    text_summary = get_text_length_summary(df)
    date_summary = get_date_range_summary(df)

    total_cells = df.size
    total_missing_cells = int(df.isnull().sum().sum())
    overall_missing_pct = round((total_missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

    # Dataset Quality Score (0-100)
    completeness_score = max(0.0, 100.0 - overall_missing_pct)
    uniqueness_score = max(0.0, 100.0 - duplicate_info.get("duplicate_percentage", 0.0))
    quality_score = round(0.7 * completeness_score + 0.3 * uniqueness_score, 1)

    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_cells": total_cells,
        "column_types": col_types,
        "duplicate_info": duplicate_info,
        "total_missing_cells": total_missing_cells,
        "overall_missing_percentage": overall_missing_pct,
        "quality_score": quality_score,
        "memory_usage": mem_usage,
        "identifier_columns": id_cols,
        "constant_columns": const_cols,
        "text_length_summary": text_summary,
        "date_range_summary": date_summary,
        "missing_summary_table": missing_df,
        "numerical_stats_table": num_stats,
        "categorical_stats_table": cat_stats,
    }


# Standalone demonstration block
if __name__ == "__main__":
    from data_loader import load_dataset
    import os

    sample_path = os.path.join("datasets", "sample_data.csv")
    df, err = load_dataset(sample_path)

    if err:
        print(f"Error loading dataset: {err}")
    else:
        print("=" * 70)
        print("DATA PROFILER REPORT")
        print("=" * 70)

        profile = generate_full_profile(df)
        print(f"Rows: {profile['total_rows']} | Columns: {profile['total_columns']}")
        print(f"Total Cells: {profile['total_cells']}")
        print(f"Duplicate Rows: {profile['duplicate_info']['duplicate_count']} ({profile['duplicate_info']['duplicate_percentage']}%)")
        print(f"Total Missing Values: {profile['total_missing_cells']} ({profile['overall_missing_percentage']}%)")

        print("\n--- Column Types ---")
        print(f"Numerical: {profile['column_types']['numerical']}")
        print(f"Categorical: {profile['column_types']['categorical']}")

        print("\n--- Numerical Statistics ---")
        print(profile['numerical_stats_table'].to_string(index=False))

        print("\n--- Categorical Statistics ---")
        print(profile['categorical_stats_table'].to_string(index=False))

        print("\n--- Missing Values Summary ---")
        print(profile['missing_summary_table'].to_string(index=False))
        print("\nSUCCESS: data_profiler.py completed profiling accurately!")
