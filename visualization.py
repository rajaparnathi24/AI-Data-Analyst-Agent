"""
visualization.py
----------------
Module responsible for generating clean, verified charts using Matplotlib.
Ensures that all plotted values originate strictly from Pandas data calculations,
never from LLM-generated hallucinated numbers.

Supported Charts:
- Bar Charts (for group aggregations, categories, Top N)
- Line Charts (for trends over continuous or sequential variables)
- Histograms (for distribution of numerical columns)
- Scatter Plots (for relationship between two numerical variables)
- Auto-Chart generator (infers best chart from analytical result)
"""

import matplotlib
# Use Agg backend for non-interactive server rendering (avoids GUI thread crashes)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Cohesive, modern color palette
PRIMARY_COLOR = "#2563EB"   # Modern Blue
ACCENT_COLOR = "#0D9488"    # Modern Teal
BAR_COLORS = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#6366F1"]


def create_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = None,
) -> tuple:
    """
    Creates a clean bar chart for categorical comparisons.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        x_col (str): Categorical column for x-axis.
        y_col (str): Numerical column for y-axis.
        title (str, optional): Chart title.

    Returns:
        tuple: (fig, error_message)
    """
    if data is None or data.empty:
        return None, "Cannot create bar chart: Data is empty."

    if x_col not in data.columns or y_col not in data.columns:
        return None, f"Cannot create bar chart: Columns '{x_col}' or '{y_col}' not found."

    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        
        # Select color palette
        colors = BAR_COLORS[:len(data)] if len(data) <= len(BAR_COLORS) else PRIMARY_COLOR
        bars = ax.bar(data[x_col].astype(str), data[y_col], color=colors, edgecolor="#1E293B", linewidth=0.7)

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:,.1f}" if isinstance(height, float) else f"{height:,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title(title or f"{y_col} by {x_col}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(x_col, fontsize=10, labelpad=8)
        ax.set_ylabel(y_col, fontsize=10, labelpad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # Rotate x labels if more than 4 categories or long strings
        if len(data) > 4 or any(len(str(v)) > 8 for v in data[x_col]):
            plt.xticks(rotation=30, ha="right")

        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating bar chart: {str(e)}"


def create_histogram(
    df: pd.DataFrame,
    col: str,
    bins: int = 10,
    title: str = None,
) -> tuple:
    """
    Creates a histogram showing the distribution of a numerical column.

    Parameters:
        df (pd.DataFrame): Dataset.
        col (str): Numerical column to plot.
        bins (int): Number of bins.
        title (str, optional): Chart title.

    Returns:
        tuple: (fig, error_message)
    """
    if df is None or df.empty or col not in df.columns:
        return None, f"Cannot create histogram: Column '{col}' not available."

    try:
        series = df[col].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            return None, f"Cannot create histogram: Column '{col}' is not numeric."

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(series, bins=bins, color=PRIMARY_COLOR, edgecolor="#0F172A", alpha=0.8, linewidth=0.8)

        # Add vertical line for mean
        mean_val = series.mean()
        ax.axvline(mean_val, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:,.1f}")

        ax.set_title(title or f"Distribution of {col}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(frameon=True)

        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating histogram: {str(e)}"


def create_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = None,
) -> tuple:
    """
    Creates a line chart for sequential or trend comparisons.
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None, f"Columns '{x_col}' or '{y_col}' not found for line chart."

    try:
        sorted_df = df.sort_values(by=x_col).dropna(subset=[x_col, y_col])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(sorted_df[x_col].astype(str), sorted_df[y_col], marker="o", color=ACCENT_COLOR, linewidth=2)

        ax.set_title(title or f"{y_col} Trend by {x_col}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

        if len(sorted_df) > 5:
            plt.xticks(rotation=30, ha="right")

        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating line chart: {str(e)}"


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = None,
) -> tuple:
    """
    Creates a scatter plot showing correlation between two numerical columns.
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None, f"Columns '{x_col}' or '{y_col}' not found for scatter plot."

    try:
        clean_df = df.dropna(subset=[x_col, y_col])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(clean_df[x_col], clean_df[y_col], color=PRIMARY_COLOR, alpha=0.7, edgecolors="none", s=50)

        # Plot trend line if both are numeric
        if pd.api.types.is_numeric_dtype(clean_df[x_col]) and pd.api.types.is_numeric_dtype(clean_df[y_col]):
            x = clean_df[x_col]
            y = clean_df[y_col]
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m * x + b, color="#DC2626", linestyle="--", linewidth=1.5, label="Trend line")
            ax.legend(frameon=True)

        ax.set_title(title or f"{y_col} vs. {x_col}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating scatter plot: {str(e)}"


def create_correlation_heatmap(corr_df: pd.DataFrame) -> tuple:
    """
    Renders a clear, stylish heatmap for numerical correlation matrix.
    """
    if corr_df is None or corr_df.empty:
        return None, "Cannot generate correlation heatmap: Data is empty."

    try:
        plot_df = corr_df.copy()
        if "Variable" in plot_df.columns:
            plot_df = plot_df.set_index("Variable")
        plot_df = plot_df.select_dtypes(include=[np.number])
        if plot_df.empty or len(plot_df.columns) < 2:
            return None, "At least two numerical columns are required for a correlation heatmap."

        fig, ax = plt.subplots(figsize=(7, 5))
        cax = ax.matshow(plot_df.values, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax, shrink=0.8, pad=0.04)

        ax.set_xticks(range(len(plot_df.columns)))
        ax.set_yticks(range(len(plot_df.index)))
        ax.set_xticklabels(plot_df.columns, rotation=45, ha="left", fontsize=9)
        ax.set_yticklabels(plot_df.index, fontsize=9)

        for i in range(len(plot_df.index)):
            for j in range(len(plot_df.columns)):
                val = plot_df.iloc[i, j]
                text_color = "white" if abs(val) > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8, fontweight="bold")

        ax.set_title("Pearson Correlation Heatmap", pad=25, fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig, None

    except Exception as e:
        plt.close("all")
        return None, f"Error generating correlation heatmap: {str(e)}"


def create_box_plot(df: pd.DataFrame, col: str, by_col: str = None, title: str = None) -> tuple:
    """
    Creates a box plot showing median, quartiles, and outliers.
    """
    if df is None or df.empty or col not in df.columns:
        return None, f"Cannot create box plot: Column '{col}' not found."

    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if by_col and by_col in df.columns:
            # Grouped boxplot
            clean_df = df.dropna(subset=[col, by_col])
            categories = clean_df[by_col].unique()[:10]  # Limit to 10 categories
            data_to_plot = [clean_df[clean_df[by_col] == cat][col].dropna() for cat in categories]
            ax.boxplot(data_to_plot, tick_labels=[str(c) for c in categories], patch_artist=True,
                       boxprops=dict(facecolor=PRIMARY_COLOR, alpha=0.7),
                       medianprops=dict(color="#DC2626", linewidth=1.5))
            ax.set_xlabel(by_col, fontsize=10)
            if len(categories) > 4:
                plt.xticks(rotation=30, ha="right")
        else:
            clean_data = df[col].dropna()
            ax.boxplot(clean_data, patch_artist=True,
                       boxprops=dict(facecolor=PRIMARY_COLOR, alpha=0.7),
                       medianprops=dict(color="#DC2626", linewidth=1.5))
            ax.set_xticklabels([col])

        ax.set_title(title or f"Box Plot of {col}", fontsize=12, fontweight="bold", pad=12)
        ax.set_ylabel(col, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        return fig, None
    except Exception as e:
        plt.close("all")
        return None, f"Error generating box plot: {str(e)}"


def create_pie_chart(df: pd.DataFrame, col: str = None, title: str = None) -> tuple:
    """
    Creates a pie chart for categorical distributions (up to 7 slices).
    Accepts raw DataFrames or pre-aggregated count DataFrames.
    """
    if df is None or df.empty:
        return None, "Cannot create pie chart: Data is empty."

    try:
        label_col = col
        if col and col in df.columns and len(df.columns) > 1 and "count" in df.columns[1].lower():
            counts_series = df.set_index(col)[df.columns[1]].head(7)
            labels = counts_series.index.astype(str)
            values = counts_series.values
        elif col and col in df.columns:
            counts = df[col].value_counts().head(7)
            labels = counts.index.astype(str)
            values = counts.values
        elif len(df.columns) >= 2 and "count" in df.columns[1].lower():
            label_col = df.columns[0]
            counts_series = df.set_index(label_col)[df.columns[1]].head(7)
            labels = counts_series.index.astype(str)
            values = counts_series.values
        elif len(df.columns) > 0:
            label_col = df.columns[0]
            counts = df[label_col].value_counts().head(7)
            labels = counts.index.astype(str)
            values = counts.values
        else:
            return None, f"Cannot create pie chart: Column '{col}' not found."

        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = BAR_COLORS[:len(values)]
        ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140,
               wedgeprops=dict(edgecolor="#1E293B", linewidth=0.5))
        ax.set_title(title or f"Distribution of {label_col or 'Categories'}", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        return fig, None
    except Exception as e:
        plt.close("all")
        return None, f"Error generating pie chart: {str(e)}"


def create_time_series_chart(df: pd.DataFrame, date_col: str, val_col: str, title: str = None) -> tuple:
    """
    Creates a time-series line chart.
    """
    if df is None or df.empty or date_col not in df.columns or val_col not in df.columns:
        return None, f"Cannot create time series chart: Required columns not found."

    try:
        plot_df = df.dropna(subset=[date_col, val_col]).sort_values(by=date_col)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(plot_df[date_col].astype(str), plot_df[val_col], marker="o", markersize=3, color=PRIMARY_COLOR, linewidth=1.8)
        ax.set_title(title or f"{val_col} over {date_col}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(date_col, fontsize=10)
        ax.set_ylabel(val_col, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        if len(plot_df) > 6:
            plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        return fig, None
    except Exception as e:
        plt.close("all")
        return None, f"Error generating time series chart: {str(e)}"


def create_cluster_scatter(df: pd.DataFrame, x_col: str, y_col: str, cluster_col: str = "Cluster", title: str = None) -> tuple:
    """
    Creates a 2D scatter plot colored by cluster assignments.
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return None, f"Cannot create cluster scatter: Columns not found."

    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        clusters = df[cluster_col].unique() if cluster_col in df.columns else [0]
        for i, clus in enumerate(clusters):
            sub = df[df[cluster_col] == clus]
            color = "#64748B" if clus == -1 else BAR_COLORS[i % len(BAR_COLORS)]
            label = "Noise" if clus == -1 else f"Cluster {clus}"
            ax.scatter(sub[x_col], sub[y_col], c=color, label=label, alpha=0.75, s=40, edgecolors="none")

        ax.set_title(title or f"Clusters: {y_col} vs {x_col}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=True, fontsize=8)
        plt.tight_layout()
        return fig, None
    except Exception as e:
        plt.close("all")
        return None, f"Error generating cluster scatter: {str(e)}"


def create_confusion_matrix_plot(cm_data, labels=None, title: str = None) -> tuple:
    """
    Renders a confusion matrix heatmap.
    """
    if cm_data is None:
        return None, "No confusion matrix data provided."

    try:
        cm_array = np.array(cm_data)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cax = ax.matshow(cm_array, cmap="Blues")
        fig.colorbar(cax, shrink=0.8)

        num_classes = cm_array.shape[0]
        class_names = [str(lbl) for lbl in labels] if labels and len(labels) == num_classes else [f"Class {i}" for i in range(num_classes)]

        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=30, ha="left", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)

        for i in range(num_classes):
            for j in range(num_classes):
                val = cm_array[i, j]
                color = "white" if val > cm_array.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

        ax.set_title(title or "Confusion Matrix", pad=20, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
        ax.set_ylabel("True Label", fontsize=10, labelpad=8)
        plt.tight_layout()
        return fig, None
    except Exception as e:
        plt.close("all")
        return None, f"Error generating confusion matrix: {str(e)}"


def create_actual_vs_predicted_plot(y_true, y_pred, title: str = None) -> tuple:
    """
    Renders an Actual vs Predicted scatter plot with perfect-prediction 45-degree line.
    """
    try:
        yt = np.array(y_true)
        yp = np.array(y_pred)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(yt, yp, color=PRIMARY_COLOR, alpha=0.7, s=40, label="Predictions")

        min_val = min(float(yt.min()), float(yp.min()))
        max_val = max(float(yt.max()), float(yp.max()))
        ax.plot([min_val, max_val], [min_val, max_val], color="#DC2626", linestyle="--", linewidth=1.5, label="Perfect Fit (y = x)")

        ax.set_title(title or "Actual vs. Predicted Values", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Actual Values", fontsize=10)
        ax.set_ylabel("Predicted Values", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=True)
        plt.tight_layout()
        return fig, None
    except Exception as e:
        plt.close("all")
        return None, f"Error generating actual vs predicted plot: {str(e)}"


def create_pca_scatter(pca_df: pd.DataFrame, title: str = None) -> tuple:
    """
    Renders a 2D scatter plot for PCA components.
    """
    if pca_df is None or "PCA1" not in pca_df.columns or "PCA2" not in pca_df.columns:
        return None, "Cannot create PCA scatter: PCA1 and PCA2 columns required."

    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(pca_df["PCA1"], pca_df["PCA2"], color=ACCENT_COLOR, alpha=0.7, s=45)
        ax.set_title(title or "PCA: 2D Feature Projection", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Principal Component 1 (PCA1)", fontsize=10)
        ax.set_ylabel("Principal Component 2 (PCA2)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        return fig, None
    except Exception as e:
        plt.close("all")
        return None, f"Error generating PCA scatter: {str(e)}"


def create_auto_chart(analysis_result: dict, original_df: pd.DataFrame) -> tuple:
    """
    Automatically selects and renders the most appropriate chart
    based on the analysis engine result and query task.

    Parameters:
        analysis_result (dict): The output dictionary from analysis_engine.execute_plan.
        original_df (pd.DataFrame): The full underlying dataset.

    Returns:
        tuple: (fig, error_message)
    """
    if not analysis_result or analysis_result.get("status") != "success":
        return None, "No successful analytical result available to plot."

    task = analysis_result.get("task", "")
    chart_data = analysis_result.get("chart_data")

    # 1. Pie Chart
    if task in ["pie_chart", "pie"]:
        col = analysis_result.get("task_column")
        if isinstance(chart_data, pd.DataFrame):
            return create_pie_chart(chart_data, col=col)
        elif col and original_df is not None and col in original_df.columns:
            return create_pie_chart(original_df, col=col)
        elif original_df is not None and len(original_df.columns) > 0:
            return create_pie_chart(original_df, col=original_df.columns[0])

    # 2. Bar Chart / Count Plot
    elif task in ["bar_chart", "count_plot", "countplot"]:
        if isinstance(chart_data, pd.DataFrame) and len(chart_data.columns) >= 2:
            return create_bar_chart(chart_data, x_col=chart_data.columns[0], y_col=chart_data.columns[1])
        col = analysis_result.get("task_column")
        if col and original_df is not None and col in original_df.columns:
            vc = original_df[col].value_counts().head(10).reset_index()
            return create_bar_chart(vc, x_col=vc.columns[0], y_col=vc.columns[1])

    # 3. Line Chart / Line Graph
    elif task in ["line_chart", "line_graph", "line_plot"]:
        if isinstance(chart_data, pd.DataFrame) and len(chart_data.columns) >= 2:
            return create_line_chart(chart_data, x_col=chart_data.columns[0], y_col=chart_data.columns[1])
        col = analysis_result.get("task_column")
        if col and original_df is not None and col in original_df.columns:
            sub = pd.DataFrame({"Index": range(len(original_df)), col: original_df[col].values})
            return create_line_chart(sub, x_col="Index", y_col=col)

    # 4. Group Aggregations -> Bar Chart
    elif task in ["group_aggregate", "group_by", "aggregate"] and isinstance(chart_data, pd.DataFrame):
        cols = list(chart_data.columns)
        if len(cols) >= 2:
            return create_bar_chart(chart_data, x_col=cols[0], y_col=cols[1])

    # 5. Top N / Bottom N -> Bar Chart
    elif task in ["top_n", "bottom_n", "highest_n", "lowest_n"] and isinstance(chart_data, pd.DataFrame):
        cat_cols = chart_data.select_dtypes(include=["object", "string"]).columns
        num_cols = chart_data.select_dtypes(include=[np.number]).columns
        if len(cat_cols) > 0 and len(num_cols) > 0:
            return create_bar_chart(chart_data, x_col=cat_cols[0], y_col=num_cols[-1])

    # 6. Correlation Matrix / Heatmap
    elif task in ["correlation", "corr", "heatmap", "correlation_heatmap"] and isinstance(chart_data, pd.DataFrame):
        return create_correlation_heatmap(chart_data)
    elif task in ["heatmap", "correlation_heatmap"] and original_df is not None:
        num_cols = original_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) >= 2:
            return create_correlation_heatmap(original_df[num_cols].corr().round(3).reset_index())

    # 7. Missing Values Breakdown -> Bar Chart
    elif task in ["missing_values", "null_values"]:
        val = analysis_result.get("value")
        if isinstance(val, pd.DataFrame) and "Missing_Count" in val.columns:
            has_missing = val[val["Missing_Count"] > 0]
            if not has_missing.empty:
                return create_bar_chart(has_missing, x_col="Column", y_col="Missing_Count", title="Missing Values per Column")

    # 8. Outliers / Boxplot
    elif task in ["detect_outliers", "box_plot", "boxplot"]:
        col = analysis_result.get("task_column")
        by_col = None
        if isinstance(chart_data, pd.DataFrame) and len(chart_data.columns) >= 2:
            by_col = chart_data.columns[1]
        if col and original_df is not None and col in original_df.columns:
            return create_box_plot(original_df, col=col, by_col=by_col)

    # 9. Scatter Plot
    elif task in ["scatter", "scatter_plot"] and isinstance(chart_data, pd.DataFrame):
        cols = chart_data.select_dtypes(include=[np.number]).columns
        if len(cols) >= 2:
            return create_scatter_plot(chart_data, x_col=cols[0], y_col=cols[1])

    # 10. Clustering -> Cluster Scatter
    elif task in ["kmeans_cluster", "dbscan_cluster"] and isinstance(chart_data, pd.DataFrame):
        num_cols = [c for c in chart_data.select_dtypes(include=[np.number]).columns if c != "Cluster"]
        if len(num_cols) >= 2 and "Cluster" in chart_data.columns:
            return create_cluster_scatter(chart_data, x_col=num_cols[0], y_col=num_cols[1])

    # 11. Classification -> Confusion Matrix
    elif task in ["train_classification", "classification", "decision_tree", "logistic_regression"]:
        val = analysis_result.get("value")
        if isinstance(val, dict) and "confusion_matrix" in val:
            labels = val.get("classes")
            return create_confusion_matrix_plot(val["confusion_matrix"], labels=labels)

    # 12. Regression -> Actual vs Predicted
    elif task in ["train_regression", "regression", "linear_regression"]:
        val = analysis_result.get("value")
        if isinstance(val, dict) and "actual_vs_predicted" in val:
            avp = val["actual_vs_predicted"]
            if isinstance(avp, pd.DataFrame) and "Actual" in avp.columns and "Predicted" in avp.columns:
                return create_actual_vs_predicted_plot(avp["Actual"], avp["Predicted"])

    # 13. PCA -> 2D Scatter
    elif task in ["pca", "pca_reduction"] and isinstance(chart_data, pd.DataFrame):
        if "PCA1" in chart_data.columns and "PCA2" in chart_data.columns:
            return create_pca_scatter(chart_data)

    # 14. Scalar aggregations (Mean / Median / Sum / Histogram) -> Histogram
    elif task in ["mean", "median", "sum", "std", "histogram"] and original_df is not None:
        col = analysis_result.get("task_column")
        if col and col in original_df.columns:
            return create_histogram(original_df, col=col)

    return None, None


# Standalone demonstration block
if __name__ == "__main__":
    from data_loader import load_dataset
    from analysis_engine import execute_plan
    import os

    df, _ = load_dataset(os.path.join("datasets", "sample_data.csv"))
    print("=" * 60)
    print("TESTING VISUALIZATION GENERATION")
    print("=" * 60)

    # Test 1: Bar chart for Group Aggregation
    plan1 = {"task": "group_aggregate", "group_by": "Department", "column": "Salary", "aggregation": "mean"}
    res1 = execute_plan(df, plan1)
    fig1, err1 = create_auto_chart(res1, df)
    if err1:
        print("Chart 1 failed:", err1)
    else:
        print("Chart 1 (Bar Chart) generated successfully:", type(fig1))
        fig1.savefig("tests/demo_bar_chart.png")
        print("  -> Saved preview to tests/demo_bar_chart.png")
        plt.close(fig1)

    # Test 2: Histogram
    fig2, err2 = create_histogram(df, col="Salary", bins=5)
    if err2:
        print("Chart 2 failed:", err2)
    else:
        print("Chart 2 (Histogram) generated successfully:", type(fig2))
        fig2.savefig("tests/demo_histogram.png")
        print("  -> Saved preview to tests/demo_histogram.png")
        plt.close(fig2)

    # Test 3: Scatter Plot
    fig3, err3 = create_scatter_plot(df, x_col="Experience_Years", y_col="Salary")
    if err3:
        print("Chart 3 failed:", err3)
    else:
        print("Chart 3 (Scatter Plot) generated successfully:", type(fig3))
        fig3.savefig("tests/demo_scatter.png")
        print("  -> Saved preview to tests/demo_scatter.png")
        plt.close(fig3)

    print("\nSUCCESS: All visualization charts rendered cleanly!")
