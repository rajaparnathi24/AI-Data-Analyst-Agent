"""
data_loader.py
--------------
Module responsible for safely reading and validating tabular datasets (CSV and Excel).
Follows best practices:
- Validates file existence and supported extensions (.csv, .xlsx, .xls)
- Gracefully handles empty, missing, or malformed files without crashing
- Standardizes column names (strips accidental spaces)
- Computes basic dataset metadata for display in the UI
"""

import os
import pandas as pd


# Allowed file extensions for tabular uploads
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

# Maximum allowed file size in bytes (25 MB)
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


def load_dataset(file_path: str, sheet_name=None):
    """
    Safely loads a CSV or Excel file into a Pandas DataFrame.

    Parameters:
        file_path (str): Full path to the uploaded file.
        sheet_name (str or int, optional): Excel sheet name or index to load.

    Returns:
        tuple: (df, error_message)
               - df: pd.DataFrame if successful, else None
               - error_message: str describing failure, or None if successful
    """
    # 1. Check if path is provided
    if not file_path:
        return None, "Error: No file path provided."

    # 2. Check file extension
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return (
            None,
            f"Error: Unsupported format '{extension}'. Please upload a CSV (.csv) or Excel (.xlsx, .xls) file.",
        )

    # 3. Check if file physically exists on disk
    if not os.path.exists(file_path):
        return None, f"Error: File not found at '{file_path}'."

    # 4. Enforce file size limit to prevent Denial-of-Service / Memory exhaustion
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return None, f"Error: File size ({size_mb:.1f} MB) exceeds maximum allowed limit of 25 MB."

    # 5. Attempt to load the data using Pandas with encoding fallback
    try:
        if extension == ".csv":
            encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
            df = None
            last_err = None
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    break
                except UnicodeDecodeError as ue:
                    last_err = ue
                    continue
            if df is None:
                return None, f"Error decoding CSV file across supported encodings: {last_err}"
        else:
            # Excel file loading
            excel_file = pd.ExcelFile(file_path)
            target_sheet = sheet_name if sheet_name is not None else excel_file.sheet_names[0]
            df = pd.read_excel(excel_file, sheet_name=target_sheet)

        # 6. Check if dataset is empty
        if df is None or df.empty or len(df.columns) == 0:
            return None, "Error: The uploaded file is empty (contains 0 rows or 0 columns)."

        # 7. Clean column names (strip leading/trailing whitespace) and deduplicate
        cleaned_cols = []
        counts = {}
        for col in df.columns:
            name = str(col).strip() if str(col).strip() else "Unnamed"
            if name in counts:
                counts[name] += 1
                cleaned_cols.append(f"{name}_{counts[name]}")
            else:
                counts[name] = 0
                cleaned_cols.append(name)
        df.columns = cleaned_cols

        return df, None

    except pd.errors.EmptyDataError:
        return None, "Error: The uploaded file has no data (EmptyDataError)."
    except pd.errors.ParserError as e:
        return None, f"Error parsing file: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error loading dataset: {str(e)}"


def get_dataset_overview(df: pd.DataFrame, dataset_name: str = "Dataset") -> dict:
    """
    Extracts an initial summary of the dataset for display upon file upload.

    Parameters:
        df (pd.DataFrame): The loaded dataset.
        dataset_name (str): The filename or label of the dataset.

    Returns:
        dict: Summary containing row count, column count, column names, data types,
              first 5 rows, and missing value counts.
    """
    if df is None or df.empty:
        return {}

    total_rows, total_cols = df.shape
    missing_per_col = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    # Format data types into human-readable strings
    column_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "column_names": list(df.columns),
        "column_dtypes": column_dtypes,
        "missing_values": missing_per_col,
        "total_missing_values": total_missing,
        "head_preview": df.head(5),
    }


# Standalone demonstration block for testing and college viva explanation
if __name__ == "__main__":
    print("=" * 60)
    print("Testing data_loader.py with sample dataset...")
    print("=" * 60)

    sample_csv_path = os.path.join("datasets", "sample_data.csv")
    loaded_df, err = load_dataset(sample_csv_path)

    if err:
        print(f"Failed to load: {err}")
    else:
        overview = get_dataset_overview(loaded_df, dataset_name="sample_data.csv")
        print(f"Dataset Name: {overview['dataset_name']}")
        print(f"Total Rows: {overview['total_rows']}")
        print(f"Total Columns: {overview['total_columns']}")
        print(f"Columns: {overview['column_names']}")
        print("\nMissing values per column:")
        for col, count in overview['missing_values'].items():
            print(f"  - {col}: {count}")
        print("\nFirst 5 rows:")
        print(overview['head_preview'])
        print("\nSUCCESS: data_loader.py works correctly!")
