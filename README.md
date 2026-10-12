# 🤖 AI Data Analyst Agent

> **An LLM-Powered Conversational System for Automated Tabular Data Analysis**  
> *Fully Migrated to Local Ollama for 100% Private, Offline LLM Communication*

---

## 1. Project Overview

The **AI Data Analyst Agent** is a full-stack, conversational data analytics application that enables non-technical users and analysts to upload tabular datasets (**CSV** or **Excel**) and query them using natural English.

Unlike conventional LLM wrappers that ask models to calculate mathematical figures directly (which leads to arithmetic hallucinations) or execute unverified LLM-generated code (which creates critical security risks), this application implements a **safe, decoupled architecture**:
1. An **NLP Machine Learning Classifier** (`Scikit-Learn`) categorizes the query intent.
2. An **LLM Query Planner** (`Ollama` local LLM via `qwen3:4b-instruct`) compiles the question into a structured JSON execution plan.
3. A **Deterministic Analysis Engine** (`Pandas` & `NumPy`) calculates the verified numerical answer.
4. An **Automated Visualization Engine** (`Matplotlib`) renders clean statistical plots.
5. An interactive **Web Application** (`Gradio`) displays conversational answers, summary tables, and charts.

All LLM communication is handled locally via **Ollama** (`http://localhost:11434`), eliminating all cloud API keys, recurring fees, and data privacy concerns.

---

## 2. Problem Statement

Business users and students often struggle to write complex SQL queries or Pandas commands to extract simple insights from tabular files. While general-purpose LLMs can converse about data, standard implementations suffer from major drawbacks:
1. **Arithmetic Hallucination**: LLMs predict text tokens; they do not perform arithmetic natively and frequently calculate incorrect averages, counts, or sums.
2. **Arbitrary Code Execution Hazards**: Systems running raw LLM-generated Python scripts via `eval()` or `exec()` expose hosts to prompt injection and accidental file loss.
3. **Data Privacy and API Costs**: Cloud LLM APIs require sending proprietary datasets to third-party cloud servers and incur usage billing.

**Solution**: This project bridges the gap by using a local Ollama LLM only for **linguistic understanding and structured planning**, while delegating all computational execution to deterministic Pandas and NumPy.

---

## 3. Project Objectives

* Enable seamless drag-and-drop upload for `.csv` and `.xlsx` files with instant schema profiling.
* Perform automatic Exploratory Data Analysis (EDA): row/column counts, missing values, duplicate checks, and descriptive statistics.
* Implement a supervised **Machine Learning Intent Classifier** (TF-IDF + Logistic Regression) evaluating precision, recall, and F1-score across 11 intent classes.
* Translate free-form English queries into validated JSON execution plans using local **Ollama** models (`qwen3:4b-instruct`).
* Safely compute 20 distinct tabular analytical tasks in Pandas without `eval()` or `exec()`.
* Automatically generate publication-grade charts (bar charts, histograms, scatter plots, trend lines).
* Maintain conversational context for follow-up questions (e.g. *"What is the maximum?"* after asking about salary).
* Run 100% locally with zero cloud dependencies or external API keys.

---

## 4. Main Features & Complete KDD Workflow

The system supports the full 5-stage **Knowledge Discovery in Databases (KDD)** methodology:

### Stage 1: Selection & Data Inspection
* **Column Selection**: Subsetting specific columns (`select_columns`).
* **Condition Filtering**: Numerical and categorical filtering with multi-condition `AND` / `OR` logic (`filter`).
* **Sampling & Slicing**: Random sampling (`sample_rows`) and range slicing (`slice_rows`).
* **Ranking & Ordering**: Sorting ascending/descending, Top N (`top_n`), and Bottom N (`bottom_n`).
* **Type-Based Selection**: Extracting all numerical, categorical, or datetime features.
* **Metadata Detection**: Heuristic identification of potential Primary Key ID columns and candidate Target labels.

### Stage 2: Preprocessing & Data Cleaning
* **Missing Value Analysis**: Complete dataset breakdown and column-specific counts/percentages.
* **Safe Imputation**: Mean, Median, Mode, Constant, Forward-Fill (`ffill`), and Backward-Fill (`bfill`).
* **De-duplication**: Automatic duplicate row detection and removal (`drop_duplicates`).
* **Text Normalization**: Leading/trailing whitespace trimming (`trim_whitespace`) and casing (`text_case`: lower, upper, title).
* **Category Standardization**: Whitespace trimming with title-casing for consistent categorical values.
* **Type Coercion with Reporting**: Converting strings to numbers (`to_numeric`) or datetimes (`to_datetime`) with invalid record logging.
* **Outlier Handling**: IQR detection, outlier row filtering, and Winsorization/capping (`cap_outliers`).
* **Infinite Value Handling**: Converting `+inf`/`-inf` to `NaN`.

### Stage 3: Data Transformation
* **Feature Scaling**: Min-Max Normalization $[0, 1]$, Z-Score Standardization (mean=0, std=1), and Robust Scaling (Median/IQR).
* **Nonlinear Transforms**: Logarithmic transformation $\log(1 + x)$ for positive distributions.
* **Discretization & Binning**: Equal-width interval binning and quantile-based binning (`bin_column`).
* **Categorical Encoding**: Label encoding (`encode_category`) and One-Hot dummy encoding (`one_hot_encode`).
* **Date Part Extraction**: Extracting year, month, day, and weekday from temporal columns.
* **Column Arithmetic**: Safe derived columns (`create_derived_column`) and conditional indicators (`create_conditional_column`).
* **Reshaping & Summaries**: Pivot tables and cross-tabulations (`crosstab`).

### Stage 4: Data Mining
* **Pattern Analysis**: Pearson and Spearman correlation matrices with ranking of top correlated pairs.
* **Unsupervised Clustering**:
  * **K-Means Clustering**: Multi-cluster grouping with centroid profiling and cluster sizing.
  * **DBSCAN**: Density-based clustering with automatic noise-point identification.
* **Supervised Learning**:
  * **Classification**: Decision Trees and Logistic Regression with 80/20 train/test splits, computing Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
  * **Regression**: Linear Regression with train/test validation, reporting MAE, RMSE, and $R^2$ scores.
* **Dimensionality Reduction**: Principal Component Analysis (PCA) 2D projection with explained variance ratios.

### Stage 5: Interpretation & Presentation
* **Plain English Insights**: Evidence-grounded conversational answers generated with zero numeric hallucination.
* **Automated Visualizations**: Matplotlib plots automatically chosen: Bar charts, Box plots, Histograms, Scatter plots, Line charts, Pie charts, Correlation heatmaps, Cluster scatters, Confusion matrices, Actual vs. Predicted plots, and PCA projections.
* **Modified Data Export**: One-click download button directly below the Ask button to export cleaned and transformed CSV datasets.

---

## 5. Technology Stack

| Category | Technology / Library | Purpose in Project |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core development language |
| **Data Processing** | `pandas` (>=2.2.0) | High-performance DataFrame operations |
| **Numerical Math** | `numpy` (>=1.26.0) | Numerical statistics and vector computations |
| **Excel Parser** | `openpyxl` (>=3.1.2) | Reading `.xlsx` workbook formats |
| **Machine Learning** | `scikit-learn` (>=1.4.0) | Intent classifier, K-Means, DBSCAN, PCA, Decision Tree, Logistic Regression, Linear Regression |
| **Visualization** | `matplotlib` (>=3.8.0) | Clean figure rendering using server-safe `Agg` backend |
| **Web Interface** | `gradio` (>=4.20.0) | Reactive web UI with chat, tables, and plot rendering |
| **Environment Mgmt** | `python-dotenv` (>=1.0.1) | Environment variable management via `.env` |
| **LLM Communication** | `requests` (>=2.31.0) | HTTP REST communication with local Ollama daemon (`/api/chat`, `/api/tags`) |
| **Local LLM Engine** | **Ollama** | Local LLM server running `qwen3:4b-instruct` on `http://localhost:11434` |

---

## 11. How to Run the Automated Test Suite

The project includes **99 comprehensive automated unit tests** covering all modules:

```powershell
# Run the entire test suite (99 tests):
python -m unittest discover -s tests

# Or run individual test suites:
python -m unittest tests/test_kdd_full_operations.py
python -m unittest tests/test_analysis_engine.py
python -m unittest tests/test_query_planner.py
python -m unittest tests/test_data_profiler.py
python -m unittest tests/test_security_safeguards.py
```

*Note: All Ollama unit tests in `test_llm_client.py` use mocked HTTP responses and do not require a running Ollama server to execute.*

---

## 12. Empirical Evaluation & Benchmark Results

Run the full evaluation benchmark:
```powershell
python evaluation.py
```

### Benchmark Summary (Measured from 17 Ground-Truth Test Cases):
* **Total Test Cases**: 17
* **Supported Analytical Queries**: 16
* **Analytical Correctness**: **100.0%** (16/16 verified ground-truth matches)
* **Execution Success Rate**: **100.0%**
* **Out-of-Scope Safety Rejection**: **PASS** (Unsupported queries safely handled)
* **ML Classifier Test Accuracy**: **85.71%** (Macro F1: 0.8545)
* **Average Response Latency**: ~355 ms

---

## 6. System Architecture

```text
               [ User Uploads CSV/Excel & Types Query in English ]
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │        Gradio Web User Interface        │
                   │                (app.py)                 │
                   └────────────────────┬────────────────────┘
                                        │
            ┌───────────────────────────┴───────────────────────────┐
            ▼                                                       ▼
┌──────────────────────────────┐                         ┌───────────────────────┐
│   Supervised ML Classifier   │                         │   LLM Query Planner   │
│    (query_classifier.py)     │                         │  (query_planner.py)   │
│   TF-IDF + Logistic Reg.     │                         │           │           │
│  Predicts 1 of 11 intents    │                         │           ▼           │
└──────────────┬───────────────┘                         │  Ollama Local LLM     │
               │                                         │    (llm_client.py)    │
               │ (Visual Badge)                          │  Strict JSON Planning │
               │                                         └───────────┬───────────┘
               │                                                     │
               │                                                     ▼
               │                                         ┌───────────────────────┐
               │                                         │  Plan Safety Checker  │
               │                                         │ Schema & type checks  │
               │                                         └───────────┬───────────┘
               │                                                     │
               │                                                     ▼
               │                                         ┌───────────────────────┐
               │                                         │    Analysis Engine    │
               │                                         │ (analysis_engine.py)  │
               │                                         │  Pandas computations  │
               │                                         └─────┬───────────┬─────┘
               │                                               │           │
               │                        ┌──────────────────────┘           └──────────────────────┐
               │                        ▼                                                         ▼
               │          ┌───────────────────────────┐                             ┌───────────────────────────┐
               │          │    Response Generator     │                             │    Visualization Engine   │
               │          │  (response_generator.py)  │                             │    (visualization.py)     │
               │          │  Ollama explains result   │                             │ Generates Matplotlib plot │
               │          └─────────────┬─────────────┘                             └─────────────┬─────────────┘
               │                        │                                                         │
               └────────────────────────┼─────────────────────────────────────────────────────────┘
                                        ▼
                          [ Render Answer & Plot in UI ]
```

---

## 7. Project Folder Structure

```text
Ai-Data-Analyst-Agent-ML-Project/
│
├── app.py                      # Main web application (Gradio UI & orchestration)
├── data_loader.py              # Secure CSV & Excel loader with format & size guards
├── data_profiler.py            # Automated dataset profiling & descriptive statistics
├── llm_client.py               # Centralized Ollama HTTP client (/api/chat, /api/tags)
├── query_planner.py            # Converts natural language into validated JSON plans
├── analysis_engine.py          # Deterministic Pandas engine executing 20 analytical tasks
├── visualization.py            # Matplotlib figure generation (Bar, Line, Histogram, Scatter)
├── query_classifier.py         # Supervised ML model (TF-IDF + Logistic Regression)
├── response_generator.py       # Grounded conversational answer formulation via Ollama
├── evaluation.py               # Empirical evaluation suite measuring correctness & latency
│
├── datasets/
│   ├── sample_data.csv         # Benchmark employee dataset with mixed features & nulls
│   └── sample_data.xlsx        # Excel workbook edition for format testing
│
├── tests/
│   ├── test_data_loader.py            # 5 tests: CSV/Excel loading, metadata, file errors
│   ├── test_data_profiler.py          # 6 tests: statistics, types, duplicate/null checks
│   ├── test_analysis_engine.py        # 10 tests: analytical tasks, bounds, edge cases
│   ├── test_llm_client.py             # 15 tests: Ollama status, chat, errors, timeouts, mocks
│   ├── test_query_planner.py          # 5 tests: validation, heuristic planning, errors
│   ├── test_visualization.py          # 6 tests: bar, line, scatter, histogram rendering
│   ├── test_query_classifier.py       # 5 tests: model convergence, accuracy threshold
│   ├── test_conversational_followup.py# 3 tests: multi-turn dialogue, context inheritance
│   └── test_security_safeguards.py    # 4 tests: code injection, length limits, file types
│
├── evaluation_report.json      # Saved JSON report from evaluation benchmark
├── .env                        # Local Ollama configuration (ignored by Git)
├── .env.example                # Configuration template
├── .gitignore                  # Git ignore rules for venv, cache, and local files
├── requirements.txt            # Minimal pip dependencies
└── README.md                   # Complete documentation and viva guide
```

---

## 8. Installation & Setup Instructions

### Step 1: Install & Start Ollama
1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Pull the default lightweight model:
   ```bash
   ollama pull qwen3:4b-instruct
   ```
3. Start the Ollama server (if not already running as a Windows background task):
   ```bash
   ollama serve
   ```
*(Note: On Windows, Ollama typically runs automatically as a system tray service at `http://localhost:11434`.)*

### Step 2: Set Up Python Virtual Environment
In a new terminal window:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment on Windows PowerShell
.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env` (or edit existing `.env`):
```ini
# Ollama base URL (defaults to http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Ollama model name (default: qwen3:4b-instruct)
OLLAMA_MODEL=qwen3:4b-instruct
```
*No cloud API keys are required!* If Ollama is not active, the system automatically uses the built-in deterministic heuristic planner so the app remains fully functional.

---

## 9. How to Run the Application

Start the local web application:
```powershell
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:7860
```

### Quick UI Tour:
1. Click **"📊 Load Sample Employee Dataset"** (or upload your own CSV/Excel file).
2. Expand **"📋 Dataset Profiling & Statistics"** to view descriptive statistics and missing values.
3. Check the **LLM Status Badge**: Shows `🟢 LLM: Connected (Ollama - qwen3:4b-instruct)` or `🟡 Mode: Heuristic Fallback` with clear instructions.
4. Use one-click prompts or ask questions in plain English:
   * *"What is the average salary?"*
   * *"Show average salary by department"*
   * *"Top 3 employees by salary"*
   * *"Show employees with salary greater than 70000"*
   * *"Are there any missing values?"*
   * *"What is the correlation between experience and salary?"*

---

## 10. How to Train and Evaluate the ML Classifier

To execute the standalone Machine Learning intent classification pipeline:
```powershell
python query_classifier.py
```

### Measured Performance Metrics:
* **Algorithm**: Multiclass Logistic Regression (`solver='lbfgs'`, $C=5.0$)
* **Feature Extraction**: TF-IDF Vectorizer with Unigrams and Bigrams (`ngram_range=(1, 2)`)
* **Split Strategy**: Stratified Train/Test Split (75% train / 25% test, `random_state=42`)
* **Overall Test Accuracy**: **85.71%**
* **Macro F1-Score**: **0.8545**
* **Weighted F1-Score**: **0.8643**

---

## 11. How to Run the Automated Test Suite

The project includes **59 comprehensive automated unit tests** covering all modules:

```powershell
# Run the entire test suite:
python -m unittest discover -s tests

# Or run individual test modules:
python -m unittest tests/test_llm_client.py
python -m unittest tests/test_data_loader.py
python -m unittest tests/test_analysis_engine.py
python -m unittest tests/test_security_safeguards.py
```

*Note: All Ollama unit tests in `test_llm_client.py` use mocked HTTP responses and do not require a running Ollama server to execute.*

---

## 12. Empirical Evaluation & Benchmark Results

Run the full evaluation benchmark:
```powershell
python evaluation.py
```

### Benchmark Summary (Measured from 12 Ground-Truth Test Cases):
* **Total Test Cases**: 12
* **Supported Analytical Queries**: 11
* **Analytical Correctness**: **100.0%** (11/11 verified numerical matches)
* **Execution Success Rate**: **100.0%**
* **Out-of-Scope Safety Rejection**: **PASS** (Unsupported queries safely handled)

---

## 13. System Limitations & Offline Processing

1. **Local Hardware Constraints**: Ollama runs on your local machine. Inference speed depends on your local CPU or GPU. Lightweight models such as `qwen3:4b-instruct` (3B) or `phi3` provide optimal performance on standard laptops.
2. **Single-Table Scope**: The current release operates on single-table datasets. Multi-table relational joins (`JOIN` across multiple CSVs) are planned for a future release.
3. **File Size Ceiling**: Uploads are restricted to 25 MB to maintain responsiveness in local memory.
4. **Context Window**: Follow-up questions track recent conversational turns. Extensive multi-turn dialogues are pruned to prevent memory bloat.

---

## 14. ML Viva Questions and Answers (Quick Revision)

### Q1: Why did you migrate from cloud Gemini to local Ollama?
> **Answer**: Local Ollama provides 100% privacy (data never leaves the host machine), eliminates external API key dependencies and billing quotas, enables offline classroom demonstrations, and ensures repeatable local evaluation.

### Q2: Why did you use TF-IDF instead of simple Count Vectorizer?
> **Answer**: Count Vectorizer gives disproportionate weight to frequent non-informative words (*"what"*, *"is"*). **TF-IDF** penalizes ubiquitous words and assigns higher weights to domain-specific analytical keywords (*"average"*, *"histogram"*, *"correlation"*).

### Q3: Why can't we just ask the LLM to calculate mathematical answers directly?
> **Answer**: LLMs are autoregressive token predictors trained on text. They do not contain arithmetic ALUs and frequently hallucinate numeric sums and averages. Delegating calculation to **Pandas and NumPy** guarantees 100% mathematical precision.

### Q4: How did you safeguard the system against arbitrary code execution attacks?
> **Answer**: We completely banned dynamic code execution functions (`eval()` and `exec()`). The Ollama LLM produces a structured JSON schema with whitelisted tasks (`mean`, `filter`, `top_n`, etc.). The **Analysis Engine** maps approved tasks to verified Pandas functions deterministically.
