# InsightForge AI

**A local, conversational AI data-analysis workspace — React + FastAPI + Ollama (Qwen3 4B).**

InsightForge AI lets users upload CSV and Excel data, ask questions in natural language, inspect results calculated with Pandas, generate charts, and explore machine-learning operations through a modern three-column interface. It is built for local development and academic demonstrations, **not public multi-user deployment**.

> **Current branch:** `v4` · **Local model:** `qwen3:4b-instruct` · **Frontend:** React / TypeScript · **Backend:** FastAPI / Python

## V4 features

| Workspace | What it does |
| --- | --- |
| **Overview** | Dataset summary and starting points for analysis |
| **AI Analyst** | Chat-style questions, grounded analytical answers, follow-up context, and one-click suggested questions |
| **Datasets** | Preview the active table, inspect columns and data quality, and export the current dataset |
| **Visualization Studio** | View charts and data tables at a larger size, browse session results, and ask more questions using the same dataset |
| **ML Studio** | Shortcuts to supported machine-learning operations in the existing analysis engine |
| **Reports** | Access available exports; automatic PDF report creation is **not implemented** |

The desktop interface has a navigation/history sidebar on the left, chat in the center, and a compact upload, dataset-context, and chart dock on the right. Users can drag dividers to resize panels. Panels adapt to narrower windows to avoid horizontal **page** scrolling; the chart dock can be expanded, pinned, or hidden.

### Upload, inspect and visualize

- Browse or drag and drop **CSV, XLSX, XLS**, or a folder containing supported files.
- Import up to **20 files per selection**, maximum **25 MB per file**.
- Select one active dataset at a time; multiple imported datasets can be switched, but **cross-file joins are not supported**.
- Inspect rows, columns, missing cells, duplicate rows, column names, data types and per-column missing values.
- After uploading, **Download current data** exports the active table as CSV, including changes made by supported cleaning/transformation tasks. The user can also replace the imported files or folder.
- Clicking a suggested quick question **submits it immediately**; an active dataset is required.
- Clicking a chart/table result card opens that result in **Visualization Studio**, which has its own chat input and shares the analysis session.
- Download generated charts as PNG images.

### Analytical capabilities

The Python engine separates language understanding from numerical computation:

1. `query_classifier.py` — scikit-learn intent classification.
2. `query_planner.py` — natural-language question to a validated, whitelisted execution plan, with deterministic fallback.
3. `analysis_engine.py` — statistics, filtering, grouping, cleaning, transformation, clustering and selected supervised-learning operations using Pandas, NumPy and scikit-learn.
4. `visualization.py` — Matplotlib charts based on analytical results.
5. `response_generator.py` — explanations grounded in the results, using Ollama when available.

Supported tasks include summary statistics, missing-value analysis, sorting/filtering, group aggregation, encoding, scaling, correlations, K-Means, DBSCAN, classification, regression and PCA. The exact task available depends on the dataset and planner.

**Important:** The LLM helps interpret questions and generate safe plans; it is **not** trusted to execute arbitrary generated Python. Actual computations come from the analysis engine. AI-produced explanations should still be reviewed.

## System architecture

```text
Browser (http://127.0.0.1:5173)
  React + TypeScript + Vite
        |  /api via Vite development proxy
        v
FastAPI (http://127.0.0.1:8000)
  insightforge_api.py
    |-- Upload and in-memory session state
    |-- Data loader and profiler
    |-- Query planner <---- Ollama / Qwen3 4B (localhost:11434)
    |-- Pandas / NumPy / scikit-learn analysis engine
    |-- Response generator
    '-- Matplotlib ---> PNG chart / result table
```

The original **Gradio** app (`app.py`) remains as an optional legacy interface. **V4's primary interface is React, not Gradio.**

## Technology stack

| Layer | Technologies |
| --- | --- |
| Frontend | React 19, TypeScript, Vite 6, Lucide React, react-markdown |
| API | FastAPI, Uvicorn, Pydantic, python-multipart |
| Local LLM | Ollama + `qwen3:4b-instruct` |
| Analysis | Python, Pandas, NumPy, scikit-learn |
| Charts | Matplotlib-generated images |
| Excel parsing | openpyxl (`.xlsx`), xlrd (`.xls`) |
| Legacy interface | Gradio (`app.py`, optional) |

## Quick start — Windows PowerShell

### Prerequisites

- **Python 3.11** recommended for dependency compatibility.
- **Node.js LTS** with `npm`.
- **Ollama** installed and running locally, with the Qwen3 model downloaded.

Clone V4 if this is your first setup:

```powershell
git clone --branch v4 https://github.com/chauhanyagnik34/Ai-Data-Analyst-Agent-ML-Project.git
cd Ai-Data-Analyst-Agent-ML-Project
```

If you already have the repository:

```powershell
cd C:\Users\LOQ\Ai-Data-Analyst-Agent-ML-Project
git switch v4
```

### 1. Python environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-v4.txt
```

If a working `.venv` already exists, use it rather than recreating it. `requirements.txt` contains core analytics/legacy dependencies; `requirements-v4.txt` adds the API layer.

### 2. Configure the local LLM

```powershell
ollama pull qwen3:4b-instruct
Copy-Item .env.example .env
```

The `.env` configuration should be:

```ini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct
```

On Windows, Ollama usually runs from its desktop app. If it is not active, start that app or run `ollama serve` in another terminal. Check the connection:

```powershell
.\.venv\Scripts\python.exe llm_client.py
```

Look for `Configured: True` and a response from the model. The first call may take longer while Ollama loads it.

### 3. Start the backend — Terminal 1

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn insightforge_api:app --host 127.0.0.1 --port 8000 --reload
```

- API health: http://127.0.0.1:8000/api/health
- Interactive API docs: http://127.0.0.1:8000/api/docs

**Keep this terminal running.**

### 4. Start the frontend — Terminal 2

```powershell
cd frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173** in the browser. **Keep this terminal running, too.** Vite proxies `/api` to the Python server on port 8000.

## How to use InsightForge

1. Upload a dataset from the right-hand card or drag in a supported folder.
2. Choose the active dataset if you imported multiple files.
3. Use the **Overview** and **Columns** tabs to inspect the table and its fields.
4. Ask questions such as **“What is the average salary?”**, **“Show missing values”**, or **“Compare average salary by department”**, using column names that actually exist in your data.
5. Open any chart/table result in **Visualization Studio** and continue asking questions from the studio's chat box.
6. Use **Download current data** to save the current table after analysis or supported transformations.

The compact chart dock supports resizing, expanding, shrinking and pinning. Pinning keeps the selected visualization visible while new results are generated.

## Local API routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Check the API and Ollama status |
| `POST` | `/api/session` | Create an in-memory analysis session |
| `GET` | `/api/session/{sid}` | Restore active dataset, chat and chart/table state |
| `POST` | `/api/upload` | Import supported files using multipart upload |
| `POST` | `/api/sample` | Load the bundled example dataset |
| `POST` | `/api/select-dataset` | Change the active imported dataset |
| `POST` | `/api/query` | Run a question against the active dataset |
| `GET` | `/api/export/{sid}` | Download the active table as CSV |
| `POST` | `/api/reset` | Clear session data |

For exact request/response schemas, see **http://127.0.0.1:8000/api/docs**.

## Repository structure

```text
Ai-Data-Analyst-Agent-ML-Project/
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Navigation, chat and visualization workspace
│   │   ├── styles.css           # Responsive panels, theme, typography
│   │   └── main.tsx             # React entry point
│   ├── package.json
│   └── vite.config.ts           # /api development proxy
├── insightforge_api.py          # V4 FastAPI bridge
├── requirements-v4.txt          # Additional API dependencies
├── v4_smoke_test.py             # API smoke tests
├── data_loader.py
├── data_profiler.py
├── query_classifier.py
├── query_planner.py
├── analysis_engine.py
├── visualization.py
├── response_generator.py
├── llm_client.py                # Local Ollama HTTP client
├── app.py                       # Legacy Gradio frontend (optional)
├── requirements.txt            # Core Python dependencies
├── .env.example                 # Local model configuration example
├── datasets/                    # Sample datasets
├── tests/                       # Python unit tests
├── evaluation.py
├── V4_SETUP.md                   # Additional setup details
└── APPLY_V4_1.md                # V4 UI functionality update notes
```

## Tests and checks

Run these from the repository root after installing dependencies:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe v4_smoke_test.py
```

The original test suite previously completed **111 tests successfully** in a local session following the Qwen3 configuration change. This is not a claim that CI or all V4 frontend interactions have been tested.

For the React build check:

```powershell
cd frontend
npm install
npm run build
```

Manually verify real dataset upload, query execution, transformations and CSV export, result-card navigation, the Studio chat, and mouse-controlled panel resizing.

## Current limitations

- **Development-only API:** No authentication or production-grade multi-user security. Do **not** expose the local API to the public internet.
- **Sessions stored in RAM:** Conversations and datasets survive a page refresh only while the Python server remains running. Restarting the backend loses them. The browser stores some preferences and chat titles, not uploaded data contents.
- **One active dataset:** Multiple files can be imported, but the engine does not join data across files.
- **Upload limits:** 25 MB per file and 20 files per upload. Unsupported formats are ignored or reported.
- **Static visualizations:** Matplotlib returns PNG images. Panels can resize the images, but chart points are not yet interactively selectable as in Plotly.
- **ML Studio scope:** Current UI offers shortcuts to supported engine operations. Dedicated AutoML and SHAP interfaces are future work.
- **Reports scope:** CSV and chart export are supported; automatic PDF reports are not yet implemented.
- **Data export:** Exports the current dataset to CSV; it does not overwrite the original Excel file.
- **Ollama status:** Some deterministic analysis can still work when Ollama is offline, but model-based interpretation and phrasing may be unavailable or reduced.
- **Local privacy:** The configured LLM runs locally, but users remain responsible for securing local files and devices.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Browser cannot open port `5173` | Restart `npm run dev` inside `frontend` |
| Frontend says backend unavailable | Start Uvicorn at `127.0.0.1:8000` from the repository root |
| `/docs` shows 404 | Use V4's documented API route: `/api/docs` |
| Ollama appears offline | Run `ollama ps`, verify `.env` and test `llm_client.py` |
| `ModuleNotFoundError` | Install both Python requirements files into the same virtual environment |
| `npm` is not recognized | Install Node.js LTS and restart PowerShell |
| No chart appears | Ask a chart-producing question; verify the Python API is running |
| Chat history disappears after API restart | Sessions are in server memory; persistence is not implemented |

## Legacy Gradio app

The earlier Gradio-based application is still available independently:

```powershell
.\.venv\Scripts\python.exe app.py
```

Its usual URL is **http://127.0.0.1:7860**. The primary V4 React frontend uses **http://127.0.0.1:5173**.

---

**Repository:** [AI Data Analyst Agent](https://github.com/chauhanyagnik34/Ai-Data-Analyst-Agent-ML-Project) · [V4 branch](https://github.com/chauhanyagnik34/Ai-Data-Analyst-Agent-ML-Project/tree/v4)

*InsightForge AI V4 — local AI-assisted data analysis in a responsive three-column workspace.*
