# InsightForge AI v4 — Functionality Update (v4.1)

This is a **two-file frontend update** for an existing working v4 installation. It uses the current FastAPI routes; you do NOT need to reinstall Python dependencies or replace the backend.

## Changes

1. **Uploaded dataset state:** Upload area changes to a compact dataset confirmation with **Download current data**. This downloads the current active dataset (including changes made through cleaning/transformation queries), using the existing `/api/export/{session_id}` route. **Replace files** and **Replace folder** remain available. Drag-and-drop still works.
2. **Active Dataset tabs:** Overview now displays duplicates, missing cells, numeric column count and a dataset-explorer link. Columns now lists all actual columns with data types/missing values and offers search. Long lists scroll inside the panel.
3. **Working analysis-result cards:** Clicking an assistant's chart/table result opens the corresponding real result in the full-size Visualization Studio page, rather than merely updating an offscreen right dock.
4. **Sidebar Visualization Studio:** Full-width visualization page with Chart/Table tabs, session result history, and **a functional Qwen3 chat box** using the same conversation and current dataset. Sending a question keeps you on the Visualization Studio page. New generated results update the display unless pinned.
5. **One-click quick questions:** Suggested prompt buttons and ML Studio shortcuts immediately send the question; no additional Enter key is needed. Suggested questions are disabled until a dataset is loaded.
6. **Styling:** Includes the previous readability patch and the new styling; no horizontal **page** scrolling. Navigation panels still resize as before.

## Apply on Windows PowerShell

1. Download `InsightForge_v4_functionality_update.zip` to your Downloads folder.
2. Run:

```powershell
cd C:\Users\LOQ\Ai-Data-Analyst-Agent-ML-Project
git branch --show-current
```

It **must show `v4`**. This patch is intended for that branch.

3. Extract the ZIP into the project root (contains `frontend/src/App.tsx` and `frontend/src/styles.css`):

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\InsightForge_v4_functionality_update.zip" -DestinationPath . -Force
```

4. With your Python server (`uvicorn`) and React Vite server (`npm run dev`) running, open `http://127.0.0.1:5173`, then press `Ctrl + Shift + R`.

5. Test: upload CSV; see the Download button; switch Overview/Columns; ask a question that returns a table; click its result card; view that table in Visualization Studio; ask a question in the studio chat; return to AI Analyst and click one suggested question.

Optional build check from `frontend`:

```powershell
npm run build
```

To commit and push only this update (no untracked export CSVs):

```powershell
cd C:\Users\LOQ\Ai-Data-Analyst-Agent-ML-Project
git add frontend/src/App.tsx frontend/src/styles.css
git diff --cached --stat
git commit -m "Improve V4 analysis workspace interactions and Visualization Studio"
git push -u origin v4
```

## Notes
- These updates operate on your existing single-active-table analysis model. They do not add multi-table joins or PDF report generation.
- “Download current data” exports the current in-memory active dataset; if it hasn't been modified, the CSV contains its current unmodified contents.
- Studio charts remain Matplotlib-generated PNGs; they're larger on the full page, not Plotly-based interactive plot points yet.
- The local API and Ollama must be running. These changes do not change their configuration.
- This package was validated for TypeScript/JSX syntax and CSS parsing. A production `npm run build` and live browser interaction should be checked in your local project with dependencies installed.
