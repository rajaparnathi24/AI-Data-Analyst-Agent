"""InsightForge v4 API: thin local HTTP adapter for the existing analysis engine.

Run from the repository root: python -m uvicorn insightforge_api:app --host 127.0.0.1 --port 8000
The original app.py, tests, and core modules remain unchanged.
"""
from __future__ import annotations

import base64
import io
import math
import os
import re
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from data_loader import load_dataset
from llm_client import check_api_status, call_llm
from query_planner import plan_query
from analysis_engine import execute_plan
from response_generator import generate_response
from visualization import create_auto_chart

APP_ROOT = Path(__file__).resolve().parent
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_BATCH_FILES = 20
SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

app = FastAPI(title='InsightForge AI', version='4.0.0', docs_url='/api/docs', redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['GET', 'POST'],
    allow_headers=['Content-Type'],
)
_sessions: dict[str, dict[str, Any]] = {}
_guard = threading.RLock()


class QueryRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=4000)


class SessionRequest(BaseModel):
    session_id: str


def _session(sid: str) -> dict[str, Any]:
    with _guard:
        entry = _sessions.get(sid)
    if entry is None:
        raise HTTPException(404, 'Session not found. Create a new analysis and upload the dataset again.')
    return entry


def _clean_value(value: Any) -> Any:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, (float, int)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _table(value)
    if isinstance(value, pd.Series):
        return _clean_value(value.to_dict())
    if isinstance(value, (str, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _table(df: pd.DataFrame | None, limit: int = 100) -> dict[str, Any] | None:
    if df is None:
        return None
    small = df.head(limit)
    records = [{str(column): _clean_value(value) for column, value in row.items()}
               for row in small.to_dict(orient='records')]
    return {'columns': [str(col) for col in df.columns], 'rows': records, 'total': len(df)}


def _meta(df: pd.DataFrame, name: str) -> dict[str, Any]:
    missing = int(df.isna().sum().sum())
    cells = max(1, df.shape[0] * df.shape[1])
    return {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'missing': missing,
        'missing_pct': round(missing / cells * 100, 2),
        'duplicates': int(df.duplicated().sum()),
        'fields': [{'name': str(col), 'type': str(dtype), 'missing': int(df[col].isna().sum())}
                   for col, dtype in df.dtypes.items()],
        'preview': _table(df, 12),
    }


def _store_dataset(entry: dict[str, Any], df: pd.DataFrame, name: str) -> dict[str, Any]:
    with _guard:
        entry['df'] = df
        entry['name'] = name
        entry['context'] = []
        entry['messages'] = []
        entry['chart'] = None
        entry['table'] = None
        entry['meta'] = _meta(df, name)
    return entry['meta']


def _chart_image(fig) -> str | None:
    if fig is None:
        return None
    try:
        stream = io.BytesIO()
        fig.savefig(stream, format='png', dpi=130, bbox_inches='tight', facecolor='white')
        return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')
    finally:
        plt.close(fig)


@app.get('/api/health')
def health():
    status = check_api_status(use_cache=True)
    return {'running': True, 'ollama': bool(status.get('configured')), 'model': status.get('model'),
            'message': status.get('message')}


@app.post('/api/session')
def new_session():
    sid = str(uuid.uuid4())
    with _guard:
        _sessions[sid] = {'id': sid, 'df': None, 'name': None, 'meta': None,
                          'context': [], 'messages': [], 'chart': None, 'table': None, 'datasets': {}}
    return {'session_id': sid}


@app.get('/api/session/{sid}')
def get_session(sid: str):
    s = _session(sid)
    with _guard:
        return {'session_id': sid, 'dataset': s['meta'], 'messages': s['messages'],
                'chart': s['chart'], 'table': s['table'], 'datasets': list(s['datasets'])}


@app.post('/api/upload')
async def upload_dataset(session_id: str = Form(...), files: list[UploadFile] = File(...)):
    entry = _session(session_id)
    if not files or len(files) > MAX_BATCH_FILES:
        raise HTTPException(400, f'Select between 1 and {MAX_BATCH_FILES} supported dataset files.')
    uploaded = []
    ignored = []
    for file in files:
        name = Path((file.filename or '').replace('\\', '/')).name
        if not name or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            ignored.append(file.filename)
            continue
        contents = await file.read(MAX_FILE_BYTES + 1)
        await file.close()
        if len(contents) > MAX_FILE_BYTES:
            raise HTTPException(413, f'{name} exceeds the 25 MB per-file limit.')
        with tempfile.TemporaryDirectory() as d:
            temporary_file = Path(d) / name
            temporary_file.write_bytes(contents)
            df, error = load_dataset(str(temporary_file))
        if error:
            ignored.append(f'{name}: {error}')
            continue
        uploaded.append((name, df))
    if not uploaded:
        raise HTTPException(400, 'No supported datasets could be loaded. ' + '; '.join(ignored[:3]))
    # One active table at a time, but keep every valid imported table for selection.
    catalog = {}
    for name, df in uploaded:
        candidate, index = name, 2
        while candidate in catalog:
            candidate = f'{Path(name).stem}_{index}{Path(name).suffix}'
            index += 1
        catalog[candidate] = df
    with _guard:
        entry['datasets'] = catalog
    name = next(iter(catalog))
    info = _store_dataset(entry, catalog[name], name)
    return {'dataset': info, 'imported': list(catalog),
            'ignored': ignored, 'notice': ('Imported multiple datasets; choose one from the Active Dataset menu. '
            'Multi-table joins are not supported yet.') if len(catalog) > 1 else None}


@app.post('/api/sample')
def load_sample(request: SessionRequest):
    entry = _session(request.session_id)
    file = APP_ROOT / 'datasets' / 'sample_data.csv'
    df, error = load_dataset(str(file))
    if error:
        raise HTTPException(500, error)
    with _guard:
        entry['datasets'] = {file.name: df}
    return {'dataset': _store_dataset(entry, df, file.name), 'imported': [file.name]}



class SelectDatasetRequest(BaseModel):
    session_id: str
    name: str


@app.post('/api/select-dataset')
def select_dataset(request: SelectDatasetRequest):
    entry = _session(request.session_id)
    with _guard:
        df = entry['datasets'].get(request.name)
    if df is None:
        raise HTTPException(404, 'The requested dataset is not in this session.')
    return {'dataset': _store_dataset(entry, df, request.name),
            'imported': list(entry['datasets'])}

def _conversational(question: str, df: pd.DataFrame, name: str) -> str:
    text = f"You are InsightForge AI, a helpful data analysis assistant. Active dataset: {name}; rows: {len(df)}; columns: {', '.join(map(str, df.columns[:20]))}. Respond concisely, suggest useful questions if relevant; never invent statistics."
    if check_api_status().get('configured'):
        answer, error = call_llm(question, system_instruction=text, timeout=45)
        if not error and answer:
            return answer
    return f"I'm ready to analyze **{name}** ({len(df):,} rows). Try asking for averages, missing values, distributions, correlations, or charts."


@app.post('/api/query')
def run_query(request: QueryRequest):
    entry = _session(request.session_id)
    question = request.question.strip()
    with _guard:
        df = entry['df']
        name = entry['name']
        history = list(entry['context'])
    if df is None:
        raise HTTPException(400, 'Upload a dataset before asking an analysis question.')
    if re.fullmatch(r'(?i)(hi|hello|hey|help|who are you|what can you do|thanks|thank you)[!? .]*', question):
        answer = _conversational(question, df, name)
        payload = {'answer': answer, 'chart': None, 'table': None, 'dataset': entry['meta'], 'task': 'conversation'}
    else:
        try:
            plan, error = plan_query(question, df, conversation_history=history)
            if error or not plan:
                answer = f"I couldn't plan that analysis: {error or 'Unsupported request.'} Try rephrasing it using a dataset column name."
                payload = {'answer': answer, 'chart': None, 'table': None, 'dataset': entry['meta'], 'task': 'unavailable'}
            else:
                result = execute_plan(df, plan)
                answer = generate_response(question, result, dataset_name=name)
                table = _table(result.get('value')) if isinstance(result.get('value'), pd.DataFrame) else None
                fig = None
                if result.get('status') == 'success':
                    fig, _ = create_auto_chart(result, df)
                chart = _chart_image(fig)
                # Apply KDD transformations just as app.py does.
                updated = result.get('modified_df')
                if isinstance(updated, pd.DataFrame):
                    _store_updated = _meta(updated, name)
                    with _guard:
                        entry['df'] = updated
                        entry['datasets'][name] = updated
                        entry['meta'] = _store_updated
                with _guard:
                    entry['context'].append({'user': question, 'task': plan.get('task'),
                                             'column': plan.get('column') or plan.get('group_by') or plan.get('condition_col'),
                                             'analyzed_column': plan.get('column') or plan.get('group_by') or plan.get('condition_col')})
                payload = {'answer': answer, 'chart': chart, 'table': table,
                           'dataset': entry['meta'], 'task': plan.get('task')}
        except Exception as exc:
            # Keep detailed traceback server-side; avoid leaking local file paths to the UI.
            import logging
            logging.exception('Analysis request failed')
            raise HTTPException(500, 'Analysis failed. Check the backend terminal for details.') from exc
    with _guard:
        entry['messages'].extend([
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': payload['answer'], 'has_chart': bool(payload['chart']),
             'has_table': bool(payload['table']), 'task': payload['task'], 'chart': payload['chart'], 'table': payload['table']},
        ])
        if payload['chart']:
            entry['chart'] = payload['chart']
        if payload['table']:
            entry['table'] = payload['table']
    return payload


@app.get('/api/export/{sid}')
def export_dataset(sid: str):
    s = _session(sid)
    if s['df'] is None:
        raise HTTPException(400, 'No active dataset to export.')
    filename = re.sub(r'[^A-Za-z0-9_.-]', '_', str(s['name'] or 'dataset'))
    filename = Path(filename).stem + '_modified.csv'
    binary = io.BytesIO(s['df'].to_csv(index=False).encode('utf-8-sig'))
    return StreamingResponse(binary, media_type='text/csv', headers={
        'Content-Disposition': f'attachment; filename="{filename}"'})


@app.post('/api/reset')
def reset_session(request: SessionRequest):
    s = _session(request.session_id)
    with _guard:
        s.update({'df': None, 'name': None, 'meta': None,
                  'context': [], 'messages': [], 'chart': None, 'table': None, 'datasets': {}})
    return {'ok': True}
