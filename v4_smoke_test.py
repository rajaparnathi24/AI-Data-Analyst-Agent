"""Lightweight smoke test for v4's HTTP API without calling Ollama.

Run from the repository root after installing requirements-v4.txt:
    .\\.venv\\Scripts\\python.exe v4_smoke_test.py
"""
from fastapi.testclient import TestClient
from insightforge_api import app

client = TestClient(app)


def run():
    created = client.post('/api/session')
    assert created.status_code == 200, created.text
    sid = created.json()['session_id']
    sample = client.post('/api/sample', json={'session_id': sid})
    assert sample.status_code == 200, sample.text
    dataset = sample.json()['dataset']
    assert dataset['rows'] > 0
    assert dataset['columns'] > 0
    # Data export must always work without an LLM.
    export = client.get(f'/api/export/{sid}')
    assert export.status_code == 200, export.text
    assert b',' in export.content
    restored = client.get(f'/api/session/{sid}')
    assert restored.status_code == 200, restored.text
    assert restored.json()['dataset']['name'] == 'sample_data.csv'
    reset = client.post('/api/reset', json={'session_id': sid})
    assert reset.status_code == 200, reset.text
    assert client.get(f'/api/session/{sid}').json()['dataset'] is None
    print('v4 smoke test PASS: sessions, sample dataset, export, restore, reset.')


if __name__ == '__main__':
    run()
