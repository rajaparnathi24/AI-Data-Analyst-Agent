"""
test_llm_client.py
------------------
Unit tests for llm_client.py validating Ollama HTTP client operations:
- Status detection and model verification via /api/tags
- Chat completion requests via /api/chat
- Markdown JSON stripping
- Graceful error handling (connection failure, timeouts, HTTP errors, 404 missing model, malformed responses)
- Fully mocked: runs offline without requiring a running Ollama server
"""

import unittest
from unittest.mock import patch, MagicMock
import requests
from llm_client import (
    check_api_status,
    _clean_json_markdown,
    call_llm,
    get_base_url,
    get_model_name,
    reset_status_cache,
)


class TestLLMClient(unittest.TestCase):

    def setUp(self):
        reset_status_cache()

    def test_status_structure(self):
        """Test status dictionary structure always returns required keys."""
        status = check_api_status()
        self.assertIn("configured", status)
        self.assertIn("provider", status)
        self.assertIn("model", status)
        self.assertIn("message", status)
        self.assertEqual(status["provider"], "Ollama")

    def test_clean_json_markdown(self):
        """Test stripping of markdown json tags."""
        raw = '```json\n{"task": "mean", "column": "Salary"}\n```'
        cleaned = _clean_json_markdown(raw)
        self.assertEqual(cleaned, '{"task": "mean", "column": "Salary"}')

        raw_generic = '```\n{"task": "sum"}\n```'
        self.assertEqual(_clean_json_markdown(raw_generic), '{"task": "sum"}')

        plain = '{"task": "count_rows"}'
        self.assertEqual(_clean_json_markdown(plain), '{"task": "count_rows"}')

    @patch("llm_client.requests.get")
    def test_check_api_status_success(self, mock_get):
        """Test check_api_status when Ollama server is running and model is installed."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3.2:latest", "model": "llama3.2:latest"},
                {"name": "nomic-embed-text:latest", "model": "nomic-embed-text:latest"},
            ]
        }
        mock_get.return_value = mock_resp

        status = check_api_status(model_name="llama3.2")
        self.assertTrue(status["configured"])
        self.assertEqual(status["provider"], "Ollama")
        self.assertIn("ready", status["message"])

    @patch("llm_client.requests.get")
    def test_check_api_status_missing_model(self, mock_get):
        """Test check_api_status when Ollama server is running but model is missing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "mistral:latest"}]
        }
        mock_get.return_value = mock_resp

        status = check_api_status(model_name="llama3.2")
        self.assertFalse(status["configured"])
        self.assertIn("ollama pull llama3.2", status["message"])

    @patch("llm_client.requests.get")
    def test_check_api_status_connection_error(self, mock_get):
        """Test check_api_status when Ollama server is not running."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        status = check_api_status()
        self.assertFalse(status["configured"])
        self.assertIn("Cannot connect to Ollama", status["message"])
        self.assertIn("ollama serve", status["message"])

    @patch("llm_client.requests.get")
    def test_check_api_status_timeout(self, mock_get):
        """Test check_api_status when request times out."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        status = check_api_status()
        self.assertFalse(status["configured"])
        self.assertIn("timed out", status["message"])

    @patch("llm_client.requests.get")
    def test_check_api_status_http_error(self, mock_get):
        """Test check_api_status when server returns an HTTP error code."""
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_get.return_value = mock_resp

        status = check_api_status()
        self.assertFalse(status["configured"])
        self.assertIn("502", status["message"])

    @patch("llm_client.requests.post")
    def test_call_llm_success(self, mock_post):
        """Test successful text completion via /api/chat."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "The average salary is 67,333.33."}
        }
        mock_post.return_value = mock_resp

        res, err = call_llm("What is the average salary?", system_instruction="Be concise.")
        self.assertIsNone(err)
        self.assertEqual(res, "The average salary is 67,333.33.")

    @patch("llm_client.requests.post")
    def test_call_llm_json_mode(self, mock_post):
        """Test JSON mode requesting structured output and stripping markdown wrappers."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "role": "assistant",
                "content": '```json\n{"task": "mean", "column": "Salary"}\n```',
            }
        }
        mock_post.return_value = mock_resp

        res, err = call_llm("Plan query", json_mode=True)
        self.assertIsNone(err)
        self.assertEqual(res, '{"task": "mean", "column": "Salary"}')

        # Verify format="json" was included in payload
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"].get("format"), "json")

    @patch("llm_client.requests.post")
    def test_call_llm_connection_error(self, mock_post):
        """Ensure call_llm returns a friendly error if Ollama server is not running."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        res, err = call_llm("Ping")
        self.assertIsNone(res)
        self.assertIsNotNone(err)
        self.assertIn("Cannot connect to Ollama server", err)
        self.assertIn("ollama serve", err)

    @patch("llm_client.requests.post")
    def test_call_llm_timeout(self, mock_post):
        """Ensure call_llm handles timeouts gracefully without crashing."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        res, err = call_llm("Ping", timeout=5)
        self.assertIsNone(res)
        self.assertIsNotNone(err)
        self.assertIn("timed out", err)

    @patch("llm_client.requests.post")
    def test_call_llm_model_not_found_404(self, mock_post):
        """Ensure call_llm provides clear model pull instruction when 404 is returned."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_post.return_value = mock_resp

        res, err = call_llm("Ping", model_name="llama3.2")
        self.assertIsNone(res)
        self.assertIsNotNone(err)
        self.assertIn("ollama pull llama3.2", err)

    @patch("llm_client.requests.post")
    def test_call_llm_http_error(self, mock_post):
        """Ensure call_llm handles unexpected HTTP status codes."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp

        res, err = call_llm("Ping")
        self.assertIsNone(res)
        self.assertIsNotNone(err)
        self.assertIn("HTTP status 500", err)

    @patch("llm_client.requests.post")
    def test_call_llm_empty_response(self, mock_post):
        """Ensure call_llm handles empty model responses gracefully."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "   "}}
        mock_post.return_value = mock_resp

        res, err = call_llm("Ping")
        self.assertIsNone(res)
        self.assertIn("empty response", err)

    @patch("llm_client.requests.post")
    def test_call_llm_malformed_json_response(self, mock_post):
        """Ensure call_llm handles malformed JSON from endpoint gracefully."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "doc", 0)
        mock_post.return_value = mock_resp

        res, err = call_llm("Ping")
        self.assertIsNone(res)
        self.assertIn("Failed to parse Ollama response as JSON", err)


if __name__ == "__main__":
    unittest.main()
