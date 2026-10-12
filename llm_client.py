"""
llm_client.py
-------------
Modular API client interface for connecting to local instruction-tuned LLMs via Ollama.
Communicates strictly via Ollama's local HTTP API (defaults to http://localhost:11434).

Features:
- Configurable base URL and model via environment variables (OLLAMA_BASE_URL, OLLAMA_MODEL)
- Health and model availability checks via Ollama's /api/tags endpoint
- Conversational chat requests via Ollama's /api/chat endpoint
- Native JSON Mode support (format="json") for structured query planning
- Robust error handling for connection failures, missing models, timeouts, and HTTP errors
- 100% offline, local execution with zero cloud API keys or dependencies
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

# Load variables from .env file into os.environ
load_dotenv()

# Default configuration parameters
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")

# Internal cache for status checks to prevent connection latency spikes
_STATUS_CACHE = {"timestamp": 0.0, "result": None, "key": None}


def reset_status_cache():
    """Resets the cached status result (useful for test isolation)."""
    global _STATUS_CACHE
    _STATUS_CACHE = {"timestamp": 0.0, "result": None, "key": None}


def get_base_url() -> str:
    """Retrieves the Ollama base URL from environment variables."""
    url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL).strip()
    return url.rstrip("/") if url else "http://localhost:11434"


def get_model_name() -> str:
    """Retrieves the Ollama model name from environment variables."""
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL).strip()
    return model if model else "qwen3:4b-instruct"


def check_api_status(
    base_url: str = None,
    model_name: str = None,
    use_cache: bool = True,
    cache_ttl: float = 5.0,
) -> dict:
    """
    Checks if the local Ollama server is reachable and if the requested model is available.
    Uses the /api/tags endpoint.

    Parameters:
        base_url (str, optional): Ollama server base URL.
        model_name (str, optional): Target model name.
        use_cache (bool): Whether to use cached check result within cache_ttl seconds.
        cache_ttl (float): Cache duration in seconds (default 5.0s).

    Returns:
        dict: {
            "configured": bool,
            "provider": "Ollama",
            "model": str,
            "message": str
        }
    """
    target_base = (base_url or get_base_url()).rstrip("/")
    target_model = (model_name or get_model_name()).strip()
    cache_key = (target_base, target_model)

    if use_cache and _STATUS_CACHE["key"] == cache_key:
        effective_ttl = 30.0 if (_STATUS_CACHE["result"] and not _STATUS_CACHE["result"].get("configured")) else cache_ttl
        if (time.time() - _STATUS_CACHE["timestamp"]) < effective_ttl:
            return _STATUS_CACHE["result"]

    result = None
    try:
        response = requests.get(f"{target_base}/api/tags", timeout=1.0)
        if response.status_code == 200:
            data = response.json()
            models_list = data.get("models", [])
            available_names = [m.get("name", "") for m in models_list]

            # Model names in Ollama can be "llama3.2:latest", "llama3.2", etc.
            model_found = False
            for m in available_names:
                m_clean = m.lower()
                req_clean = target_model.lower()
                if (
                    m_clean == req_clean
                    or m_clean.startswith(f"{req_clean}:")
                    or m_clean.split(":")[0] == req_clean
                ):
                    model_found = True
                    break

            if model_found:
                result = {
                    "configured": True,
                    "provider": "Ollama",
                    "model": target_model,
                    "message": f"Ollama is running at {target_base} and model '{target_model}' is ready.",
                }
            else:
                result = {
                    "configured": False,
                    "provider": "Ollama",
                    "model": target_model,
                    "message": (
                        f"Ollama server is running, but model '{target_model}' was not found. "
                        f"Please pull it by running: 'ollama pull {target_model}'"
                    ),
                }
        else:
            result = {
                "configured": False,
                "provider": "Ollama",
                "model": target_model,
                "message": f"Ollama server returned HTTP {response.status_code}.",
            }

    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
        result = {
            "configured": False,
            "provider": "Ollama",
            "model": target_model,
            "message": (
                f"Cannot connect to Ollama at {target_base}. "
                "Please start the Ollama server using 'ollama serve' or open the Ollama desktop app."
            ),
        }
    except requests.exceptions.Timeout:
        result = {
            "configured": False,
            "provider": "Ollama",
            "model": target_model,
            "message": f"Connection to Ollama server at {target_base} timed out.",
        }
    except Exception as e:
        result = {
            "configured": False,
            "provider": "Ollama",
            "model": target_model,
            "message": f"Error connecting to Ollama: {str(e)}",
        }

    if use_cache:
        _STATUS_CACHE["timestamp"] = time.time()
        _STATUS_CACHE["key"] = cache_key
        _STATUS_CACHE["result"] = result

    return result


def call_llm(
    prompt: str,
    system_instruction: str = None,
    json_mode: bool = False,
    model_name: str = None,
    base_url: str = None,
    timeout: int = 60,
) -> tuple:
    """
    Calls Ollama's /api/chat endpoint with a prompt and returns the text response safely.

    Parameters:
        prompt (str): The user/task prompt.
        system_instruction (str, optional): System level instruction or persona.
        json_mode (bool): When True, instructs model to return strict JSON using format="json".
        model_name (str, optional): Target model name (defaults to OLLAMA_MODEL).
        base_url (str, optional): Ollama server base URL (defaults to OLLAMA_BASE_URL).
        timeout (int): Request timeout in seconds (default 60).

    Returns:
        tuple: (response_text, error_message)
               - response_text: str if successful, else None
               - error_message: str describing error, or None if successful
    """
    target_base = (base_url or get_base_url()).rstrip("/")
    target_model = (model_name or get_model_name()).strip()

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": target_model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }

    if json_mode:
        payload["format"] = "json"

    try:
        response = requests.post(
            f"{target_base}/api/chat",
            json=payload,
            timeout=timeout,
        )

        if response.status_code == 404:
            return None, (
                f"Error: Model '{target_model}' was not found on Ollama. "
                f"Please pull it by running: 'ollama pull {target_model}'"
            )
        elif response.status_code != 200:
            return None, f"Error: Ollama API returned HTTP status {response.status_code}."

        res_json = response.json()
        message_obj = res_json.get("message", {})
        content = message_obj.get("content", "")

        if not content or not content.strip():
            return None, "Error: The model returned an empty response."

        output_text = content.strip()

        if json_mode:
            output_text = _clean_json_markdown(output_text)

        return output_text, None

    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
        return None, (
            f"Error: Cannot connect to Ollama server at {target_base}. "
            "Please ensure Ollama is running ('ollama serve')."
        )
    except requests.exceptions.Timeout:
        return None, f"Error: Request to Ollama server at {target_base} timed out after {timeout} seconds."
    except json.JSONDecodeError:
        return None, "Error: Failed to parse Ollama response as JSON."
    except Exception as e:
        return None, f"LLM API Error: {str(e)}"


def _clean_json_markdown(text: str) -> str:
    """Strips markdown ```json ``` wrappers if returned by the LLM."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


# Standalone demonstration block
if __name__ == "__main__":
    print("=" * 60)
    print("CHECKING OLLAMA LLM CLIENT CONFIGURATION")
    print("=" * 60)

    status = check_api_status()
    print(f"Provider:    {status['provider']}")
    print(f"Model:       {status['model']}")
    print(f"Configured:  {status['configured']}")
    print(f"Message:     {status['message']}")

    if status["configured"]:
        print(f"\nSending a test ping to Ollama /api/chat ({status['model']})...")
        res, err = call_llm("Return only the word: PONG", json_mode=False)
        if err:
            print(f"Ping failed: {err}")
        else:
            print(f"LLM Response: {res}")
            print("SUCCESS: Ollama LLM client is live and operational!")
    else:
        print("\nNote: Start Ollama with 'ollama serve' and pull your model ('ollama pull llama3.2') for live local LLM inference.")
