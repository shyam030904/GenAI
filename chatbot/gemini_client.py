"""
chatbot/gemini_client.py
────────────────────────
Thin wrapper around the Google Gemini SDK (google-genai >= 1.0).

Responsibilities:
  - Build a Gemini client for a given API key.
  - Convert our internal history format to Gemini Content objects.
  - Call client.models.generate_content().
  - Classify the response as success / retryable / permanent failure.
  - Raise the appropriate exception so ai_service.py can act on it.

This module knows NOTHING about key rotation.
Key rotation lives entirely in ai_service.py + api_key_manager.py.
"""

import os
import logging

from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError, ClientError, ServerError

from .api_key_manager import PermanentAPIError

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# Google recommended model: gemini-3.8-flash
DEFAULT_MODEL     = "gemini-3.8-flash"
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE       = 0.7
MAX_HISTORY_TURNS = 10   # truncate history to limit token usage

# HTTP codes that mean "try the next key"
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

# HTTP codes that mean "the request is broken, rotating keys won't help"
# NOTE: 400 may also mean invalid API key — see _is_key_error() below.
PERMANENT_HTTP_CODES = {403, 404, 422}


# ── Internal retryable marker ─────────────────────────────────────────────────
class _RetryableGeminiError(Exception):
    """Signals that the caller (ai_service) should try the next API key."""
    def __init__(self, message: str, http_status: int | None = None):
        super().__init__(message)
        self.http_status = http_status


# ── Key-error detector ────────────────────────────────────────────────────────
def _is_key_error(exc: "ClientError") -> bool:
    """
    Return True when a ClientError is caused by an invalid/disabled API key.

    Gemini returns HTTP 400 (INVALID_ARGUMENT) for bad keys, NOT 401.
    We check the error reason field so we only rotate on key problems,
    not on genuinely malformed requests.
    """
    details_str = str(getattr(exc, "details", "") or "").upper()
    message_str = (getattr(exc, "message", "") or str(exc)).lower()
    return (
        "api_key_invalid" in details_str          # machine-readable reason
        or "api key not valid" in message_str     # human-readable message
        or ("key" in message_str and "valid" in message_str)
    )


# ── History conversion ────────────────────────────────────────────────────────
def _build_contents(prompt: str, history: list[dict]) -> list:
    """
    Convert our internal message list + new prompt into Gemini Content objects.

    Our format (from Django DB / views.py):
        [{"role": "user"|"model", "parts": ["text"]}, ...]

    Gemini SDK format:
        [Content(role="user"|"model", parts=[Part(text="...")])]
    """
    trimmed = (
        history[-MAX_HISTORY_TURNS:]
        if len(history) > MAX_HISTORY_TURNS
        else history
    )

    contents = []
    for msg in trimmed:
        role = msg.get("role", "user")
        if role not in ("user", "model"):
            role = "user"

        parts_val = msg.get("parts")
        if isinstance(parts_val, list) and parts_val:
            text = str(parts_val[0])
        else:
            text = str(msg.get("content", ""))

        contents.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=text)],
            )
        )

    # Add the new user turn
    contents.append(
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        )
    )
    return contents


# ── Main call function ────────────────────────────────────────────────────────
def call_gemini(
    api_key: str,
    key_label: str,
    prompt: str,
    history: list[dict],
) -> str:
    """
    Make a single Gemini API call.

    Parameters
    ----------
    api_key   : raw key value — never logged
    key_label : safe label like "API_KEY_1" — used in logs only
    prompt    : the user's latest message
    history   : previous messages in our internal dict format

    Returns
    -------
    str : the AI text reply

    Raises
    ------
    _RetryableGeminiError  : 429 / 5xx / timeout / invalid key → try next key
    PermanentAPIError      : 400 / 403 / bad model → don't rotate
    """
    model_name = (
        os.environ.get("AI_MODEL", "").strip() or DEFAULT_MODEL
    )

    client   = genai.Client(api_key=api_key)
    contents = _build_contents(prompt, history)
    config   = genai_types.GenerateContentConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        # ── Extract text ──────────────────────────────────────────────────
        text = response.text
        if not text:
            finish = None
            if response.candidates:
                finish = getattr(response.candidates[0], "finish_reason", None)
            logger.warning(
                "GeminiClient: %s returned empty text (finish_reason=%s).",
                key_label, finish,
            )
            raise PermanentAPIError(
                f"Gemini returned an empty response (finish_reason={finish})."
            )

        logger.info(
            "GeminiClient: %s succeeded (model=%s).", key_label, model_name
        )
        return text

    # ── SDK-native errors (google-genai >= 1.0) ───────────────────────────
    except ClientError as exc:
        # 4xx errors
        code = getattr(exc, "code", None)
        logger.warning(
            "GeminiClient: %s -> ClientError HTTP %s: %s",
            key_label, code, exc.message or str(exc),
        )
        if code == 429:
            # 429 = quota exhausted -> rotate to next key
            raise _RetryableGeminiError(str(exc), http_status=429) from exc
        if code == 400 and _is_key_error(exc):
            # Gemini returns HTTP 400 (not 401) for invalid/disabled API keys.
            # Treat as retryable so the manager tries the next configured key.
            logger.warning(
                "GeminiClient: %s -> invalid API key (HTTP 400 API_KEY_INVALID) "
                "- rotating to next key.",
                key_label,
            )
            raise _RetryableGeminiError(str(exc), http_status=400) from exc
        if code in PERMANENT_HTTP_CODES:
            # e.g. malformed prompt, unsupported model — rotating keys won't help
            raise PermanentAPIError(str(exc), http_status=code) from exc
        # 401, and any other 4xx -> retryable (try next key)
        raise _RetryableGeminiError(str(exc), http_status=code) from exc

    except ServerError as exc:
        # 5xx errors → always retryable
        code = getattr(exc, "code", None)
        logger.warning(
            "GeminiClient: %s → ServerError HTTP %s.", key_label, code
        )
        raise _RetryableGeminiError(str(exc), http_status=code) from exc

    except APIError as exc:
        # Generic SDK API error — check the code
        code = getattr(exc, "code", None)
        logger.warning(
            "GeminiClient: %s → APIError HTTP %s.", key_label, code
        )
        if code in RETRYABLE_HTTP_CODES:
            raise _RetryableGeminiError(str(exc), http_status=code) from exc
        if code in PERMANENT_HTTP_CODES:
            raise PermanentAPIError(str(exc), http_status=code) from exc
        # Unknown code — rotate to be safe
        raise _RetryableGeminiError(str(exc), http_status=code) from exc

    except PermanentAPIError:
        raise  # pass-through, don't rewrap

    except Exception as exc:
        # Connection error, timeout, unexpected — treat as retryable
        exc_type = type(exc).__name__
        logger.warning(
            "GeminiClient: %s → unexpected %s — treating as retryable.",
            key_label, exc_type,
        )
        raise _RetryableGeminiError(
            f"{exc_type}: {str(exc)}", http_status=None
        ) from exc
