"""
chatbot/ai_service.py
─────────────────────
The single public entry-point for AI generation in this project.

Orchestrates:
    Chat View  →  ai_service.generate_response()
                      │
                      ├─ APIKeyManager.get_active_key()
                      │
                      ├─ gemini_client.call_gemini()   ←── success → return
                      │
                      ├─ retryable error → report_failure() → try next key
                      │
                      └─ all keys exhausted → return user-friendly error string

This module deliberately contains NO key values, NO provider URLs,
and NO raw HTTP logic. It is the thin orchestration layer only.
"""

import logging
from .api_key_manager import get_key_manager, AllKeysExhaustedError, PermanentAPIError
from .gemini_client import call_gemini, _RetryableGeminiError

logger = logging.getLogger(__name__)

# Maximum key-rotation attempts per single user request
# (guards against infinite loops if cooldowns are all reset simultaneously)
MAX_ATTEMPTS = 3


def generate_response(prompt: str, history: list) -> str:
    """
    Generate an AI response for the given prompt + history.

    Returns a plain string — either the AI reply or a user-friendly error
    message. Never raises; never exposes key values or internal stack traces
    to the caller.

    Parameters
    ----------
    prompt  : the user's latest message
    history : list of previous messages, each dict with keys
              "role" ("user" | "model") and "parts" ([str])
    """
    manager = get_key_manager()

    if not manager.has_keys():
        logger.error(
            "ai_service: No API keys configured. "
            "Set API_KEY_1 (and optionally API_KEY_2, API_KEY_3) "
            "in Render environment variables."
        )
        return (
            "⚠️ The AI service is not configured. "
            "Please set API_KEY_1 in your Render environment variables."
        )

    last_error_msg = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # ── Pick the current best key ─────────────────────────────────────
        try:
            key_value, key_label = manager.get_active_key()
        except AllKeysExhaustedError as exc:
            logger.error(
                "ai_service: All keys exhausted on attempt %d/%d. %s",
                attempt, MAX_ATTEMPTS, exc,
            )
            return (
                "⚠️ AI service is temporarily unavailable. "
                "All configured API keys are on cooldown. "
                "Please try again in a moment."
            )

        logger.info(
            "ai_service: Attempt %d/%d using %s.", attempt, MAX_ATTEMPTS, key_label
        )

        # ── Call the Gemini client ────────────────────────────────────────
        try:
            reply = call_gemini(
                api_key=key_value,
                key_label=key_label,
                prompt=prompt,
                history=history,
            )
            manager.report_success(key_label)
            logger.info("ai_service: %s succeeded on attempt %d.", key_label, attempt)
            return reply

        except PermanentAPIError as exc:
            # Do NOT rotate — this is a problem with the request, not the key
            logger.error(
                "ai_service: Permanent error from %s (HTTP %s): %s",
                key_label,
                exc.http_status,
                str(exc),
            )
            return (
                "⚠️ The AI service returned an error with your request. "
                "Please rephrase your message and try again."
            )

        except _RetryableGeminiError as exc:
            logger.warning(
                "ai_service: Retryable error from %s (HTTP %s): %s — rotating key.",
                key_label,
                exc.http_status,
                str(exc),
            )
            manager.report_failure(key_label, http_status=exc.http_status)
            last_error_msg = str(exc)
            # Loop continues → picks next key

        except Exception as exc:
            # Unexpected error — log safely and rotate
            logger.error(
                "ai_service: Unexpected error from %s: %s — rotating key.",
                key_label,
                type(exc).__name__,   # class name only, not the full message (may contain key)
            )
            manager.report_failure(key_label)
            last_error_msg = type(exc).__name__

    # ── Exhausted all MAX_ATTEMPTS ────────────────────────────────────────
    logger.error(
        "ai_service: All %d attempt(s) failed. Last error type: %s",
        MAX_ATTEMPTS,
        last_error_msg,
    )
    return (
        "⚠️ AI service is temporarily unavailable. "
        "Please try again in a few moments."
    )
