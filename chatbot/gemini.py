"""
chatbot/gemini.py
─────────────────
BACKWARD-COMPATIBILITY SHIM

This file previously contained the OpenAI (and before that, Groq) API client.
It has been replaced by the multi-key Gemini failover architecture:

    chatbot/api_key_manager.py   ─── key rotation & cooldown
    chatbot/gemini_client.py     ─── Gemini SDK wrapper
    chatbot/ai_service.py        ─── public entry-point (orchestrator)

This shim re-exports generate_response() from ai_service so that any
code still importing from chatbot.gemini continues to work without changes.
"""

from .ai_service import generate_response  # noqa: F401  (re-export)

__all__ = ["generate_response"]
