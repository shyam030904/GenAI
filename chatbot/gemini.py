import os
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Max history messages to keep token usage low
MAX_HISTORY = 10


def get_api_key() -> str:
    """Safely retrieves API key from environment."""
    return (
        os.environ.get("GROQ_API_KEY", "").strip()
        or os.environ.get("GROK_API_KEY", "").strip()
        or os.environ.get("XAI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )


def generate_response(prompt: str, history: list) -> str:
    """
    Sends a prompt to Groq (Llama 3.3 70B) with conversation history.
    - Lazily initializes Groq client so application startup never fails (avoids 502 Bad Gateway).
    """
    api_key = get_api_key()
    if not api_key:
        logger.error("❌ No API key found in environment variables.")
        return "⚠️ Error: No GROQ_API_KEY or GROK_API_KEY set in environment variables. Please configure it in Render Dashboard settings."

    try:
        client = Groq(api_key=api_key)

        # ── Convert Django DB history format → Groq messages format ──
        trimmed = history[-MAX_HISTORY:] if len(history) > MAX_HISTORY else history

        messages = []
        for msg in trimmed:
            role = msg.get("role", "user")
            # Groq uses 'assistant' not 'model'
            if role == "model":
                role = "assistant"
            content = msg.get("parts", [""])[0] if isinstance(msg.get("parts"), list) else msg.get("content", "")
            messages.append({"role": role, "content": content})

        # Add the new user message
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        logger.info("✅ Groq response received successfully.")
        return reply

    except Exception as e:
        logger.error(f"❌ Groq error: {e}")
        return f"⚠️ Error: {str(e)}"
