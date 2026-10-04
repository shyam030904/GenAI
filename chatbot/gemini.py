import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Max history messages to keep token usage low
MAX_HISTORY = 10


def get_api_key() -> str:
    """Safely retrieves OpenAI API key from environment."""
    return os.environ.get("OPENAI_API_KEY", "").strip()


def generate_response(prompt: str, history: list) -> str:
    """
    Sends a prompt to OpenAI (gpt-4o-mini) with conversation history.
    - Lazily initializes OpenAI client so application startup never fails (avoids 502 Bad Gateway).
    """
    api_key = get_api_key()
    if not api_key:
        logger.error("❌ No API key found in environment variables.")
        return "⚠️ Error: No OPENAI_API_KEY set in environment variables. Please configure it in Render Dashboard settings."

    try:
        client = OpenAI(api_key=api_key)

        # ── Convert Django DB history format → OpenAI messages format ──
        trimmed = history[-MAX_HISTORY:] if len(history) > MAX_HISTORY else history

        messages = []
        for msg in trimmed:
            role = msg.get("role", "user")
            # OpenAI uses 'assistant' not 'model'
            if role == "model":
                role = "assistant"
            content = msg.get("parts", [""])[0] if isinstance(msg.get("parts"), list) else msg.get("content", "")
            messages.append({"role": role, "content": content})

        # Add the new user message
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        logger.info("✅ OpenAI response received successfully.")
        return reply

    except Exception as e:
        logger.error(f"❌ OpenAI error: {e}")
        return f"⚠️ Error: {str(e)}"
