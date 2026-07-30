import os
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Load Groq / Grok API key ──
GROQ_API_KEY = (
    os.environ.get("GROQ_API_KEY", "").strip()
    or os.environ.get("GROK_API_KEY", "").strip()
    or os.environ.get("XAI_API_KEY", "").strip()
)

if not GROQ_API_KEY:
    raise ValueError("No GROQ_API_KEY or GROK_API_KEY found in .env file.")

logger.info("✅ Groq API key loaded successfully.")

# Max history messages to keep token usage low
MAX_HISTORY = 10

# Groq client
client = Groq(api_key=GROQ_API_KEY)


def generate_response(prompt: str, history: list) -> str:
    """
    Sends a prompt to Groq (Llama 3.3 70B) with conversation history.
    - Trims history to last MAX_HISTORY messages to reduce token usage.
    - Groq free tier: 14,400 requests/day — very generous.
    """

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

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Free, fast, highly capable
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
