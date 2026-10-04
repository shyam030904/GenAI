"""
chatbot/api_key_manager.py
──────────────────────────
Thread-safe, in-memory API key rotation manager.

Loads up to 3 keys from environment variables:
    API_KEY_1, API_KEY_2, API_KEY_3

Key rotation rules:
  - On success          → keep using the current key
  - On retryable error  → mark key as cooling down, advance to next
  - On permanent error  → do NOT rotate; surface immediately
  - All keys exhausted  → raise AllKeysExhaustedError

Retryable HTTP codes: 429, 500, 502, 503, 504
Permanent HTTP codes : 400, 401 (bad key treated as permanent once
                       all slots have been tried), 403, 404, 422

Cooldown per key: COOLDOWN_SECONDS (default 60 s).
After cooldown the key is eligible to be retried again.
"""

import os
import time
import threading
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────
COOLDOWN_SECONDS = 60          # seconds to back off a failed key
ENV_VAR_NAMES    = ["API_KEY_1", "API_KEY_2", "API_KEY_3"]

# HTTP status codes we should retry with the next key
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


# ── Custom exceptions ─────────────────────────────────────────────────────────
class AllKeysExhaustedError(Exception):
    """Raised when every configured API key has failed for this request."""


class PermanentAPIError(Exception):
    """Raised for non-retryable errors (malformed request, unsupported model, etc.)."""
    def __init__(self, message: str, http_status: int | None = None):
        super().__init__(message)
        self.http_status = http_status


# ── Key slot dataclass ────────────────────────────────────────────────────────
@dataclass
class _KeySlot:
    label: str          # "API_KEY_1" etc. — safe to log
    value: str          # never logged in full
    failed_at: float = field(default=0.0)   # epoch seconds when last failure occurred

    @property
    def is_cooling_down(self) -> bool:
        if self.failed_at == 0.0:
            return False
        return (time.monotonic() - self.failed_at) < COOLDOWN_SECONDS

    def mark_failed(self) -> None:
        self.failed_at = time.monotonic()

    def reset(self) -> None:
        self.failed_at = 0.0


# ── Manager (module-level singleton) ─────────────────────────────────────────
class APIKeyManager:
    """
    Manages a list of API keys and provides intelligent round-robin failover.

    Usage
    -----
    manager = APIKeyManager()
    key, label = manager.get_active_key()
    # ... make API call ...
    manager.report_success()
    # or
    manager.report_failure(label, http_status=429)
    """

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._slots: list[_KeySlot] = []
        self._current_idx: int = 0
        self._load_keys()

    # ── Loading ──────────────────────────────────────────────────────────────
    def _load_keys(self) -> None:
        loaded = 0
        for var in ENV_VAR_NAMES:
            val = os.environ.get(var, "").strip()
            if val:
                self._slots.append(_KeySlot(label=var, value=val))
                loaded += 1
                logger.info("APIKeyManager: %s loaded (value hidden).", var)
            else:
                logger.debug("APIKeyManager: %s not set or empty — skipped.", var)

        if not self._slots:
            logger.error(
                "APIKeyManager: No API keys found. "
                "Set at least API_KEY_1 in environment variables."
            )
        else:
            logger.info(
                "APIKeyManager: %d key(s) loaded: %s",
                loaded,
                ", ".join(s.label for s in self._slots),
            )

    # ── Public interface ──────────────────────────────────────────────────────
    def has_keys(self) -> bool:
        return bool(self._slots)

    def key_count(self) -> int:
        return len(self._slots)

    def get_active_key(self) -> tuple[str, str]:
        """
        Returns (key_value, label) for the best available key.
        Advances past cooling-down slots automatically.

        Raises
        ------
        AllKeysExhaustedError   if every configured key is cooling down.
        """
        with self._lock:
            if not self._slots:
                raise AllKeysExhaustedError(
                    "No API keys are configured. "
                    "Please set API_KEY_1 (and optionally API_KEY_2, API_KEY_3) "
                    "in your Render environment variables."
                )

            n = len(self._slots)
            for _ in range(n):
                slot = self._slots[self._current_idx % n]
                if not slot.is_cooling_down:
                    return slot.value, slot.label
                # Skip this cooling-down key
                logger.debug(
                    "APIKeyManager: %s is cooling down — skipping.", slot.label
                )
                self._current_idx = (self._current_idx + 1) % n

            raise AllKeysExhaustedError(
                "All configured API keys are temporarily unavailable. "
                "They will recover after the cooldown period."
            )

    def report_success(self, label: str) -> None:
        """Call after a successful API response. Resets the key's cooldown."""
        with self._lock:
            for slot in self._slots:
                if slot.label == label:
                    slot.reset()
                    return

    def report_failure(self, label: str, http_status: int | None = None) -> None:
        """
        Call after a retryable failure. Marks the key as cooling down and
        advances the active-key pointer to the next slot.
        """
        with self._lock:
            n = len(self._slots)
            for i, slot in enumerate(self._slots):
                if slot.label == label:
                    slot.mark_failed()
                    logger.warning(
                        "APIKeyManager: %s failed (HTTP %s). "
                        "Marking as cooling down for %ds.",
                        label,
                        http_status or "unknown",
                        COOLDOWN_SECONDS,
                    )
                    # Advance pointer to next slot
                    self._current_idx = (i + 1) % n
                    next_slot = self._slots[self._current_idx]
                    logger.info(
                        "APIKeyManager: Next candidate is %s.", next_slot.label
                    )
                    return

    def status_summary(self) -> list[dict]:
        """Returns safe (no values) status info for diagnostics."""
        with self._lock:
            return [
                {
                    "label":        slot.label,
                    "cooling_down": slot.is_cooling_down,
                    "cooldown_remaining_s": max(
                        0,
                        round(COOLDOWN_SECONDS - (time.monotonic() - slot.failed_at))
                    ) if slot.is_cooling_down else 0,
                }
                for slot in self._slots
            ]


# ── Module-level singleton (shared across all requests in one process) ────────
_manager: APIKeyManager | None = None
_manager_lock = threading.Lock()


def get_key_manager() -> APIKeyManager:
    """Return the module-level singleton, creating it on first call."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = APIKeyManager()
    return _manager
