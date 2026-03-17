"""RANKIGI Python SDK — tamper-evident audit trails for AI agents.

Non-blocking. Errors are never raised; they are routed to an optional
``on_error`` callback and silently swallowed. RANKIGI observability must
never impact your agent's critical path.

Usage::

    from rankigi import Rankigi

    rk = Rankigi(
        api_key="rk_live_...",
        agent_id="your-agent-uuid",
    )

    rk.track_tool_call("web_search", {"query": "Q4 revenue"}, results)
    rk.track_agent_output("The Q4 revenue was $12M.")
    rk.track_error(some_exception)
    rk.track_custom_event("policy_check", {"rule": "pii_filter", "passed": True})

    # Graceful shutdown (optional — flushes pending events)
    rk.close()
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Union

from rankigi._transport import Transport

__version__ = "1.0.0"
__all__ = ["Rankigi", "__version__"]

Severity = str  # "info" | "warn" | "critical"
_VALID_SEVERITIES = frozenset(("info", "warn", "critical"))
_SDK_TAG = f"rankigi-python/{__version__}"


# -- Hashing helpers ----------------------------------------------------------

def _canonical_json(value: Any) -> str:
    """Deterministic JSON serialization with sorted keys and compact separators."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    """SHA-256 hex digest of a value's canonical JSON representation."""
    text = value if isinstance(value, str) else _canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _byte_len(value: Any) -> int:
    """Byte length of a value when serialized."""
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(_canonical_json(value).encode("utf-8"))


def _truncate(value: Any, max_len: int = 256) -> str:
    """Truncate a value's string representation."""
    text = value if isinstance(value, str) else _canonical_json(value)
    if len(text) > max_len:
        return text[:max_len] + "...[truncated]"
    return text


# -- Main client --------------------------------------------------------------

class Rankigi:
    """RANKIGI SDK client.

    All tracking methods are non-blocking — events are queued and sent on a
    background daemon thread. If the API is unreachable, events are buffered
    locally and retried with exponential backoff.

    Parameters
    ----------
    api_key : str
        Ingest API key (from /dashboard/keys).
    agent_id : str
        Agent UUID (from /dashboard/agents).
    base_url : str
        Base URL of your RANKIGI deployment.
    on_error : callable, optional
        Called when a tracking call fails silently.
    max_retries : int
        Maximum retry attempts per event (with exponential backoff).
    max_buffer : int
        Maximum number of events buffered locally when the API is down.
    timeout : float
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        base_url: str = "https://rankigi.com",
        on_error: Optional[Callable[[Exception], None]] = None,
        max_retries: int = 3,
        max_buffer: int = 1000,
        timeout: float = 5.0,
        signing_key: Optional[str] = None,
        passport_id: Optional[str] = None,
        intent_key: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not agent_id:
            raise ValueError("agent_id is required")

        self._agent_id = agent_id
        self._on_error = on_error
        self._signing_key = signing_key
        self._passport_id = passport_id
        self._intent_key = intent_key
        self._transport = Transport(
            base_url=base_url,
            api_key=api_key,
            on_error=on_error,
            max_retries=max_retries,
            max_buffer=max_buffer,
            timeout=timeout,
        )

    # -- Public tracking methods ----------------------------------------------

    def track_tool_call(
        self,
        tool: str,
        input: Any,
        output: Any,
        severity: Severity = "info",
        intent: Optional[str] = None,
    ) -> None:
        """Track a tool call made by the agent.

        Both *input* and *output* are SHA-256 hashed before transmission —
        no raw data leaves the process.
        """
        self._track(
            action="tool_call",
            tool=tool,
            severity=severity,
            payload={
                "tool": tool,
                "input_hash": _sha256(input),
                "output_hash": _sha256(output),
                "input_size": _byte_len(input),
                "output_size": _byte_len(output),
            },
            intent=intent,
        )

    def track_agent_output(
        self,
        output: Any,
        severity: Severity = "info",
        intent: Optional[str] = None,
    ) -> None:
        """Track a final agent output/response.

        The output is hashed; only a truncated preview is transmitted.
        """
        self._track(
            action="response_generated",
            severity=severity,
            payload={
                "output_hash": _sha256(output),
                "output_preview": _truncate(output, 256),
            },
            intent=intent,
        )

    def track_error(
        self,
        error: Union[Exception, Any],
        severity: Severity = "warn",
        intent: Optional[str] = None,
    ) -> None:
        """Track an error that occurred during agent execution."""
        if isinstance(error, BaseException):
            error_type = type(error).__name__
            error_message = str(error)
            error_stack = "".join(
                __import__("traceback").format_exception(type(error), error, error.__traceback__)
            ) if error.__traceback__ else ""
        else:
            error_type = "UnknownError"
            error_message = str(error)
            error_stack = ""

        self._track(
            action="error",
            severity=severity,
            payload={
                "error_type": error_type,
                "error_message": error_message,
                "error_hash": _sha256(error_message + error_stack),
            },
            intent=intent,
        )

    def track_custom_event(
        self,
        action_type: str,
        metadata: Dict[str, Any],
        severity: Severity = "info",
        intent: Optional[str] = None,
    ) -> None:
        """Track a custom event with arbitrary metadata."""
        self._track(
            action=action_type,
            severity=severity,
            payload=metadata,
            intent=intent,
        )

    # -- Lifecycle ------------------------------------------------------------

    def flush(self, timeout: float = 5.0) -> None:
        """Block until all queued events are sent or *timeout* elapses."""
        self._transport.flush(timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Flush pending events and shut down the background worker."""
        self._transport.flush(timeout)
        self._transport.shutdown()

    def __enter__(self) -> "Rankigi":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- Internal -------------------------------------------------------------

    def _track(
        self,
        action: str,
        severity: Severity = "info",
        tool: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
    ) -> None:
        if severity not in _VALID_SEVERITIES:
            severity = "info"

        body: Dict[str, Any] = {
            "agent_id": self._agent_id,
            "action": action,
            "severity": severity,
            "payload": {
                **(payload or {}),
                "_sdk": _SDK_TAG,
                "_ts": datetime.now(timezone.utc).isoformat(),
            },
        }
        if tool is not None:
            body["tool"] = tool

        # Encrypt intent reasoning if intent_key is configured and intent provided
        if self._intent_key and intent:
            try:
                from rankigi._intent import encrypt_intent
                packed, intent_hash = encrypt_intent(intent, self._intent_key)
                body["intent"] = {"reasoning": packed, "intent_hash": intent_hash}
            except ImportError:
                if self._on_error:
                    self._on_error(ImportError("Install rankigi[intent] for Intent Chain support (requires cryptography)"))
            except Exception as exc:
                if self._on_error:
                    self._on_error(exc)

        # Sign event with passport key if configured
        if self._signing_key and self._passport_id:
            try:
                from rankigi._signing import sign_payload
                signing_payload = {
                    "agent_id": self._agent_id,
                    "action": action,
                    "tool": tool,
                    "payload": body["payload"],
                    "occurred_at": body["payload"]["_ts"],
                }
                body["passport_id"] = self._passport_id
                body["signature"] = sign_payload(self._signing_key, signing_payload)
            except ImportError:
                if self._on_error:
                    self._on_error(ImportError("Install rankigi[signing] for Ed25519 signing"))
            except Exception as exc:
                if self._on_error:
                    self._on_error(exc)

        self._transport.send(body)
