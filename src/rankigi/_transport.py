"""Background transport layer with retry and local buffering.

All network I/O runs on a daemon thread so the calling agent is never blocked.
If the RANKIGI API is unreachable, events are buffered in a bounded queue and
retried with exponential backoff.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

_SDK_VERSION = "rankigi-python/1.0.0"


class Transport:
    """Non-blocking, buffered HTTP transport for the RANKIGI ingest API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        on_error: Optional[Callable[[Exception], None]] = None,
        max_retries: int = 3,
        max_buffer: int = 1000,
        timeout: float = 5.0,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/api/ingest"
        self._api_key = api_key
        self._on_error = on_error
        self._max_retries = max_retries
        self._timeout = timeout
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_buffer)
        self._shutdown = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="rankigi-transport")
        self._worker.start()

    # -- Public API -----------------------------------------------------------

    def send(self, payload: dict[str, Any]) -> None:
        """Enqueue a payload for async delivery. Never blocks, never raises."""
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self._safe_error(BufferError("RANKIGI send buffer full, event dropped"))

    def flush(self, timeout: float = 5.0) -> None:
        """Block until all queued events are sent or *timeout* seconds elapse."""
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.05)

    def shutdown(self) -> None:
        """Signal the worker to stop after draining the queue."""
        self._shutdown.set()

    # -- Worker ---------------------------------------------------------------

    def _run(self) -> None:
        while not self._shutdown.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._send_with_retry(payload)
            self._queue.task_done()

    def _send_with_retry(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        for attempt in range(self._max_retries + 1):
            try:
                req = urllib.request.Request(
                    self._url,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    resp.read()
                return  # success
            except Exception as exc:
                if attempt < self._max_retries:
                    backoff = min(2 ** attempt, 30)
                    time.sleep(backoff)
                else:
                    self._safe_error(exc)

    # -- Helpers --------------------------------------------------------------

    def _safe_error(self, exc: Exception) -> None:
        if self._on_error:
            try:
                self._on_error(exc if isinstance(exc, Exception) else Exception(str(exc)))
            except Exception:
                pass  # never propagate
