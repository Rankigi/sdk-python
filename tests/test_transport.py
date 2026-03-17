"""Tests for non-blocking transport, local buffering, and retry."""

import json
import queue
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import MagicMock

from rankigi._transport import Transport


def _free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CollectorHandler(BaseHTTPRequestHandler):
    """Test HTTP handler that collects received payloads."""

    received: list = []
    fail_count: int = 0
    _fail_counter: int = 0
    lock = threading.Lock()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        with self.lock:
            if self._fail_counter < self.fail_count:
                self.__class__._fail_counter += 1
                self.send_response(500)
                self.end_headers()
                return
            self.received.append(json.loads(body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, *args):
        pass  # suppress logging


def _reset_handler():
    _CollectorHandler.received = []
    _CollectorHandler.fail_count = 0
    _CollectorHandler._fail_counter = 0


class TestNonBlocking:
    """Transport.send() must return immediately without blocking the caller."""

    def test_send_returns_instantly(self):
        port = _free_port()
        _reset_handler()
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            transport = Transport(
                base_url=f"http://127.0.0.1:{port}",
                api_key="test-key",
                max_retries=0,
            )
            start = time.monotonic()
            transport.send({"agent_id": "test", "action": "ping", "severity": "info", "payload": {}})
            elapsed = time.monotonic() - start
            # send() must return in < 50ms (it just enqueues)
            assert elapsed < 0.05, f"send() took {elapsed:.3f}s — not non-blocking"

            transport.flush(timeout=3.0)
            assert len(_CollectorHandler.received) == 1
        finally:
            server.shutdown()
            transport.shutdown()

    def test_multiple_events_queued(self):
        port = _free_port()
        _reset_handler()
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            transport = Transport(
                base_url=f"http://127.0.0.1:{port}",
                api_key="test-key",
                max_retries=0,
            )
            for i in range(10):
                transport.send({"agent_id": "test", "action": f"event_{i}", "severity": "info", "payload": {}})

            transport.flush(timeout=5.0)
            assert len(_CollectorHandler.received) == 10
        finally:
            server.shutdown()
            transport.shutdown()


class TestBuffering:
    """Events must buffer locally when the API is unreachable."""

    def test_buffer_overflow_calls_on_error(self):
        errors = []
        transport = Transport(
            base_url="http://127.0.0.1:1",  # unreachable
            api_key="test-key",
            on_error=errors.append,
            max_buffer=3,
            max_retries=0,
        )
        # Fill the buffer
        for i in range(3):
            transport.send({"agent_id": "test", "action": f"e{i}", "severity": "info", "payload": {}})

        # Next one should overflow
        transport.send({"agent_id": "test", "action": "overflow", "severity": "info", "payload": {}})

        # on_error should have been called for the overflow
        assert len(errors) >= 1
        assert isinstance(errors[0], BufferError)
        transport.shutdown()

    def test_events_survive_temporary_failure(self):
        """Events retry and eventually succeed after transient failures."""
        port = _free_port()
        _reset_handler()
        _CollectorHandler.fail_count = 2  # fail the first 2 attempts

        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            transport = Transport(
                base_url=f"http://127.0.0.1:{port}",
                api_key="test-key",
                max_retries=3,
            )
            transport.send({"agent_id": "test", "action": "retry_me", "severity": "info", "payload": {}})

            # Wait for retries (backoff: 1s + 2s = 3s, plus margin)
            # flush() only checks queue emptiness; the item is dequeued before
            # retries begin, so we must poll the received list directly.
            deadline = time.monotonic() + 12.0
            while time.monotonic() < deadline:
                with _CollectorHandler.lock:
                    if len(_CollectorHandler.received) >= 1:
                        break
                time.sleep(0.25)

            assert len(_CollectorHandler.received) >= 1
            assert _CollectorHandler.received[0]["action"] == "retry_me"
        finally:
            server.shutdown()
            transport.shutdown()


class TestAuth:
    """Transport must send the Bearer token in the Authorization header."""

    def test_bearer_token_sent(self):
        port = _free_port()
        _reset_handler()
        auth_headers = []

        class _AuthHandler(_CollectorHandler):
            def do_POST(self):
                auth_headers.append(self.headers.get("Authorization"))
                super().do_POST()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", port), _AuthHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            transport = Transport(
                base_url=f"http://127.0.0.1:{port}",
                api_key="rk_live_secret123",
                max_retries=0,
            )
            transport.send({"agent_id": "test", "action": "auth_check", "severity": "info", "payload": {}})
            transport.flush(timeout=3.0)

            assert len(auth_headers) == 1
            assert auth_headers[0] == "Bearer rk_live_secret123"
        finally:
            server.shutdown()
            transport.shutdown()
