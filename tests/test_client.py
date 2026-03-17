"""Tests for Rankigi client payload structure and field validation."""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import patch

from rankigi import Rankigi, _sha256, _truncate


def _free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _CollectorHandler(BaseHTTPRequestHandler):
    received: list = []
    lock = threading.Lock()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        with self.lock:
            self.received.append(json.loads(body))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, *args):
        pass


def _make_client(port):
    return Rankigi(
        api_key="test-key",
        agent_id="agent-uuid-123",
        base_url=f"http://127.0.0.1:{port}",
        max_retries=0,
    )


class TestTrackToolCall:
    def test_payload_structure(self):
        port = _free_port()
        _CollectorHandler.received = []
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            rk = _make_client(port)
            rk.track_tool_call("web_search", {"query": "test"}, {"results": [1, 2]})
            rk.flush(timeout=3.0)

            assert len(_CollectorHandler.received) == 1
            msg = _CollectorHandler.received[0]

            assert msg["agent_id"] == "agent-uuid-123"
            assert msg["action"] == "tool_call"
            assert msg["tool"] == "web_search"
            assert msg["severity"] == "info"

            payload = msg["payload"]
            assert payload["tool"] == "web_search"
            assert payload["input_hash"] == _sha256({"query": "test"})
            assert payload["output_hash"] == _sha256({"results": [1, 2]})
            assert isinstance(payload["input_size"], int)
            assert isinstance(payload["output_size"], int)
            assert payload["_sdk"].startswith("rankigi-python/")
            assert "_ts" in payload
        finally:
            server.shutdown()
            rk.close()

    def test_no_raw_data_transmitted(self):
        """Input/output must be hashed — no raw content in the payload."""
        port = _free_port()
        _CollectorHandler.received = []
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            secret = "super-secret-api-key-12345"
            rk = _make_client(port)
            rk.track_tool_call("vault", secret, "access granted")
            rk.flush(timeout=3.0)

            raw = json.dumps(_CollectorHandler.received[0])
            assert secret not in raw
            assert "access granted" not in raw
        finally:
            server.shutdown()
            rk.close()


class TestTrackAgentOutput:
    def test_payload_structure(self):
        port = _free_port()
        _CollectorHandler.received = []
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            rk = _make_client(port)
            rk.track_agent_output("The revenue was $12M.")
            rk.flush(timeout=3.0)

            msg = _CollectorHandler.received[0]
            assert msg["action"] == "response_generated"
            assert msg["severity"] == "info"
            assert "tool" not in msg

            payload = msg["payload"]
            assert payload["output_hash"] == _sha256("The revenue was $12M.")
            assert payload["output_preview"] == "The revenue was $12M."
        finally:
            server.shutdown()
            rk.close()

    def test_long_output_truncated(self):
        port = _free_port()
        _CollectorHandler.received = []
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            rk = _make_client(port)
            rk.track_agent_output("x" * 1000)
            rk.flush(timeout=3.0)

            payload = _CollectorHandler.received[0]["payload"]
            assert payload["output_preview"].endswith("...[truncated]")
            assert len(payload["output_preview"]) < 300
        finally:
            server.shutdown()
            rk.close()


class TestTrackError:
    def test_exception_payload(self):
        port = _free_port()
        _CollectorHandler.received = []
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            rk = _make_client(port)
            try:
                raise ValueError("something broke")
            except ValueError as e:
                rk.track_error(e)
            rk.flush(timeout=3.0)

            msg = _CollectorHandler.received[0]
            assert msg["action"] == "error"
            assert msg["severity"] == "warn"

            payload = msg["payload"]
            assert payload["error_type"] == "ValueError"
            assert payload["error_message"] == "something broke"
            assert len(payload["error_hash"]) == 64  # SHA-256 hex
        finally:
            server.shutdown()
            rk.close()

    def test_string_error(self):
        port = _free_port()
        _CollectorHandler.received = []
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            rk = _make_client(port)
            rk.track_error("something went wrong")
            rk.flush(timeout=3.0)

            payload = _CollectorHandler.received[0]["payload"]
            assert payload["error_type"] == "UnknownError"
            assert payload["error_message"] == "something went wrong"
        finally:
            server.shutdown()
            rk.close()


class TestTrackCustomEvent:
    def test_payload_structure(self):
        port = _free_port()
        _CollectorHandler.received = []
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            rk = _make_client(port)
            rk.track_custom_event("policy_check", {"rule": "pii_filter", "passed": True})
            rk.flush(timeout=3.0)

            msg = _CollectorHandler.received[0]
            assert msg["action"] == "policy_check"
            assert msg["severity"] == "info"

            payload = msg["payload"]
            assert payload["rule"] == "pii_filter"
            assert payload["passed"] is True
        finally:
            server.shutdown()
            rk.close()


class TestSeverity:
    def test_custom_severity(self):
        port = _free_port()
        _CollectorHandler.received = []
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            rk = _make_client(port)
            rk.track_tool_call("risky_tool", "in", "out", severity="critical")
            rk.flush(timeout=3.0)

            assert _CollectorHandler.received[0]["severity"] == "critical"
        finally:
            server.shutdown()
            rk.close()

    def test_invalid_severity_defaults_to_info(self):
        port = _free_port()
        _CollectorHandler.received = []
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            rk = _make_client(port)
            rk.track_custom_event("test", {}, severity="bogus")
            rk.flush(timeout=3.0)

            assert _CollectorHandler.received[0]["severity"] == "info"
        finally:
            server.shutdown()
            rk.close()


class TestSdkMetadata:
    def test_sdk_tag_and_timestamp_present(self):
        port = _free_port()
        _CollectorHandler.received = []
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            rk = _make_client(port)
            rk.track_custom_event("ping", {})
            rk.flush(timeout=3.0)

            payload = _CollectorHandler.received[0]["payload"]
            assert payload["_sdk"] == "rankigi-python/1.0.0"
            assert "_ts" in payload
            # Timestamp should be ISO 8601
            assert "T" in payload["_ts"]
        finally:
            server.shutdown()
            rk.close()


class TestContextManager:
    def test_context_manager_flushes(self):
        port = _free_port()
        _CollectorHandler.received = []
        server = HTTPServer(("127.0.0.1", port), _CollectorHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        try:
            with Rankigi(
                api_key="test-key",
                agent_id="agent-uuid",
                base_url=f"http://127.0.0.1:{port}",
                max_retries=0,
            ) as rk:
                rk.track_custom_event("inside_context", {"ok": True})

            # After exiting context, event should have been flushed
            time.sleep(0.5)
            assert len(_CollectorHandler.received) >= 1
        finally:
            server.shutdown()


class TestValidation:
    def test_empty_api_key_raises(self):
        import pytest
        with pytest.raises(ValueError, match="api_key"):
            Rankigi(api_key="", agent_id="test")

    def test_empty_agent_id_raises(self):
        import pytest
        with pytest.raises(ValueError, match="agent_id"):
            Rankigi(api_key="test", agent_id="")
