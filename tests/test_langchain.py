"""Tests for the RANKIGI LangChain callback handler.

These tests mock the Rankigi SDK client to verify that:
1. The handler instantiates correctly
2. Tool events fire the right tracking calls
3. Agent events fire the right tracking calls
4. LLM events fire the right tracking calls
5. Error events never throw (sidecar principle)
6. The handler never blocks agent execution
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from unittest.mock import MagicMock, patch

import pytest

from rankigi import Rankigi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


@pytest.fixture
def server():
    port = _free_port()
    _CollectorHandler.received = []
    srv = HTTPServer(("127.0.0.1", port), _CollectorHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port
    srv.shutdown()


def _make_handler(port):
    from rankigi.langchain import RangigiCallbackHandler
    return RangigiCallbackHandler(
        api_key="test-key",
        agent_id="agent-uuid-123",
        base_url=f"http://127.0.0.1:{port}",
    )


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_handler_creates_without_error(self, server):
        handler = _make_handler(server)
        assert handler is not None
        assert handler.client is not None
        handler.close(timeout=1.0)

    def test_handler_creates_with_verbose(self, server):
        from rankigi.langchain import RangigiCallbackHandler
        handler = RangigiCallbackHandler(
            api_key="test-key",
            agent_id="agent-uuid-123",
            base_url=f"http://127.0.0.1:{server}",
            verbose=True,
        )
        assert handler.verbose is True
        handler.close(timeout=1.0)


# ---------------------------------------------------------------------------
# Tool events
# ---------------------------------------------------------------------------

class TestToolEvents:
    def test_on_tool_start_and_end_tracks_tool_call(self, server):
        handler = _make_handler(server)
        from uuid import uuid4
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "web_search"},
            input_str="python tutorials",
            run_id=run_id,
        )

        handler.on_tool_end(
            output="Found 10 results",
            run_id=run_id,
        )

        handler.client.flush(timeout=3.0)
        handler.close(timeout=1.0)

        # Should have: tool_call + tool_latency
        assert len(_CollectorHandler.received) >= 1

        tool_call = _CollectorHandler.received[0]
        assert tool_call["action"] == "tool_call"
        assert tool_call["tool"] == "web_search"
        assert "input_hash" in tool_call["payload"]
        assert "output_hash" in tool_call["payload"]

    def test_on_tool_end_computes_latency(self, server):
        handler = _make_handler(server)
        from uuid import uuid4
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "slow_tool"},
            input_str="input",
            run_id=run_id,
        )

        time.sleep(0.05)  # Small delay to get measurable latency

        handler.on_tool_end(
            output="output",
            run_id=run_id,
        )

        handler.client.flush(timeout=3.0)
        handler.close(timeout=1.0)

        # Find the latency event
        latency_events = [
            m for m in _CollectorHandler.received
            if m.get("action") == "tool_latency"
        ]
        assert len(latency_events) == 1
        assert latency_events[0]["payload"]["tool"] == "slow_tool"
        assert latency_events[0]["payload"]["latency_ms"] > 0

    def test_on_tool_error_tracks_error(self, server):
        handler = _make_handler(server)
        from uuid import uuid4
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "failing_tool"},
            input_str="bad input",
            run_id=run_id,
        )

        handler.on_tool_error(
            error=RuntimeError("tool crashed"),
            run_id=run_id,
        )

        handler.client.flush(timeout=3.0)
        handler.close(timeout=1.0)

        error_events = [
            m for m in _CollectorHandler.received
            if m.get("action") == "error"
        ]
        assert len(error_events) == 1
        assert error_events[0]["payload"]["error_type"] == "RuntimeError"
        assert error_events[0]["payload"]["error_message"] == "tool crashed"
        assert error_events[0]["severity"] == "warn"

    def test_on_tool_error_never_throws(self, server):
        handler = _make_handler(server)

        # Should not raise even with no prior on_tool_start
        handler.on_tool_error(
            error=RuntimeError("unexpected"),
        )

        handler.close(timeout=1.0)


# ---------------------------------------------------------------------------
# Agent events
# ---------------------------------------------------------------------------

class TestAgentEvents:
    def test_on_agent_action_tracks_decision(self, server):
        handler = _make_handler(server)
        from rankigi.langchain import AgentAction

        action = AgentAction(
            tool="calculator",
            tool_input="2 + 2",
            log="Thought: I need to calculate.\nAction: calculator",
        )

        handler.on_agent_action(action)
        handler.client.flush(timeout=3.0)
        handler.close(timeout=1.0)

        agent_events = [
            m for m in _CollectorHandler.received
            if m.get("action") == "agent_action"
        ]
        assert len(agent_events) == 1
        assert agent_events[0]["payload"]["tool"] == "calculator"
        assert "log" in agent_events[0]["payload"]

    def test_on_agent_finish_tracks_output(self, server):
        handler = _make_handler(server)
        from rankigi.langchain import AgentFinish

        finish = AgentFinish(
            return_values={"output": "The answer is 4"},
            log="Final Answer: The answer is 4",
        )

        handler.on_agent_finish(finish)
        handler.client.flush(timeout=3.0)
        handler.close(timeout=1.0)

        output_events = [
            m for m in _CollectorHandler.received
            if m.get("action") == "response_generated"
        ]
        assert len(output_events) == 1


# ---------------------------------------------------------------------------
# LLM events
# ---------------------------------------------------------------------------

class TestLLMEvents:
    def test_on_llm_error_tracks_critical(self, server):
        handler = _make_handler(server)

        handler.on_llm_error(error=ConnectionError("API unreachable"))
        handler.client.flush(timeout=3.0)
        handler.close(timeout=1.0)

        error_events = [
            m for m in _CollectorHandler.received
            if m.get("action") == "error"
        ]
        assert len(error_events) == 1
        assert error_events[0]["severity"] == "critical"
        assert error_events[0]["payload"]["error_type"] == "ConnectionError"

    def test_on_llm_error_never_throws(self, server):
        handler = _make_handler(server)

        # Even with bizarre error types, should never raise
        handler.on_llm_error(error=Exception("LLM exploded"))
        handler.close(timeout=1.0)


# ---------------------------------------------------------------------------
# Chain events
# ---------------------------------------------------------------------------

class TestChainEvents:
    def test_on_chain_error_tracks_critical(self, server):
        handler = _make_handler(server)

        handler.on_chain_error(error=ValueError("Chain broke"))
        handler.client.flush(timeout=3.0)
        handler.close(timeout=1.0)

        error_events = [
            m for m in _CollectorHandler.received
            if m.get("action") == "error"
        ]
        assert len(error_events) == 1
        assert error_events[0]["severity"] == "critical"

    def test_on_chain_error_never_throws(self, server):
        handler = _make_handler(server)
        handler.on_chain_error(error=RuntimeError("bad chain"))
        handler.close(timeout=1.0)


# ---------------------------------------------------------------------------
# Non-blocking behavior
# ---------------------------------------------------------------------------

class TestNonBlocking:
    def test_handler_never_blocks_on_api_failure(self):
        """Handler should return immediately even if RANKIGI API is unreachable."""
        from rankigi.langchain import RangigiCallbackHandler
        from uuid import uuid4

        handler = RangigiCallbackHandler(
            api_key="bad-key",
            agent_id="agent-uuid",
            base_url="http://127.0.0.1:1",  # nothing listening
        )

        start = time.monotonic()

        handler.on_tool_start(
            serialized={"name": "test"},
            input_str="input",
            run_id=uuid4(),
        )
        handler.on_tool_end(output="output")
        handler.on_tool_error(error=RuntimeError("err"))
        handler.on_chain_error(error=RuntimeError("chain err"))
        handler.on_llm_error(error=RuntimeError("llm err"))

        elapsed = time.monotonic() - start

        # All calls should complete in under 1 second (non-blocking)
        assert elapsed < 1.0

        handler.close(timeout=0.1)

    def test_all_methods_return_none(self, server):
        """Every callback method must return None (LangChain contract)."""
        handler = _make_handler(server)
        from uuid import uuid4
        from rankigi.langchain import AgentAction, AgentFinish

        run_id = uuid4()

        assert handler.on_tool_start({"name": "t"}, "i", run_id=run_id) is None
        assert handler.on_tool_end("output", run_id=run_id) is None
        assert handler.on_tool_error(RuntimeError("e"), run_id=run_id) is None
        assert handler.on_chain_error(RuntimeError("e")) is None
        assert handler.on_llm_error(RuntimeError("e")) is None

        action = AgentAction(tool="t", tool_input="i", log="l")
        assert handler.on_agent_action(action) is None

        finish = AgentFinish(return_values={"output": "o"}, log="l")
        assert handler.on_agent_finish(finish) is None

        handler.close(timeout=1.0)
