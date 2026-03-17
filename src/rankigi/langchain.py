"""LangChain callback integration for RANKIGI.

Passive, non-blocking governance for LangChain agents. Add the callback
handler and every tool call, agent action, LLM response, and error is
automatically captured in your tamper-evident audit trail.

Requires the ``langchain-core`` dependency::

    pip install rankigi langchain-core

Usage::

    from rankigi.langchain import RangigiCallbackHandler

    handler = RangigiCallbackHandler(
        api_key="rk_live_...",
        agent_id="your-agent-uuid",
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        callbacks=[handler],
    )

    # RANKIGI now governs this agent automatically.
    # Every tool call is hashed and chained.
    # No other changes needed.
    result = agent.run("your task here")
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_core.agents import AgentAction, AgentFinish

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

    # Minimal stubs so the handler can be imported and tested without
    # langchain-core installed.  When consumers ``pip install langchain-core``
    # the real types are used instead via the try-block above.
    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Stub base class — duck-typing compatible with LangChain runtime."""
        name: str = "BaseCallbackHandler"

    class AgentAction:  # type: ignore[no-redef]
        def __init__(self, tool: str = "", tool_input: Any = "", log: str = "") -> None:
            self.tool = tool
            self.tool_input = tool_input
            self.log = log

    class AgentFinish:  # type: ignore[no-redef]
        def __init__(self, return_values: Any = None, log: str = "") -> None:
            self.return_values = return_values
            self.log = log

    class _Generation:
        def __init__(self, text: str = "") -> None:
            self.text = text

    class LLMResult:  # type: ignore[no-redef]
        def __init__(self, generations: Any = None) -> None:
            self.generations = generations or []

from rankigi import Rankigi

__all__ = ["RangigiCallbackHandler", "AgentAction", "AgentFinish", "LLMResult"]


class RangigiCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that streams governance events to RANKIGI.

    Hooks into tool execution, agent decisions, LLM responses, and error
    events. All tracking is non-blocking and never interferes with chain
    execution — the sidecar principle applied to LangChain.

    Parameters
    ----------
    api_key : str
        RANKIGI ingest API key (from /dashboard/keys).
    agent_id : str
        Agent UUID (from /dashboard/agents).
    base_url : str
        Base URL of your RANKIGI deployment.
    verbose : bool
        If True, print tracking events to stderr for debugging.
    """

    def __init__(
        self,
        api_key: str,
        agent_id: str,
        base_url: str = "https://rankigi.com",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.client = Rankigi(
            api_key=api_key,
            agent_id=agent_id,
            base_url=base_url,
        )
        self.verbose = verbose
        self._tool_start_times: Dict[str, float] = {}
        self._tool_inputs: Dict[str, Dict[str, Any]] = {}

    # -- Tool events ----------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record tool invocation start time and input for latency tracking."""
        try:
            run_key = str(run_id) if run_id else "latest"
            tool_name = serialized.get("name", "unknown")
            self._tool_start_times[run_key] = time.monotonic()
            self._tool_inputs[run_key] = {
                "name": tool_name,
                "input": inputs if inputs is not None else input_str,
            }
            if self.verbose:
                print(f"[rankigi] tool_start: {tool_name}", file=sys.stderr)
        except Exception:
            pass  # never interfere with chain execution

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Record tool completion with output and latency."""
        try:
            run_key = str(run_id) if run_id else "latest"
            ctx = self._tool_inputs.pop(run_key, {"name": "unknown", "input": ""})
            start = self._tool_start_times.pop(run_key, None)
            latency_ms = round((time.monotonic() - start) * 1000, 1) if start else None

            self.client.track_tool_call(
                tool=ctx["name"],
                input=ctx["input"],
                output=str(output),
                severity="info",
            )

            if latency_ms is not None:
                self.client.track_custom_event(
                    "tool_latency",
                    {
                        "tool": ctx["name"],
                        "latency_ms": latency_ms,
                    },
                )

            if self.verbose:
                lat = f" ({latency_ms}ms)" if latency_ms is not None else ""
                print(f"[rankigi] tool_end: {ctx['name']}{lat}", file=sys.stderr)
        except Exception:
            pass

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Record tool error. Never throws."""
        try:
            run_key = str(run_id) if run_id else "latest"
            ctx = self._tool_inputs.pop(run_key, {"name": "unknown"})
            self._tool_start_times.pop(run_key, None)

            self.client.track_error(error, severity="warn")

            if self.verbose:
                print(f"[rankigi] tool_error: {ctx['name']}: {error}", file=sys.stderr)
        except Exception:
            pass

    # -- Agent events ---------------------------------------------------------

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Record agent decision (tool selection)."""
        try:
            self.client.track_custom_event(
                "agent_action",
                {
                    "tool": action.tool,
                    "tool_input_preview": str(action.tool_input)[:256],
                    "log": (action.log or "")[:512],
                },
            )

            if self.verbose:
                print(f"[rankigi] agent_action: {action.tool}", file=sys.stderr)
        except Exception:
            pass

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Record agent completion with return values."""
        try:
            output = finish.return_values
            output_str = str(output) if output else ""
            self.client.track_agent_output(output_str, severity="info")

            if self.verbose:
                print(f"[rankigi] agent_finish", file=sys.stderr)
        except Exception:
            pass

    # -- LLM events -----------------------------------------------------------

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Record LLM output (first generation only)."""
        try:
            text = ""
            if response.generations and response.generations[0]:
                text = response.generations[0][0].text or ""
            if text:
                self.client.track_agent_output(text)

            if self.verbose and text:
                preview = text[:80] + "..." if len(text) > 80 else text
                print(f"[rankigi] llm_end: {preview}", file=sys.stderr)
        except Exception:
            pass

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Record LLM error. Never throws."""
        try:
            self.client.track_error(error, severity="critical")

            if self.verbose:
                print(f"[rankigi] llm_error: {error}", file=sys.stderr)
        except Exception:
            pass

    # -- Chain events ---------------------------------------------------------

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Record chain error. Never throws."""
        try:
            self.client.track_error(error, severity="critical")

            if self.verbose:
                print(f"[rankigi] chain_error: {error}", file=sys.stderr)
        except Exception:
            pass

    # -- Lifecycle ------------------------------------------------------------

    def close(self, timeout: float = 5.0) -> None:
        """Flush pending events and shut down the transport."""
        self.client.close(timeout)
