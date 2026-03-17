# rankigi

[![PyPI version](https://img.shields.io/pypi/v/rankigi.svg)](https://pypi.org/project/rankigi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Tamper-evident audit trails for AI agents. Non-blocking, zero-overhead governance that never impacts your agent's critical path.

## Install

```bash
pip install rankigi
```

Optional extras:

```bash
pip install rankigi[langchain]   # LangChain callback handler
pip install rankigi[signing]     # Ed25519 passport signing
```

## Quick Start

```python
from rankigi import Rankigi

rk = Rankigi(
    api_key="rk_live_...",
    agent_id="your-agent-uuid",
)

# Track a tool call — input/output are SHA-256 hashed before transmission
rk.track_tool_call("web_search", {"query": "Q4 revenue"}, results)

# Track with intent reasoning (encrypted with AES-256-GCM)
rk.track_tool_call(
    "web_search",
    {"query": "Q4 revenue"},
    results,
    intent="User asked about quarterly financials, searching for latest data",
)

# Track agent output
rk.track_agent_output("The Q4 revenue was $12M.")

# Track errors
rk.track_error(some_exception)

# Track custom events
rk.track_custom_event("policy_check", {"rule": "pii_filter", "passed": True})

# Graceful shutdown (optional — flushes pending events)
rk.close()
```

## Context Manager

```python
with Rankigi(api_key="rk_live_...", agent_id="your-agent-uuid") as rk:
    rk.track_tool_call("web_search", {"query": "test"}, {"results": []})
# Events are automatically flushed on exit
```

## LangChain Integration

```python
from rankigi.langchain import RangigiCallbackHandler

handler = RangigiCallbackHandler(
    api_key="rk_live_...",
    agent_id="your-agent-uuid",
)

# Add to any LangChain agent — governance is now automatic
agent = initialize_agent(
    tools=tools,
    llm=llm,
    callbacks=[handler],
)
```

## Features

- **Non-blocking** — events are queued and sent on a background daemon thread
- **Privacy-first** — inputs/outputs are SHA-256 hashed before transmission
- **Intent Chain** — optional AES-256-GCM encrypted agent reasoning
- **Passport signing** — Ed25519 event signatures for tamper-evident chains
- **LangChain support** — drop-in callback handler for automatic governance
- **Zero dependencies** — stdlib only (extras for LangChain, signing)
- **Automatic retry** — exponential backoff with local buffering

## Configuration

```python
rk = Rankigi(
    api_key="rk_live_...",           # Required: API key from /dashboard/keys
    agent_id="your-agent-uuid",      # Required: Agent ID from /dashboard/agents
    base_url="https://rankigi.com",  # Optional: custom deployment URL
    on_error=lambda e: log(e),       # Optional: error callback
    max_retries=3,                   # Optional: retry attempts (default 3)
    max_buffer=1000,                 # Optional: max buffered events (default 1000)
    timeout=5.0,                     # Optional: HTTP timeout in seconds
    signing_key="base64...",         # Optional: Ed25519 key for passport signing
    passport_id="passport-uuid",     # Optional: passport ID for event signing
    intent_key="hex-string",         # Optional: AES-256 key for intent chain
)
```

## Documentation

Full documentation at [rankigi.com/docs](https://rankigi.com/docs)

## License

MIT
