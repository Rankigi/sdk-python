"""
Basic usage example for the rankigi Python SDK.

Run with:
    python examples/basic.py

Environment variables:
    RANKIGI_API_KEY  — from /dashboard/keys
    RANKIGI_AGENT_ID — from /dashboard/agents
"""

import os

from rankigi import Rankigi

rk = Rankigi(
    api_key=os.environ["RANKIGI_API_KEY"],
    agent_id=os.environ["RANKIGI_AGENT_ID"],
    on_error=lambda e: print(f"[rankigi error] {e}"),
)

# 1. Track a tool call
search_input = {"query": "Q4 revenue figures"}
search_output = {"results": ["Revenue was $12M in Q4 2025"]}

rk.track_tool_call("web_search", search_input, search_output)
print("Tracked tool call: web_search")

# 2. Track with intent reasoning
rk.track_tool_call(
    "database_query",
    {"sql": "SELECT * FROM revenue WHERE quarter = 'Q4'"},
    {"rows": 42},
    intent="User asked about revenue, querying the financial database for Q4 data",
)
print("Tracked tool call with intent: database_query")

# 3. Track agent output
rk.track_agent_output("Based on the data, Q4 revenue was $12M, up 15% YoY.")
print("Tracked agent output")

# 4. Track an error
rk.track_error(RuntimeError("Rate limit exceeded"), severity="warn")
print("Tracked error")

# 5. Track custom event
rk.track_custom_event("policy_check", {
    "rule": "pii_filter",
    "passed": True,
    "scanned_fields": 12,
})
print("Tracked custom event: policy_check")

# 6. Graceful shutdown
rk.close()
print("\nAll events tracked successfully!")
