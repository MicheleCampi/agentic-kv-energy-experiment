"""Middleware serializing tool execution (ADR-013: controlled concurrency).

The model may emit multiple tool calls per response; llama.cpp ignores
parallel_tool_calls=false, and the ToolNode executes calls concurrently.
Truncating tool_calls would alter the model-decided trajectory, so instead
we keep the trajectory intact and serialize execution with a lock: tool
segments become pairwise disjoint by construction.
"""

import threading
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage


class SerializeToolCalls(AgentMiddleware):
    _lock = threading.Lock()

    def wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Any:
        with self._lock:
            return handler(request)
