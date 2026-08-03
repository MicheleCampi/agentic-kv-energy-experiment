"""Callback handler emitting an inferscope ADR-013 steps-file (JSONL).

One record per completed step:
    {"step_id": u64, "kind": "llm_call"|"tool", "t_start_unix_ns": int, "t_end_unix_ns": int}

(step_id is the ADR-013 contract: a plain driver-assigned integer, unique
within the file; kind is already its own field, so no id-encoded kind.)

Timestamps are taken with time.time_ns() inside the handler (wall-clock UTC ns),
never from framework-internal timestamps. Start/end pairing is keyed on run_id.
"""

import json
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


class StepsFileCallback(BaseCallbackHandler):
    def __init__(self, path: str) -> None:
        self._path = path
        self._open_runs: dict[UUID, tuple[str, int]] = {}
        self._seq = 0
        # truncate on init: one file per trajectory run
        with open(self._path, "w", encoding="utf-8"):
            pass

    def begin_step(self, key: UUID, kind: str) -> None:
        """Open a step segment. Public: the replay arm has no
        framework to emit callbacks, so it drives the handler
        directly. Both arms must open and close segments through
        the same code, or the two steps-files are not comparable.
        """
        self._open_runs[key] = (kind, time.time_ns())

    def _start(self, run_id: UUID, kind: str) -> None:
        self.begin_step(run_id, kind)

    def end_step(self, key: UUID) -> None:
        """Close a step segment opened by begin_step."""
        self._end(key)

    def _end(self, run_id: UUID) -> None:
        entry = self._open_runs.pop(run_id, None)
        if entry is None:
            return  # end without start: ignore, never fabricate a segment
        kind, t_start = entry
        t_end = time.time_ns()
        self._seq += 1
        record = {
            "step_id": self._seq,
            "kind": kind,
            "t_start_unix_ns": t_start,
            "t_end_unix_ns": t_end,
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # NOTE: explicit positional args required; *args breaks the
    # on_llm_start fallback path in langchain-core (see base.py docstring).
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start(run_id, "llm_call")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        # drop the open segment: a failed call must not appear as a step
        self._open_runs.pop(run_id, None)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start(run_id, "tool")

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._open_runs.pop(run_id, None)

    @property
    def open_runs(self) -> int:
        """Non-zero after a trajectory completes = boundary anomaly, fail the run."""
        return len(self._open_runs)
