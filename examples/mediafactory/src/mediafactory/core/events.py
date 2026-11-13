"""`events.jsonl` trace contract (`SPEC.md` §7.3, Module 2 & Module 4.3 waterfall)."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EventKind = Literal["llm", "gpu_tool", "judge", "check", "ticket", "budget", "lifecycle"]
EventStatus = Literal["ok", "retry", "failed", "cache_hit", "refused"]
WorkloadKind = Literal["endpoint", "job", "token_factory", "local"]
VerdictKind = Literal["PASS", "RETRY", "WEAK"]


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class TraceEvent(BaseModel):
    """One closed event line in `events.jsonl` (`SPEC.md` §7.3)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_event_id: str | None = None
    timestamp: str = Field(default_factory=_now_iso)
    run_id: str = Field(..., min_length=1)
    record_id: str | None = None
    kind: EventKind
    name: str = Field(..., min_length=1)
    call_id: str | None = None
    workload: WorkloadKind | None = None
    workload_id: str | None = None
    attempt: int = Field(default=1, ge=1)
    status: EventStatus = "ok"
    verdict: VerdictKind | None = None
    t_gen_s: float | None = Field(default=None, ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    rubric_version: str | None = None
    judge_model: str | None = None
    score: int | None = Field(default=None, ge=0, le=100)
    reasons: list[str] | None = None
    detail: str | None = None


class EventLogger:
    """Thread-safe JSONL event writer for `runs/<run_id>/events.jsonl`."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        *,
        kind: EventKind,
        name: str,
        record_id: str | None = None,
        parent_event_id: str | None = None,
        call_id: str | None = None,
        workload: WorkloadKind | None = None,
        workload_id: str | None = None,
        attempt: int = 1,
        status: EventStatus = "ok",
        verdict: VerdictKind | None = None,
        t_gen_s: float | None = None,
        cost_usd: float = 0.0,
        rubric_version: str | None = None,
        judge_model: str | None = None,
        score: int | None = None,
        reasons: list[str] | None = None,
        detail: str | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            record_id=record_id,
            parent_event_id=parent_event_id,
            kind=kind,
            name=name,
            call_id=call_id,
            workload=workload,
            workload_id=workload_id,
            attempt=attempt,
            status=status,
            verdict=verdict,
            t_gen_s=t_gen_s,
            cost_usd=round(cost_usd, 6),
            rubric_version=rubric_version,
            judge_model=judge_model,
            score=score,
            reasons=reasons,
            detail=detail,
        )
        line = json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        return event

    def read_all(self) -> list[TraceEvent]:
        if not self.path.is_file():
            return []
        events: list[TraceEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(TraceEvent.model_validate_json(line))
        return events
