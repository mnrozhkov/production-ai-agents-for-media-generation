"""Persisted async ticket state machine (`intent -> submitted -> completed | failed`) (`FR-08`).

Implements Lesson 2.4 (`02-multi-agent-media-workflows/04-long-running-tools.md`):
1. Write ticket with ``state="intent"`` before making the network submit call.
2. Save ``workload_id`` with ``state="submitted"`` immediately after submit returns.
3. On resume, reconcile open tickets before launching any new GPU work:
   - If ``state == "submitted"``, query ``provider.get_by_id(workload_id)``.
   - If ``state == "intent"`` (crash occurred between intent write and workload_id save), probe
     ``provider.find_by_name(deterministic_name)`` before submitting a new workload.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from mediafactory.core.storage import staged_write

TicketState = Literal["intent", "submitted", "completed", "failed"]
ProviderState = Literal["running", "completed", "failed", "not_found"]


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def deterministic_workload_name(run_id: str, call_id: str) -> str:
    return f"mf-{run_id[:16]}-{call_id[:12]}"


class Ticket(BaseModel):
    """Closed persisted async ticket (`SPEC.md` §8.2)."""

    model_config = ConfigDict(extra="forbid")

    ticket_version: Literal[1] = 1
    call_id: str = Field(..., min_length=1)
    spec_hash: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    record_id: str | None = None
    role: str = Field(..., min_length=1)
    deterministic_name: str = Field(..., min_length=1)
    state: TicketState = "intent"
    workload_id: str | None = None
    endpoint_or_job_id: str | None = None
    created_at: str = Field(default_factory=_now_iso)
    submitted_at: str | None = None
    finished_at: str | None = None
    reconciled_count: int = 0
    error: str | None = None

    def save(self, path: str | Path) -> Path:
        payload = json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        return staged_write(Path(path), payload)

    @classmethod
    def load(cls, path: str | Path) -> Ticket:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class AsyncWorkloadProvider(Protocol):
    """Protocol for Endpoint async video API or Nebius Job service used during ticket execution."""

    def submit(self, *, deterministic_name: str, call_id: str) -> str:
        """Submit workload and return `workload_id`."""
        ...

    def get_status(self, workload_id: str) -> ProviderState:
        """Query status of an existing `workload_id`."""
        ...

    def find_by_name(self, deterministic_name: str) -> tuple[str, ProviderState] | None:
        """Probe provider for a workload submitted just before a crash in `intent` state."""
        ...


class InMemoryWorkloadProvider:
    """Testable workload provider tracking submissions and statuses."""

    def __init__(self) -> None:
        self.submissions: list[tuple[str, str]] = []
        self.by_id: dict[str, ProviderState] = {}
        self.by_name: dict[str, str] = {}

    def submit(self, *, deterministic_name: str, call_id: str) -> str:
        wid = f"wl-{len(self.submissions) + 1:04d}-{call_id[:8]}"
        self.submissions.append((deterministic_name, wid))
        self.by_id[wid] = "completed"
        self.by_name[deterministic_name] = wid
        return wid

    def get_status(self, workload_id: str) -> ProviderState:
        return self.by_id.get(workload_id, "not_found")

    def find_by_name(self, deterministic_name: str) -> tuple[str, ProviderState] | None:
        wid = self.by_name.get(deterministic_name)
        if wid is None:
            return None
        return wid, self.by_id.get(wid, "completed")


def write_intent_ticket(
    path: str | Path,
    *,
    call_id: str,
    spec_hash: str,
    run_id: str,
    role: str,
    record_id: str | None = None,
    endpoint_or_job_id: str | None = None,
) -> Ticket:
    ticket = Ticket(
        call_id=call_id,
        spec_hash=spec_hash,
        run_id=run_id,
        record_id=record_id,
        role=role,
        deterministic_name=deterministic_workload_name(run_id, call_id),
        state="intent",
        endpoint_or_job_id=endpoint_or_job_id,
    )
    ticket.save(path)
    return ticket


def mark_ticket_submitted(path: str | Path, ticket: Ticket, workload_id: str) -> Ticket:
    ticket.state = "submitted"
    ticket.workload_id = workload_id
    ticket.submitted_at = _now_iso()
    ticket.save(path)
    return ticket


def mark_ticket_completed(path: str | Path, ticket: Ticket) -> Ticket:
    ticket.state = "completed"
    ticket.finished_at = _now_iso()
    ticket.save(path)
    return ticket


def mark_ticket_failed(path: str | Path, ticket: Ticket, error: str) -> Ticket:
    ticket.state = "failed"
    ticket.finished_at = _now_iso()
    ticket.error = error
    ticket.save(path)
    return ticket


def reconcile_or_submit(
    ticket_path: str | Path,
    *,
    call_id: str,
    spec_hash: str,
    run_id: str,
    role: str,
    provider: AsyncWorkloadProvider,
    record_id: str | None = None,
) -> tuple[Ticket, bool]:
    """Reconcile an existing ticket if present, or submit a new workload (`SPEC.md` §8.2).

    Returns ``(ticket, was_reconciled)`` where ``was_reconciled=True`` means an existing
    in-flight or completed workload was recovered without launching a duplicate job/video.
    """
    path = Path(ticket_path)
    if path.is_file():
        existing = Ticket.load(path)
        if existing.spec_hash == spec_hash:
            if existing.state == "completed" and existing.workload_id:
                existing.reconciled_count += 1
                existing.save(path)
                return existing, True

            if existing.state == "submitted" and existing.workload_id:
                status = provider.get_status(existing.workload_id)
                if status in ("running", "completed"):
                    existing.reconciled_count += 1
                    existing.state = "completed"
                    existing.finished_at = _now_iso()
                    existing.save(path)
                    return existing, True

            if existing.state == "intent":
                probed = provider.find_by_name(existing.deterministic_name)
                if probed is not None:
                    wid, status = probed
                    if status in ("running", "completed"):
                        existing.workload_id = wid
                        existing.reconciled_count += 1
                        existing.state = "completed"
                        existing.submitted_at = existing.submitted_at or _now_iso()
                        existing.finished_at = _now_iso()
                        existing.save(path)
                        return existing, True

    ticket = write_intent_ticket(
        path,
        call_id=call_id,
        spec_hash=spec_hash,
        run_id=run_id,
        role=role,
        record_id=record_id,
    )
    wid = provider.submit(deterministic_name=ticket.deterministic_name, call_id=call_id)
    mark_ticket_submitted(path, ticket, wid)
    mark_ticket_completed(path, ticket)
    return ticket, False
