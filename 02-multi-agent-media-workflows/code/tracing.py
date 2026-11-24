"""JSONL trace for Module 2: one event per model request, tool call, judge call, check,
ticket step and budget step, and the totals a trace adds up to."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

RUNS = Path(__file__).resolve().parent / "runs"  # one folder per lesson: runs/<lesson>/<run_id>/
REPO_ROOT = Path(__file__).resolve().parents[2]  # code/ -> module folder -> repository root

Kind = Literal["model_request", "tool_call", "judge", "check", "ticket", "budget", "node"]
STOP_REASONS = (
    "done",
    "attempts_exhausted",
    "budget_exhausted",
    "judge_unparseable",
    "weak",
    "needs_reconcile",
    "poll_timeout",
    "unchanged_retry",
    "video_denied",
)


class Event(BaseModel):
    """One line of events.jsonl. Field names are the Module 2 trace contract."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    run_id: str
    framework_run_id: str | None = None
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    kind: Kind
    name: str
    call_id: str | None = None
    tool_call_ids: list[str] | None = None  # model_request only: the calls this response made
    workload: Literal["endpoint", "job", "llm", "none"] = "none"
    workload_id: str | None = None
    request_id: str | None = None
    attempt: int | None = None
    verdict: Literal["PASS", "RETRY", "WEAK"] | None = None
    rubric_version: str | None = None
    judge_model: str | None = None
    reasons: list[str] | None = None
    cache_hit: bool = False
    possible_duplicate: bool = False
    stop_reason: str | None = None
    t_gen_s: float | None = None
    wall_s: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float = 0.0
    asset_ref: str | None = None


class LLMPrice(BaseModel):
    """USD per 1M tokens, from the Token Factory pricing page on the day you run."""

    input_per_m: float
    output_per_m: float

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        """Return the USD cost of one call."""
        return (tokens_in * self.input_per_m + tokens_out * self.output_per_m) / 1e6


# ---------------------------------------------------------------------------
# Run folder and the event writer
# ---------------------------------------------------------------------------


def new_run_id() -> str:
    """Return a sortable run ID, e.g. 20261130T101500Z-3f2a."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]


def repo_path(path: str | Path) -> str:
    """Return a path relative to the repository root, for printing and for what the model sees.

    Short, the same on every machine, and no home folder in a prompt. A path outside the
    repository comes back unchanged.
    """
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def run_folder(run_id: str, lesson: str) -> Path:
    """Return code/runs/<lesson>/<run_id>/, created if missing. lesson: e.g. "02-tools"."""
    folder = RUNS / lesson / run_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_events(path: str | Path) -> list[Event]:
    """Read events.jsonl into Event objects."""
    path = Path(path)
    if not path.exists():
        return []
    return [Event.model_validate_json(line) for line in path.read_text().splitlines() if line]


class Tracer:
    """Append events to <run folder>/events.jsonl. Reopening a run keeps appending to it."""

    def __init__(self, run_id: str, lesson: str):
        self.run_id = run_id
        self.folder = run_folder(run_id, lesson)
        self.path = self.folder / "events.jsonl"
        self.framework_run_id: str | None = None
        # tool call ID -> event_id of the model request that issued it (parent of the tool call)
        self._parents: dict[str, str] = {}
        for event in self.events():
            for call_id in event.tool_call_ids or []:
                self._parents[call_id] = event.event_id

    def emit(self, kind: Kind, name: str, **fields: Any) -> Event:
        """Write one event and return it."""
        call_id = fields.get("call_id")
        if call_id and "parent_id" not in fields:
            fields["parent_id"] = self._parents.get(call_id)
        event = Event(
            run_id=self.run_id,
            framework_run_id=self.framework_run_id,
            kind=kind,
            name=name,
            **fields,
        )
        for issued in event.tool_call_ids or []:
            self._parents[issued] = event.event_id
        with self.path.open("a") as f:
            f.write(event.model_dump_json() + "\n")
        return event

    def stop(self, reason: str, **fields: Any) -> Event:
        """Write the event that ends the run, with its stop reason."""
        assert reason in STOP_REASONS, reason
        return self.emit("check", "run_end", stop_reason=reason, **fields)

    def events(self) -> list[Event]:
        """Return every event written so far."""
        return read_events(self.path)


# ---------------------------------------------------------------------------
# Tracing a PydanticAI run
# ---------------------------------------------------------------------------


def save_messages(folder: Path, messages: list) -> Path:
    """Save the conversation to messages.json, so a new process can continue it."""
    path = folder / "messages.json"
    path.write_bytes(ModelMessagesTypeAdapter.dump_json(messages, indent=2))
    return path


def load_messages(folder: Path) -> list:
    """Load the conversation saved by save_messages."""
    return ModelMessagesTypeAdapter.validate_json((folder / "messages.json").read_bytes())


def last_tool_return(messages: list[ModelMessage]) -> Any:
    """The content of the most recent tool result in the conversation."""
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, ToolReturnPart):
                return part.content
    return None


def model_request_event(
    tracer: Tracer, response: ModelResponse, wall_s: float, price: LLMPrice | None
) -> Event:
    """Write the model_request event for one model response."""
    calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
    tokens_in, tokens_out = response.usage.input_tokens, response.usage.output_tokens
    return tracer.emit(
        "model_request",
        ",".join(c.tool_name for c in calls) or "text",
        tool_call_ids=[c.tool_call_id for c in calls],
        workload="llm",
        workload_id=response.model_name,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=price.cost(tokens_in, tokens_out) if price else 0.0,
        wall_s=round(wall_s, 3),
    )


async def run_traced(
    agent: Agent,
    prompt: Any = None,
    *,
    tracer: Tracer,
    deps: Any = None,
    price: LLMPrice | None = None,
    **run_kwargs: Any,
):
    """Run the agent node by node; trace each model response and save messages.json after it.

    run_kwargs go to agent.iter(): message_history, deferred_tool_results, usage_limits.
    Tools write their own tool_call events. Returns the run result.
    """
    started = time.perf_counter()
    async with agent.iter(prompt, deps=deps, **run_kwargs) as run:
        async for node in run:
            if Agent.is_call_tools_node(node):
                response = node.model_response
                # Recent PydanticAI versions stamp each response with the framework's run ID
                tracer.framework_run_id = getattr(response, "run_id", None)
                model_request_event(tracer, response, time.perf_counter() - started, price)
                save_messages(tracer.folder, run.all_messages())
            started = time.perf_counter()
    save_messages(tracer.folder, run.all_messages())
    return run.result


# ---------------------------------------------------------------------------
# Totals: the aggregation rules of the trace contract
# ---------------------------------------------------------------------------


def step_of(event: Event) -> str:
    """Return the cost_by_step bucket an event's cost goes to."""
    if event.kind == "model_request":
        return "llm"
    if event.kind == "judge":
        return "judge"
    if event.kind == "ticket" or event.name == "generate_video":
        return "video"
    if event.name == "generate_image":
        return "keyframe"
    return event.name


def summarize(events: list[Event]) -> dict[str, Any]:
    """Add up a trace: steps, judge calls, marginal cost, rejected share and the rest."""
    judged = [e for e in events if e.kind == "judge" and e.name == "judge" and e.verdict]
    passed = {e.attempt for e in judged if e.verdict == "PASS"}
    keyframes = [
        e for e in events if e.name == "generate_image" and e.kind == "tool_call" and e.t_gen_s
    ]
    videos = [e for e in events if e.kind == "ticket" and e.t_gen_s is not None]
    all_gen_s = sum(e.t_gen_s for e in events if e.t_gen_s is not None)
    rejected_s = sum(e.t_gen_s for e in keyframes if e.attempt not in passed)

    cost_by_step: dict[str, float] = {}
    for e in events:
        if e.cost_usd:
            cost_by_step[step_of(e)] = cost_by_step.get(step_of(e), 0.0) + e.cost_usd
    stops = [e.stop_reason for e in events if e.stop_reason]
    return {
        "steps": sum(e.kind in ("model_request", "tool_call") for e in events),
        "judge_calls": sum(e.kind == "judge" for e in events),
        "attempts": {
            "keyframe": len(keyframes),
            "video": len({e.request_id for e in events if e.kind == "ticket" and e.request_id}),
        },
        "accept_rate": len(passed) / len(judged) if judged else None,
        "rejected_share": rejected_s / all_gen_s if judged and all_gen_s else None,
        "llm_tokens": sum(
            (e.tokens_in or 0) + (e.tokens_out or 0)
            for e in events
            if e.kind in ("model_request", "judge")
        ),
        "latency_s": [e.t_gen_s for e in keyframes],
        "video_latency_s": [e.t_gen_s for e in videos],
        "cost_by_step": cost_by_step,
        "marginal_cost_usd": sum(e.cost_usd for e in events),
        "possible_duplicates": sum(e.possible_duplicate for e in events),
        "stop_reason": stops[-1] if stops else None,
    }


PAID_KINDS = ("model_request", "judge", "tool_call", "ticket")


def check(events: list[Event]) -> list[str]:
    """Return every violation of the trace contract; an empty list means the trace passes."""
    problems: list[str] = []
    seen: dict[str, Event] = {}
    for i, e in enumerate(events, 1):
        where = f"line {i} ({e.kind} {e.name})"
        if e.event_id in seen:
            problems.append(f"{where}: duplicate event_id")
        if e.parent_id and e.parent_id not in seen:
            problems.append(f"{where}: parent_id points to no earlier event")
        if e.cost_usd < 0:
            problems.append(f"{where}: negative cost")
        if e.cost_usd and e.kind not in PAID_KINDS:
            problems.append(f"{where}: {e.kind} events cost nothing")
        if e.kind == "ticket" and e.cost_usd and e.name not in ("completed", "failed"):
            problems.append(f"{where}: a video is charged once, on completed or failed")
        if e.cache_hit and (e.cost_usd or e.t_gen_s is not None):
            problems.append(f"{where}: a cache hit has cost_usd 0 and t_gen_s null")
        if e.possible_duplicate and e.kind != "ticket":
            problems.append(f"{where}: possible_duplicate is for ticket events")
        if e.stop_reason and e.stop_reason not in STOP_REASONS:
            problems.append(f"{where}: unknown stop_reason {e.stop_reason!r}")
        if e.kind == "node" and (e.tokens_in or e.tokens_out):
            problems.append(f"{where}: node events carry no tokens")
        if e.kind == "model_request" and (e.workload != "llm" or e.tokens_in is None):
            problems.append(f"{where}: model_request needs workload llm and tokens")
        if e.kind == "tool_call":
            parent = seen.get(e.parent_id or "")
            if not e.call_id:
                problems.append(f"{where}: tool_call without call_id")
            elif parent is None or e.call_id not in (parent.tool_call_ids or []):
                problems.append(f"{where}: call_id is not linked to the model request")
        if e.kind == "judge":
            if e.rubric_version is None or e.judge_model is None or e.reasons is None:
                problems.append(f"{where}: judge needs rubric_version, judge_model, reasons")
        elif e.verdict or e.rubric_version:
            problems.append(f"{where}: verdict and rubric_version are for judge events")
        seen[e.event_id] = e
    return problems


def write_metrics(tracer: Tracer, **extra: Any) -> dict[str, Any]:
    """Write metrics.json next to events.jsonl and return it.

    Also checks the trace against the contract. Problems are printed and saved as
    contract_problems; they never stop the run, because the GPU work is already paid for.
    """
    events = tracer.events()
    problems = check(events)
    for problem in problems:
        print(f"Trace contract: {problem}")
    metrics = {"schema_version": 1, "run_id": tracer.run_id, **summarize(events)}
    metrics["contract_problems"] = problems
    metrics.update(extra)
    (tracer.folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
