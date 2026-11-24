"""Lesson 2.4: video as a long-running tool. Submit, defer, poll, resume, with a ticket.

    uv run python video_tool.py run --fake                    # offline, scripted model
    uv run python video_tool.py run --fake --kill-at pending  # kill the agent on purpose
    uv run python video_tool.py resume <run_id> --fake        # continue the same run
    uv run python video_tool.py run --keyframe keyframe.png   # live: Token Factory + Wan2.2

Promise, with one process per ticket: a completed video is reused, a known pending request
is polled and never submitted again, and an uncertain one stops with needs_reconcile.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import requests
import typer
from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    CallDeferred,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    RunContext,
    ToolDenied,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from rich import print

from fake import PlaceholderImages, PlaceholderVideos, block_real_models, placeholder_png
from models import WAN_I2V
from tools import (
    Store,
    ToolDeps,
    gpu_cost,
    llm_model,
    optional_setting,
    settings,
    spec_hash,
)
from tracing import (
    LLMPrice,
    Tracer,
    load_messages,
    new_run_id,
    repo_path,
    run_traced,
    write_metrics,
)

LESSON = "04-long-running-tools"  # runs go to code/runs/04-long-running-tools/<run_id>/
WAN_IMAGE = "vllm/vllm-omni:v0.24.0"  # the template's container image, part of the spec hash
KILL_AT: str | None = None  # --kill-at: where this process kills itself, for the lesson demo

# ---------------------------------------------------------------------------
# The Videos API on the Wan2.2 Endpoint (vLLM-Omni)
# ---------------------------------------------------------------------------


class VideosAPI:
    """POST /v1/videos, GET /v1/videos/{id}, GET .../content, DELETE, GET /v1/videos."""

    workload = "endpoint"

    def __init__(self, base_url: str, token: str, endpoint_id: str):
        self.url = base_url.rstrip("/").removesuffix("/v1") + "/v1/videos"
        self.headers = {"Authorization": f"Bearer {token}"}
        self.workload_id = endpoint_id

    def create(self, prompt: str, image: bytes, *, user: str, params: dict) -> dict:
        """Submit one image-to-video request; returns the video object with its id."""
        data = {"prompt": prompt, "model": WAN_I2V.model, "user": user, **params}
        files = {"input_reference": ("keyframe.png", image, "image/png")}
        r = requests.post(self.url, data=data, files=files, headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json()

    def get(self, video_id: str) -> dict:
        """Status of one video. KeyError if the Endpoint doesn't know it (restarted)."""
        r = requests.get(f"{self.url}/{video_id}", headers=self.headers, timeout=30)
        if r.status_code == 404:
            raise KeyError(video_id)
        r.raise_for_status()
        return r.json()

    def list(self) -> list[dict]:
        """Every video the Endpoint stores. [verify: the list returns `user` on v0.24.0]"""
        r = requests.get(self.url, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json().get("data", [])

    def content(self, video_id: str) -> bytes:
        r = requests.get(f"{self.url}/{video_id}/content", headers=self.headers, timeout=300)
        r.raise_for_status()
        return r.content

    def delete(self, video_id: str) -> None:
        requests.delete(f"{self.url}/{video_id}", headers=self.headers, timeout=30)


# ---------------------------------------------------------------------------
# The ticket: what survives the agent process
# ---------------------------------------------------------------------------


def now() -> str:
    return datetime.now(UTC).isoformat()


class Ticket(BaseModel):
    """tickets/<spec_hash>.json in the run folder."""

    spec_hash: str
    run_id: str  # the course run ID, the same before and after a resume
    tool_call_id: str | None  # the tool call that waits for this video
    backend: Literal["endpoint", "job"] = "endpoint"
    deployment_id: str  # which server: the Endpoint ID
    request_id: str | None = None  # which request: the video_id
    state: Literal["intent", "pending", "completed", "failed", "unknown"] = "intent"
    submit_attempts: int = 0
    max_submits: int = 1
    budget_reserved_usd: float = 0.0
    created_at: str
    updated_at: str
    asset_ref: str | None = None

    @staticmethod
    def path(folder: Path, key: str) -> Path:
        return folder / "tickets" / f"{key}.json"

    def save(self, folder: Path) -> None:
        """Write the ticket. Called before submit (intent) and right after it (pending)."""
        self.updated_at = now()
        path = self.path(folder, self.spec_hash)
        path.parent.mkdir(exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def load_ticket(folder: Path, key: str) -> Ticket | None:
    path = Ticket.path(folder, key)
    return Ticket.model_validate_json(path.read_text()) if path.exists() else None


def find_ticket(folder: Path, tool_call_id: str) -> Ticket | None:
    """The ticket a tool call is waiting for."""
    for path in (folder / "tickets").glob("*.json"):
        ticket = Ticket.model_validate_json(path.read_text())
        if ticket.tool_call_id == tool_call_id:
            return ticket
    return None


@dataclass
class VideoDeps(ToolDeps):
    """ToolDeps plus the video backend and the two gates lesson 2.5 adds."""

    videos: Any = None  # VideosAPI, or PlaceholderVideos with --fake
    video_gate_open: bool = True  # 2.5: the tool is hidden until a keyframe gets PASS
    approve_video: bool = False  # 2.5 also try: a human says yes before the submit


def ticket_event(deps: VideoDeps, ticket: Ticket, name: str, **fields: Any) -> None:
    deps.tracer.emit(
        "ticket",
        name,
        call_id=ticket.tool_call_id,
        workload=ticket.backend,
        workload_id=ticket.deployment_id,
        request_id=ticket.request_id,
        **fields,
    )


def maybe_kill(point: str) -> None:
    """Kill this process at a restart boundary, without any cleanup, when --kill-at says so."""
    if KILL_AT == point:
        print(f"[red]Killed at {point}[/red]. Resume with: video_tool.py resume <run_id>")
        os._exit(1)


# ---------------------------------------------------------------------------
# Submit, reconcile, poll
# ---------------------------------------------------------------------------


class NeedsReconcile(Exception):
    """We can't tell whether the request exists. A person decides; we never resend blindly."""


def submit(ticket: Ticket, deps: VideoDeps, motion_prompt: str, keyframe: Path, seed: int) -> None:
    """intent -> submit -> pending. The request_id is saved right after the submit returns."""
    ticket.save(deps.store.folder)
    ticket_event(deps, ticket, "intent")
    maybe_kill("after-intent")
    params = {**WAN_I2V.params, "seed": seed}
    video = deps.videos.create(
        motion_prompt, keyframe.read_bytes(), user=ticket.spec_hash, params=params
    )
    ticket.submit_attempts += 1
    maybe_kill("after-submit")
    ticket.request_id, ticket.state = video["id"], "pending"
    ticket.save(deps.store.folder)
    ticket_event(deps, ticket, "pending")


def reconcile(ticket: Ticket, deps: VideoDeps) -> None:
    """Find the request by its spec hash (sent as `user`): found -> pending, else stop."""
    found = [v for v in deps.videos.list() if v.get("user") == ticket.spec_hash]
    if not found:
        ticket.state = "unknown"
        ticket.save(deps.store.folder)
        ticket_event(deps, ticket, "unknown", reasons=["no request with this spec hash"])
        raise NeedsReconcile(ticket.spec_hash)
    ticket.request_id, ticket.state = found[0]["id"], "pending"
    ticket.save(deps.store.folder)
    ticket_event(deps, ticket, "reconciled")


def wait_video(ticket: Ticket, deps: VideoDeps, max_wait_s: float, poll_s: float) -> None:
    """Poll a pending video until it completes, fails or the deadline passes.

    Completed: copy it to the run folder (and bucket), then delete it from the Endpoint.
    """
    deadline = time.monotonic() + max_wait_s
    while True:
        try:
            status = deps.videos.get(ticket.request_id)["status"]
        except KeyError:  # the Endpoint restarted and lost the request
            ticket.state = "unknown"
            ticket.save(deps.store.folder)
            ticket_event(deps, ticket, "unknown", reasons=["the Endpoint doesn't know this id"])
            return
        if status not in ("queued", "in_progress"):
            break
        maybe_kill("pending")
        if time.monotonic() + poll_s > deadline:
            return  # still pending: the run stops with poll_timeout, resume continues later
        time.sleep(poll_s)

    t_gen_s = round(time.time() - datetime.fromisoformat(ticket.created_at).timestamp(), 3)
    cost = gpu_cost(deps.gpu_price_per_hour, t_gen_s)
    if status != "completed":  # the failure status name is [verify] on v0.24.0
        ticket.state = "failed"
        ticket.save(deps.store.folder)
        ticket_event(deps, ticket, "failed", t_gen_s=t_gen_s, cost_usd=cost, reasons=[status])
        return
    video = deps.videos.content(ticket.request_id)
    ticket.asset_ref = deps.store.put(f"video-{ticket.spec_hash[:12]}.mp4", video)
    deps.videos.delete(ticket.request_id)
    ticket.state = "completed"
    ticket.save(deps.store.folder)
    ticket_event(
        deps, ticket, "completed", t_gen_s=t_gen_s, cost_usd=cost, asset_ref=ticket.asset_ref
    )


def video_ref(ticket: Ticket) -> dict:
    """What the model gets back for a finished video."""
    return {
        "uri": ticket.asset_ref,
        "spec_hash": ticket.spec_hash,
        "duration_s": WAN_I2V.duration_s,
    }


# ---------------------------------------------------------------------------
# The deferred tool
# ---------------------------------------------------------------------------


async def generate_video(
    ctx: RunContext[VideoDeps], motion_prompt: str, keyframe_uri: str, seed: int = WAN_I2V.seed
) -> dict:
    """Make a 5-second video from a keyframe image. The video renders in the background.

    Args:
        motion_prompt: What moves in the clip. The keyframe sets the look.
        keyframe_uri: The uri of the keyframe, as generate_image returned it.
        seed: Change it only if a previous video with the same inputs failed.
    """
    deps = ctx.deps
    if deps.approve_video and not ctx.tool_call_approved:
        raise ApprovalRequired(metadata={"keyframe": keyframe_uri})
    keyframe = deps.store.path(Path(keyframe_uri).name)
    key = spec_hash(
        model=WAN_I2V.model,
        model_revision=WAN_I2V.model_revision,
        image=WAN_IMAGE,
        prompt=motion_prompt,
        seed=seed,
        params=WAN_I2V.params,
        inputs=[keyframe],  # the keyframe's bytes, not its name
    )
    ticket = load_ticket(deps.store.folder, key)
    event = {
        "call_id": ctx.tool_call_id,
        "workload": deps.videos.workload,
        "workload_id": deps.videos.workload_id,
    }

    if ticket and ticket.state == "completed":  # same spec already rendered: reuse it
        deps.tracer.emit(
            "tool_call", "generate_video", cache_hit=True, asset_ref=ticket.asset_ref, **event
        )
        return video_ref(ticket)
    if ticket and ticket.state == "failed":
        raise ModelRetry("The video with these inputs failed. Call again with another seed.")

    if ticket is None:
        stamp = now()
        ticket = Ticket(
            spec_hash=key,
            run_id=deps.tracer.run_id,
            tool_call_id=ctx.tool_call_id,
            deployment_id=deps.videos.workload_id,
            created_at=stamp,
            updated_at=stamp,
        )
        submit(ticket, deps, motion_prompt, keyframe, seed)
    else:
        ticket.tool_call_id = ctx.tool_call_id  # a new call now waits for the same request
        ticket.save(deps.store.folder)
        if ticket.state in ("intent", "unknown"):
            reconcile(ticket, deps)
    deps.tracer.emit("tool_call", "generate_video", request_id=ticket.request_id, **event)
    raise CallDeferred(metadata={"spec_hash": key})


# ---------------------------------------------------------------------------
# Run and resume
# ---------------------------------------------------------------------------


def open_video_calls(messages: list[ModelMessage]) -> list[ToolCallPart]:
    """generate_video calls in the last model response that have no result yet."""
    last = max((i for i, m in enumerate(messages) if isinstance(m, ModelResponse)), default=None)
    if last is None:
        return []
    answered = {
        p.tool_call_id
        for m in messages[last + 1 :]
        for p in m.parts
        if isinstance(p, ToolReturnPart | RetryPromptPart)
    }
    return [
        p
        for p in messages[last].parts
        if isinstance(p, ToolCallPart)
        and p.tool_name == "generate_video"
        and p.tool_call_id not in answered
    ]


def collect_results(
    messages: list[ModelMessage], deps: VideoDeps, max_wait_s: float, poll_s: float
) -> dict[str, Any]:
    """Answer every open video call from its ticket: reconcile, poll, then the result."""
    results: dict[str, Any] = {}
    for call in open_video_calls(messages):
        ticket = find_ticket(deps.store.folder, call.tool_call_id)
        if ticket is None:
            continue  # killed before the ticket existed: the agent runs this call again
        if ticket.state in ("intent", "unknown"):
            reconcile(ticket, deps)
        if ticket.state == "pending":
            wait_video(ticket, deps, max_wait_s, poll_s)
        if ticket.state == "unknown":
            reconcile(ticket, deps)
        if ticket.state == "pending":
            raise TimeoutError(ticket.spec_hash)
        if ticket.state == "failed":
            results[call.tool_call_id] = ModelRetry("The video failed. Call again with a new seed.")
        else:
            results[call.tool_call_id] = video_ref(ticket)
    return results


async def drive(
    agent: Agent,
    deps: VideoDeps,
    prompt: str | None,
    *,
    price: LLMPrice | None,
    max_wait_s: float,
    poll_s: float,
    approve: Any = None,
    **run_kwargs: Any,
) -> str | None:
    """Run (prompt) or resume (prompt=None) the agent until it answers or a stop condition.

    Between agent runs, our code polls the videos: waiting costs no LLM turns.
    approve(call) -> bool answers approval requests (2.5 also try).
    """
    tracer = deps.tracer
    history = None if prompt else load_messages(tracer.folder)
    approvals: dict[str, Any] = {}
    while True:
        try:
            calls = collect_results(history, deps, max_wait_s, poll_s) if history else {}
            results = DeferredToolResults(calls=calls, approvals=approvals)
            result = await run_traced(
                agent,
                prompt,
                tracer=tracer,
                deps=deps,
                price=price,
                message_history=history,
                deferred_tool_results=results if calls or approvals else None,
                **run_kwargs,
            )
        except NeedsReconcile:
            print("[red]Uncertain submission.[/red] Check the Endpoint's videos, then resume.")
            tracer.stop("needs_reconcile")
            return None
        except TimeoutError:
            print("[yellow]Still rendering.[/yellow] Resume later, or delete the video.")
            tracer.stop("poll_timeout")
            return None
        if not isinstance(result.output, DeferredToolRequests):
            return result.output
        prompt, history = None, result.all_messages()
        approvals = {
            c.tool_call_id: True if approve(c) else ToolDenied("A person declined the video.")
            for c in result.output.approvals
        }


def fake_video_model(motion_prompt: str, keyframe_uri: str) -> FunctionModel:
    """Scripted model for --fake: call generate_video, then answer with the video URI.

    It reads the conversation instead of following a fixed script, so it also works
    after a resume in a new process.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        result = next(
            (p.content for p in messages[-1].parts if isinstance(p, ToolReturnPart)), None
        )
        if isinstance(result, dict):
            return ModelResponse(parts=[TextPart(f"Video ready: {result['uri']}")])
        if result is not None:  # a person declined the video (2.5)
            return ModelResponse(parts=[TextPart(f"No video: {result}")])
        retries = sum(
            isinstance(p, RetryPromptPart)
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
        )
        args = {"motion_prompt": motion_prompt, "keyframe_uri": keyframe_uri, "seed": retries}
        return ModelResponse(parts=[ToolCallPart("generate_video", args)])

    return FunctionModel(respond, model_name="scripted")


INSTRUCTIONS = (
    "You turn a keyframe into a short video. Call generate_video once with the motion "
    "prompt and the keyframe URI you are given, then reply with the video URI."
)


def video_agent(model: Any) -> Agent[VideoDeps, str | DeferredToolRequests]:
    """The agent of this lesson: one deferred tool."""
    return Agent(
        model,
        deps_type=VideoDeps,
        output_type=[str, DeferredToolRequests],
        instructions=INSTRUCTIONS,
        tools=[generate_video],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(no_args_is_help=True)
MOTION = "The developer lifts the cup, takes a slow sip and leans back; steam rises."


def make_deps(run_id: str, fake: bool, gpu_price_per_hour: float | None) -> VideoDeps:
    tracer = Tracer(run_id, LESSON)
    if fake:
        videos = PlaceholderVideos(tracer.folder, render_s=3.0)
        return VideoDeps(tracer, PlaceholderImages(), Store(tracer.folder), videos=videos)
    env = settings(
        "WAN22_I2V_BASE_URL",
        "WAN22_I2V_TOKEN",
        "WAN22_I2V_ENDPOINT_ID",
        secret=("WAN22_I2V_TOKEN",),
    )
    videos = VideosAPI(
        env["WAN22_I2V_BASE_URL"], env["WAN22_I2V_TOKEN"], env["WAN22_I2V_ENDPOINT_ID"]
    )
    store = Store(tracer.folder, bucket=optional_setting("BUCKET"))
    return VideoDeps(tracer, PlaceholderImages(), store, gpu_price_per_hour, videos=videos)


def finish(deps: VideoDeps, output: str | None) -> None:
    if output is not None:
        deps.tracer.stop("done")
        print(f"[green]{output}[/green]")
    write_metrics(deps.tracer)
    deps.store.upload_run()
    print(f"Trace: {repo_path(deps.tracer.path)}")


@app.command()
def run(
    fake: bool = typer.Option(False, help="Scripted model and placeholder video. Offline."),
    keyframe: Path | None = typer.Option(
        None, help="Keyframe PNG (default with --fake: a placeholder)"
    ),
    motion_prompt: str = MOTION,
    kill_at: str | None = typer.Option(None, help="after-intent | after-submit | pending"),
    max_wait_s: float | None = typer.Option(
        None, help="Polling deadline. Default: 10 with --fake, 1200 live"
    ),
    poll_s: float = 1.0,
    gpu_price_per_hour: float | None = typer.Option(None, help="USD/h of the Wan Endpoint's GPU"),
    llm_price_in: float | None = typer.Option(None, help="USD per 1M input tokens"),
    llm_price_out: float | None = typer.Option(None, help="USD per 1M output tokens"),
) -> None:
    """Start a run: submit the video and wait for it."""
    global KILL_AT
    KILL_AT = kill_at
    run_id = new_run_id()
    deps = make_deps(run_id, fake, gpu_price_per_hour)
    image = keyframe.read_bytes() if keyframe else placeholder_png("keyframe", 832, 480)
    uri = deps.store.put("keyframe.png", image)
    print(f"Run ID: [bold]{run_id}[/bold]")
    if fake:
        block_real_models()
    model = fake_video_model(motion_prompt, uri) if fake else llm_model()
    price = LLMPrice(input_per_m=llm_price_in, output_per_m=llm_price_out) if llm_price_in else None
    prompt = f"Motion prompt: {motion_prompt}\nKeyframe URI: {uri}"
    output = asyncio.run(
        drive(
            video_agent(model),
            deps,
            prompt,
            price=price,
            max_wait_s=max_wait_s or (10 if fake else 1200),
            poll_s=poll_s,
        )
    )
    finish(deps, output)


@app.command()
def resume(
    run_id: str,
    fake: bool = False,
    motion_prompt: str = MOTION,
    kill_at: str | None = None,
    max_wait_s: float | None = typer.Option(
        None, help="Polling deadline. Default: 10 with --fake, 1200 live"
    ),
    poll_s: float = 1.0,
    gpu_price_per_hour: float | None = None,
    llm_price_in: float | None = None,
    llm_price_out: float | None = None,
) -> None:
    """Continue a run from its tickets and messages.json. Never submits a known request again."""
    global KILL_AT
    KILL_AT = kill_at
    deps = make_deps(run_id, fake, gpu_price_per_hour)
    if fake:
        block_real_models()
    uri = deps.store.uri("keyframe.png")
    model = fake_video_model(motion_prompt, uri) if fake else llm_model()
    price = LLMPrice(input_per_m=llm_price_in, output_per_m=llm_price_out) if llm_price_in else None
    output = asyncio.run(
        drive(
            video_agent(model),
            deps,
            None,
            price=price,
            max_wait_s=max_wait_s or (10 if fake else 1200),
            poll_s=poll_s,
        )
    )
    finish(deps, output)


if __name__ == "__main__":
    app()
