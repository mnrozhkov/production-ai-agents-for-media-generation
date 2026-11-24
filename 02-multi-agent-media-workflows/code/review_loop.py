"""Lesson 2.5: planner, generator, reviewer. Judge every keyframe, regenerate on RETRY,
stop cleanly at the caps. This is the loop Homework 2 builds.

    uv run python review_loop.py --fake                                  # RETRY-free happy path
    uv run python review_loop.py --fake --fixture ../assets/fixtures/bad_keyframe.png
    uv run python review_loop.py --fake --fault always-retry             # also: weak,
    #   tight-budget, unparseable, unchanged-retry
    uv run python review_loop.py --fake --two-agents                     # read the handoff
    uv run python review_loop.py --brief ../assets/brief.yaml --approve-video   # live
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import typer
import yaml
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, BinaryContent, DeferredToolRequests, RunContext, Tool, UsageLimits
from pydantic_ai.tools import ToolDefinition
from rich import print

from fake import (
    PlaceholderImages,
    PlaceholderVideos,
    block_real_models,
    call,
    placeholder_png,
    scripted_model,
)
from models import JUDGE, LLM, SANA, WAN_I2V, Brief, Plan
from tools import (
    AssetRef,
    SanaEndpoint,
    Store,
    generate_image,
    gpu_cost,
    llm_model,
    optional_setting,
    settings,
)
from tracing import (
    LLMPrice,
    Tracer,
    last_tool_return,
    new_run_id,
    repo_path,
    run_traced,
    write_metrics,
)
from video_tool import VideoDeps, VideosAPI, drive, fake_video_model, generate_video

ASSETS = Path(__file__).resolve().parent.parent / "assets"
LESSON = "05-judge-loop"  # runs go to code/runs/05-judge-loop/<run_id>/

# Caps live in code, not in the prompt
KEYFRAME_ATTEMPTS = 3
VIDEO_ATTEMPTS = 2
JUDGE_PARSE_RETRIES = 2  # one format slip plus one more; a third points to a prompt problem

# ---------------------------------------------------------------------------
# Rubric and verdict
# ---------------------------------------------------------------------------


class Criterion(BaseModel):
    id: str
    brief_field: str  # the Brief field this criterion reads
    question: str
    must_have: bool
    scope: Literal["keyframe", "video_only"]
    pass_example: str
    fail_example: str


class Rubric(BaseModel):
    """rubric.yaml: criteria that read Brief fields, so one rubric fits many briefs."""

    version: str
    criteria: list[Criterion]

    @classmethod
    def from_yaml(cls, path: Path) -> Rubric:
        return cls.model_validate(yaml.safe_load(path.read_text()))


class CriterionResult(BaseModel):
    id: str
    score: Literal[0, 0.5, 1]
    assessable: bool  # false when the keyframe can't show it reliably
    note: str


class Fix(BaseModel):
    change_seed: bool
    prompt_suffix: str | None = None


class JudgeReply(BaseModel):
    """What the judge model must return, as JSON."""

    criteria: list[CriterionResult]
    reasons: list[str]
    fix: Fix


class Verdict(BaseModel):
    """The typed verdict. The verdict itself is decided by code, not by the model."""

    verdict: Literal["PASS", "RETRY", "WEAK"]
    score: float
    criteria: list[CriterionResult]
    reasons: list[str]
    fix: Fix
    rubric_version: str
    judge_model: str


def map_verdict(reply: JudgeReply, rubric: Rubric, judge_model: str) -> Verdict:
    """Can't assess a keyframe criterion -> WEAK; a must-have below 1 -> RETRY; else PASS."""
    keyframe = {c.id: c for c in rubric.criteria if c.scope == "keyframe"}
    results = [r for r in reply.criteria if r.id in keyframe]
    if len(results) < len(keyframe) or not all(r.assessable for r in results):
        verdict = "WEAK"
    elif any(keyframe[r.id].must_have and r.score < 1 for r in results):
        verdict = "RETRY"
    else:
        verdict = "PASS"
    return Verdict(
        verdict=verdict,
        score=sum(r.score for r in results) / max(len(results), 1),
        criteria=results,
        reasons=reply.reasons,
        fix=reply.fix,
        rubric_version=rubric.version,
        judge_model=judge_model,
    )


def judge_instructions(rubric: Rubric, brief: Brief) -> str:
    """The judge's prompt: the brief, the keyframe criteria and the reply schema."""
    lines = ["You review one keyframe image for a video ad against a brief. Be strict."]
    lines.append(f"Brief: {brief.model_dump_json()}")
    for c in rubric.criteria:
        if c.scope == "keyframe":
            field = getattr(brief, c.brief_field)
            lines.append(
                f"- {c.id}: {c.question} (brief {c.brief_field}: {field}). "
                f"Pass: {c.pass_example}. Fail: {c.fail_example}."
            )
    lines.append(
        "Score each criterion 0, 0.5 or 1. Set assessable=false if the image can't show it."
    )
    lines.append("Reply with JSON only, matching this schema:")
    lines.append(json.dumps(JudgeReply.model_json_schema()))
    return "\n".join(lines)


def parse_reply(text: str) -> JudgeReply:
    """Parse the judge's JSON, tolerating a ```json fence around it."""
    return JudgeReply.model_validate_json(text.strip().strip("`").removeprefix("json"))


async def judge_keyframe(
    judge: Agent,
    rubric: Rubric,
    brief: Brief,
    png: Path,
    *,
    tracer: Tracer,
    attempt: int | None,
    price: LLMPrice | None,
    asset_ref: str | None = None,
) -> Verdict | None:
    """Judge one keyframe. Re-asks on an unparseable reply; None if it never parses."""
    image = BinaryContent(data=png.read_bytes(), media_type="image/png")
    history = None
    for _ in range(1 + JUDGE_PARSE_RETRIES):
        prompt = (
            [judge_instructions(rubric, brief), image]
            if history is None
            else ("That wasn't valid JSON for the schema. Reply with the JSON object only.")
        )
        result = await judge.run(prompt, message_history=history)
        response = result.all_messages()[-1]
        tokens_in, tokens_out = response.usage.input_tokens, response.usage.output_tokens
        event = {
            "attempt": attempt,
            "workload": "llm",
            "workload_id": response.model_name,
            "rubric_version": rubric.version,
            "judge_model": response.model_name,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": price.cost(tokens_in, tokens_out) if price else 0.0,
            "asset_ref": asset_ref or repo_path(png),
        }
        try:
            reply = parse_reply(result.output)
        except ValidationError as exc:
            tracer.emit(
                "judge", "judge", reasons=[f"unparseable: {exc.errors()[0]['msg']}"], **event
            )
            history = result.all_messages()
            continue
        verdict = map_verdict(reply, rubric, response.model_name)
        tracer.emit("judge", "judge", verdict=verdict.verdict, reasons=verdict.reasons, **event)
        return verdict
    return None


# ---------------------------------------------------------------------------
# Runtime gates: deterministic checks before and after the judge
# ---------------------------------------------------------------------------


def check_keyframe(png: Path, tracer: Tracer, attempt: int) -> bool:
    """The file exists and has the size the video model expects."""
    from PIL import Image  # Pillow is a repo dependency; imported here, only the gate needs it

    ok = png.exists() and Image.open(png).size == (WAN_I2V.width, WAN_I2V.height)
    tracer.emit(
        "check",
        "keyframe_file",
        attempt=attempt,
        asset_ref=repo_path(png),
        reasons=[] if ok else ["missing, or not 832x480"],
    )
    return ok


def check_video(path: Path, tracer: Tracer, fake: bool) -> bool:
    """The video exists; with ffprobe installed, its duration matches the request."""
    reasons = [] if path.exists() and path.stat().st_size else ["missing or empty"]
    if not reasons and not fake:
        if shutil.which("ffprobe"):
            out = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            if abs(float(out) - WAN_I2V.duration_s) > 0.5:
                reasons.append(f"duration {float(out):.2f} s, asked {WAN_I2V.duration_s:.2f} s")
        else:
            print("[yellow]ffprobe not found: video duration not checked[/yellow]")
    tracer.emit("check", "video_file", asset_ref=repo_path(path), reasons=reasons)
    return not reasons


# ---------------------------------------------------------------------------
# Budget guard: an application guard, not a billing ceiling
# ---------------------------------------------------------------------------


class BudgetGuard:
    """Reserve the estimated GPU cost before each paid generation; settle it after.

    State lives in budget.json in the run folder, so a resumed run keeps its reservations.
    """

    def __init__(self, tracer: Tracer, limit_usd: float):
        self.tracer = tracer
        self.path = tracer.folder / "budget.json"
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.limit_usd = limit_usd
        self.committed_usd = state.get("committed_usd", 0.0)  # spent + reserved

    def reserve(self, step: str, estimate_usd: float) -> bool:
        """True if the estimate fits; then it counts as committed until settled."""
        if self.committed_usd + estimate_usd > self.limit_usd:
            self.tracer.emit(
                "budget",
                f"refuse:{step}",
                reasons=[
                    f"committed {self.committed_usd:.4f} + estimate {estimate_usd:.4f} "
                    f"> limit {self.limit_usd:.4f} USD"
                ],
            )
            return False
        self.committed_usd += estimate_usd
        self._save()
        self.tracer.emit("budget", f"reserve:{step}", reasons=[f"estimate {estimate_usd:.4f} USD"])
        return True

    def settle(self, step: str, estimate_usd: float, actual_usd: float) -> None:
        """Replace the estimate with the real cost."""
        self.committed_usd += actual_usd - estimate_usd
        self._save()
        self.tracer.emit("budget", f"settle:{step}", reasons=[f"actual {actual_usd:.4f} USD"])

    def _save(self) -> None:
        self.path.write_text(
            json.dumps({"limit_usd": self.limit_usd, "committed_usd": self.committed_usd})
        )


# ---------------------------------------------------------------------------
# The agents: planner and reviewer are typed calls; only the generator picks tools
# ---------------------------------------------------------------------------


async def video_gate(ctx: RunContext[VideoDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
    """generate_video exists for the model only after PASS and under the video cap."""
    tickets = list((ctx.deps.store.folder / "tickets").glob("*.json"))
    if ctx.deps.video_gate_open and len(tickets) < VIDEO_ATTEMPTS:
        return tool_def
    return None


def make_agents(model, judge_model) -> tuple[Agent, Agent, Agent]:
    """Planner (Brief -> Plan), generator (GPU tools), judge (keyframe -> JSON reply)."""
    planner = Agent(
        model,
        output_type=Plan,
        instructions=(
            "Turn the ad brief into one keyframe prompt and one motion prompt for a 5-second clip. "
            "Show everything in must_show; nothing in must_not."
        ),
    )
    generator = Agent(
        model,
        deps_type=VideoDeps,
        output_type=[str, DeferredToolRequests],
        instructions="Call the tool you are asked for, with the arguments given.",
        tools=[generate_image, Tool(generate_video, prepare=video_gate)],
    )
    judge = Agent(judge_model, output_type=str)
    return planner, generator, judge


# ---------------------------------------------------------------------------
# Scripted replies for --fake, one per fault
# ---------------------------------------------------------------------------


def judge_reply(
    rubric: Rubric,
    score: float,
    *,
    assessable: bool = True,
    change_seed: bool = True,
    suffix: str | None = None,
) -> str:
    criteria = [
        {"id": c.id, "score": score, "assessable": assessable, "note": "scripted"}
        for c in rubric.criteria
        if c.scope == "keyframe"
    ]
    reasons = [] if score == 1 else ["scripted: a must_show item is missing"]
    return json.dumps(
        {
            "criteria": criteria,
            "reasons": reasons,
            "fix": {"change_seed": change_seed, "prompt_suffix": suffix},
        }
    )


def fake_judge_script(rubric: Rubric, fault: str, fixture: bool) -> list[str]:
    retry = judge_reply(rubric, 0, suffix="every must_show item clearly in frame")
    passed = judge_reply(rubric, 1)
    return {
        "none": [retry, passed] if fixture else [passed],
        "always-retry": [retry] * KEYFRAME_ATTEMPTS,
        "weak": [judge_reply(rubric, 0.5, assessable=False)],
        "unparseable": ["The keyframe looks great!"] * (1 + JUDGE_PARSE_RETRIES),
        "unchanged-retry": [judge_reply(rubric, 0, change_seed=False)] * 2,
        "tight-budget": [passed],
    }[fault]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def review(
    brief: Brief,
    rubric: Rubric,
    *,
    fake: bool,
    fault: str = "none",
    fixture: Path | None = None,
    approve_video: bool = False,
    budget_usd: float = 2.0,
    image_gpu_price: float | None = None,
    video_gpu_price: float | None = None,
    keyframe_s: float = 10.0,
    video_s: float = 600.0,
    price: LLMPrice | None = None,
) -> str:
    """Plan -> keyframe -> judge -> (regenerate) -> video -> checks. Returns the stop reason."""
    tracer = Tracer(new_run_id(), LESSON)
    print(f"Run ID: [bold]{tracer.run_id}[/bold]")
    store = Store(tracer.folder, bucket=None if fake else optional_setting("BUCKET"))
    if fake:
        block_real_models()
        model, judge_model = (
            scripted_model(),
            scripted_model(*fake_judge_script(rubric, fault, bool(fixture))),
        )
        images, videos = PlaceholderImages(), PlaceholderVideos(tracer.folder)
        if fault == "tight-budget":  # [illustrative] prices, so the guard has something to refuse
            image_gpu_price, budget_usd = 1.0, 0.001
    else:
        env = settings(
            "SANA_BASE_URL",
            "SANA_TOKEN",
            "SANA_ENDPOINT_ID",
            "WAN22_I2V_BASE_URL",
            "WAN22_I2V_TOKEN",
            "WAN22_I2V_ENDPOINT_ID",
            secret=("SANA_TOKEN", "WAN22_I2V_TOKEN"),
        )
        model, judge_model = llm_model(LLM), llm_model(JUDGE)
        images = SanaEndpoint(env["SANA_BASE_URL"], env["SANA_TOKEN"], env["SANA_ENDPOINT_ID"])
        videos = VideosAPI(
            env["WAN22_I2V_BASE_URL"], env["WAN22_I2V_TOKEN"], env["WAN22_I2V_ENDPOINT_ID"]
        )
    planner, generator, judge = make_agents(model, judge_model)
    deps = VideoDeps(
        tracer,
        images,
        store,
        image_gpu_price,
        videos=videos,
        video_gate_open=False,
        approve_video=approve_video,
    )
    guard = BudgetGuard(tracer, budget_usd)

    def stop(reason: str) -> str:
        tracer.stop(reason)
        write_metrics(tracer, fixture_run=bool(fixture), session_spend_usd=None)
        store.upload_run()
        print(f"Stop: [bold]{reason}[/bold]. Trace: {repo_path(tracer.path)}")
        return reason

    # 1. Plan: a typed call inside a fixed stage
    with (
        planner.override(
            model=scripted_model(
                {
                    "image_prompt": f"{brief.shot}, photorealistic, warm light",
                    "motion_prompt": "slow push-in, steam rises from the cup",
                }
            )
        )
        if fake
        else nullcontext()
    ):
        plan = (
            await run_traced(planner, brief.model_dump_json(), tracer=tracer, price=price)
        ).output

    # 2. Keyframe: generate, gate, judge; regenerate on RETRY
    seed, suffix, last_spec, keyframe = SANA.seed, "", None, None
    for attempt in range(1, KEYFRAME_ATTEMPTS + 1):
        deps.attempt = attempt
        if attempt == 1 and fixture:  # recovery demo: the fixture is attempt 1, no generation
            keyframe = AssetRef(
                uri=store.put("fixture.png", read_png(fixture, fake)),
                spec_hash="fixture",
                width=WAN_I2V.width,
                height=WAN_I2V.height,
                workload="none",
                workload_id="fixture",
            )
        else:
            prompt = f"{plan.image_prompt}. {suffix}".strip(". ")
            if (prompt, seed) == last_spec:  # the same spec again would be a cache hit
                return stop("unchanged_retry")
            last_spec = (prompt, seed)
            estimate = gpu_cost(image_gpu_price, keyframe_s)
            if not guard.reserve("keyframe", estimate):
                return stop("budget_exhausted")
            ask = f"Call generate_image with prompt={prompt!r} and seed={seed}."
            with (
                generator.override(
                    model=scripted_model(
                        call("generate_image", prompt=prompt, seed=seed), "Keyframe done."
                    )
                )
                if fake
                else nullcontext()
            ):
                result = await run_traced(
                    generator,
                    ask,
                    tracer=tracer,
                    deps=deps,
                    price=price,
                    usage_limits=UsageLimits(tool_calls_limit=1),
                )
            keyframe = AssetRef.model_validate(last_tool_return(result.all_messages()))
            guard.settle("keyframe", estimate, gpu_cost(image_gpu_price, keyframe.t_gen_s or 0))
        png = store.path(Path(keyframe.uri).name)
        if not check_keyframe(png, tracer, attempt):
            seed += 1
            continue
        verdict = await judge_keyframe(
            judge,
            rubric,
            brief,
            png,
            tracer=tracer,
            attempt=attempt,
            price=price,
            asset_ref=keyframe.uri,
        )
        if verdict is None:
            return stop("judge_unparseable")
        print(f"Attempt {attempt}: [bold]{verdict.verdict}[/bold] {verdict.reasons}")
        if verdict.verdict == "WEAK":
            return stop("weak")
        if verdict.verdict == "PASS":
            break
        seed += 1 if verdict.fix.change_seed else 0
        suffix = verdict.fix.prompt_suffix or suffix
    else:
        return stop("attempts_exhausted")

    # 3. Video: PASS opens the gate. It means "allowed to make the video", not "the ad is good"
    estimate = gpu_cost(video_gpu_price, video_s)
    if not guard.reserve("video", estimate):
        return stop("budget_exhausted")
    deps.video_gate_open, deps.attempt, deps.gpu_price_per_hour = True, None, video_gpu_price
    ask = f"Call generate_video with motion_prompt={plan.motion_prompt!r} and keyframe_uri={keyframe.uri!r}."
    with (
        generator.override(model=fake_video_model(plan.motion_prompt, keyframe.uri))
        if fake
        else nullcontext()
    ):
        output = await drive(
            generator,
            deps,
            ask,
            price=price,
            max_wait_s=10 if fake else 1200,
            poll_s=1.0,
            approve=ask_person,
            usage_limits=UsageLimits(tool_calls_limit=VIDEO_ATTEMPTS),
        )
    if output is None:  # drive() wrote the stop event: needs_reconcile or poll_timeout
        write_metrics(tracer, fixture_run=bool(fixture), session_spend_usd=None)
        return tracer.events()[-1].stop_reason
    videos_done = list(store.folder.glob("video-*.mp4"))
    if not videos_done:
        return stop("video_denied" if approve_video else "attempts_exhausted")
    actual = sum(e.cost_usd for e in tracer.events() if e.kind == "ticket")
    guard.settle("video", estimate, actual)
    check_video(videos_done[-1], tracer, fake)
    print("Watch the video once and write one sentence: does it match the brief?")
    return stop("done")


def read_png(path: Path, fake: bool) -> bytes:
    """A saved keyframe. With --fake, a placeholder if the file isn't there yet."""
    if fake and not path.exists():
        print(f"[yellow]{path.name} not found: using a placeholder[/yellow]")
        return placeholder_png(path.stem, WAN_I2V.width, WAN_I2V.height)
    return path.read_bytes()


def ask_person(call) -> bool:
    """Human approval before the video (also try). Look at the PASS keyframe first."""
    print(f"Keyframe: {call.args}")
    return typer.confirm("Generate the video from this keyframe?")


# ---------------------------------------------------------------------------
# Also: one agent or two. The planner delegates review to a reviewer agent
# ---------------------------------------------------------------------------


async def two_agents(brief: Brief, rubric: Rubric, keyframes: list[Path]) -> None:
    """Scripted, no GPU: the planner calls a reviewer agent as a tool and gets a Verdict back."""
    block_real_models()
    tracer = Tracer(new_run_id(), LESSON)
    paths = []
    for k in keyframes:  # copies in the run folder: placeholders if the fixtures aren't saved yet
        paths.append(tracer.folder / k.name)
        paths[-1].write_bytes(read_png(k, fake=True))
    reviewer = Agent(
        scripted_model(*[judge_reply(rubric, 0, suffix="add the cup"), judge_reply(rubric, 1)]),
        output_type=str,
    )

    async def ask_reviewer(ctx: RunContext[None], keyframe_path: str) -> Verdict | str:
        """Ask the reviewer agent for a verdict on one keyframe."""
        tracer.emit("tool_call", "ask_reviewer", call_id=ctx.tool_call_id, asset_ref=keyframe_path)
        verdict = await judge_keyframe(
            reviewer, rubric, brief, Path(keyframe_path), tracer=tracer, attempt=None, price=None
        )
        return verdict or "unparseable"

    planner = Agent(
        scripted_model(
            *[call("ask_reviewer", keyframe_path=str(k)) for k in paths], "Use the second keyframe."
        ),
        tools=[ask_reviewer],
    )
    await run_traced(planner, "Pick the keyframe that meets the brief.", tracer=tracer)
    events = tracer.events()
    print(
        f"Two agents: {sum(e.kind == 'model_request' for e in events)} planner requests, "
        f"{sum(e.kind == 'tool_call' for e in events)} handoffs, "
        f"{sum(e.kind == 'judge' for e in events)} judge calls. Trace: {repo_path(tracer.path)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(
    fake: bool = typer.Option(False, help="Scripted models, placeholder GPU tools. Offline."),
    brief: Path = ASSETS / "brief.yaml",
    rubric: Path = ASSETS / "rubric.yaml",
    fixture: Path | None = typer.Option(None, help="A keyframe to judge as attempt 1"),
    fault: str = typer.Option(
        "none", help="--fake only: always-retry, weak, tight-budget, unparseable, unchanged-retry"
    ),
    two_agents_demo: bool = typer.Option(False, "--two-agents", help="Scripted handoff demo"),
    approve_video: bool = typer.Option(False, help="Ask a person before the video"),
    budget_usd: float = typer.Option(2.0, help="GPU budget for this run"),
    image_gpu_price: float | None = typer.Option(None, help="USD/h of the Sana Endpoint GPU"),
    video_gpu_price: float | None = typer.Option(None, help="USD/h of the Wan Endpoint GPU"),
    keyframe_s: float = typer.Option(10.0, help="Estimated keyframe t_gen for the guard [measure]"),
    video_s: float = typer.Option(600.0, help="Estimated video t_gen for the guard [measure]"),
    llm_price_in: float | None = typer.Option(None, help="USD per 1M input tokens"),
    llm_price_out: float | None = typer.Option(None, help="USD per 1M output tokens"),
) -> None:
    """Run the planner, generator and reviewer loop on one brief."""
    the_brief, the_rubric = Brief.from_yaml(brief), Rubric.from_yaml(rubric)
    if two_agents_demo:
        fixtures = ASSETS / "fixtures"
        asyncio.run(
            two_agents(
                the_brief,
                the_rubric,
                [fixtures / "bad_keyframe.png", fixtures / "good_keyframe.png"],
            )
        )
        return
    price = LLMPrice(input_per_m=llm_price_in, output_per_m=llm_price_out) if llm_price_in else None
    asyncio.run(
        review(
            the_brief,
            the_rubric,
            fake=fake,
            fault=fault,
            fixture=fixture,
            approve_video=approve_video,
            budget_usd=budget_usd,
            image_gpu_price=image_gpu_price,
            video_gpu_price=video_gpu_price,
            keyframe_s=keyframe_s,
            video_s=video_s,
            price=price,
        )
    )


if __name__ == "__main__":
    typer.run(main)
