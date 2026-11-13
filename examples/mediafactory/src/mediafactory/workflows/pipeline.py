"""Single-record planner -> generator -> judge loop (`SPEC.md` §4, §7.3, §8.2, §8.3)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mediafactory.config import Settings
from mediafactory.core.cost import Budget, BudgetExceeded, marginal_cost_usd
from mediafactory.core.events import EventLogger
from mediafactory.core.result import ResultReport
from mediafactory.core.spec import GenerationSpec, InputImage, Pricing
from mediafactory.core.storage import staged_write
from mediafactory.profile.application import ApplicationBundle, StageName
from mediafactory.profile.records import Brief
from mediafactory.worker.adapters import FakeAdapter
from mediafactory.worker.runner import ExecutionMode, run_single
from mediafactory.workflows.tickets import (
    AsyncWorkloadProvider,
    InMemoryWorkloadProvider,
    reconcile_or_submit,
)


def deterministic_seed(app_name: str, record_id: str, role: str, attempt: int = 1) -> int:
    """Stable per-record seed (`SPEC.md` §8): retries change seed; resumes repeat it."""
    raw = f"{app_name}:{record_id}:{role}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def build_role_spec(
    bundle: ApplicationBundle,
    settings: Settings,
    brief: Brief,
    copy_data: dict[str, str],
    *,
    role: str,
    run_id: str,
    attempt: int = 1,
    input_image: InputImage | None = None,
    preemptible: bool = False,
) -> GenerationSpec:
    """Build a closed `GenerationSpec` for ``role`` from ``brief`` and ``copy_data``."""
    rec_id = str(brief.record["id"])
    _, style_preset = bundle.config.style_for_record(brief.record)
    role_cfg = settings.role(role)
    platform = role_cfg.compute.platform if role_cfg.compute else "gpu-l40s-a"
    preset = role_cfg.compute.preset if role_cfg.compute else "1gpu-8vcpu-32gb"
    pricing = Pricing(mode="preemptible" if preemptible else "on-demand")
    seed = deterministic_seed(bundle.config.name, rec_id, role, attempt)
    output_prefix = f"runs/{run_id}/items/{rec_id}/{role}"
    brief_ref = f"runs/{run_id}/items/{rec_id}/brief.json"

    if role == "image":
        return GenerationSpec(
            role="image",
            model=role_cfg.model,
            model_revision=role_cfg.model_revision,
            lora_adapter=style_preset.lora,
            prompt=copy_data["image_prompt"],
            negative_prompt=style_preset.negative or None,
            brief_ref=brief_ref,
            record_id=rec_id,
            attempt=attempt,
            seed=seed,
            size=bundle.config.image_size(),
            steps=bundle.config.constraints.image.steps,
            guidance=bundle.config.constraints.image.guidance,
            platform=platform,
            preset=preset,
            pricing=pricing,
            output_prefix=output_prefix,
        )

    if role == "speech":
        return GenerationSpec(
            role="speech",
            model=role_cfg.model,
            model_revision=role_cfg.model_revision,
            prompt=copy_data["speech_text"],
            voice=style_preset.voice or "af_heart",
            brief_ref=brief_ref,
            record_id=rec_id,
            attempt=attempt,
            seed=seed,
            duration_s=bundle.config.constraints.speech.duration_s,
            platform=platform,
            preset=preset,
            pricing=pricing,
            output_prefix=output_prefix,
        )

    if role == "video":
        v_cfg = bundle.config.constraints.video
        return GenerationSpec(
            role="video",
            model=role_cfg.model,
            model_revision=role_cfg.model_revision,
            prompt=copy_data["motion_prompt"],
            negative_prompt=style_preset.negative or None,
            input_image=input_image,
            brief_ref=brief_ref,
            record_id=rec_id,
            attempt=attempt,
            seed=seed,
            size=bundle.config.video_size(),
            steps=v_cfg.steps,
            guidance=v_cfg.guidance,
            duration_s=v_cfg.duration_s,
            fps=v_cfg.fps,
            frames=v_cfg.frames,
            platform=platform,
            preset=preset,
            pricing=pricing,
            output_prefix=output_prefix,
        )

    raise ValueError(f"unsupported stage role: {role}")


@dataclass
class ItemLedger:
    record_id: str
    status: str = "pending"
    stages_requested: list[StageName] = field(default_factory=list)
    stages_completed: list[str] = field(default_factory=list)
    image_attempts: int = 0
    video_attempts: int = 0
    verdict: str | None = None
    score: int | None = None
    rubric_version: str | None = None
    judge_model: str | None = None
    reasons: list[str] = field(default_factory=list)
    marginal_gpu_usd: float = 0.0
    llm_and_judge_usd: float = 0.0
    regenerated_on_resume: bool = False
    tickets_reconciled: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "status": self.status,
            "stages_requested": self.stages_requested,
            "stages_completed": self.stages_completed,
            "image_attempts": self.image_attempts,
            "video_attempts": self.video_attempts,
            "verdict": self.verdict,
            "score": self.score,
            "rubric_version": self.rubric_version,
            "judge_model": self.judge_model,
            "reasons": self.reasons,
            "marginal_gpu_usd": round(self.marginal_gpu_usd, 6),
            "llm_and_judge_usd": round(self.llm_and_judge_usd, 6),
            "total_marginal_usd": round(self.marginal_gpu_usd + self.llm_and_judge_usd, 6),
            "regenerated_on_resume": self.regenerated_on_resume,
            "tickets_reconciled": self.tickets_reconciled,
            "error": self.error,
        }


def run_record(
    bundle: ApplicationBundle,
    settings: Settings,
    brief: Brief,
    *,
    root: str | Path,
    run_id: str,
    mode: ExecutionMode = "fake",
    budget: Budget,
    event_logger: EventLogger,
    adapter: FakeAdapter | None = None,
    provider: AsyncWorkloadProvider | None = None,
    verify_speech: bool = False,
    force_first_verdict: str | None = None,
    unparseable_judge_replies_before_valid: int = 0,
) -> ItemLedger:
    """Run one record through its active stages (`plan -> image -> judge -> speech -> video`)."""
    root_path = Path(root)
    rec_id = str(brief.record["id"])
    item_dir = root_path / "runs" / run_id / "items" / rec_id
    item_dir.mkdir(parents=True, exist_ok=True)
    fake_adapter = adapter or FakeAdapter()
    active_provider = provider or InMemoryWorkloadProvider()

    # Check if already completed (for --resume)
    item_json_path = item_dir / "item.json"
    if item_json_path.is_file():
        existing_item = json.loads(item_json_path.read_text(encoding="utf-8"))
        if existing_item.get("status") == "completed":
            ledger = ItemLedger(
                record_id=rec_id,
                status="completed",
                stages_requested=list(brief.stages),
                stages_completed=list(existing_item.get("stages_completed", [])),
                image_attempts=int(existing_item.get("image_attempts", 1)),
                video_attempts=int(existing_item.get("video_attempts", 0)),
                verdict=existing_item.get("verdict"),
                score=existing_item.get("score"),
                rubric_version=existing_item.get("rubric_version"),
                judge_model=existing_item.get("judge_model"),
                reasons=list(existing_item.get("reasons", [])),
                marginal_gpu_usd=float(existing_item.get("marginal_gpu_usd", 0.0)),
                llm_and_judge_usd=float(existing_item.get("llm_and_judge_usd", 0.0)),
                regenerated_on_resume=False,
            )
            return ledger

    staged_write(
        item_dir / "brief.json",
        json.dumps(brief.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )

    ledger = ItemLedger(record_id=rec_id, stages_requested=list(brief.stages))
    stage_map = bundle.config.stage_map()
    llm_cfg = settings.role("llm")
    judge_cfg = settings.role("judge")

    try:
        # 1. Plan + Image + Judge loop
        copy_data: dict[str, str] = {}
        fix_text = ""
        winning_image_report: ResultReport | None = None
        max_img_attempts = settings.policy.max_image_attempts if "image" in stage_map else 1

        for attempt in range(1, max_img_attempts + 1):
            # Plan step
            res_llm = budget.reserve(rec_id, 0.002)
            copy_data, plan_s, plan_cost = fake_adapter.plan(
                bundle,
                brief.record,
                fix=fix_text,
                price_in_per_1m_usd=llm_cfg.price_in_per_1m_usd,
                price_out_per_1m_usd=llm_cfg.price_out_per_1m_usd,
            )
            budget.commit(rec_id, res_llm, plan_cost)
            ledger.llm_and_judge_usd += plan_cost
            plan_ev = event_logger.emit(
                kind="llm",
                name="plan",
                record_id=rec_id,
                workload="token_factory",
                attempt=attempt,
                status="ok",
                t_gen_s=plan_s,
                cost_usd=plan_cost,
            )
            staged_write(
                item_dir / "copy.json",
                json.dumps(copy_data, indent=2, sort_keys=True) + "\n",
            )
            if "plan" in stage_map and "plan" not in ledger.stages_completed:
                ledger.stages_completed.append("plan")

            if "image" not in stage_map:
                break

            img_spec = build_role_spec(
                bundle,
                settings,
                brief,
                copy_data,
                role="image",
                run_id=run_id,
                attempt=attempt,
            )
            est_img_usd = marginal_cost_usd(
                img_spec.platform,
                img_spec.preset,
                settings.estimate_role_seconds("image"),
            )
            res_img = budget.reserve(rec_id, est_img_usd)
            img_report = run_single(
                img_spec,
                root=root_path,
                mode=mode,
                adapter=fake_adapter,
                event_logger=event_logger,
                parent_event_id=plan_ev.event_id,
            )
            actual_img_usd = img_report.cost.marginal_compute_usd
            budget.commit(rec_id, res_img, actual_img_usd)
            ledger.marginal_gpu_usd += actual_img_usd
            ledger.image_attempts = attempt

            if not stage_map["image"].judge:
                winning_image_report = img_report
                ledger.verdict = "PASS"
                ledger.stages_completed.append("image")
                break

            # Judge step (with retry on unparseable replies per FR-13)
            img_bytes = (root_path / img_spec.asset_key()).read_bytes()
            for parse_try in range(unparseable_judge_replies_before_valid):
                res_j = budget.reserve(rec_id, 0.001)
                parse_cost = 0.0001
                budget.commit(rec_id, res_j, parse_cost)
                ledger.llm_and_judge_usd += parse_cost
                event_logger.emit(
                    kind="judge",
                    name="judge",
                    record_id=rec_id,
                    parent_event_id=plan_ev.event_id,
                    workload="token_factory",
                    attempt=attempt,
                    status="retry",
                    t_gen_s=0.01,
                    cost_usd=parse_cost,
                    rubric_version=bundle.rubric.rubric_version,
                    judge_model=judge_cfg.model,
                    reasons=[f"unparseable judge reply #{parse_try + 1}; re-querying judge"],
                )

            res_j = budget.reserve(rec_id, 0.001)
            forced = force_first_verdict if attempt == 1 else None
            verdict_obj, judge_s, judge_cost = fake_adapter.judge_image(
                record_id=rec_id,
                attempt=attempt,
                image_bytes=img_bytes,
                rubric=bundle.rubric,
                judge_model=judge_cfg.model,
                force_verdict=forced,
            )
            budget.commit(rec_id, res_j, judge_cost)
            ledger.llm_and_judge_usd += judge_cost
            ledger.verdict = verdict_obj.verdict
            ledger.score = verdict_obj.score
            ledger.rubric_version = verdict_obj.rubric_version
            ledger.judge_model = verdict_obj.judge_model
            ledger.reasons = list(verdict_obj.reasons)

            staged_write(
                item_dir / "image" / "verdict.json",
                json.dumps(verdict_obj.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            )
            event_logger.emit(
                kind="judge",
                name="judge",
                record_id=rec_id,
                parent_event_id=plan_ev.event_id,
                workload="token_factory",
                attempt=attempt,
                status="ok" if verdict_obj.verdict == "PASS" else "retry",
                verdict=verdict_obj.verdict,
                t_gen_s=judge_s,
                cost_usd=judge_cost,
                rubric_version=verdict_obj.rubric_version,
                judge_model=verdict_obj.judge_model,
                score=verdict_obj.score,
                reasons=verdict_obj.reasons,
            )

            if verdict_obj.verdict == "PASS":
                winning_image_report = img_report
                ledger.stages_completed.append("image")
                break
            if verdict_obj.verdict == "WEAK":
                ledger.status = "rejected_weak"
                staged_write(
                    item_json_path,
                    json.dumps(ledger.to_dict(), indent=2, sort_keys=True) + "\n",
                )
                return ledger
            fix_text = verdict_obj.fix

        if "image" in stage_map and winning_image_report is None:
            ledger.status = "rejected_retry_exhausted"
            staged_write(
                item_json_path,
                json.dumps(ledger.to_dict(), indent=2, sort_keys=True) + "\n",
            )
            return ledger

        # 2. Speech stage
        if "speech" in brief.stages:
            speech_stage = stage_map["speech"]
            if not speech_stage.gate_on_image_pass or ledger.verdict in ("PASS", None):
                sp_spec = build_role_spec(
                    bundle,
                    settings,
                    brief,
                    copy_data,
                    role="speech",
                    run_id=run_id,
                    attempt=1,
                )
                est_sp_usd = marginal_cost_usd(
                    sp_spec.platform,
                    sp_spec.preset,
                    settings.estimate_role_seconds("speech"),
                )
                res_sp = budget.reserve(rec_id, est_sp_usd)
                sp_report = run_single(
                    sp_spec,
                    root=root_path,
                    mode=mode,
                    adapter=fake_adapter,
                    verify_speech=verify_speech,
                    event_logger=event_logger,
                )
                sp_cost = sp_report.cost.marginal_compute_usd
                asr_cost = (
                    sp_report.checks.speech_check.cost_usd
                    if sp_report.checks and sp_report.checks.speech_check
                    else 0.0
                )
                budget.commit(rec_id, res_sp, sp_cost + asr_cost)
                ledger.marginal_gpu_usd += sp_cost
                ledger.llm_and_judge_usd += asr_cost
                ledger.stages_completed.append("speech")

        # 3. Video stage (gated on image PASS when from == "image", ticketed async call)
        if "video" in brief.stages:
            vid_stage = stage_map["video"]
            from_src = vid_stage.from_source or "image"
            if from_src == "image" and ledger.verdict != "PASS":
                ledger.status = "rejected_before_video"
                staged_write(
                    item_json_path,
                    json.dumps(ledger.to_dict(), indent=2, sort_keys=True) + "\n",
                )
                return ledger

            keyframe_input: InputImage | None = None
            if from_src == "image":
                if winning_image_report is None or winning_image_report.asset is None:
                    raise RuntimeError("video stage requires a passing image asset")
                keyframe_input = InputImage(
                    uri=winning_image_report.asset.key,
                    sha256=winning_image_report.asset.sha256,
                    mime="image/png",
                )

            vid_spec = build_role_spec(
                bundle,
                settings,
                brief,
                copy_data,
                role="video",
                run_id=run_id,
                attempt=1,
                input_image=keyframe_input,
            )
            ticket_path = item_dir / "tickets" / f"video-{vid_spec.spec_hash()[:12]}.json"
            ticket, was_reconciled = reconcile_or_submit(
                ticket_path,
                call_id=f"video-{rec_id}",
                spec_hash=vid_spec.spec_hash(),
                run_id=run_id,
                role="video",
                provider=active_provider,
                record_id=rec_id,
            )
            if was_reconciled:
                ledger.tickets_reconciled += 1
            event_logger.emit(
                kind="ticket",
                name="video_ticket",
                record_id=rec_id,
                call_id=vid_spec.spec_hash(),
                workload="endpoint",
                workload_id=ticket.workload_id,
                status="cache_hit" if was_reconciled else "ok",
                t_gen_s=0.0,
                cost_usd=0.0,
            )

            est_vid_usd = marginal_cost_usd(
                vid_spec.platform,
                vid_spec.preset,
                settings.estimate_role_seconds("video"),
            )
            res_vid = budget.reserve(rec_id, est_vid_usd)
            vid_report = run_single(
                vid_spec,
                root=root_path,
                mode=mode,
                adapter=fake_adapter,
                event_logger=event_logger,
            )
            budget.commit(rec_id, res_vid, vid_report.cost.marginal_compute_usd)
            ledger.marginal_gpu_usd += vid_report.cost.marginal_compute_usd
            ledger.video_attempts = 1
            ledger.stages_completed.append("video")

        ledger.status = "completed"
    except BudgetExceeded as exc:
        ledger.status = "refused_budget"
        ledger.error = str(exc)
        event_logger.emit(
            kind="budget",
            name=f"budget_{exc.scope}",
            record_id=rec_id,
            status="refused",
            t_gen_s=0.0,
            cost_usd=0.0,
            detail=str(exc),
        )

    staged_write(
        item_json_path,
        json.dumps(ledger.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    return ledger
