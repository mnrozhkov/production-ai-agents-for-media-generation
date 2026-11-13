"""Batch orchestration, deterministic stage sampling, resume, and `summary.json` (`M3/M4`)."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mediafactory.config import Settings
from mediafactory.core.cost import Budget, BudgetExceeded, hourly_rate, marginal_cost_usd
from mediafactory.core.events import EventLogger
from mediafactory.core.result import ResultReport
from mediafactory.core.spec import GenerationSpec, InputImage
from mediafactory.core.storage import staged_write
from mediafactory.profile.application import ApplicationBundle, StageName
from mediafactory.profile.records import (
    Brief,
    DeadLetterEntry,
    build_brief,
    load_records,
    write_dead_letters,
)
from mediafactory.worker.adapters import FakeAdapter
from mediafactory.worker.runner import ExecutionMode, run_batch_specs
from mediafactory.workflows.pipeline import ItemLedger, build_role_spec
from mediafactory.workflows.tickets import (
    AsyncWorkloadProvider,
    InMemoryWorkloadProvider,
    reconcile_or_submit,
)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac, 4)


@dataclass
class BatchRunSummary:
    run_id: str
    run_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)
    ledgers: list[ItemLedger] = field(default_factory=list)


def partition_into_batches(
    specs: list[GenerationSpec], num_batches: int
) -> list[list[GenerationSpec]]:
    if not specs or num_batches <= 0:
        return []
    k = min(num_batches, len(specs))
    buckets: list[list[GenerationSpec]] = [[] for _ in range(k)]
    for idx, spec in enumerate(specs):
        buckets[idx % k].append(spec)
    return [b for b in buckets if b]


def run_batch(
    bundle: ApplicationBundle,
    settings: Settings,
    *,
    records_path: str | Path | None = None,
    root: str | Path = ".",
    run_id: str = "run-001",
    jobs: int = 10,
    mode: ExecutionMode = "fake",
    resume: bool = False,
    limit: int | None = None,
    stop_after_records: int | None = None,
    verify_speech: bool = True,
    adapter: FakeAdapter | None = None,
    provider: AsyncWorkloadProvider | None = None,
) -> BatchRunSummary:
    """Execute or resume a parallel batch run (`SPEC.md` §7.3, §8.1, §8.2, §8.3)."""
    root_path = Path(root)
    run_dir = root_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = run_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    rec_file = (
        Path(records_path)
        if records_path
        else (bundle.root / bundle.config.records.path).resolve()
    )
    load_res = load_records(
        rec_file, required_columns=bundle.config.records.required, limit=limit
    )

    dead_letter_path = run_dir / "dead_letter.jsonl"
    if not resume and dead_letter_path.exists():
        dead_letter_path.unlink()
    if not resume:
        write_dead_letters(dead_letter_path, load_res.dead_letters)
    elif not dead_letter_path.exists():
        dead_letter_path.touch()

    staged_write(
        run_dir / "application.json",
        json.dumps(
            bundle.snapshot_dict(models_snapshot=settings.file.model_dump(mode="json")),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    valid_ids = [str(r["id"]) for r in load_res.valid_records]
    sampled_stage_ids: dict[StageName, set[str]] = {}
    for stage in bundle.config.stages:
        sampled_stage_ids[stage.name] = bundle.config.sampled_ids_for_stage(
            stage.name, valid_ids
        )

    briefs: list[Brief] = []
    manifest_lines: list[str] = []
    for row in load_res.valid_records:
        b = build_brief(bundle, row, sampled_stage_ids=sampled_stage_ids)
        briefs.append(b)
        manifest_lines.append(json.dumps(b.model_dump(mode="json"), sort_keys=True))
    staged_write(
        run_dir / "manifest.jsonl",
        "\n".join(manifest_lines) + ("\n" if manifest_lines else ""),
    )

    event_logger = EventLogger(run_dir / "events.jsonl", run_id=run_id)
    fake_adapter = adapter or FakeAdapter()
    active_provider = provider or InMemoryWorkloadProvider()
    budget = Budget(
        record_budget_usd=bundle.config.constraints.record_budget_usd,
        run_budget_usd=bundle.config.constraints.run_budget_usd,
    )

    # Track items already completed before this invocation (for resume proof)
    completed_before_resume: set[str] = set()
    regenerated_after_resume = 0
    if resume:
        for b in briefs:
            rec_id = str(b.record["id"])
            item_file = run_dir / "items" / rec_id / "item.json"
            if item_file.is_file():
                data = json.loads(item_file.read_text(encoding="utf-8"))
                if data.get("status") == "completed":
                    completed_before_resume.add(rec_id)

    active_briefs = (
        briefs[:stop_after_records] if stop_after_records is not None else briefs
    )

    stage_map = bundle.config.stage_map()
    llm_cfg = settings.role("llm")
    judge_cfg = settings.role("judge")

    ledgers_by_id: dict[str, ItemLedger] = {}
    copy_by_id: dict[str, dict[str, str]] = {}

    # Load already-completed items on resume without touching them
    pending_briefs: list[Brief] = []
    for b in active_briefs:
        rec_id = str(b.record["id"])
        item_dir = run_dir / "items" / rec_id
        item_file = item_dir / "item.json"
        if resume and rec_id in completed_before_resume and item_file.is_file():
            existing = json.loads(item_file.read_text(encoding="utf-8"))
            ledgers_by_id[rec_id] = ItemLedger(
                record_id=rec_id,
                status="completed",
                stages_requested=list(b.stages),
                stages_completed=list(existing.get("stages_completed", [])),
                image_attempts=int(existing.get("image_attempts", 1)),
                video_attempts=int(existing.get("video_attempts", 0)),
                verdict=existing.get("verdict"),
                score=existing.get("score"),
                rubric_version=existing.get("rubric_version"),
                judge_model=existing.get("judge_model"),
                reasons=list(existing.get("reasons", [])),
                marginal_gpu_usd=float(existing.get("marginal_gpu_usd", 0.0)),
                llm_and_judge_usd=float(existing.get("llm_and_judge_usd", 0.0)),
            )
            budget.commit(
                rec_id,
                0.0,
                ledgers_by_id[rec_id].marginal_gpu_usd
                + ledgers_by_id[rec_id].llm_and_judge_usd,
            )
            continue
        pending_briefs.append(b)

    # Step 1: Plan copy for pending records
    image_specs: list[GenerationSpec] = []
    for b in pending_briefs:
        rec_id = str(b.record["id"])
        item_dir = run_dir / "items" / rec_id
        item_dir.mkdir(parents=True, exist_ok=True)
        staged_write(
            item_dir / "brief.json",
            json.dumps(b.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        )
        ledger = ItemLedger(record_id=rec_id, stages_requested=list(b.stages))
        ledgers_by_id[rec_id] = ledger
        try:
            res_llm = budget.reserve(rec_id, 0.002)
            copy_data, plan_s, plan_cost = fake_adapter.plan(
                bundle,
                b.record,
                price_in_per_1m_usd=llm_cfg.price_in_per_1m_usd,
                price_out_per_1m_usd=llm_cfg.price_out_per_1m_usd,
            )
            budget.commit(rec_id, res_llm, plan_cost)
            ledger.llm_and_judge_usd += plan_cost
            copy_by_id[rec_id] = copy_data
            staged_write(
                item_dir / "copy.json",
                json.dumps(copy_data, indent=2, sort_keys=True) + "\n",
            )
            event_logger.emit(
                kind="llm",
                name="plan",
                record_id=rec_id,
                workload="token_factory",
                attempt=1,
                status="ok",
                t_gen_s=plan_s,
                cost_usd=plan_cost,
            )
            if "plan" in stage_map:
                ledger.stages_completed.append("plan")
            if "image" in stage_map:
                img_spec = build_role_spec(
                    bundle,
                    settings,
                    b,
                    copy_data,
                    role="image",
                    run_id=run_id,
                    attempt=1,
                    preemptible=True,
                )
                est_img = marginal_cost_usd(
                    img_spec.platform,
                    img_spec.preset,
                    settings.estimate_role_seconds("image"),
                    preemptible=True,
                )
                res_img = budget.reserve(rec_id, est_img)
                budget.release(rec_id, res_img)
                image_specs.append(img_spec)
        except BudgetExceeded as exc:
            ledger.status = "refused_budget"
            ledger.error = str(exc)
            write_dead_letters(
                dead_letter_path,
                [
                    DeadLetterEntry(
                        row_index=0,
                        record_id=rec_id,
                        error_class="BudgetExceeded",
                        reason=str(exc),
                        raw_row=b.record,
                    )
                ],
            )
            event_logger.emit(
                kind="budget",
                name=f"budget_{exc.scope}",
                record_id=rec_id,
                status="refused",
                t_gen_s=0.0,
                cost_usd=0.0,
                detail=str(exc),
            )

    all_batch_metrics: list[dict[str, Any]] = []
    total_jobs_submitted = 0
    peak_parallel = 0
    tickets_reconciled = 0

    def _dispatch_role_batches(role: str, specs: list[GenerationSpec]) -> dict[str, ResultReport]:
        nonlocal total_jobs_submitted, peak_parallel, tickets_reconciled, regenerated_after_resume
        if not specs:
            return {}
        batches = partition_into_batches(specs, jobs)
        peak_parallel = max(peak_parallel, len(batches))
        results_by_record: dict[str, ResultReport] = {}

        for batch_idx, batch_specs in enumerate(batches):
            batch_file = batches_dir / f"batch-{batch_idx:02d}-{role}.jsonl"
            batch_lines = [
                json.dumps(s.model_dump(mode="json"), sort_keys=True) for s in batch_specs
            ]
            staged_write(batch_file, "\n".join(batch_lines) + "\n")
            batch_hash = batch_specs[0].spec_hash()
            ticket_path = batches_dir / f"batch-{batch_idx:02d}-{role}-ticket.json"
            ticket, was_reconciled = reconcile_or_submit(
                ticket_path,
                call_id=f"batch-{batch_idx:02d}-{role}",
                spec_hash=batch_hash,
                run_id=run_id,
                role=role,
                provider=active_provider,
            )
            if was_reconciled:
                tickets_reconciled += 1
            total_jobs_submitted += 1

            metrics_file = batches_dir / f"batch-{batch_idx:02d}-{role}-metrics.json"
            batch_res = run_batch_specs(
                batch_file,
                root=root_path,
                mode=mode,
                adapter=fake_adapter,
                verify_speech=(role == "speech" and verify_speech),
                event_logger=event_logger,
                metrics_out=metrics_file,
                worker_run_id=ticket.workload_id or f"batch-{batch_idx:02d}-{role}",
            )
            if metrics_file.is_file():
                all_batch_metrics.append(json.loads(metrics_file.read_text(encoding="utf-8")))
            for rep in batch_res.results:
                results_by_record[rep.spec.record_id] = rep
                if rep.spec.record_id in completed_before_resume and not rep.cache_hit:
                    regenerated_after_resume += 1
        return results_by_record

    # Step 2: Batch image generation + Judge + retry batch if needed
    image_results = _dispatch_role_batches("image", image_specs)
    winning_images: dict[str, ResultReport] = {}
    retry_image_specs: list[GenerationSpec] = []
    first_attempt_passes = 0
    total_judged_records = 0

    for b in pending_briefs:
        rec_id = str(b.record["id"])
        ledger = ledgers_by_id[rec_id]
        if ledger.status == "refused_budget" or rec_id not in image_results:
            continue
        rep = image_results[rec_id]
        budget.commit(rec_id, 0.0, rep.cost.marginal_compute_usd)
        ledger.marginal_gpu_usd += rep.cost.marginal_compute_usd
        ledger.image_attempts = 1

        if stage_map["image"].judge:
            total_judged_records += 1
            img_bytes = (root_path / rep.spec.asset_key()).read_bytes()
            verdict_obj, judge_s, judge_cost = fake_adapter.judge_image(
                record_id=rec_id,
                attempt=1,
                image_bytes=img_bytes,
                rubric=bundle.rubric,
                judge_model=judge_cfg.model,
            )
            budget.commit(rec_id, 0.0, judge_cost)
            ledger.llm_and_judge_usd += judge_cost
            ledger.verdict = verdict_obj.verdict
            ledger.score = verdict_obj.score
            ledger.rubric_version = verdict_obj.rubric_version
            ledger.judge_model = verdict_obj.judge_model
            ledger.reasons = list(verdict_obj.reasons)
            staged_write(
                run_dir / "items" / rec_id / "image" / "verdict.json",
                json.dumps(verdict_obj.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            )
            event_logger.emit(
                kind="judge",
                name="judge",
                record_id=rec_id,
                workload="token_factory",
                attempt=1,
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
                first_attempt_passes += 1
                winning_images[rec_id] = rep
                ledger.stages_completed.append("image")
            elif verdict_obj.verdict == "RETRY" and settings.policy.max_image_attempts >= 2:
                copy_data, plan_s, plan_cost = fake_adapter.plan(
                    bundle,
                    b.record,
                    fix=verdict_obj.fix,
                    price_in_per_1m_usd=llm_cfg.price_in_per_1m_usd,
                    price_out_per_1m_usd=llm_cfg.price_out_per_1m_usd,
                )
                budget.commit(rec_id, 0.0, plan_cost)
                ledger.llm_and_judge_usd += plan_cost
                copy_by_id[rec_id] = copy_data
                retry_image_specs.append(
                    build_role_spec(
                        bundle,
                        settings,
                        b,
                        copy_data,
                        role="image",
                        run_id=run_id,
                        attempt=2,
                        preemptible=True,
                    )
                )
            else:
                ledger.status = "rejected_weak"
        else:
            winning_images[rec_id] = rep
            ledger.verdict = "PASS"
            ledger.stages_completed.append("image")

    if retry_image_specs:
        retry_results = _dispatch_role_batches("image-retry", retry_image_specs)
        for rec_id, rep in retry_results.items():
            ledger = ledgers_by_id[rec_id]
            budget.commit(rec_id, 0.0, rep.cost.marginal_compute_usd)
            ledger.marginal_gpu_usd += rep.cost.marginal_compute_usd
            ledger.image_attempts = 2
            img_bytes = (root_path / rep.spec.asset_key()).read_bytes()
            verdict_obj, judge_s, judge_cost = fake_adapter.judge_image(
                record_id=rec_id,
                attempt=2,
                image_bytes=img_bytes,
                rubric=bundle.rubric,
                judge_model=judge_cfg.model,
            )
            budget.commit(rec_id, 0.0, judge_cost)
            ledger.llm_and_judge_usd += judge_cost
            ledger.verdict = verdict_obj.verdict
            ledger.score = verdict_obj.score
            staged_write(
                run_dir / "items" / rec_id / "image" / "verdict.json",
                json.dumps(verdict_obj.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            )
            if verdict_obj.verdict == "PASS":
                winning_images[rec_id] = rep
                ledger.stages_completed.append("image")
            else:
                ledger.status = "rejected_weak"

    # Step 3: Speech batches for records that passed image
    speech_specs: list[GenerationSpec] = []
    for b in pending_briefs:
        rec_id = str(b.record["id"])
        ledger = ledgers_by_id[rec_id]
        if "speech" in b.stages and ledger.verdict == "PASS":
            speech_specs.append(
                build_role_spec(
                    bundle,
                    settings,
                    b,
                    copy_by_id[rec_id],
                    role="speech",
                    run_id=run_id,
                    attempt=1,
                    preemptible=True,
                )
            )
    speech_results = _dispatch_role_batches("speech", speech_specs)
    for rec_id, rep in speech_results.items():
        ledger = ledgers_by_id[rec_id]
        sp_cost = rep.cost.marginal_compute_usd
        asr_cost = (
            rep.checks.speech_check.cost_usd
            if rep.checks and rep.checks.speech_check
            else 0.0
        )
        budget.commit(rec_id, 0.0, sp_cost + asr_cost)
        ledger.marginal_gpu_usd += sp_cost
        ledger.llm_and_judge_usd += asr_cost
        ledger.stages_completed.append("speech")

    # Step 4: Video batches for sampled records that passed image
    video_specs: list[GenerationSpec] = []
    for b in pending_briefs:
        rec_id = str(b.record["id"])
        ledger = ledgers_by_id[rec_id]
        if "video" in b.stages and ledger.verdict == "PASS" and rec_id in winning_images:
            win_img = winning_images[rec_id]
            if win_img.asset is not None:
                video_specs.append(
                    build_role_spec(
                        bundle,
                        settings,
                        b,
                        copy_by_id[rec_id],
                        role="video",
                        run_id=run_id,
                        attempt=1,
                        input_image=InputImage(
                            uri=win_img.asset.key,
                            sha256=win_img.asset.sha256,
                            mime="image/png",
                        ),
                        preemptible=True,
                    )
                )
    video_results = _dispatch_role_batches("video", video_specs)
    for rec_id, rep in video_results.items():
        ledger = ledgers_by_id[rec_id]
        budget.commit(rec_id, 0.0, rep.cost.marginal_compute_usd)
        ledger.marginal_gpu_usd += rep.cost.marginal_compute_usd
        ledger.video_attempts = 1
        ledger.stages_completed.append("video")

    # Finalize per-item ledgers
    for b in pending_briefs:
        rec_id = str(b.record["id"])
        ledger = ledgers_by_id[rec_id]
        if ledger.status == "pending":
            ledger.status = "completed" if ledger.verdict == "PASS" else "rejected_weak"
        staged_write(
            run_dir / "items" / rec_id / "item.json",
            json.dumps(ledger.to_dict(), indent=2, sort_keys=True) + "\n",
        )

    # Aggregate run-level metrics.json and summary.json (including existing batch metrics on resume)
    all_batch_files = sorted(batches_dir.glob("*-metrics.json"))
    merged_batch_metrics = [
        json.loads(p.read_text(encoding="utf-8")) for p in all_batch_files
    ]
    summary_dict, run_metrics_dict = _build_run_reports(
        bundle=bundle,
        run_id=run_id,
        mode=mode,
        jobs_requested=jobs,
        peak_parallel=peak_parallel or min(jobs, max(len(briefs), 1)),
        total_jobs_submitted=len(merged_batch_metrics),
        total_records=len(load_res.valid_records) + len(load_res.dead_letters),
        valid_records=len(load_res.valid_records),
        dead_letters_count=len(load_res.dead_letters),
        video_sampled_count=len(sampled_stage_ids.get("video", set())),
        ledgers=list(ledgers_by_id.values()),
        batch_metrics=merged_batch_metrics,
        events=event_logger.read_all(),
        items_completed_before_resume=len(completed_before_resume),
        items_regenerated_after_resume=regenerated_after_resume,
        tickets_reconciled=tickets_reconciled,
    )

    staged_write(
        run_dir / "metrics.json",
        json.dumps(run_metrics_dict, indent=2, sort_keys=True) + "\n",
    )
    staged_write(
        run_dir / "summary.json",
        json.dumps(summary_dict, indent=2, sort_keys=True) + "\n",
    )

    return BatchRunSummary(
        run_id=run_id,
        run_dir=run_dir,
        summary=summary_dict,
        ledgers=list(ledgers_by_id.values()),
    )


def _build_run_reports(
    *,
    bundle: ApplicationBundle,
    run_id: str,
    mode: ExecutionMode,
    jobs_requested: int,
    peak_parallel: int,
    total_jobs_submitted: int,
    total_records: int,
    valid_records: int,
    dead_letters_count: int,
    video_sampled_count: int,
    ledgers: list[ItemLedger],
    batch_metrics: list[dict[str, Any]],
    events: list[Any],
    items_completed_before_resume: int,
    items_regenerated_after_resume: int,
    tickets_reconciled: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    is_fake = mode == "fake"
    completed_accepted = sum(1 for item in ledgers if item.status == "completed")
    rejected_weak = sum(1 for item in ledgers if item.status.startswith("rejected"))

    marginal_gpu_usd = round(sum(item.marginal_gpu_usd for item in ledgers), 6)
    llm_and_judge_usd = round(sum(item.llm_and_judge_usd for item in ledgers), 6)
    fixed_gpu_overhead_usd = round(
        sum(float(m.get("fixed_overhead_usd") or 0.0) for m in batch_metrics), 6
    )
    total_run_usd = round(marginal_gpu_usd + fixed_gpu_overhead_usd + llm_and_judge_usd, 6)

    observed_wall_proxy_usd = 0.0
    saved_vs_on_demand_usd = 0.0
    all_latencies: list[float] = []
    for m in batch_metrics:
        plat = str(m.get("platform") or "gpu-l40s-a")
        pres = str(m.get("preset") or "1gpu-8vcpu-32gb")
        preemptible = m.get("pricing_mode") == "preemptible"
        billable_s = float(m.get("billable_s") or 0.0)
        rate = hourly_rate(plat, pres, preemptible=preemptible)
        observed_wall_proxy_usd += rate.usd_per_hour * (billable_s / 3600.0)
        if preemptible:
            saved_vs_on_demand_usd += (
                rate.on_demand_usd_per_hour - rate.usd_per_hour
            ) * (billable_s / 3600.0)
        all_latencies.extend(float(x) for x in (m.get("latency_s") or []))

    # Per-role latency distributions and rejected GPU share from trace events
    role_latencies: dict[str, list[float]] = {"image": [], "speech": [], "video": []}
    total_gpu_gen_s = 0.0
    rejected_gpu_gen_s = 0.0
    for ev in events:
        if ev.kind == "gpu_tool" and ev.t_gen_s is not None and ev.status != "cache_hit":
            role_latencies.setdefault(ev.name, []).append(float(ev.t_gen_s))
            total_gpu_gen_s += float(ev.t_gen_s)
            if ev.attempt > 1:
                rejected_gpu_gen_s += float(ev.t_gen_s)

    latency_summary: dict[str, Any] = {}
    for role_name, vals in role_latencies.items():
        latency_summary[role_name] = {
            "p50": _percentile(vals, 50),
            "p95": _percentile(vals, 95),
            "median": round(statistics.median(vals), 4) if vals else None,
            "count": len(vals),
        }

    judged = [item for item in ledgers if item.verdict is not None]
    pass_count = sum(1 for item in judged if item.verdict == "PASS")
    first_pass_count = sum(
        1 for item in judged if item.verdict == "PASS" and item.image_attempts == 1
    )
    judge_pass_rate = round(pass_count / len(judged), 4) if judged else 1.0
    first_attempt_pass_rate = round(first_pass_count / len(judged), 4) if judged else 1.0

    cost_per_accepted = (
        round(total_run_usd / completed_accepted, 6) if completed_accepted > 0 else None
    )
    rejected_share = (
        round(rejected_gpu_gen_s / total_gpu_gen_s, 4) if total_gpu_gen_s > 0 else 0.0
    )

    summary_payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "application": bundle.config.name,
        "fake": is_fake,
        "records": {
            "total": total_records,
            "valid": valid_records,
            "completed_accepted": completed_accepted,
            "rejected_weak": rejected_weak,
            "dead_lettered": dead_letters_count,
            "video_sampled": video_sampled_count,
        },
        "jobs": {
            "requested_parallel": jobs_requested,
            "peak_parallel": peak_parallel,
            "total_submitted": total_jobs_submitted,
            "preempted_or_killed": 0,
        },
        "cost": {
            "marginal_gpu_usd": marginal_gpu_usd,
            "fixed_gpu_overhead_usd": fixed_gpu_overhead_usd,
            "llm_and_judge_usd": llm_and_judge_usd,
            "total_run_usd": total_run_usd,
            "observed_wall_proxy_usd": round(observed_wall_proxy_usd, 6),
            "cost_per_accepted_asset_usd": cost_per_accepted,
            "rejected_gpu_share": rejected_share,
            "saved_vs_on_demand_usd": round(saved_vs_on_demand_usd, 6),
        },
        "latency_s": latency_summary,
        "quality": {
            "judge_pass_rate": judge_pass_rate,
            "first_attempt_pass_rate": first_attempt_pass_rate,
            "rubric_version": bundle.rubric.rubric_version,
        },
        "resume_proof": {
            "items_completed_before_resume": items_completed_before_resume,
            "items_regenerated_after_resume": items_regenerated_after_resume,
            "tickets_reconciled": tickets_reconciled,
        },
    }

    run_metrics_payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "fake": is_fake,
        "status": "completed",
        "batches_count": len(batch_metrics),
        "models_loaded_total": sum(int(m.get("models_loaded") or 0) for m in batch_metrics),
        "startup_s": round(sum(float(m.get("startup_s") or 0.0) for m in batch_metrics), 6),
        "load_s": round(sum(float(m.get("load_s") or 0.0) for m in batch_metrics), 6),
        "latency_s": all_latencies,
        "billable_s": round(sum(float(m.get("billable_s") or 0.0) for m in batch_metrics), 6),
        "fixed_overhead_usd": fixed_gpu_overhead_usd,
    }

    return summary_payload, run_metrics_payload
