"""Unified single-spec and model-reusing batch worker (`SPEC.md` §5, §8.1, `FR-07`, `FR-14`).

Can also be invoked directly as the Nebius GPU Job entrypoint:
    python -m mediafactory.worker.runner --spec /data/spec.json
    python -m mediafactory.worker.runner --batch /data/batches/batch-00-image.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mediafactory.core.cost import (
    cost_basis,
    fixed_overhead_usd,
    hourly_rate,
    marginal_cost_usd,
)
from mediafactory.core.events import EventLogger
from mediafactory.core.result import CostReport, Measurements, ModelRef, ResultReport
from mediafactory.core.spec import GenerationSpec
from mediafactory.core.storage import ContentCache, staged_write
from mediafactory.worker.adapters import FakeAdapter, LoadedModel, ModelLoader, ModelRenderer
from mediafactory.worker.checks import build_check_report, serialize_transcript

ExecutionMode = Literal["dry_run", "fake", "local", "cloud"]


@dataclass(frozen=True)
class JobRequest:
    """Resolved `nebius ai job create` command and cost cap for dry-run or live submission."""

    name: str
    platform: str
    preset: str
    pricing_mode: str
    timeout_min: int
    estimated_cap_usd: float
    argv: list[str]


@dataclass
class BatchWorkerResult:
    worker_run_id: str
    results: list[ResultReport] = field(default_factory=list)
    metrics_path: Path | None = None
    models_loaded: int = 0
    items_generated: int = 0
    items_cached: int = 0


def require_real_local_runtime() -> None:
    """Enforce `FR-14`: `--local` must run real local inference or fail clearly if unavailable."""
    missing = [
        pkg
        for pkg in ("torch", "diffusers")
        if importlib.util.find_spec(pkg) is None
    ]
    if missing:
        raise RuntimeError(
            f"--local requires local GPU/model packages {missing} which are not installed. "
            "Install the `gpu` extra (`uv sync --extra gpu`) on a supported host, or use "
            "`--fake` for offline CI/walkthroughs."
        )


def build_job_request(
    *,
    target_path: str | Path,
    is_batch: bool,
    platform: str,
    preset: str,
    preemptible: bool = True,
    timeout_min: int = 60,
    image: str = "cr.eu-north1.nebius.cloud/mediafactory/worker:latest",
    mount_path: str = "/data",
    run_id: str = "dryrun",
) -> JobRequest:
    """Build the resolved `nebius ai job create` argv and cost ceiling (`FR-06`, `FR-14`)."""
    rate = hourly_rate(platform, preset, preemptible=preemptible)
    cap_usd = round(rate.usd_per_hour * (timeout_min / 60.0), 4)
    flag = "--batch" if is_batch else "--spec"
    job_name = f"mf-{run_id[:16]}-{Path(target_path).stem[:16]}"
    argv = [
        "nebius",
        "ai",
        "job",
        "create",
        "--name",
        job_name,
        "--image",
        image,
        "--platform",
        platform,
        "--preset",
        preset,
        "--timeout",
        f"{timeout_min}m",
        "--mount",
        f"{mount_path}",
    ]
    if preemptible:
        argv.extend(["--preemptible", "--follows-spot-price"])
    argv.extend(
        [
            "--command",
            f"python -m mediafactory.worker.runner {flag} {target_path}",
        ]
    )
    return JobRequest(
        name=job_name,
        platform=platform,
        preset=preset,
        pricing_mode="preemptible" if preemptible else "on-demand",
        timeout_min=timeout_min,
        estimated_cap_usd=cap_usd,
        argv=argv,
    )


def _execute_spec(
    spec: GenerationSpec,
    *,
    root: Path,
    cache: ContentCache,
    loaded: LoadedModel,
    renderer: ModelRenderer,
    worker_run_id: str,
    fake: bool,
    record_load_in_item: bool,
    startup_s: float | None = None,
    verify_speech: bool = False,
    event_logger: EventLogger | None = None,
    parent_event_id: str | None = None,
) -> ResultReport:
    """Render one `GenerationSpec` with an already-loaded model and write `asset` + `result.json`."""
    asset_bytes, gen_s, peak_vram_gb = renderer(spec, loaded)
    meta, checks = build_check_report(
        spec, asset_bytes, verify_speech=verify_speech
    )
    preemptible = spec.pricing.mode == "preemptible"
    rate = hourly_rate(spec.platform, spec.preset, preemptible=preemptible)
    c_usd = marginal_cost_usd(spec.platform, spec.preset, gen_s, preemptible=preemptible)

    report = ResultReport(
        status="ok" if checks.deterministic.ok else "failed",
        fake=fake,
        cache_hit=False,
        spec_hash=spec.spec_hash(),
        spec=spec,
        worker_run_id=worker_run_id,
        model=ModelRef(repo=spec.model, revision=spec.model_revision, digest=None),
        asset=meta,
        measurements=Measurements(
            startup_s=startup_s if record_load_in_item else None,
            image_pull_s=None,
            load_s=loaded.load_s if record_load_in_item else None,
            gen_s=round(gen_s, 6),
            peak_vram_gb=round(peak_vram_gb, 3),
            gpu=loaded.gpu_name,
        ),
        cost=CostReport(
            rate_usd_per_hour=rate.usd_per_hour,
            marginal_compute_usd=c_usd,
            pricing_mode=spec.pricing.mode,
            basis=cost_basis(spec.platform, spec.preset, preemptible=preemptible),
        ),
        checks=checks,
        error="; ".join(checks.deterministic.errors) if not checks.deterministic.ok else None,
    )

    asset_dest = root / spec.asset_key()
    result_dest = root / spec.result_key()
    staged_write(asset_dest, asset_bytes)
    staged_write(result_dest, report.to_json())

    if spec.role == "speech" and checks.speech_check is not None:
        transcript_dest = root / spec.output_prefix / "transcript.json"
        staged_write(
            transcript_dest,
            json.dumps(serialize_transcript(checks.speech_check), indent=2, sort_keys=True) + "\n",
        )

    if report.status == "ok":
        cache.store(report, asset_bytes)

    if event_logger is not None:
        gpu_ev = event_logger.emit(
            kind="gpu_tool",
            name=spec.role,
            record_id=spec.record_id,
            parent_event_id=parent_event_id,
            call_id=report.spec_hash,
            workload="job",
            workload_id=worker_run_id,
            attempt=spec.attempt,
            status="ok" if report.status == "ok" else "failed",
            t_gen_s=round(gen_s, 6),
            cost_usd=c_usd,
        )
        event_logger.emit(
            kind="check",
            name="deterministic_check",
            record_id=spec.record_id,
            parent_event_id=gpu_ev.event_id,
            call_id=report.spec_hash,
            attempt=spec.attempt,
            status="ok" if checks.deterministic.ok else "failed",
            t_gen_s=0.001,
            cost_usd=0.0,
        )
        if checks.speech_check is not None:
            event_logger.emit(
                kind="check",
                name="asr_check",
                record_id=spec.record_id,
                parent_event_id=gpu_ev.event_id,
                call_id=report.spec_hash,
                attempt=spec.attempt,
                status="ok" if checks.speech_check.ok else "failed",
                t_gen_s=0.25,
                cost_usd=checks.speech_check.cost_usd,
            )

    return report


def run_single(
    spec: GenerationSpec,
    *,
    root: str | Path = ".",
    mode: ExecutionMode = "fake",
    adapter: FakeAdapter | None = None,
    loader: ModelLoader | None = None,
    renderer: ModelRenderer | None = None,
    startup_s: float | None = None,
    verify_speech: bool = False,
    event_logger: EventLogger | None = None,
    parent_event_id: str | None = None,
    metrics_out: str | Path | None = None,
) -> ResultReport:
    """Execute a single `GenerationSpec` (`SPEC.md` §5, §7.1, §7.2)."""
    if mode == "local" and loader is None and renderer is None:
        require_real_local_runtime()

    root_path = Path(root)
    cache = ContentCache(root_path)
    worker_run_id = f"worker-{uuid.uuid4().hex[:10]}"

    hit = cache.lookup(spec)
    if hit is not None:
        cached_report, cached_bytes = hit
        materialized = cache.materialize_hit(spec, cached_report, cached_bytes)
        if event_logger is not None:
            event_logger.emit(
                kind="gpu_tool",
                name=spec.role,
                record_id=spec.record_id,
                parent_event_id=parent_event_id,
                call_id=materialized.spec_hash,
                workload="local" if mode == "fake" else "job",
                workload_id=worker_run_id,
                attempt=spec.attempt,
                status="cache_hit",
                t_gen_s=0.0,
                cost_usd=0.0,
            )
        return materialized

    fake_adapter = adapter or FakeAdapter()
    active_loader = loader or fake_adapter.load_model
    active_renderer = renderer or fake_adapter.render_asset
    is_fake = mode == "fake"

    loaded = active_loader(spec.worker_group_key())
    report = _execute_spec(
        spec,
        root=root_path,
        cache=cache,
        loaded=loaded,
        renderer=active_renderer,
        worker_run_id=worker_run_id,
        fake=is_fake,
        record_load_in_item=True,
        startup_s=startup_s,
        verify_speech=verify_speech,
        event_logger=event_logger,
        parent_event_id=parent_event_id,
    )

    if metrics_out is not None:
        preemptible = spec.pricing.mode == "preemptible"
        f_usd = fixed_overhead_usd(
            spec.platform,
            spec.preset,
            startup_s=startup_s if startup_s is not None else 0.0,
            ready_or_load_s=loaded.load_s,
            preemptible=preemptible,
        )
        metrics_payload = {
            "schema_version": 1,
            "worker_run_id": worker_run_id,
            "mode": mode,
            "fake": is_fake,
            "status": "completed" if report.status == "ok" else "failed",
            "startup_s": startup_s,
            "ready_wait_s": None,
            "load_s": loaded.load_s,
            "latency_s": [report.measurements.gen_s] if report.measurements.gen_s else [],
            "billable_s": round(
                (startup_s or 0.0) + loaded.load_s + (report.measurements.gen_s or 0.0), 6
            ),
            "fixed_overhead_usd": f_usd,
            "platform": spec.platform,
            "preset": spec.preset,
            "pricing_mode": spec.pricing.mode,
            "assets": [
                {"key": report.asset.key, "sha256": report.asset.sha256}
                for _ in [1]
                if report.asset is not None
            ],
        }
        staged_write(
            Path(metrics_out), json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n"
        )

    return report


def load_batch_specs(batch_path: str | Path) -> list[GenerationSpec]:
    path = Path(batch_path)
    if not path.is_file():
        raise FileNotFoundError(f"batch file not found: {path}")
    specs: list[GenerationSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            specs.append(GenerationSpec.from_mapping(json.loads(line)))
    return specs


def run_batch_specs(
    specs_or_path: list[GenerationSpec] | str | Path,
    *,
    root: str | Path = ".",
    mode: ExecutionMode = "fake",
    adapter: FakeAdapter | None = None,
    loader: ModelLoader | None = None,
    renderer: ModelRenderer | None = None,
    startup_s: float = 15.0,
    verify_speech: bool = False,
    event_logger: EventLogger | None = None,
    metrics_out: str | Path | None = None,
    worker_run_id: str | None = None,
) -> BatchWorkerResult:
    """Process a batch of `GenerationSpec` items, loading each unique model **once** (`FR-07`)."""
    if mode == "local" and loader is None and renderer is None:
        require_real_local_runtime()

    specs = (
        load_batch_specs(specs_or_path)
        if isinstance(specs_or_path, (str, Path))
        else list(specs_or_path)
    )
    root_path = Path(root)
    cache = ContentCache(root_path)
    run_id = worker_run_id or f"batch-worker-{uuid.uuid4().hex[:10]}"
    fake_adapter = adapter or FakeAdapter()
    active_loader = loader or fake_adapter.load_model
    active_renderer = renderer or fake_adapter.render_asset
    is_fake = mode == "fake"

    # Preserve input order for results while grouping pending specs by model key
    groups: dict[tuple[str, str, str | None, str | None], list[tuple[int, GenerationSpec]]] = {}
    ordered_results: dict[int, ResultReport] = {}
    items_cached = 0

    for idx, spec in enumerate(specs):
        hit = cache.lookup(spec)
        if hit is not None:
            cached_report, cached_bytes = hit
            materialized = cache.materialize_hit(spec, cached_report, cached_bytes)
            ordered_results[idx] = materialized
            items_cached += 1
            if event_logger is not None:
                event_logger.emit(
                    kind="gpu_tool",
                    name=spec.role,
                    record_id=spec.record_id,
                    call_id=materialized.spec_hash,
                    workload="job",
                    workload_id=run_id,
                    attempt=spec.attempt,
                    status="cache_hit",
                    t_gen_s=0.0,
                    cost_usd=0.0,
                )
        else:
            key = spec.worker_group_key()
            groups.setdefault(key, []).append((idx, spec))

    total_load_s = 0.0
    models_loaded = 0
    items_generated = 0
    latencies: list[float] = []

    for group_key, indexed_specs in groups.items():
        # Load the pipeline ONCE for all specs in this group
        loaded = active_loader(group_key)
        models_loaded += 1
        total_load_s += loaded.load_s

        for idx, spec in indexed_specs:
            report = _execute_spec(
                spec,
                root=root_path,
                cache=cache,
                loaded=loaded,
                renderer=active_renderer,
                worker_run_id=run_id,
                fake=is_fake,
                record_load_in_item=False,  # load_s is recorded once on the batch worker metrics.json
                startup_s=None,
                verify_speech=verify_speech,
                event_logger=event_logger,
            )
            ordered_results[idx] = report
            items_generated += 1
            if report.measurements.gen_s is not None:
                latencies.append(report.measurements.gen_s)

    final_results = [ordered_results[i] for i in range(len(specs))]
    metrics_path: Path | None = None
    if metrics_out is not None and specs:
        first = specs[0]
        preemptible = first.pricing.mode == "preemptible"
        effective_startup = startup_s if models_loaded > 0 else 0.0
        f_usd = fixed_overhead_usd(
            first.platform,
            first.preset,
            startup_s=effective_startup,
            ready_or_load_s=total_load_s,
            preemptible=preemptible,
        )
        billable_s = round(effective_startup + total_load_s + sum(latencies), 6)
        metrics_payload: dict[str, Any] = {
            "schema_version": 1,
            "worker_run_id": run_id,
            "mode": mode,
            "fake": is_fake,
            "status": "completed",
            "models_loaded": models_loaded,
            "items_generated": items_generated,
            "items_cached": items_cached,
            "startup_s": effective_startup,
            "ready_wait_s": None,
            "load_s": round(total_load_s, 6),
            "latency_s": latencies,
            "billable_s": billable_s,
            "fixed_overhead_usd": f_usd or 0.0,
            "platform": first.platform,
            "preset": first.preset,
            "pricing_mode": first.pricing.mode,
            "assets": [
                {"key": r.asset.key, "sha256": r.asset.sha256}
                for r in final_results
                if r.asset is not None
            ],
        }
        metrics_path = staged_write(
            Path(metrics_out), json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n"
        )

    return BatchWorkerResult(
        worker_run_id=run_id,
        results=final_results,
        metrics_path=metrics_path,
        models_loaded=models_loaded,
        items_generated=items_generated,
        items_cached=items_cached,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MediaFactory single-spec and batch worker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec", type=Path, help="Path to spec.json")
    group.add_argument("--batch", type=Path, help="Path to batch.jsonl")
    parser.add_argument("--root", type=Path, default=Path("."), help="Artifact root directory")
    parser.add_argument("--fake", action="store_true", help="Use deterministic fake backend")
    parser.add_argument("--local", action="store_true", help="Run real local inference")
    parser.add_argument("--metrics-out", type=Path, default=None, help="Path to write metrics.json")
    args = parser.parse_args(argv)

    mode: ExecutionMode = "fake" if args.fake else ("local" if args.local else "fake")
    if args.spec:
        spec = GenerationSpec.from_file(args.spec)
        run_single(spec, root=args.root, mode=mode, metrics_out=args.metrics_out)
    else:
        run_batch_specs(args.batch, root=args.root, mode=mode, metrics_out=args.metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
