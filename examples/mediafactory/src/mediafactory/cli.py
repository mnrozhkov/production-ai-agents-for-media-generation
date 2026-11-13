"""`mediafactory` CLI: `doctor`, `worker`, `run`, `batch`, `finetune`, `teardown` (`SPEC.md` §9)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mediafactory.config import DEFAULT_LORA_YAML, Settings
from mediafactory.core.cost import Budget, hourly_rate, marginal_cost_usd
from mediafactory.core.events import EventLogger
from mediafactory.core.spec import GenerationSpec
from mediafactory.ops import run_doctor_checks, run_teardown
from mediafactory.profile.application import load_application
from mediafactory.profile.records import build_brief, load_records, write_dead_letters
from mediafactory.worker.runner import (
    ExecutionMode,
    build_job_request,
    load_batch_specs,
    run_batch_specs,
    run_single,
)
from mediafactory.workflows.batch import run_batch
from mediafactory.workflows.finetune import (
    LoraConfig,
    get_status,
    prepare_dataset,
    start_finetune,
)
from mediafactory.workflows.pipeline import run_record

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Personalized Media Factory — domain-agnostic multimodal agent and batch engine.",
)
finetune_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Lesson 3.6 style LoRA dataset preparation and ticketed training Jobs.",
)
app.add_typer(finetune_app, name="finetune")

console = Console()


def _resolve_mode(*, fake: bool, local: bool, confirm_create: bool) -> ExecutionMode:
    chosen = sum(bool(x) for x in (fake, local, confirm_create))
    if chosen > 1:
        raise typer.BadParameter(
            "Pass at most one execution mode flag: --fake, --local, or --confirm-create."
        )
    if fake:
        return "fake"
    if local:
        return "local"
    if confirm_create:
        return "cloud"
    return "dry_run"


@app.command()
def doctor(
    app_dir: Path | None = typer.Option(
        None, "--app", help="Optional application directory to validate (e.g. applications/storybook)."
    ),
) -> None:
    """Validate `models.yaml`, application profile, templates, records, and environment."""
    checks = run_doctor_checks(app_dir=app_dir)
    table = Table(header_style="bold")
    table.add_column("Status")
    table.add_column("Check")
    table.add_column("Detail")
    table.add_column("Fix")
    failed = False
    for c in checks:
        if c.status == "FAIL":
            failed = True
        table.add_row(c.status, c.name, c.detail, c.fix)
    console.print(table)
    if failed:
        raise typer.Exit(code=1)


@app.command()
def worker(
    spec: Path | None = typer.Option(None, "--spec", help="Path to single spec.json (M1)"),
    batch_file: Path | None = typer.Option(None, "--batch", help="Path to batch.jsonl (M3)"),
    root: Path = typer.Option(Path("."), "--root", help="Artifact root directory"),
    fake: bool = typer.Option(False, "--fake", help="Run with deterministic fake fixtures"),
    local: bool = typer.Option(False, "--local", help="Run real local inference"),
    confirm_create: bool = typer.Option(
        False, "--confirm-create", help="Submit live GPU Job on Nebius Serverless AI"
    ),
) -> None:
    """Execute one `spec.json` (`--spec`) or a model-reusing `batch.jsonl` (`--batch`)."""
    if bool(spec) == bool(batch_file):
        raise typer.BadParameter("Provide exactly one of --spec or --batch.")
    mode = _resolve_mode(fake=fake, local=local, confirm_create=confirm_create)

    if spec is not None:
        parsed = GenerationSpec.from_file(spec)
        if mode == "dry_run":
            req = build_job_request(
                target_path=spec,
                is_batch=False,
                platform=parsed.platform,
                preset=parsed.preset,
                preemptible=(parsed.pricing.mode == "preemptible"),
            )
            console.print(
                json.dumps(
                    {
                        "mode": "dry_run",
                        "spec_hash": parsed.spec_hash(),
                        "argv": req.argv,
                        "estimated_cap_usd": req.estimated_cap_usd,
                    },
                    indent=2,
                )
            )
            return
        report = run_single(parsed, root=root, mode=mode)
        console.print(report.to_json())
        return

    assert batch_file is not None
    specs = load_batch_specs(batch_file)
    if mode == "dry_run":
        first = specs[0]
        req = build_job_request(
            target_path=batch_file,
            is_batch=True,
            platform=first.platform,
            preset=first.preset,
            preemptible=(first.pricing.mode == "preemptible"),
        )
        console.print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "batch_items": len(specs),
                    "argv": req.argv,
                    "estimated_cap_usd": req.estimated_cap_usd,
                },
                indent=2,
            )
        )
        return
    res = run_batch_specs(specs, root=root, mode=mode)
    console.print(
        json.dumps(
            {
                "worker_run_id": res.worker_run_id,
                "models_loaded": res.models_loaded,
                "items_generated": res.items_generated,
                "items_cached": res.items_cached,
            },
            indent=2,
        )
    )


@app.command()
def run(
    app_dir: Path = typer.Option(..., "--app", help="Application profile directory"),
    records: Path | None = typer.Option(None, "--records", help="Override records CSV/JSONL path"),
    run_id: str = typer.Option("run-local", "--run-id", help="Run identifier"),
    root: Path = typer.Option(Path("."), "--root", help="Artifact root directory"),
    profile: str | None = typer.Option(None, "--profile", help="Hardware profile (e.g. l40s)"),
    fake: bool = typer.Option(False, "--fake", help="Run with deterministic fake fixtures"),
    local: bool = typer.Option(False, "--local", help="Run real local inference"),
    confirm_create: bool = typer.Option(
        False, "--confirm-create", help="Execute against live cloud endpoints/jobs"
    ),
) -> None:
    """Run the single-record planner -> generator -> judge pipeline over a record file."""
    mode = _resolve_mode(fake=fake, local=local, confirm_create=confirm_create)
    bundle = load_application(app_dir)
    settings = Settings.load(profile=profile)
    rec_file = records if records else (bundle.root / bundle.config.records.path).resolve()
    loaded = load_records(rec_file, required_columns=bundle.config.records.required)

    valid_ids = [str(r["id"]) for r in loaded.valid_records]
    sampled = {
        s.name: bundle.config.sampled_ids_for_stage(s.name, valid_ids)
        for s in bundle.config.stages
    }

    if mode == "dry_run":
        img_cfg = settings.role("image")
        img_plat = img_cfg.compute.platform if img_cfg.compute else "gpu-l40s-a"
        img_pres = img_cfg.compute.preset if img_cfg.compute else "1gpu-8vcpu-32gb"
        est_img = marginal_cost_usd(img_plat, img_pres, settings.estimate_role_seconds("image"))
        console.print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "application": bundle.config.name,
                    "valid_records": len(loaded.valid_records),
                    "dead_letters": len(loaded.dead_letters),
                    "stages": [s.name for s in bundle.config.stages],
                    "video_sampled_records": len(sampled.get("video", set())),
                    "estimated_marginal_image_usd_per_record": est_img,
                    "record_budget_usd": bundle.config.constraints.record_budget_usd,
                    "run_budget_usd": bundle.config.constraints.run_budget_usd,
                },
                indent=2,
            )
        )
        return

    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_dead_letters(run_dir / "dead_letter.jsonl", loaded.dead_letters)
    event_logger = EventLogger(run_dir / "events.jsonl", run_id=run_id)
    budget = Budget(
        record_budget_usd=bundle.config.constraints.record_budget_usd,
        run_budget_usd=bundle.config.constraints.run_budget_usd,
    )

    completed = 0
    for row in loaded.valid_records:
        brief = build_brief(bundle, row, sampled_stage_ids=sampled)
        ledger = run_record(
            bundle,
            settings,
            brief,
            root=root,
            run_id=run_id,
            mode=mode,
            budget=budget,
            event_logger=event_logger,
        )
        if ledger.status == "completed":
            completed += 1

    console.print(
        json.dumps(
            {
                "run_id": run_id,
                "mode": mode,
                "completed": completed,
                "total_valid": len(loaded.valid_records),
                "run_spent_usd": budget.run_spent,
            },
            indent=2,
        )
    )


@app.command()
def batch(
    app_dir: Path = typer.Option(..., "--app", help="Application profile directory"),
    records: Path | None = typer.Option(None, "--records", help="Override records CSV/JSONL path"),
    jobs: int = typer.Option(10, "--jobs", min=1, max=100, help="Number of parallel batch Jobs"),
    run_id: str = typer.Option("batch-001", "--run-id", help="Run identifier"),
    root: Path = typer.Option(Path("."), "--root", help="Artifact root directory"),
    resume: bool = typer.Option(False, "--resume", help="Resume an interrupted batch run"),
    profile: str | None = typer.Option(None, "--profile", help="Hardware profile (e.g. l40s)"),
    fake: bool = typer.Option(False, "--fake", help="Run with deterministic fake fixtures"),
    confirm_create: bool = typer.Option(
        False, "--confirm-create", help="Submit live batch GPU Jobs"
    ),
) -> None:
    """Fan out records across K parallel batch Jobs (`M3`), checkpoint per item, and write `summary.json`."""
    mode = _resolve_mode(fake=fake, local=False, confirm_create=confirm_create)
    bundle = load_application(app_dir)
    settings = Settings.load(profile=profile)
    rec_file = records if records else (bundle.root / bundle.config.records.path).resolve()
    loaded = load_records(rec_file, required_columns=bundle.config.records.required)
    valid_ids = [str(r["id"]) for r in loaded.valid_records]
    video_sampled = bundle.config.sampled_ids_for_stage("video", valid_ids)

    if mode == "dry_run":
        l40s_spot = hourly_rate("gpu-l40s-a", "1gpu-8vcpu-32gb", preemptible=True)
        console.print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "application": bundle.config.name,
                    "records": len(loaded.valid_records),
                    "jobs": jobs,
                    "video_sampled": len(video_sampled),
                    "spot_l40s_usd_per_hour": l40s_spot.usd_per_hour,
                    "run_budget_usd": bundle.config.constraints.run_budget_usd,
                },
                indent=2,
            )
        )
        return

    res = run_batch(
        bundle,
        settings,
        records_path=rec_file,
        root=root,
        run_id=run_id,
        jobs=jobs,
        mode=mode,
        resume=resume,
    )
    console.print(json.dumps(res.summary, indent=2))


@finetune_app.command("prepare")
def finetune_prepare(
    images: Path = typer.Option(..., "--images", help="Directory of training images"),
    trigger: str = typer.Option(..., "--trigger", help="Style trigger word"),
    out: Path = typer.Option(..., "--out", help="Output directory for prepared dataset"),
    fake: bool = typer.Option(False, "--fake", help="Use deterministic offline captioning"),
    confirm_create: bool = typer.Option(False, "--confirm-create", help="Use live VLM captioning"),
) -> None:
    """Deduplicate training images, generate/check captions, and write a contact sheet (Lesson 3.6)."""
    mode = _resolve_mode(fake=fake, local=False, confirm_create=confirm_create)
    if mode == "dry_run":
        console.print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "images_dir": str(images),
                    "trigger_word": trigger,
                    "output_dir": str(out),
                },
                indent=2,
            )
        )
        return
    prepared = prepare_dataset(images, trigger_word=trigger, output_dir=out)
    console.print(
        json.dumps(
            {
                "output_dir": str(prepared.output_dir),
                "total_found": prepared.total_found,
                "unique_kept": prepared.unique_kept,
                "duplicates_removed": prepared.duplicates_removed,
                "flagged_trigger_leaks": prepared.flagged_trigger_leaks,
                "manifest": str(prepared.manifest_path),
                "contact_sheet": str(prepared.contact_sheet_path),
            },
            indent=2,
        )
    )


@finetune_app.command("submit")
def finetune_submit(
    config: Path = typer.Option(DEFAULT_LORA_YAML, "--config", help="Path to lora_config.yaml"),
    dataset: Path = typer.Option(..., "--dataset", help="Prepared dataset directory"),
    out: Path = typer.Option(Path("runs/finetune-01"), "--out", help="Run output directory"),
    fake: bool = typer.Option(False, "--fake", help="Simulate ticketed LoRA Job offline"),
    confirm_create: bool = typer.Option(False, "--confirm-create", help="Submit live GPU Job"),
) -> None:
    """Submit or dry-run a ticketed LoRA style fine-tuning Job with `--timeout` (Lesson 3.6)."""
    lora_cfg = LoraConfig.from_file(config)
    res = start_finetune(
        lora_cfg,
        dataset_dir=dataset,
        run_dir=out,
        confirm_create=confirm_create,
        fake=fake,
    )
    console.print(json.dumps(res, indent=2))


@finetune_app.command("status")
def finetune_status(
    ticket: Path = typer.Option(..., "--ticket", help="Path to finetune ticket.json"),
) -> None:
    """Check the status of a ticketed LoRA fine-tuning Job."""
    console.print(json.dumps(get_status(ticket), indent=2))


@app.command()
def teardown(
    dry_run: bool = typer.Option(False, "--dry-run", help="List resources without stopping them"),
) -> None:
    """Stop active Endpoints and cancel lingering GPU Jobs."""
    report = run_teardown(dry_run=dry_run)
    console.print(
        json.dumps(
            {
                "dry_run": report.dry_run,
                "stopped_endpoints": report.stopped_endpoints,
                "cancelled_jobs": report.cancelled_jobs,
                "hourly_usd_saved": report.hourly_usd_saved,
            },
            indent=2,
        )
    )
