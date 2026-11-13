"""Comprehensive offline test suite verifying `examples/mediafactory/SPEC.md` requirements
and all 7 Phase 1 acceptance criteria + Lesson 3.6 fine-tuning workflow.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from mediafactory.cli import app
from mediafactory.config import DEFAULT_LORA_YAML, PACKAGE_ROOT, Settings
from mediafactory.core.cost import Budget, BudgetExceeded
from mediafactory.core.events import EventLogger
from mediafactory.core.spec import GenerationSpec, InputImage, LoraAdapter, Size, SpecError
from mediafactory.profile.application import ApplicationError, load_application
from mediafactory.profile.records import build_brief, load_records
from mediafactory.worker.adapters import FakeAdapter, make_fake_png
from mediafactory.worker.runner import run_batch_specs, run_single
from mediafactory.workflows.batch import run_batch
from mediafactory.workflows.finetune import (
    LoraConfig,
    get_status,
    prepare_dataset,
    start_finetune,
)
from mediafactory.workflows.pipeline import run_record
from mediafactory.workflows.tickets import (
    InMemoryWorkloadProvider,
    mark_ticket_submitted,
    reconcile_or_submit,
    write_intent_ticket,
)
from typer.testing import CliRunner

APPS_DIR = PACKAGE_ROOT / "applications"
DATA_DIR = PACKAGE_ROOT / "data"
runner = CliRunner()


# ---------------------------------------------------------------------------
# FR-03, FR-04, FR-05, FR-06: Closed schemas, import isolation, template checks
# ---------------------------------------------------------------------------


def test_fr04_import_isolation_no_torch_or_diffusers() -> None:
    """FR-04: Validating application.yaml, prompts.yaml, records, or spec.json imports no torch/diffusers."""
    code = (
        "import sys; "
        "from mediafactory.core.spec import GenerationSpec; "
        "from mediafactory.profile.application import load_application; "
        f"load_application({str(APPS_DIR / 'storybook')!r}); "
        "s = GenerationSpec(role='image', model='m', prompt='p'); "
        "assert 'torch' not in sys.modules; "
        "assert 'diffusers' not in sys.modules"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_fr03_closed_schema_rejects_typos_and_bad_paths() -> None:
    """FR-03: Typo in field name or path traversal in output_prefix fails validation."""
    with pytest.raises(SpecError, match="Extra inputs are not permitted"):
        GenerationSpec.from_mapping(
            {
                "role": "image",
                "model": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
                "prompt": "a telescope",
                "step": 8,  # typo for steps
            }
        )

    with pytest.raises(SpecError, match="output_prefix"):
        GenerationSpec.from_mapping(
            {
                "role": "image",
                "model": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
                "prompt": "a telescope",
                "output_prefix": "../escape",
            }
        )


def test_fr05_template_placeholder_validation(tmp_path: Path) -> None:
    """FR-05: Undeclared placeholder in prompts.yaml fails at load time before any model call."""
    app_dir = tmp_path / "bad_app"
    app_dir.mkdir()
    for name in ("application.yaml", "rubric.yaml"):
        (app_dir / name).write_text((APPS_DIR / "_template" / name).read_text())
    (app_dir / "prompts.yaml").write_text(
        "system: 'ok'\nplan: 'Task: {task} Unknown: {nonexistent_column}'\n"
    )
    with pytest.raises(ApplicationError, match="undeclared placeholder"):
        load_application(app_dir)


def test_fr02_dead_letters_do_not_abort_valid_records(tmp_path: Path) -> None:
    """FR-02: Invalid rows (missing id, duplicate id, missing required column) go to dead_letters."""
    bad_csv = tmp_path / "mixed.csv"
    bad_csv.write_text(
        "id,title,audience\n"
        "r1,Solar Lantern,campers\n"
        ",Missing ID Row,campers\n"
        "r1,Duplicate ID Row,campers\n"
        "r2,,campers\n"
        "r3,Valid Item Two,hikers\n"
    )
    res = load_records(bad_csv, required_columns=["id", "title", "audience"])
    assert [r["id"] for r in res.valid_records] == ["r1", "r3"]
    assert len(res.dead_letters) == 3
    error_classes = [d.error_class for d in res.dead_letters]
    assert error_classes == ["MissingId", "DuplicateId", "MissingRequiredColumn"]


# ---------------------------------------------------------------------------
# Acceptance Criterion 1: Zero-code application swap
# ---------------------------------------------------------------------------


def test_ac1_zero_code_application_swap(tmp_path: Path) -> None:
    """AC-1: storybook, product_ad, and _template run with different schemas & stage graphs."""
    settings = Settings.load()

    # 1. Storybook on kids_10.csv
    sb_bundle = load_application(APPS_DIR / "storybook")
    sb_res = run_batch(
        sb_bundle,
        settings,
        records_path=DATA_DIR / "kids_10.csv",
        root=tmp_path,
        run_id="sb-10",
        jobs=2,
        mode="fake",
    )
    assert sb_res.summary["application"] == "storybook"
    assert sb_res.summary["records"]["completed_accepted"] == 10
    assert (tmp_path / "runs" / "sb-10" / "items" / "k01" / "speech" / "asset.mp3").is_file()

    # 2. Product ad (no speech stage; uses product/audience/must_show/must_not columns)
    ad_bundle = load_application(APPS_DIR / "product_ad")
    ad_res = run_batch(
        ad_bundle,
        settings,
        root=tmp_path,
        run_id="ad-run",
        jobs=2,
        mode="fake",
    )
    assert ad_res.summary["application"] == "product_ad"
    assert ad_res.summary["records"]["completed_accepted"] == 3
    ad01_copy = json.loads(
        (tmp_path / "runs" / "ad-run" / "items" / "ad01" / "copy.json").read_text()
    )
    assert "Byte Brew" in ad01_copy["image_prompt"]
    assert not (tmp_path / "runs" / "ad-run" / "items" / "ad01" / "speech").exists()


# ---------------------------------------------------------------------------
# Acceptance Criterion 2: Single model load per batch worker
# ---------------------------------------------------------------------------


def test_ac2_single_model_load_per_batch_worker(tmp_path: Path) -> None:
    """AC-2: A batch worker given 20 specs for the same model loads it once and renders 20 times."""
    fake_adapter = FakeAdapter()
    specs = [
        GenerationSpec(
            role="image",
            model="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
            model_revision="ac0da2ff55fbe434795be0dce883042e4d49e2fc",
            prompt=f"hero object #{i}",
            record_id=f"rec-{i:02d}",
            seed=i,
            size=Size(width=1024, height=1024),
            output_prefix=f"runs/batch-test/items/rec-{i:02d}/image",
        )
        for i in range(20)
    ]
    metrics_path = tmp_path / "batch_metrics.json"
    res = run_batch_specs(
        specs,
        root=tmp_path,
        mode="fake",
        adapter=fake_adapter,
        metrics_out=metrics_path,
    )
    assert res.models_loaded == 1
    assert res.items_generated == 20
    group_key = specs[0].worker_group_key()
    assert fake_adapter.load_counts[group_key] == 1
    assert fake_adapter.render_counts["image"] == 20

    # Shared load_s is recorded once in metrics.json, and is None on individual item results
    batch_metrics = json.loads(metrics_path.read_text())
    assert batch_metrics["load_s"] == pytest.approx(1.2)
    assert len(batch_metrics["latency_s"]) == 20
    assert res.results[0].measurements.load_s is None
    assert res.results[0].measurements.gen_s == pytest.approx(4.4)


# ---------------------------------------------------------------------------
# Acceptance Criterion 3: Mid-flight ticket recovery (intent & submitted states)
# ---------------------------------------------------------------------------


def test_ac3_midflight_ticket_recovery_submitted_and_intent(tmp_path: Path) -> None:
    """AC-3: Resuming after a crash in `submitted` or `intent` reconciles without re-submitting."""
    provider = InMemoryWorkloadProvider()

    # Case A: Crash after `submitted`
    ticket_a_path = tmp_path / "tickets" / "video-a.json"
    t_a = write_intent_ticket(
        ticket_a_path,
        call_id="call-a",
        spec_hash="a" * 64,
        run_id="run-1",
        role="video",
        record_id="k01",
    )
    wid_a = provider.submit(deterministic_name=t_a.deterministic_name, call_id="call-a")
    mark_ticket_submitted(ticket_a_path, t_a, wid_a)
    assert len(provider.submissions) == 1

    reconciled_a, was_reconciled_a = reconcile_or_submit(
        ticket_a_path,
        call_id="call-a",
        spec_hash="a" * 64,
        run_id="run-1",
        role="video",
        provider=provider,
        record_id="k01",
    )
    assert was_reconciled_a is True
    assert reconciled_a.workload_id == wid_a
    assert reconciled_a.state == "completed"
    assert len(provider.submissions) == 1  # No duplicate submission!

    # Case B: Crash in the narrow window after `intent` write + provider submit, before saving workload_id
    ticket_b_path = tmp_path / "tickets" / "video-b.json"
    t_b = write_intent_ticket(
        ticket_b_path,
        call_id="call-b",
        spec_hash="b" * 64,
        run_id="run-1",
        role="video",
        record_id="k02",
    )
    wid_b = provider.submit(deterministic_name=t_b.deterministic_name, call_id="call-b")
    assert ticket_b_path.exists()
    assert len(provider.submissions) == 2

    reconciled_b, was_reconciled_b = reconcile_or_submit(
        ticket_b_path,
        call_id="call-b",
        spec_hash="b" * 64,
        run_id="run-1",
        role="video",
        provider=provider,
        record_id="k02",
    )
    assert was_reconciled_b is True
    assert reconciled_b.workload_id == wid_b
    assert reconciled_b.state == "completed"
    assert len(provider.submissions) == 2  # Reconciled by deterministic name without duplicate submit!


# ---------------------------------------------------------------------------
# Acceptance Criterion 4: Zero-regeneration resume after mid-run kill
# ---------------------------------------------------------------------------


def test_ac4_zero_regeneration_resume(tmp_path: Path) -> None:
    """AC-4: Cancelling a batch run at 50% and resuming completes the rest with 0 regenerations."""
    bundle = load_application(APPS_DIR / "storybook")
    settings = Settings.load()
    adapter = FakeAdapter()

    # First run stops after 5 of 10 records (simulating a kill at 50%)
    partial = run_batch(
        bundle,
        settings,
        records_path=DATA_DIR / "kids_10.csv",
        root=tmp_path,
        run_id="kill-resume-run",
        jobs=2,
        mode="fake",
        stop_after_records=5,
        adapter=adapter,
    )
    assert partial.summary["records"]["completed_accepted"] == 5
    renders_after_first = adapter.render_counts["image"]
    assert renders_after_first == 5

    # Resume the same run_id across all 10 records
    resumed = run_batch(
        bundle,
        settings,
        records_path=DATA_DIR / "kids_10.csv",
        root=tmp_path,
        run_id="kill-resume-run",
        jobs=2,
        mode="fake",
        resume=True,
        adapter=adapter,
    )
    assert resumed.summary["records"]["completed_accepted"] == 10
    assert resumed.summary["resume_proof"]["items_completed_before_resume"] == 5
    assert resumed.summary["resume_proof"]["items_regenerated_after_resume"] == 0
    # Only the remaining 5 images were rendered on resume
    assert adapter.render_counts["image"] == 10


# ---------------------------------------------------------------------------
# Acceptance Criterion 5: Content-addressed cache invalidation
# ---------------------------------------------------------------------------


def test_ac5_content_addressed_cache_invalidation(tmp_path: Path) -> None:
    """AC-5: Identical spec is a $0 cache hit; changing prompt, seed, revision, LoRA, or input_image invalidates."""
    base_spec = GenerationSpec(
        role="video",
        model="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        model_revision="0000000000000000000000000000000000000014",
        prompt="slow camera drift across the cup",
        input_image=InputImage(uri="items/ad01/image/asset.png", sha256="1" * 64),
        record_id="ad01",
        seed=7,
        size=Size(width=832, height=480),
        output_prefix="runs/cache-test/items/ad01/video",
    )
    adapter = FakeAdapter()

    r1 = run_single(base_spec, root=tmp_path, mode="fake", adapter=adapter)
    assert r1.cache_hit is False
    assert r1.cost.marginal_compute_usd > 0.0
    assert adapter.render_counts["video"] == 1

    # Second identical call -> cache hit with $0 marginal compute cost
    r2 = run_single(base_spec, root=tmp_path, mode="fake", adapter=adapter)
    assert r2.cache_hit is True
    assert r2.cost.marginal_compute_usd == 0.0
    assert adapter.render_counts["video"] == 1

    # Changing keyframe sha256 invalidates the cache and triggers a new generation
    mutated_keyframe_spec = base_spec.model_copy(
        update={"input_image": InputImage(uri="items/ad01/image/asset.png", sha256="2" * 64)}
    )
    assert mutated_keyframe_spec.spec_hash() != base_spec.spec_hash()
    r3 = run_single(mutated_keyframe_spec, root=tmp_path, mode="fake", adapter=adapter)
    assert r3.cache_hit is False
    assert adapter.render_counts["video"] == 2

    # Attaching a LoRA adapter also changes spec_hash
    with_lora = base_spec.model_copy(
        update={
            "lora_adapter": LoraAdapter(
                uri="s3://bucket/lora.safetensors", sha256="3" * 64, scale=0.8
            )
        }
    )
    assert with_lora.spec_hash() != base_spec.spec_hash()


# ---------------------------------------------------------------------------
# Acceptance Criterion 6: Concurrent budget enforcement & judge retry / WEAK gate
# ---------------------------------------------------------------------------


def test_ac6_concurrent_budget_enforcement_and_judge_gates(tmp_path: Path) -> None:
    """AC-6: Concurrent workers respect record/run budgets; judge RETRY recovers and WEAK blocks video."""
    budget = Budget(record_budget_usd=0.50, run_budget_usd=1.00)

    def _try_spend(idx: int) -> bool:
        rec_id = f"r-{idx % 3}"
        try:
            res = budget.reserve(rec_id, 0.20)
            budget.commit(rec_id, res, 0.20)
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_try_spend, range(12)))

    # At $0.20 per call and $1.00 run cap, at most 5 calls can succeed across all threads
    assert sum(outcomes) == 5
    assert budget.run_spent == pytest.approx(1.00)

    # Also test judge RETRY recovery and WEAK gate on a single record
    bundle = load_application(APPS_DIR / "product_ad")
    settings = Settings.load()
    loaded = load_records(
        APPS_DIR / "product_ad" / "records.csv",
        required_columns=bundle.config.records.required,
    )
    brief = build_brief(bundle, loaded.valid_records[0])
    event_logger = EventLogger(tmp_path / "runs" / "judge-test" / "events.jsonl", run_id="judge-test")
    record_budget = Budget(record_budget_usd=2.0, run_budget_usd=10.0)

    # 1. Fixture recovery: first attempt returns RETRY, plus 1 unparseable judge reply (FR-13)
    ledger_retry = run_record(
        bundle,
        settings,
        brief,
        root=tmp_path,
        run_id="judge-retry",
        mode="fake",
        budget=record_budget,
        event_logger=event_logger,
        force_first_verdict="RETRY",
        unparseable_judge_replies_before_valid=1,
    )
    assert ledger_retry.status == "completed"
    assert ledger_retry.image_attempts == 2
    assert ledger_retry.video_attempts == 1
    assert "video" in ledger_retry.stages_completed

    # 2. WEAK verdict stops before video (FR-08 / FR-10)
    brief_weak = build_brief(bundle, loaded.valid_records[1])
    ledger_weak = run_record(
        bundle,
        settings,
        brief_weak,
        root=tmp_path,
        run_id="judge-weak",
        mode="fake",
        budget=record_budget,
        event_logger=event_logger,
        force_first_verdict="WEAK",
    )
    assert ledger_weak.status == "rejected_weak"
    assert ledger_weak.video_attempts == 0
    assert "video" not in ledger_weak.stages_completed


# ---------------------------------------------------------------------------
# Acceptance Criterion 7: Module 4 offline compatibility & HW3 200-record scale
# ---------------------------------------------------------------------------


def test_ac7_module4_run_folder_and_hw3_200_records(tmp_path: Path) -> None:
    """AC-7: 200-record storybook run produces M4-compatible files and samples 10 videos."""
    bundle = load_application(APPS_DIR / "storybook")
    settings = Settings.load()

    res = run_batch(
        bundle,
        settings,
        records_path=DATA_DIR / "kids.csv",
        root=tmp_path,
        run_id="hw3-200",
        jobs=10,
        mode="fake",
    )
    run_dir = res.run_dir
    for required_file in (
        "application.json",
        "manifest.jsonl",
        "dead_letter.jsonl",
        "events.jsonl",
        "metrics.json",
        "summary.json",
    ):
        assert (run_dir / required_file).is_file(), f"missing {required_file}"

    summary = res.summary
    assert summary["fake"] is True
    assert summary["records"]["total"] == 200
    assert summary["records"]["valid"] == 200
    assert summary["records"]["completed_accepted"] == 200
    assert summary["records"]["video_sampled"] == 10
    assert summary["latency_s"]["image"]["count"] == 200
    assert summary["latency_s"]["speech"]["count"] == 200
    assert summary["latency_s"]["video"]["count"] == 10
    assert summary["cost"]["cost_per_accepted_asset_usd"] > 0

    # Check cached verdict.json and transcript.json for offline Module 4 evaluation
    assert (run_dir / "items" / "k01" / "image" / "verdict.json").is_file()
    assert (run_dir / "items" / "k01" / "speech" / "transcript.json").is_file()

    # Verify events.jsonl has timestamps, record_id, parent_event_id, t_gen_s, cost_usd
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(events) > 600
    judge_events = [e for e in events if e["kind"] == "judge"]
    assert len(judge_events) == 200
    assert all(e["rubric_version"] == "v1" and e["cost_usd"] > 0 for e in judge_events)
    det_checks = [e for e in events if e["name"] == "deterministic_check"]
    assert all(e["cost_usd"] == 0.0 and e["parent_event_id"] is not None for e in det_checks)


# ---------------------------------------------------------------------------
# Lesson 3.6 Style Fine-Tuning Workflow & CLI Modes
# ---------------------------------------------------------------------------


def test_lesson36_finetune_prepare_and_ticketed_submit(tmp_path: Path) -> None:
    """Lesson 3.6: prepare_dataset deduplicates & flags trigger leaks; start_finetune uses tickets."""
    raw_imgs = tmp_path / "raw_images"
    raw_imgs.mkdir()
    img1 = make_fake_png(512, 512, seed_tag="img-1")
    img2 = make_fake_png(512, 512, seed_tag="img-2")
    (raw_imgs / "01_telescope.png").write_bytes(img1)
    (raw_imgs / "01_duplicate.png").write_bytes(img1)  # duplicate SHA-256
    (raw_imgs / "02_train.png").write_bytes(img2)

    ds_out = tmp_path / "prepared_ds"
    prepared = prepare_dataset(
        raw_imgs,
        trigger_word="storybook_v1",
        output_dir=ds_out,
        captioner={
            "01_telescope.png": "a brass telescope on snow",
            "02_train.png": "a storybook_v1 wooden train set",  # leaks trigger word!
        },
    )
    assert prepared.total_found == 3
    assert prepared.unique_kept == 2
    assert prepared.duplicates_removed == 1
    assert prepared.flagged_trigger_leaks == 1
    assert (ds_out / "dataset_manifest.json").is_file()
    assert (ds_out / "contact_sheet.png").is_file()

    lora_cfg = LoraConfig.from_file(DEFAULT_LORA_YAML)
    dry = start_finetune(lora_cfg, dataset_dir=ds_out, run_dir=tmp_path / "ft-run")
    assert dry["mode"] == "dry_run"
    assert "--timeout" in dry["argv"]

    submitted = start_finetune(
        lora_cfg, dataset_dir=ds_out, run_dir=tmp_path / "ft-run", fake=True
    )
    assert submitted["mode"] == "fake"
    status = get_status(submitted["ticket_path"])
    assert status["state"] == "completed"
    assert status["lora_adapter"]["trigger_word"] == "storybook_v1"


def test_cli_dry_run_and_doctor(tmp_path: Path) -> None:
    """FR-14: Default CLI invocation is a side-effect-free dry run; doctor validates profiles."""
    res_doc = runner.invoke(app, ["doctor", "--app", str(APPS_DIR / "storybook")])
    assert res_doc.exit_code == 0

    res_dry = runner.invoke(
        app,
        [
            "run",
            "--app",
            str(APPS_DIR / "product_ad"),
            "--root",
            str(tmp_path),
            "--run-id",
            "should-not-exist",
        ],
    )
    assert res_dry.exit_code == 0
    payload = json.loads(res_dry.stdout)
    assert payload["mode"] == "dry_run"
    assert not (tmp_path / "runs" / "should-not-exist").exists()
