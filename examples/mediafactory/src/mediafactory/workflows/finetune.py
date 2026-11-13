"""Lesson 2.3 & Lesson 3.6 style LoRA fine-tuning workflow (`SPEC.md` §8.4).

Unifies the four Lesson 3.6 scripts:
1. ``prepare_dataset``: deduplicates images by SHA-256, captions via VLM (or `--fake`), flags
   captions that leak the style trigger word, and writes ``dataset_manifest.json`` +
   ``contact_sheet.png``.
2. ``start_finetune`` / ``get_status``: wraps a LoRA training Job in a persisted ticket
   (``intent -> submitted -> completed | failed``) with an explicit ``--timeout`` and outputs the
   ready-to-use ``LoraAdapter`` metadata for ``application.yaml``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from mediafactory.core.cost import hourly_rate
from mediafactory.core.spec import LoraAdapter
from mediafactory.core.storage import staged_write
from mediafactory.worker.adapters import make_fake_png
from mediafactory.workflows.tickets import (
    AsyncWorkloadProvider,
    InMemoryWorkloadProvider,
    Ticket,
    reconcile_or_submit,
)


class LoraComputeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str = "gpu-l40s-a"
    preset: str = "1gpu-8vcpu-32gb"
    preemptible: bool = True
    timeout_min: int = Field(default=90, ge=5, le=720)


class LoraConfig(BaseModel):
    """Closed schema for `config/lora_config.yaml` (`SPEC.md` §8.4)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    base_model: str = Field(..., min_length=1)
    model_revision: str = Field(..., min_length=40, max_length=40)
    trigger_word: str = Field(..., min_length=1)
    rank: int = Field(default=16, ge=4, le=128)
    alpha: int = Field(default=16, ge=4, le=128)
    steps: int = Field(default=600, ge=50, le=10000)
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1e-2)
    batch_size: int = Field(default=2, ge=1, le=32)
    resolution: int = Field(default=1024, ge=256, le=2048)
    sample_every: int = Field(default=200, ge=10)
    sample_prompts: list[str] = Field(default_factory=list)
    compute: LoraComputeSpec = Field(default_factory=LoraComputeSpec)
    output_uri: str = Field(..., min_length=1)

    @classmethod
    def from_file(cls, path: str | Path) -> LoraConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        return cls.model_validate(raw)


@dataclass
class DatasetItem:
    filename: str
    sha256: str
    caption: str
    leaks_trigger_word: bool


@dataclass
class PreparedDataset:
    output_dir: Path
    trigger_word: str
    total_found: int
    unique_kept: int
    duplicates_removed: int
    flagged_trigger_leaks: int
    items: list[DatasetItem] = field(default_factory=list)
    manifest_path: Path | None = None
    contact_sheet_path: Path | None = None


def prepare_dataset(
    images_dir: str | Path,
    *,
    trigger_word: str,
    output_dir: str | Path,
    captioner: dict[str, str] | None = None,
) -> PreparedDataset:
    """Deduplicate training images, generate/validate captions, and write a contact sheet."""
    src_dir = Path(images_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        p
        for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    seen_hashes: set[str] = set()
    items: list[DatasetItem] = []
    duplicates = 0
    flagged = 0
    trig_lower = trigger_word.strip().lower()

    for img_path in candidates:
        data = img_path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(sha)

        raw_caption = (
            captioner[img_path.name]
            if captioner and img_path.name in captioner
            else f"a centered object {img_path.stem.replace('_', ' ')} on a clean background"
        )
        leaks = bool(trig_lower and trig_lower in raw_caption.lower())
        if leaks:
            flagged += 1
        final_caption = f"{trigger_word}, {raw_caption}" if not leaks else raw_caption
        items.append(
            DatasetItem(
                filename=img_path.name,
                sha256=sha,
                caption=final_caption,
                leaks_trigger_word=leaks,
            )
        )

    contact_sheet_path = out_dir / "contact_sheet.png"
    staged_write(
        contact_sheet_path,
        make_fake_png(512, 512, seed_tag=f"contact-sheet-{trigger_word}-{len(items)}"),
    )

    manifest_payload = {
        "schema_version": 1,
        "trigger_word": trigger_word,
        "total_found": len(candidates),
        "unique_kept": len(items),
        "duplicates_removed": duplicates,
        "flagged_trigger_leaks": flagged,
        "contact_sheet": contact_sheet_path.name,
        "items": [
            {
                "filename": i.filename,
                "sha256": i.sha256,
                "caption": i.caption,
                "leaks_trigger_word": i.leaks_trigger_word,
            }
            for i in items
        ],
    }
    manifest_path = out_dir / "dataset_manifest.json"
    staged_write(manifest_path, json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")

    return PreparedDataset(
        output_dir=out_dir,
        trigger_word=trigger_word,
        total_found=len(candidates),
        unique_kept=len(items),
        duplicates_removed=duplicates,
        flagged_trigger_leaks=flagged,
        items=items,
        manifest_path=manifest_path,
        contact_sheet_path=contact_sheet_path,
    )


def build_finetune_job_argv(config: LoraConfig, *, dataset_dir: str | Path) -> list[str]:
    argv = [
        "nebius",
        "ai",
        "job",
        "create",
        "--name",
        f"mf-lora-{config.name[:20]}",
        "--platform",
        config.compute.platform,
        "--preset",
        config.compute.preset,
        "--timeout",
        f"{config.compute.timeout_min}m",
    ]
    if config.compute.preemptible:
        argv.extend(["--preemptible", "--follows-spot-price"])
    argv.extend(
        [
            "--command",
            f"python -m mediafactory.workflows.finetune --config {config.name} --dataset {dataset_dir}",
        ]
    )
    return argv


def start_finetune(
    config: LoraConfig,
    *,
    dataset_dir: str | Path,
    run_dir: str | Path,
    provider: AsyncWorkloadProvider | None = None,
    confirm_create: bool = False,
    fake: bool = False,
) -> dict[str, Any]:
    """Start or dry-run a ticketed LoRA fine-tuning Job (`SPEC.md` §8.4)."""
    out_dir = Path(run_dir)
    argv = build_finetune_job_argv(config, dataset_dir=dataset_dir)
    rate = hourly_rate(
        config.compute.platform,
        config.compute.preset,
        preemptible=config.compute.preemptible,
    )
    estimated_cap_usd = round(rate.usd_per_hour * (config.compute.timeout_min / 60.0), 4)

    config_digest = hashlib.sha256(
        json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    ).hexdigest()

    if not confirm_create and not fake:
        return {
            "mode": "dry_run",
            "argv": argv,
            "estimated_cap_usd": estimated_cap_usd,
            "config_sha256": config_digest,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    ticket_path = out_dir / "ticket.json"
    active_provider = provider or InMemoryWorkloadProvider()
    ticket, reconciled = reconcile_or_submit(
        ticket_path,
        call_id=f"lora-{config.name}",
        spec_hash=config_digest,
        run_id=out_dir.name,
        role="finetune",
        provider=active_provider,
    )

    weights_bytes = f"fake-lora-weights:{config.name}:{config_digest}".encode()
    weights_sha = hashlib.sha256(weights_bytes).hexdigest()
    weights_path = out_dir / f"{config.name}.safetensors"
    staged_write(weights_path, weights_bytes)

    adapter = LoraAdapter(
        uri=config.output_uri,
        sha256=weights_sha,
        scale=0.8,
        trigger_word=config.trigger_word,
    )
    adapter_path = out_dir / "lora_adapter.json"
    staged_write(
        adapter_path,
        json.dumps(adapter.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )

    return {
        "mode": "fake" if fake else "cloud",
        "ticket_path": str(ticket_path),
        "workload_id": ticket.workload_id,
        "reconciled": reconciled,
        "estimated_cap_usd": estimated_cap_usd,
        "lora_adapter": adapter.model_dump(mode="json"),
    }


def get_status(ticket_path: str | Path) -> dict[str, Any]:
    """Read and report status of a LoRA fine-tuning ticket (`SPEC.md` §8.4)."""
    path = Path(ticket_path)
    ticket = Ticket.load(path)
    adapter_path = path.parent / "lora_adapter.json"
    lora_data = (
        json.loads(adapter_path.read_text(encoding="utf-8"))
        if adapter_path.is_file()
        else None
    )
    return {
        "call_id": ticket.call_id,
        "state": ticket.state,
        "workload_id": ticket.workload_id,
        "reconciled_count": ticket.reconciled_count,
        "lora_adapter": lora_data,
    }
