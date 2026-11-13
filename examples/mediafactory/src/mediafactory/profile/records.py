"""CSV/JSONL record loading, `dead_letter.jsonl` reporting, and `brief.json` materialization."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mediafactory.profile.application import ApplicationBundle, StageName


class DeadLetterEntry(BaseModel):
    """One row in `runs/<run_id>/dead_letter.jsonl` (`SPEC.md` §3, §7.3, `FR-02`)."""

    model_config = ConfigDict(extra="forbid")

    row_index: int
    record_id: str | None = None
    error_class: str
    reason: str
    raw_row: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )


class Brief(BaseModel):
    """Materialized `items/<id>/brief.json` (`SPEC.md` §3)."""

    model_config = ConfigDict(extra="forbid")

    application: dict[str, Any]
    task: str
    record: dict[str, Any]
    style_name: str
    style: dict[str, Any]
    stages: list[StageName]
    constraints: dict[str, Any]


@dataclass
class RecordLoadResult:
    valid_records: list[dict[str, Any]] = field(default_factory=list)
    dead_letters: list[DeadLetterEntry] = field(default_factory=list)


def load_records(
    path: str | Path,
    *,
    required_columns: list[str],
    limit: int | None = None,
) -> RecordLoadResult:
    """Load CSV or JSONL records, isolating invalid rows into ``dead_letters`` (`FR-02`)."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"records file not found: {source}")

    raw_rows: list[dict[str, Any]] = []
    if source.suffix.lower() == ".jsonl":
        for idx, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raw_rows.append({"__parse_error__": f"invalid JSON on line {idx}: {exc.msg}"})
                continue
            if not isinstance(parsed, dict):
                raw_rows.append({"__parse_error__": f"line {idx} is not a JSON object"})
                continue
            raw_rows.append(parsed)
    else:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                clean = {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k is not None}
                raw_rows.append(clean)

    if limit is not None:
        raw_rows = raw_rows[:limit]

    result = RecordLoadResult()
    seen_ids: set[str] = set()

    for idx, row in enumerate(raw_rows, start=1):
        if "__parse_error__" in row:
            result.dead_letters.append(
                DeadLetterEntry(
                    row_index=idx,
                    record_id=None,
                    error_class="ParseError",
                    reason=str(row["__parse_error__"]),
                    raw_row={},
                )
            )
            continue

        rec_id = str(row.get("id") or "").strip()
        if not rec_id:
            result.dead_letters.append(
                DeadLetterEntry(
                    row_index=idx,
                    record_id=None,
                    error_class="MissingId",
                    reason="record 'id' is missing or blank",
                    raw_row=row,
                )
            )
            continue

        if rec_id in seen_ids:
            result.dead_letters.append(
                DeadLetterEntry(
                    row_index=idx,
                    record_id=rec_id,
                    error_class="DuplicateId",
                    reason=f"duplicate record id {rec_id!r}",
                    raw_row=row,
                )
            )
            continue

        missing = [col for col in required_columns if not str(row.get(col) or "").strip()]
        if missing:
            result.dead_letters.append(
                DeadLetterEntry(
                    row_index=idx,
                    record_id=rec_id,
                    error_class="MissingRequiredColumn",
                    reason=f"missing required columns: {missing}",
                    raw_row=row,
                )
            )
            continue

        seen_ids.add(rec_id)
        normalized = dict(row)
        normalized["id"] = rec_id
        result.valid_records.append(normalized)

    return result


def write_dead_letters(path: str | Path, entries: list[DeadLetterEntry]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.model_dump(mode="json"), sort_keys=True) + "\n")


def build_brief(
    bundle: ApplicationBundle,
    record: dict[str, Any],
    *,
    sampled_stage_ids: dict[StageName, set[str]] | None = None,
) -> Brief:
    """Build the per-record `Brief` object (`SPEC.md` §3)."""
    rec_id = str(record["id"])
    style_name, style_preset = bundle.config.style_for_record(record)
    active_stages: list[StageName] = []
    for stage in bundle.config.stages:
        if sampled_stage_ids and stage.name in sampled_stage_ids:
            if rec_id not in sampled_stage_ids[stage.name]:
                continue
        active_stages.append(stage.name)

    return Brief(
        application={"name": bundle.config.name, "version": bundle.config.version},
        task=bundle.config.task.strip(),
        record=record,
        style_name=style_name,
        style=style_preset.model_dump(mode="json"),
        stages=active_stages,
        constraints=bundle.config.constraints.model_dump(mode="json"),
    )
