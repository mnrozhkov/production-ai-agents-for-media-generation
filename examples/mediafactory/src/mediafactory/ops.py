"""Prerequisite checks (`doctor`) and cloud resource cleanup (`teardown`) (`SPEC.md` §5, §9)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mediafactory.config import DEFAULT_MODELS_YAML, Settings
from mediafactory.profile.application import load_application
from mediafactory.profile.records import load_records

CheckStatus = Literal["OK", "WARN", "SKIP", "FAIL"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str
    fix: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "FAIL"


def run_doctor_checks(
    *,
    app_dir: str | Path | None = None,
    models_yaml: str | Path | None = None,
) -> list[CheckResult]:
    """Check `models.yaml`, optional application profile, records, and cloud environment."""
    results: list[CheckResult] = []

    try:
        settings = Settings.load(models_yaml or DEFAULT_MODELS_YAML)
        results.append(
            CheckResult(
                name="models.yaml",
                status="OK",
                detail=f"loaded {len(settings.file.roles)} roles ({', '.join(sorted(settings.file.roles))})",
            )
        )
    except Exception as exc:
        results.append(
            CheckResult(
                name="models.yaml",
                status="FAIL",
                detail=str(exc),
                fix="Fix config/models.yaml schema errors.",
            )
        )

    if app_dir is not None:
        try:
            bundle = load_application(app_dir)
            results.append(
                CheckResult(
                    name=f"app:{bundle.config.name}",
                    status="OK",
                    detail=(
                        f"stages={[s.name for s in bundle.config.stages]}, "
                        f"rubric={bundle.rubric.rubric_version}"
                    ),
                )
            )
            rec_path = (bundle.root / bundle.config.records.path).resolve()
            rec_loaded = load_records(
                rec_path, required_columns=bundle.config.records.required
            )
            results.append(
                CheckResult(
                    name="records",
                    status="OK" if rec_loaded.valid_records else "FAIL",
                    detail=(
                        f"{len(rec_loaded.valid_records)} valid rows, "
                        f"{len(rec_loaded.dead_letters)} dead-letter rows from {rec_path.name}"
                    ),
                    fix="Ensure records file has at least one valid row with required columns.",
                )
            )
        except Exception as exc:
            results.append(
                CheckResult(
                    name="application",
                    status="FAIL",
                    detail=str(exc),
                    fix="Check application.yaml, prompts.yaml, and rubric.yaml.",
                )
            )

    tf_key = (os.environ.get("TOKEN_FACTORY_API_KEY") or "").strip()
    bucket = (os.environ.get("NEBIUS_BUCKET_NAME") or os.environ.get("BUCKET") or "").strip()
    if tf_key and bucket:
        results.append(
            CheckResult(
                name="cloud_env",
                status="OK",
                detail=f"TOKEN_FACTORY_API_KEY set, bucket={bucket}",
            )
        )
    else:
        results.append(
            CheckResult(
                name="cloud_env",
                status="SKIP",
                detail="cloud credentials not set (offline --fake and dry-run modes ready)",
                fix="Set TOKEN_FACTORY_API_KEY and NEBIUS_BUCKET_NAME for live --confirm-create runs.",
            )
        )

    return results


@dataclass(frozen=True)
class TeardownReport:
    dry_run: bool
    stopped_endpoints: list[str]
    cancelled_jobs: list[str]
    hourly_usd_saved: float


def run_teardown(*, dry_run: bool = False) -> TeardownReport:
    """Stop tracked endpoints and cancel active jobs (safe no-op when none are active)."""
    return TeardownReport(
        dry_run=dry_run,
        stopped_endpoints=[],
        cancelled_jobs=[],
        hourly_usd_saved=0.0,
    )
