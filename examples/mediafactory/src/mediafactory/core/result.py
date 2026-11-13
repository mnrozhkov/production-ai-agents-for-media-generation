"""`result.json` — per-asset provenance, measurements, marginal cost, and check report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mediafactory.core.spec import GenerationSpec, PricingMode

RESULT_VERSION = 1
ResultStatus = Literal["ok", "failed", "skipped"]
MediaMime = Literal[
    "image/png",
    "image/jpeg",
    "image/webp",
    "audio/mpeg",
    "audio/wav",
    "video/mp4",
    "video/webm",
]


class ResultError(ValueError):
    """Raised when a result report cannot be constructed or validated."""


class ModelRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(..., min_length=1)
    revision: str | None = None
    digest: str | None = None

    @field_validator("digest")
    @classmethod
    def _digest_shape(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("sha256:"):
            raise ValueError("digest must start with 'sha256:' when present")
        return value


class AssetMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., min_length=1)
    sha256: str = Field(..., min_length=64, max_length=64)
    bytes: int = Field(..., ge=1)
    mime: MediaMime
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    duration_s: float | None = Field(default=None, gt=0.0)


class Measurements(BaseModel):
    """Timing and memory windows (`SPEC.md` §7.2). Unmeasured fields are ``None``, never ``0``."""

    model_config = ConfigDict(extra="forbid")

    startup_s: float | None = Field(default=None, ge=0.0)
    image_pull_s: float | None = Field(default=None, ge=0.0)
    load_s: float | None = Field(default=None, ge=0.0)
    gen_s: float | None = Field(default=None, ge=0.0)
    peak_vram_gb: float | None = Field(default=None, ge=0.0)
    gpu: str | None = None


class CostReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rate_usd_per_hour: float = Field(..., ge=0.0)
    marginal_compute_usd: float = Field(..., ge=0.0)
    pricing_mode: PricingMode = "on-demand"
    basis: str = Field(..., min_length=1)


class DeterministicCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    errors: list[str] = Field(default_factory=list)


class SpeechCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str
    reference_text: str
    wer: float = Field(..., ge=0.0)
    ok: bool
    cost_usd: float = Field(default=0.0, ge=0.0)


class CheckReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deterministic: DeterministicCheck
    speech_check: SpeechCheck | None = None


class ResultReport(BaseModel):
    """Closed per-asset provenance and measurement document (`SPEC.md` §7.2)."""

    model_config = ConfigDict(extra="forbid")

    result_version: Literal[1] = RESULT_VERSION
    status: ResultStatus
    fake: bool = False
    cache_hit: bool = False
    spec_hash: str = Field(..., min_length=64, max_length=64)
    spec: GenerationSpec
    worker_run_id: str | None = None
    model: ModelRef
    asset: AssetMeta | None = None
    measurements: Measurements = Field(default_factory=Measurements)
    cost: CostReport
    checks: CheckReport | None = None
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_mapping(cls, data: Any) -> ResultReport:
        if not isinstance(data, dict):
            raise ResultError(f"result must be a JSON object, got {type(data).__name__}")
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise ResultError(f"invalid result report: {exc}") from exc

    @classmethod
    def from_file(cls, path: str | Path) -> ResultReport:
        source = Path(path)
        try:
            raw = source.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ResultError(f"result file not found: {source}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ResultError(f"result file is not valid JSON: {source}: {exc}") from exc
        return cls.from_mapping(data)
