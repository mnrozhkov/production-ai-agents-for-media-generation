"""`spec.json` — closed generation contract for a single media asset (`FR-03`, `FR-04`, `FR-06`).

Zero imports of ``torch``, ``diffusers``, or any cloud SDK so spec validation works anywhere.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mediafactory.core.cost import default_platform, default_preset, hourly_rate

SPEC_VERSION = 1
Role = Literal["image", "video", "speech", "audio"]
PricingMode = Literal["on-demand", "preemptible"]

_SAFE_PREFIX = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class SpecError(ValueError):
    """Raised when a generation spec fails validation."""


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: PricingMode = "on-demand"
    spot_pricing_policy_id: str | None = None

    @model_validator(mode="after")
    def _check_policy(self) -> Pricing:
        if self.spot_pricing_policy_id and self.mode != "preemptible":
            raise ValueError("spot_pricing_policy_id is only valid when mode='preemptible'")
        return self


class Size(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(..., ge=64, le=4096)
    height: int = Field(..., ge=64, le=4096)

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_32(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError(f"dimension must be a multiple of 32, got {value}")
        return value


class InputImage(BaseModel):
    """Typed keyframe reference for image-to-video (`SPEC.md` §7.1)."""

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(..., min_length=1)
    sha256: str = Field(...)
    mime: Literal["image/png", "image/jpeg", "image/webp"] = "image/png"

    @field_validator("sha256")
    @classmethod
    def _valid_sha(cls, value: str) -> str:
        if not _HEX64.fullmatch(value.lower()):
            raise ValueError("input_image.sha256 must be a 64-character hex SHA-256 digest")
        return value.lower()


class LoraAdapter(BaseModel):
    """Optional style LoRA weights produced by Lesson 3.6 fine-tuning (`SPEC.md` §7.1, §8.4)."""

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(..., min_length=1)
    sha256: str = Field(...)
    scale: float = Field(default=0.8, gt=0.0, le=2.0)
    trigger_word: str | None = None

    @field_validator("sha256")
    @classmethod
    def _valid_sha(cls, value: str) -> str:
        if not _HEX64.fullmatch(value.lower()):
            raise ValueError("lora_adapter.sha256 must be a 64-character hex SHA-256 digest")
        return value.lower()


class GenerationSpec(BaseModel):
    """Closed request contract for one media asset (`SPEC.md` §7.1)."""

    model_config = ConfigDict(extra="forbid")

    spec_version: Literal[1] = SPEC_VERSION
    role: Role
    model: str = Field(..., min_length=1)
    model_revision: str | None = None
    lora_adapter: LoraAdapter | None = None

    prompt: str = Field(..., min_length=1)
    negative_prompt: str | None = None
    voice: str | None = None
    input_image: InputImage | None = None

    brief_ref: str | None = None
    record_id: str = Field(default="single", min_length=1)
    attempt: int = Field(default=1, ge=1)

    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    size: Size | None = None
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0.0, le=30.0)
    duration_s: float | None = Field(default=None, gt=0.0, le=600.0)
    fps: int | None = Field(default=None, ge=1, le=120)
    frames: int | None = Field(default=None, ge=1, le=1200)

    platform: str = Field(default_factory=default_platform)
    preset: str = Field(default_factory=default_preset)
    pricing: Pricing = Field(default_factory=Pricing)
    output_prefix: str = Field(default="runs/default")

    @model_validator(mode="before")
    @classmethod
    def _role_aware_placement_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        role = data.get("role")
        if not isinstance(role, str):
            return data
        filled = dict(data)
        if "platform" not in filled:
            filled["platform"] = default_platform(role)
        if "preset" not in filled:
            filled["preset"] = default_preset(role)
        return filled

    @field_validator("model", "platform", "preset", "record_id")
    @classmethod
    def _no_surrounding_whitespace(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("identifier must be non-empty and free of surrounding whitespace")
        return value

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("model_revision")
    @classmethod
    def _revision_hex(cls, value: str | None) -> str | None:
        if value is not None and not _HEX40.fullmatch(value.lower()):
            raise ValueError("model_revision must be a 40-hex commit SHA or null")
        return value.lower() if value else None

    @field_validator("brief_ref")
    @classmethod
    def _brief_ref_relative(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.replace("\\", "/")
        if text.startswith("/") or text.startswith("../") or "/../" in text:
            raise ValueError("brief_ref must be a relative path with no '..' traversal")
        return text

    @field_validator("output_prefix")
    @classmethod
    def _prefix_relative(cls, value: str) -> str:
        raw = value.replace("\\", "/")
        if raw.startswith("/") or raw.startswith("../") or "/../" in raw or raw in {"", ".", ".."}:
            raise ValueError(f"output_prefix must be a relative POSIX path, got {value!r}")
        text = raw.strip("/")
        if not _SAFE_PREFIX.match(text):
            raise ValueError(
                f"output_prefix must contain only [A-Za-z0-9._-] segments, got {value!r}"
            )
        return text

    def spec_hash(self) -> str:
        """Content-addressed SHA-256 idempotency key over all generation-affecting inputs."""
        canonical = {
            "role": self.role,
            "model": self.model,
            "model_revision": self.model_revision,
            "lora_sha256": self.lora_adapter.sha256 if self.lora_adapter else None,
            "lora_scale": self.lora_adapter.scale if self.lora_adapter else None,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "voice": self.voice,
            "input_image_sha256": self.input_image.sha256 if self.input_image else None,
            "seed": self.seed,
            "size": self.size.model_dump() if self.size else None,
            "steps": self.steps,
            "guidance": self.guidance,
            "duration_s": self.duration_s,
            "fps": self.fps,
            "frames": self.frames,
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def worker_group_key(self) -> tuple[str, str, str | None, str | None]:
        """Grouping key for batch workers so each unique pipeline loads once."""
        return (
            self.role,
            self.model,
            self.model_revision,
            self.lora_adapter.sha256 if self.lora_adapter else None,
        )

    def asset_filename(self) -> str:
        return {
            "image": "asset.png",
            "video": "asset.mp4",
            "speech": "asset.mp3",
            "audio": "asset.mp3",
        }[self.role]

    def asset_key(self, filename: str | None = None) -> str:
        return f"{self.output_prefix}/{filename or self.asset_filename()}"

    def result_key(self) -> str:
        return f"{self.output_prefix}/result.json"

    def estimated_hourly_usd(self) -> float:
        preemptible = self.pricing.mode == "preemptible"
        return hourly_rate(self.platform, self.preset, preemptible=preemptible).usd_per_hour

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_mapping(cls, data: Any) -> GenerationSpec:
        if not isinstance(data, dict):
            raise SpecError(f"spec must be a JSON object, got {type(data).__name__}")
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise SpecError(f"invalid generation spec: {exc}") from exc

    @classmethod
    def from_file(cls, path: str | Path) -> GenerationSpec:
        source = Path(path)
        try:
            raw = source.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SpecError(f"spec file not found: {source}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SpecError(f"spec file is not valid JSON: {source}: {exc}") from exc
        return cls.from_mapping(data)
