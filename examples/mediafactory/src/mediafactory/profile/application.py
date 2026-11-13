"""`application.yaml`, `prompts.yaml`, and `rubric.yaml` schemas, template validator, and sampler."""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mediafactory.core.spec import LoraAdapter, Size

StageName = Literal["plan", "image", "speech", "audio", "video", "compose"]
_APP_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_STYLE_ATTRS = frozenset({"visual", "negative", "motion", "voice", "music"})


class ApplicationError(ValueError):
    """Raised when an application profile or prompt template is invalid."""


class RecordsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1)
    required: list[str] = Field(..., min_length=1)

    @field_validator("required")
    @classmethod
    def _must_include_id(cls, value: list[str]) -> list[str]:
        if "id" not in value:
            raise ValueError("records.required must include 'id'")
        if len(set(value)) != len(value):
            raise ValueError("records.required must not contain duplicate column names")
        return value


class StylePreset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual: str = Field(..., min_length=1)
    negative: str = ""
    motion: str = ""
    voice: str = "af_heart"
    music: str = "warm"
    lora: LoraAdapter | None = None


class StylesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str = Field(..., min_length=1)
    column: str | None = "style"
    presets: dict[str, StylePreset] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _default_exists(self) -> StylesSpec:
        if self.default not in self.presets:
            raise ValueError(f"styles.default {self.default!r} is not in styles.presets")
        return self


class SampleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(..., ge=1)
    seed: int = Field(default=42, ge=0)


class StageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: StageName
    judge: bool = False
    gate_on_image_pass: bool = False
    from_source: Literal["image", "prompt"] | None = Field(default=None, alias="from")
    sample: SampleSpec | None = None


class ImageConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    steps: int = Field(default=20, ge=1, le=200)
    guidance: float = Field(default=4.5, ge=0.0, le=30.0)


class SpeechConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_s: float = Field(default=15.0, gt=0.0, le=600.0)


class AudioConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration_s: float = Field(default=8.0, gt=0.0, le=600.0)


class VideoConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=832, ge=64, le=4096)
    height: int = Field(default=480, ge=64, le=4096)
    duration_s: float = Field(default=5.0, gt=0.0, le=600.0)
    fps: int = Field(default=24, ge=1, le=120)
    frames: int = Field(default=121, ge=1, le=1200)
    steps: int = Field(default=30, ge=1, le=200)
    guidance: float = Field(default=5.0, ge=0.0, le=30.0)


class ConstraintsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: ImageConstraint = Field(default_factory=ImageConstraint)
    speech: SpeechConstraint = Field(default_factory=SpeechConstraint)
    audio: AudioConstraint = Field(default_factory=AudioConstraint)
    video: VideoConstraint = Field(default_factory=VideoConstraint)
    record_budget_usd: float = Field(default=1.50, gt=0.0)
    run_budget_usd: float = Field(default=25.00, gt=0.0)
    pass_rate_floor: float = Field(default=0.70, ge=0.0, le=1.0)


class CopyFieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "enum"] = "string"
    min_length: int = Field(default=1, ge=1)
    max_words: int | None = Field(default=None, ge=1)
    values: list[str] | None = None

    @model_validator(mode="after")
    def _check_enum(self) -> CopyFieldSpec:
        if self.type == "enum" and not self.values:
            raise ValueError("enum copy field requires a non-empty 'values' list")
        return self


class CopySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: dict[str, CopyFieldSpec] = Field(..., min_length=1)

    def validate_output(self, payload: Any) -> dict[str, str]:
        if not isinstance(payload, dict):
            raise ApplicationError(f"planner output must be a JSON object, got {type(payload).__name__}")
        validated: dict[str, str] = {}
        for name, spec in self.fields.items():
            raw = payload.get(name)
            if not isinstance(raw, str) or not raw.strip():
                raise ApplicationError(f"planner output missing or empty field {name!r}")
            text = raw.strip()
            if len(text) < spec.min_length:
                raise ApplicationError(
                    f"planner field {name!r} is shorter than min_length={spec.min_length}"
                )
            if spec.max_words is not None and len(text.split()) > spec.max_words:
                raise ApplicationError(
                    f"planner field {name!r} exceeds max_words={spec.max_words}"
                )
            if spec.type == "enum" and spec.values and text not in spec.values:
                raise ApplicationError(
                    f"planner field {name!r}={text!r} must be one of {spec.values}"
                )
            validated[name] = text
        return validated


class ApplicationConfig(BaseModel):
    """Closed `application.yaml` schema (`SPEC.md` §6.1)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: int = Field(default=1, ge=1)
    task: str = Field(..., min_length=1)
    records: RecordsSpec
    styles: StylesSpec
    stages: list[StageSpec] = Field(..., min_length=1)
    constraints: ConstraintsSpec = Field(default_factory=ConstraintsSpec)
    copy_schema: CopySchema

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _APP_NAME_RE.fullmatch(value):
            raise ValueError(f"application name {value!r} must match ^[a-z][a-z0-9_]{{0,40}}$")
        return value

    @model_validator(mode="after")
    def _validate_stage_graph(self) -> ApplicationConfig:
        seen: list[StageName] = []
        for stage in self.stages:
            if stage.name in seen:
                raise ValueError(f"duplicate stage {stage.name!r} in stages")
            if stage.name == "video" and (stage.from_source or "image") == "image":
                if "image" not in seen:
                    raise ValueError("video stage with 'from: image' requires 'image' earlier in stages")
            if stage.name == "compose":
                if "video" not in seen or not ("speech" in seen or "audio" in seen):
                    raise ValueError(
                        "compose stage requires 'video' and at least one of 'speech' or 'audio'"
                    )
            seen.append(stage.name)

        fields = self.copy_schema.fields
        if "image" in seen and "image_prompt" not in fields:
            raise ValueError("image stage requires 'image_prompt' in copy_schema.fields")
        if "speech" in seen and "speech_text" not in fields:
            raise ValueError("speech stage requires 'speech_text' in copy_schema.fields")
        if "video" in seen and "motion_prompt" not in fields:
            raise ValueError("video stage requires 'motion_prompt' in copy_schema.fields")
        if "audio" in seen and "music_tag" not in fields:
            raise ValueError("audio stage requires 'music_tag' in copy_schema.fields")
        return self

    def stage_map(self) -> dict[StageName, StageSpec]:
        return {s.name: s for s in self.stages}

    def style_for_record(self, record: dict[str, Any]) -> tuple[str, StylePreset]:
        col = self.styles.column
        chosen = str(record.get(col) or "").strip() if col else ""
        if chosen and chosen in self.styles.presets:
            return chosen, self.styles.presets[chosen]
        return self.styles.default, self.styles.presets[self.styles.default]

    def sampled_ids_for_stage(self, stage_name: StageName, valid_record_ids: list[str]) -> set[str]:
        """Deterministically sample record IDs for ``stage_name`` (`SPEC.md` §6.1)."""
        stage = self.stage_map().get(stage_name)
        if stage is None:
            return set()
        sorted_ids = sorted(valid_record_ids)
        if stage.sample is None or stage.sample.n >= len(sorted_ids):
            return set(sorted_ids)
        rng = random.Random(stage.sample.seed)
        return set(rng.sample(sorted_ids, stage.sample.n))

    def image_size(self) -> Size:
        return Size(width=self.constraints.image.width, height=self.constraints.image.height)

    def video_size(self) -> Size:
        return Size(width=self.constraints.video.width, height=self.constraints.video.height)


class PromptsConfig(BaseModel):
    """Closed `prompts.yaml` schema (`SPEC.md` §6.2)."""

    model_config = ConfigDict(extra="forbid")

    system: str = Field(..., min_length=1)
    plan: str = Field(..., min_length=1)


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class RubricConfig(BaseModel):
    """Closed `rubric.yaml` schema (`SPEC.md` §6.3)."""

    model_config = ConfigDict(extra="forbid")

    rubric_version: str = Field(..., min_length=1)
    pass_score: int = Field(default=70, ge=1, le=100)
    weak_score: int = Field(default=35, ge=0, le=99)
    criteria: list[RubricCriterion] = Field(..., min_length=1)


class JudgeVerdict(BaseModel):
    """Closed judge output contract saved to `items/<id>/image/verdict.json` (`SPEC.md` §6.3)."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "RETRY", "WEAK"]
    score: int = Field(..., ge=0, le=100)
    reasons: list[str] = Field(default_factory=list)
    fix: str = ""
    rubric_version: str = "v1"
    judge_model: str = ""


class _NamespaceView:
    """Safe attribute/item wrapper for `{record.<col>}` and `{style.<attr>}` in `str.format_map`."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._data[name]
        raise KeyError(name)

    def __getitem__(self, name: str) -> Any:
        return self._data[name]


def validate_template_placeholders(
    template: str, *, required_columns: list[str]
) -> None:
    """Validate that every `{placeholder}` in ``template`` belongs to the closed namespace."""
    allowed_top = {"task", "fix", "style_names"} | set(required_columns)
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None:
            continue
        expr = field_name.strip()
        if not expr:
            raise ApplicationError("empty '{}' placeholder is not allowed in prompts.yaml")
        if "." in expr:
            prefix, attr = expr.split(".", 1)
            if prefix == "style" and attr in _STYLE_ATTRS:
                continue
            if prefix == "record" and attr in required_columns:
                continue
            raise ApplicationError(
                f"undeclared dotted placeholder {{{expr}}} in prompts.yaml "
                f"(allowed: style.{sorted(_STYLE_ATTRS)} or record.{required_columns})"
            )
        if expr not in allowed_top:
            raise ApplicationError(
                f"undeclared placeholder {{{expr}}} in prompts.yaml "
                f"(allowed: {sorted(allowed_top)}, style.*, record.*)"
            )


@dataclass(frozen=True)
class ApplicationBundle:
    """Loaded and cross-validated application directory."""

    root: Path
    config: ApplicationConfig
    prompts: PromptsConfig
    rubric: RubricConfig

    def render_plan_prompt(self, record: dict[str, Any], *, fix: str = "") -> str:
        _, style_preset = self.config.style_for_record(record)
        style_dict = {
            "visual": style_preset.visual,
            "negative": style_preset.negative,
            "motion": style_preset.motion,
            "voice": style_preset.voice,
            "music": style_preset.music,
        }
        ctx: dict[str, Any] = {
            "task": self.config.task.strip(),
            "fix": fix.strip(),
            "style_names": ", ".join(sorted(self.config.styles.presets)),
            "style": _NamespaceView(style_dict),
            "record": _NamespaceView(record),
        }
        for col in self.config.records.required:
            ctx[col] = record.get(col, "")
        return self.prompts.plan.format_map(ctx).strip()

    def snapshot_dict(self, *, models_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "application": self.config.model_dump(mode="json", by_alias=True),
            "prompts": self.prompts.model_dump(mode="json"),
            "rubric": self.rubric.model_dump(mode="json"),
            "models": models_snapshot or {},
        }


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ApplicationError(f"missing required application file: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ApplicationError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ApplicationError(f"{path} must contain a YAML mapping")
    return loaded


def load_application(app_dir: str | Path) -> ApplicationBundle:
    """Load and validate `application.yaml`, `prompts.yaml`, and `rubric.yaml` (`FR-03`, `FR-05`)."""
    root = Path(app_dir)
    if not root.is_dir():
        raise ApplicationError(f"application directory not found: {root}")
    try:
        config = ApplicationConfig.model_validate(_read_yaml(root / "application.yaml"))
        prompts = PromptsConfig.model_validate(_read_yaml(root / "prompts.yaml"))
        rubric = RubricConfig.model_validate(_read_yaml(root / "rubric.yaml"))
    except ApplicationError:
        raise
    except Exception as exc:
        raise ApplicationError(f"invalid application in {root}: {exc}") from exc

    validate_template_placeholders(
        prompts.system, required_columns=config.records.required
    )
    validate_template_placeholders(
        prompts.plan, required_columns=config.records.required
    )
    return ApplicationBundle(root=root, config=config, prompts=prompts, rubric=rubric)
