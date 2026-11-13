"""`config/models.yaml` loader, role resolution, and GPU placement defaults (`SPEC.md` §3, §5)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_YAML = PACKAGE_ROOT / "config" / "models.yaml"
DEFAULT_LORA_YAML = PACKAGE_ROOT / "config" / "lora_config.yaml"


class ConfigError(ValueError):
    """Raised when `models.yaml` or a role configuration is invalid."""


class ComputeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str
    preset: str


class RoleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: str
    model: str
    model_revision: str | None = None
    base_url: str | None = None
    url_env: str | None = None
    api_key_env: str | None = None
    token_env: str | None = None
    price_in_per_1m_usd: float = 0.15
    price_out_per_1m_usd: float = 0.60
    compute: ComputeSpec | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class PolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_image_attempts: int = Field(default=3, ge=1)
    max_video_attempts: int = Field(default=2, ge=1)
    max_judge_parse_retries: int = Field(default=2, ge=1)
    record_budget_usd: float = Field(default=1.50, gt=0.0)
    run_budget_usd: float = Field(default=25.00, gt=0.0)
    gate_video_on_image_pass: bool = True
    gate_speech_on_image_pass: bool = True
    estimate_image_seconds: float = Field(default=12.0, gt=0.0)
    estimate_speech_seconds: float = Field(default=4.0, gt=0.0)
    estimate_video_seconds: float = Field(default=120.0, gt=0.0)
    estimate_audio_seconds: float = Field(default=8.0, gt=0.0)


class JobDefaultsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str = ""
    platform: str = "gpu-l40s-a"
    preset: str = "1gpu-8vcpu-32gb"
    preemptible: bool = True
    disk_gb: int = 200
    timeout_min: int = 60
    mount_path: str = "/data"
    hf_home: str = "/data/models"


class ModelsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles: dict[str, RoleSpec]
    profiles: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)
    policy: PolicySpec = Field(default_factory=PolicySpec)
    job: JobDefaultsSpec = Field(default_factory=JobDefaultsSpec)


class RoleConfig(BaseModel):
    """Resolved role configuration ready for an adapter or spec builder."""

    model_config = ConfigDict(extra="forbid")

    name: str
    adapter: str
    model: str
    model_revision: str | None = None
    base_url: str = ""
    api_key: str = ""
    price_in_per_1m_usd: float = 0.15
    price_out_per_1m_usd: float = 0.60
    compute: ComputeSpec | None = None
    options: dict[str, Any] = Field(default_factory=dict)


def _env(name: str | None) -> str:
    return (os.environ.get(name) or "").strip() if name else ""


class Settings:
    """Loaded `models.yaml` settings with optional profile overrides (e.g. `--profile l40s`)."""

    def __init__(self, models_file: ModelsFile, *, profile: str | None = None) -> None:
        self.file = models_file
        self.profile = profile
        self.policy = models_file.policy
        self.job = models_file.job

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        profile: str | None = None,
    ) -> Settings:
        target = Path(path) if path else DEFAULT_MODELS_YAML
        if not target.is_file():
            raise ConfigError(f"models.yaml not found at {target}")
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigError(f"{target} must contain a YAML mapping")
        models_file = ModelsFile.model_validate(raw)
        if profile and profile not in models_file.profiles:
            raise ConfigError(
                f"unknown hardware profile {profile!r}; available: {sorted(models_file.profiles)}"
            )
        return cls(models_file, profile=profile)

    def role(self, name: str) -> RoleConfig:
        if name not in self.file.roles:
            raise ConfigError(f"unknown role {name!r} in models.yaml")
        base_dict = self.file.roles[name].model_dump()
        if self.profile:
            overrides = self.file.profiles.get(self.profile, {}).get(name, {})
            for k, v in overrides.items():
                if isinstance(v, dict) and isinstance(base_dict.get(k), dict):
                    base_dict[k] = {**base_dict[k], **v}
                else:
                    base_dict[k] = v
        spec = RoleSpec.model_validate(base_dict)
        url = _env(spec.url_env) or (spec.base_url or "")
        key = _env(spec.api_key_env) or _env(spec.token_env)
        return RoleConfig(
            name=name,
            adapter=spec.adapter,
            model=spec.model,
            model_revision=spec.model_revision,
            base_url=url,
            api_key=key,
            price_in_per_1m_usd=spec.price_in_per_1m_usd,
            price_out_per_1m_usd=spec.price_out_per_1m_usd,
            compute=spec.compute,
            options=dict(spec.options),
        )

    def estimate_role_seconds(self, role: str) -> float:
        return {
            "image": self.policy.estimate_image_seconds,
            "speech": self.policy.estimate_speech_seconds,
            "video": self.policy.estimate_video_seconds,
            "audio": self.policy.estimate_audio_seconds,
        }.get(role, 10.0)
