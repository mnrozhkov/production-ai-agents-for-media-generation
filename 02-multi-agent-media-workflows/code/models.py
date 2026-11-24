"""Pinned model specs and the task data (brief, plan) for Module 2."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# The default LLM provider. Token Factory is OpenAI-compatible. Base URL from its quickstart.
TOKEN_FACTORY_URL = "https://api.tokenfactory.nebius.com/v1/"


# ---------------------------------------------------------------------------
# Model specs: every model ID and revision used in Module 2 lives here
# ---------------------------------------------------------------------------


class LLMSpec(BaseModel):
    """An LLM on any OpenAI-compatible API.

    api_key_env names the key in code/.env (or the environment) that holds the API key.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    base_url: str = TOKEN_FACTORY_URL
    api_key_env: str = "NEBIUS_API_KEY"


class ImageSpec(BaseModel):
    """Text-to-image settings for the keyframe."""

    model_config = ConfigDict(frozen=True)

    model: str
    model_revision: str | None = None
    width: int = 832
    height: int = 480
    num_inference_steps: int = 20
    guidance_scale: float = 4.5
    seed: int = 7
    gpu: str = "L40S"

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def params(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }


class VideoSpec(BaseModel):
    """Image-to-video settings."""

    model_config = ConfigDict(frozen=True)

    model: str
    model_revision: str | None = None
    width: int = 832
    height: int = 480
    num_frames: int = 81
    fps: int = 16
    num_inference_steps: int = 20
    guidance_scale: float = 1.0
    seed: int = 0
    gpu: str = "H100"

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    @property
    def params(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }


# Candidate IDs from the Token Factory docs examples. [verify: T3 bench pins the LLM,
# T9 pins the judge]. Model IDs on Token Factory change; change them here only.
#
# The agent's LLM. Any OpenAI-compatible API with tool calling works: set model, base_url
# and api_key_env. The notebooks ask for the key once and save it to code/.env. Examples:
#   LLM = LLMSpec(model="<model id>", base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY")
#   LLM = LLMSpec(model="<model id>", base_url="http://localhost:11434/v1", api_key_env="OLLAMA_API_KEY")
# Ollama accepts any non-empty key.
LLM = LLMSpec(model="Qwen/Qwen3-30B-A3B-Instruct-2507")
JUDGE = LLMSpec(model="Qwen/Qwen2-VL-72B-Instruct")

# Same Sana revision as Module 1. 832x480 so the keyframe matches the video frame.
SANA = ImageSpec(
    model="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    model_revision="ac0da2ff55fbe434795be0dce883042e4d49e2fc",
)
WAN_I2V = VideoSpec(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers")


# ---------------------------------------------------------------------------
# Task data: what the learner writes (brief.yaml) and what the planner returns
# ---------------------------------------------------------------------------


class Brief(BaseModel):
    """A short video ad brief. The fields of the Homework 2 brief.yaml."""

    product: str
    audience: str
    format: str
    shot: str
    must_show: str
    must_not: str

    @field_validator("format", mode="before")
    @classmethod
    def aspect_ratio(cls, value: object) -> object:
        """YAML 1.1 reads an unquoted 16:9 as the base-60 number 969; turn it back."""
        return f"{value // 60}:{value % 60}" if isinstance(value, int) else value

    @classmethod
    def from_yaml(cls, path: str | Path) -> Brief:
        """Read a brief.yaml file."""
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


class Plan(BaseModel):
    """The planner's output: one prompt per generation step."""

    image_prompt: str = Field(description="Prompt for one keyframe image that shows the shot")
    motion_prompt: str = Field(description="What moves in the 5-second clip made from it")
