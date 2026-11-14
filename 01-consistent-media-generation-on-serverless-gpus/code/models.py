"""Model specs for Module 1: one frozen Pydantic spec per modality."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class LLMSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    base_url: str = "https://api.tokenfactory.nebius.com/v1"
    temperature: float = 0.8

    def with_overrides(self, **overrides: object) -> LLMSpec:
        return self.model_copy(update=overrides)


class ImageSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    model_revision: str | None = None
    variant: str = "fp16"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 4.5
    negative_prompt: str = ""
    seed: int = 7
    gpu: str = "L40S"

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def params(self) -> dict[str, int | float | str]:
        return {
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
        }

    def with_overrides(self, **overrides: object) -> ImageSpec:
        return self.model_copy(update=overrides)


class ImageEditSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 20
    output_format: str = "png"
    seed: int = 7
    gpu: str = "H100"

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    def with_overrides(self, **overrides: object) -> ImageEditSpec:
        return self.model_copy(update=overrides)


class VideoSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    width: int = 832
    height: int = 480
    num_frames: int = 81
    fps: int = 16
    num_inference_steps: int = 20
    guidance_scale: float = 1.0
    guidance_scale_2: float = 1.0  # Wan2.2 A14B has two experts; this is the second one's guidance
    seed: int = 0
    gpu: str = "H100"

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    @property
    def params(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "guidance_scale_2": self.guidance_scale_2,
        }

    def with_overrides(self, **overrides: object) -> VideoSpec:
        return self.model_copy(update=overrides)


class SpeechSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    lang_code: str = "a"
    voice: str = "af_bella"
    sample_rate: int = 24000
    gpu: str = "L40S"

    def with_overrides(self, **overrides: object) -> SpeechSpec:
        return self.model_copy(update=overrides)


class MusicSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    task_type: str = "text2music"
    thinking: bool = False
    audio_duration: int = 8
    inference_steps: int = 8
    audio_format: str = "mp3"
    gpu: str = "H100"

    def with_overrides(self, **overrides: object) -> MusicSpec:
        return self.model_copy(update=overrides)


GPT_OSS = LLMSpec(model="openai/gpt-oss-120b")

SANA = ImageSpec(
    model="Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    model_revision="ac0da2ff55fbe434795be0dce883042e4d49e2fc",
)

QWEN_EDIT = ImageEditSpec(model="Qwen/Qwen-Image-Edit-2511")

WAN_I2V = VideoSpec(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers")

KOKORO = SpeechSpec(model="hexgrad/Kokoro-82M")

ACE_STEP = MusicSpec(model="ACE-Step/Ace-Step1.5")
