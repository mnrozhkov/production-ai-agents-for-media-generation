"""Offline stand-ins for Module 2: a scripted model and placeholder GPU backends.

No network, no GPU, no credits. MODE = "fake" uses all of them; MODE = "fake-gpu" keeps
the real LLM and uses only the placeholder GPU backends.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from pathlib import Path
from typing import Any

import pydantic_ai.models
from PIL import Image, ImageDraw
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def block_real_models(block: bool = True) -> None:
    """Make any request to a real LLM fail, so a fake run can't spend tokens by accident.

    The switch is process-wide and stays set in a notebook kernel. block=False turns real
    requests back on, so a notebook can change MODE without a restart.
    """
    pydantic_ai.models.ALLOW_MODEL_REQUESTS = not block


# ---------------------------------------------------------------------------
# Scripted model
# ---------------------------------------------------------------------------


def call(tool: str, **args: Any) -> ToolCallPart:
    """One tool call for a script."""
    return ToolCallPart(tool_name=tool, args=args)


def scripted_model(*turns: Any) -> FunctionModel:
    """A model that answers with the given turns, in order, whatever it is asked.

    A turn is a tool call (call(...)), a list of tool calls, a str (final text),
    a dict (the structured output, sent through the agent's output tool) or a function
    of the messages that returns the final text.
    """
    script = list(turns)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not script:
            raise RuntimeError("The scripted model ran out of turns")
        turn = script.pop(0)
        if isinstance(turn, str):
            parts = [TextPart(turn)]
        elif isinstance(turn, dict):
            parts = [ToolCallPart(info.output_tools[0].name, turn)]
        elif callable(turn):
            parts = [TextPart(turn(messages))]
        elif isinstance(turn, ToolCallPart):
            parts = [turn]
        else:
            parts = list(turn)
        return ModelResponse(parts=parts)

    return FunctionModel(respond, model_name="scripted")


# ---------------------------------------------------------------------------
# Placeholder GPU backends
# ---------------------------------------------------------------------------


def placeholder_png(text: str, width: int, height: int, seed: int = 0) -> bytes:
    """A flat-colour PNG with the prompt written on it. Same inputs, same bytes."""
    shade = (seed * 47) % 200
    img = Image.new("RGB", (width, height), (40 + shade // 4, 60, 90 + shade // 2))
    draw = ImageDraw.Draw(img)
    for i in range(0, len(text), 60):
        draw.text((16, 16 + 14 * (i // 60)), text[i : i + 60], fill="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class PlaceholderImages:
    """Stands in for the Sana Endpoint: same arguments, a placeholder PNG after sleep_s."""

    workload = "none"
    workload_id = "placeholder-image"

    def __init__(self, sleep_s: float = 0.2):
        self.sleep_s = sleep_s
        self.requests = 0  # what the "GPU" received, including requests the client gave up on

    async def generate(self, prompt: str, *, seed: int, width: int, height: int) -> bytes:
        """Return PNG bytes, like the Endpoint."""
        self.requests += 1
        await asyncio.sleep(self.sleep_s)
        return placeholder_png(prompt, width, height, seed)


FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 64


class PlaceholderVideos:
    """Stands in for the Videos API on the Wan2.2 Endpoint (create, get, list, content, delete).

    Its state lives in a JSON file, so a video "keeps rendering" while the agent is dead.
    A video is queued for 0.5 s, in_progress until render_s, then completed.
    """

    workload = "none"
    workload_id = "placeholder-video"

    def __init__(self, state_dir: Path, render_s: float = 3.0):
        self.path = Path(state_dir) / "fake_endpoint.json"
        self.render_s = render_s

    def _load(self) -> dict[str, dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self, videos: dict[str, dict]) -> None:
        self.path.write_text(json.dumps(videos, indent=2))

    def create(self, prompt: str, image: bytes, *, user: str, params: dict) -> dict:
        """POST /v1/videos."""
        videos = self._load()
        video = {"id": f"video-{uuid.uuid4().hex[:8]}", "user": user, "created_at": time.time()}
        videos[video["id"]] = video
        self._save(videos)
        return {**video, "status": "queued"}

    def get(self, video_id: str) -> dict:
        """GET /v1/videos/{video_id}. KeyError if the Endpoint doesn't know the video."""
        video = self._load()[video_id]
        age = time.time() - video["created_at"]
        status = "queued" if age < 0.5 else "in_progress" if age < self.render_s else "completed"
        return {**video, "status": status}

    def list(self) -> list[dict]:
        """GET /v1/videos."""
        return [self.get(video_id) for video_id in self._load()]

    def content(self, video_id: str) -> bytes:
        """GET /v1/videos/{video_id}/content."""
        assert self.get(video_id)["status"] == "completed"
        return FAKE_MP4

    def delete(self, video_id: str) -> None:
        """DELETE /v1/videos/{video_id}."""
        videos = self._load()
        videos.pop(video_id, None)
        self._save(videos)

    def restart(self) -> None:
        """Simulate an Endpoint restart: every stored video is gone."""
        self._save({})
