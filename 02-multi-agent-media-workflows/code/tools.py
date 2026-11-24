"""Shared tools for lessons 2.2-2.5: typed arguments, the spec hash cache and the GPU backends.

Three kinds of retry, kept apart on purpose:
- Validation retry: bad arguments go back to the model (schema error or ModelRetry). No GPU call.
- Transport retry: the connection never opened (connect timeout), so nothing reached the
  Endpoint. Retried with backoff, at most TRANSPORT_RETRIES times.
- Uncertain submission: the request may have reached the GPU (read timeout, dropped
  connection, or 502/503/504 after sending: a gateway status doesn't prove the generation
  never started). Never resent here: POST isn't idempotent (RFC 9110, section 9.2.2).
  Short tools raise UncertainSubmission; long tools keep a ticket (video_tool.py).
  Call wait_ready() before the first request, so a loading Endpoint doesn't cause these.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import boto3
import requests
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from models import LLM, SANA, LLMSpec
from tracing import Tracer, repo_path

CODE_DIR = Path(__file__).resolve().parent
TRANSPORT_RETRIES = 3


class UncertainSubmission(RuntimeError):
    """The request may have reached the GPU. Check the Endpoint before you send it again."""


# ---------------------------------------------------------------------------
# Settings and credentials: read on the tool side, never sent to the model
# ---------------------------------------------------------------------------


def read_env() -> dict[str, str]:
    """KEY=VALUE lines of code/.env, or {} if there is none."""
    env_file = CODE_DIR / ".env"
    values: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def optional_setting(key: str) -> str | None:
    """A setting from code/.env or the environment, or None. Never asks."""
    return read_env().get(key) or os.getenv(key) or None


def settings(*keys: str, secret: tuple[str, ...] = (), ask: bool = True) -> dict[str, str]:
    """Return the keys from code/.env or the environment; ask for missing ones and save them.

    Keys listed in `secret` are read with getpass, so they never show in notebook output.
    ask=False raises KeyError instead (an MCP server can't prompt: stdin is the protocol).
    """
    saved = read_env()
    for key in keys:
        value = saved.get(key) or os.getenv(key, "")
        if not value and not ask:
            raise KeyError(f"{key} is not set in code/.env or the environment")
        if not value:
            value = (getpass if key in secret else input)(f"{key}: ").strip()
        saved[key] = value
    env_file = CODE_DIR / ".env"
    env_file.write_text("".join(f"{k}={v}\n" for k, v in saved.items()))
    env_file.chmod(0o600)
    return {key: saved[key] for key in keys}


def llm_api_key(spec: LLMSpec = LLM) -> str:
    """Return the API key for an LLM spec from code/.env or the environment; ask once if missing."""
    return settings(spec.api_key_env, secret=(spec.api_key_env,))[spec.api_key_env]


def llm_model(spec: LLMSpec = LLM) -> OpenAIChatModel:
    """Return a PydanticAI model for an LLM on any OpenAI-compatible API (Token Factory by default)."""
    provider = OpenAIProvider(base_url=spec.base_url, api_key=llm_api_key(spec))
    return OpenAIChatModel(spec.model, provider=provider)


# ---------------------------------------------------------------------------
# Run assets: the run folder, copied to the bucket when BUCKET is set
# ---------------------------------------------------------------------------


class Store:
    """Assets of one run. Local run folder always; s3://BUCKET/runs/... too when bucket is set."""

    def __init__(self, folder: Path, bucket: str | None = None):
        self.folder = folder
        self.bucket = bucket
        self.prefix = "runs/" + "/".join(folder.parts[-2:])  # runs/<lesson>/<run_id>
        endpoint = optional_setting("S3_ENDPOINT_URL") or "https://storage.eu-north1.nebius.cloud"
        self._s3 = boto3.client("s3", endpoint_url=endpoint) if bucket else None

    def path(self, name: str) -> Path:
        return self.folder / name

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def uri(self, name: str) -> str:
        """Bucket URI when a bucket is set, else the local path relative to the repository root."""
        if self.bucket:
            return f"s3://{self.bucket}/{self.prefix}/{name}"
        return repo_path(self.path(name))

    def put(self, name: str, data: bytes) -> str:
        """Save bytes to the run folder (and the bucket); return the URI."""
        self.path(name).parent.mkdir(parents=True, exist_ok=True)
        self.path(name).write_bytes(data)
        if self._s3:
            self._s3.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{name}", Body=data)
        return self.uri(name)

    def upload_run(self) -> None:
        """Copy the trace, metrics, tickets and messages to the bucket at the end of a run."""
        if not self._s3:
            return
        for path in self.folder.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".jsonl"):
                key = f"{self.prefix}/{path.relative_to(self.folder)}"
                self._s3.upload_file(str(path), self.bucket, key)


# ---------------------------------------------------------------------------
# Spec hash: same spec, same asset
# ---------------------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec_hash(
    *,
    model: str,
    prompt: str,
    seed: int,
    params: dict[str, Any],
    model_revision: str | None = None,
    image: str | None = None,
    inputs: Sequence[str | Path] = (),
    adapter: str | None = None,
) -> str:
    """sha256 of everything that changes the output, including the bytes of every input file."""
    spec = {
        "model": model,
        "model_revision": model_revision,
        "image": image,  # container image of the Endpoint, as in Module 1 config
        "prompt": prompt,
        "seed": seed,
        "params": params,
        "inputs": [file_sha256(p) for p in inputs],
        "adapter": adapter,
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# GPU backends: anything with generate(prompt, seed=, width=, height=) -> PNG bytes
# ---------------------------------------------------------------------------


class ImageBackend(Protocol):
    workload: str
    workload_id: str

    async def generate(self, prompt: str, *, seed: int, width: int, height: int) -> bytes: ...


class SanaEndpoint:
    """The Sana Endpoint from Module 1: POST /v1/images/generations, PNG bytes back."""

    workload = "endpoint"

    def __init__(self, base_url: str, token: str, endpoint_id: str):
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.workload_id = endpoint_id

    def wait_ready(self, deadline_s: float = 900) -> float:
        """Poll /v1/models until the server lists a model; return the seconds waited.

        RUNNING in the console means the container started, not that the weights are loaded.
        """
        start = time.monotonic()
        while time.monotonic() - start < deadline_s:
            try:
                r = requests.get(f"{self.base_url}/v1/models", headers=self.headers, timeout=30)
                if r.ok and r.json().get("data"):
                    return time.monotonic() - start
            except requests.RequestException:
                pass  # not up yet
            time.sleep(5)
        raise TimeoutError("Endpoint not ready: check its logs in the console, then stop it")

    async def generate(self, prompt: str, *, seed: int, width: int, height: int) -> bytes:
        """Generate one PNG. Retries only a connection that never opened; see the module docstring."""
        body = {
            "model": SANA.model,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": SANA.num_inference_steps,
            "guidance_scale": SANA.guidance_scale,
            "seed": seed,
        }
        url = f"{self.base_url}/v1/images/generations"
        for attempt in range(TRANSPORT_RETRIES + 1):
            try:
                r = await asyncio.to_thread(
                    requests.post,
                    url,
                    json=body,
                    headers=self.headers,
                    timeout=(10, 300),
                )
                break
            except requests.ConnectTimeout:  # never connected: nothing reached the Endpoint
                if attempt == TRANSPORT_RETRIES:
                    raise RuntimeError("Endpoint unreachable after retries") from None
                await asyncio.sleep(2**attempt)
        if r.status_code in (502, 503, 504):  # sent, then the gateway failed: uncertain
            raise UncertainSubmission(
                f"HTTP {r.status_code} after sending: the image may still be generating. "
                "Check the Endpoint before you send it again."
            )
        r.raise_for_status()  # 4xx: our request is wrong; read timeouts propagate (uncertain)
        return base64.b64decode(r.json()["data"][0]["b64_json"])


def gpu_cost(price_per_hour: float | None, t_gen_s: float) -> float:
    """c = rate x t_gen, the Module 1 marginal cost. 0 when no price is given."""
    return (price_per_hour or 0.0) / 3600 * t_gen_s


# ---------------------------------------------------------------------------
# The image tool
# ---------------------------------------------------------------------------


@dataclass
class ToolDeps:
    """What a tool needs at run time. The model never sees any of it."""

    tracer: Tracer
    images: ImageBackend
    store: Store
    gpu_price_per_hour: float | None = None  # USD/h of the Endpoint's GPU, from the pricing page
    attempt: int | None = None  # set by the review loop (2.5)


class AssetRef(BaseModel):
    """What a GPU tool returns to the model: a reference, never the bytes."""

    uri: str
    spec_hash: str
    width: int
    height: int
    workload: str
    workload_id: str
    cache_hit: bool = False
    t_gen_s: float | None = None


Size = Literal["832x480", "480x832", "1024x1024"]
Prompt = Annotated[str, Field(min_length=10, max_length=600)]


async def image_asset(
    images: ImageBackend, store: Store, prompt: str, seed: int, size: Size
) -> AssetRef:
    """Return the keyframe for this spec: from the run folder if it exists, else generate it."""
    width, height = (int(side) for side in size.split("x"))
    params = {**SANA.params, "width": width, "height": height}
    key = spec_hash(
        model=SANA.model,
        model_revision=SANA.model_revision,
        prompt=prompt,
        seed=seed,
        params=params,
    )
    name = f"keyframe-{key[:12]}.png"
    ref = {
        "spec_hash": key,
        "width": width,
        "height": height,
        "uri": store.uri(name),
        "workload": images.workload,
        "workload_id": images.workload_id,
    }
    if store.exists(name):  # same spec as a finished generation: reuse it
        return AssetRef(cache_hit=True, **ref)

    start = time.perf_counter()
    png = await images.generate(prompt, seed=seed, width=width, height=height)
    t_gen_s = round(time.perf_counter() - start, 3)
    store.put(name, png)
    return AssetRef(t_gen_s=t_gen_s, **ref)


def trace_asset(deps: ToolDeps, call_id: str | None, name: str, asset: AssetRef) -> None:
    """Write the tool_call event for one GPU tool result. A cache hit costs nothing."""
    deps.tracer.emit(
        "tool_call",
        name,
        call_id=call_id,
        workload=asset.workload,
        workload_id=asset.workload_id,
        attempt=deps.attempt,
        cache_hit=asset.cache_hit,
        t_gen_s=asset.t_gen_s,
        wall_s=asset.t_gen_s or 0.0,
        cost_usd=gpu_cost(deps.gpu_price_per_hour, asset.t_gen_s or 0.0),
        asset_ref=asset.uri,
    )


async def generate_image(
    ctx: RunContext[ToolDeps],
    prompt: Prompt,
    seed: int = SANA.seed,
    size: Size = "832x480",
) -> AssetRef:
    """Generate one keyframe image and return a reference to the PNG.

    Args:
        prompt: One scene, described visually. No on-screen text.
        seed: Change it to get a different image for the same prompt.
        size: WIDTHxHEIGHT in pixels. 832x480 matches the video frame.
    """
    asset = await image_asset(ctx.deps.images, ctx.deps.store, prompt, seed, size)
    trace_asset(ctx.deps, ctx.tool_call_id, "generate_image", asset)
    return asset
