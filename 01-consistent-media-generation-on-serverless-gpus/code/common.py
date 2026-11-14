"""Shared helpers for configuration, run directories, metrics, traces, and cost calculations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

# The module's Sana settings: the model pinned to a Hub commit, and the sampling settings of lesson 02.
SANA = {
    "model": "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
    "model_revision": "ac0da2ff55fbe434795be0dce883042e4d49e2fc",
    "variant": "fp16",
    "prompt": (
        "A centered Christmas storybook illustration of a brass telescope glowing softly on snow under a starry "
        "night sky, surrounded by a small decorated Christmas tree, wrapped presents, twinkling lights, gentle "
        "falling snow, warm vibrant colors, whimsical children's-book art style, no people, no text"
    ),
    "seed": 7,
    "params": {"width": 512, "height": 512, "num_inference_steps": 20, "guidance_scale": 4.5, "negative_prompt": ""},
    "image": None,
}
MODULE = "01-consistent-media-generation-on-serverless-gpus"


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def load_pipeline(cfg=SANA):
    """Load the Sana Diffusers pipeline onto the GPU. Used by image.py (lesson 02) and batch_inference.py (lesson 04)."""
    import torch
    from diffusers import SanaPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a GPU runtime with CUDA-enabled PyTorch")
    pipe = SanaPipeline.from_pretrained(
        cfg["model"],
        revision=cfg.get("model_revision"),
        variant=cfg.get("variant", "fp16"),
        torch_dtype=torch.float16,
    ).to("cuda")
    # Sana recipe: fp16 transformer, bf16 text encoder and VAE to prevent black images.
    pipe.text_encoder.to(torch.bfloat16)
    pipe.vae.to(torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_dotenv(path):
    """Read KEY=VALUE lines from a .env file; missing file -> {}."""
    values = {}
    path = Path(path)
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def save_dotenv(path, values):
    """Write KEY=VALUE lines to a .env file readable only by you (it holds the Endpoint token)."""
    path = Path(path)
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    os.chmod(path, 0o600)
    return path


def setup_endpoint(env_file=None):
    """Return the Endpoint URL and token.

    Reads code/.env, then the environment. Prompts for anything still missing.
    Uses getpass for the token so it is never echoed to the notebook output.
    """
    env_file = Path(env_file) if env_file else Path(__file__).with_name(".env")
    saved = load_dotenv(env_file)
    for key in ("GPU", "PRESET", "ENDPOINT_ID"):
        saved.pop(key, None)
    base = saved.get("BASE_URL") or os.getenv("BASE_URL") or ""
    token = saved.get("TOKEN") or os.getenv("TOKEN") or ""
    if not base:
        base = input("Endpoint HTTPS URL: ").strip()
    if not token:
        token = getpass("Endpoint token: ").strip()
    base = base.strip().rstrip("/").removesuffix("/v1")
    assert base.startswith("https://"), "Paste the Endpoint's https:// URL"
    assert token, "Paste the Endpoint token"

    os.environ.update(BASE_URL=base, TOKEN=token)
    saved.update(BASE_URL=base, TOKEN=token)
    save_dotenv(env_file, saved)
    print(f"Endpoint configured (***{token[-4:]})")
    return base, token


def request(base, token, route, body=None, timeout=30):
    """GET (no body) or POST JSON to an Endpoint route with a bearer token; return parsed JSON."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        base.rstrip("/") + route,
        headers=headers,
        data=None if body is None else json.dumps(body).encode(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_ready(base, token, deadline_s=600):
    """Poll /v1/models until the Endpoint returns a non-empty model list.

    RUNNING only means the container started; the server may still be loading weights (502).
    """
    deadline = time.monotonic() + deadline_s
    last_error = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Readiness deadline reached (last error: {last_error}); inspect logs and stop the Endpoint"
            )
        try:
            data = request(base, token, "/v1/models", timeout=min(30, remaining))
            if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
                return data
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 502, 503, 504):
                raise
            last_error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
        time.sleep(min(2, max(0, deadline - time.monotonic())))


def output_dir(path):
    p = (
        Path(path)
        if path
        else Path("runs") / MODULE / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    p.mkdir(parents=True, exist_ok=False)
    return p


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metrics(cfg, mode):
    return {
        "schema_version": 1,
        "mode": mode,
        "startup_s": None,
        "ready_wait_s": None,
        "load_s": None,
        "latency_s": [],
        "billable_s": None,
        "billing_basis": "unavailable",
        "config": cfg,
        "env": {
            k: None
            for k in ("driver", "cuda", "torch", "diffusers", "peak_vram_gb")
        },
        "assets": [],
        "status": "running",
    }


def asset_record(path):
    data = Path(path).read_bytes()
    return {
        "file": Path(path).name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def calculate_cost(
    startup_s: float,
    ready_s: float,
    latency_s: list[float],
    price_per_hour: float,
    counts: tuple[int, ...] | list[int] = (1, 10, 100),
    idle_s: float = 0.0,
    api_price: float | None = None,
) -> dict:
    """Compute marginal cost c, fixed cost F, projected cost per asset at N, and API breakeven."""
    values = [startup_s, ready_s, *latency_s, price_per_hour, idle_s]
    if not latency_s or any(
        v is None or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values
    ):
        raise ValueError("Need finite nonnegative startup, readiness/load and latency measurements")
    if not counts or any(n < 1 for n in counts):
        raise ValueError("Asset counts must be positive")
    if api_price is not None and (not math.isfinite(api_price) or api_price < 0):
        raise ValueError("Invalid API price")

    rate = price_per_hour / 3600.0
    t_gen = statistics.median(latency_s)
    marginal = rate * t_gen
    fixed = rate * (startup_s + ready_s + idle_s)
    breakeven = None
    if api_price is not None:
        breakeven = max(1, math.ceil(fixed / (api_price - marginal))) if api_price > marginal else 0

    return {
        "marginal_compute_usd": marginal,
        "fixed_run_usd": fixed,
        "projected_compute_usd_per_asset": {str(n): fixed / n + marginal for n in counts},
        "breakeven_assets": breakeven,
    }

