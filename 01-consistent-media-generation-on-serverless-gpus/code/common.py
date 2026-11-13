"""Shared helpers for configuration, run directories, metrics, traces, and cost calculations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("generation.json")
MODULE = "01-consistent-media-generation-on-serverless-gpus"
SPOT_PRICE_API = "https://api.nebius.cloud/billing/v1alpha1/spot-price"  # public, no auth


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def validate_config(value):
    """Check a generation config dict (from a notebook cell or generation.json)."""
    revision = value.get("model_revision")
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("model_revision must be an immutable 40-char Hub commit")
    if not value["prompt"].strip():
        raise ValueError("prompt must not be empty")
    if not isinstance(value["seed"], int) or not 0 <= value["seed"] < 2**63:
        raise ValueError("seed must be a nonnegative integer below 2**63")
    p = value["params"]
    if not math.isfinite(p["guidance_scale"]) or p["guidance_scale"] < 0:
        raise ValueError("guidance_scale must be finite and nonnegative")
    for k in ("width", "height"):
        if not isinstance(p[k], int) or p[k] % 32 or not 256 <= p[k] <= 2048:
            raise ValueError(f"invalid {k}: use a multiple of 32 between 256 and 2048")
    if not 1 <= p["num_inference_steps"] <= 100:
        raise ValueError("steps must be between 1 and 100")
    return value


def config(path=DEFAULT_CONFIG):
    """Load and check a generation config file (used by the Job worker in lesson 04)."""
    value = json.loads(Path(path).read_text())
    if value.get("model_revision") is None:
        raise ValueError("model_revision must be pinned in a config file")
    return validate_config(value)


def as_config(cfg_or_path):
    """Accept a config dict or a path to a JSON config; return a checked copy."""
    if isinstance(cfg_or_path, dict):
        return validate_config(json.loads(json.dumps(cfg_or_path)))
    return config(cfg_or_path)


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


def emit(out, event, **details):
    with (Path(out) / "trace.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "run_id": Path(out).name,
                    "event": event,
                    **details,
                },
                allow_nan=False,
            )
            + "\n"
        )


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


def interval(start, finish):
    a, b = (datetime.fromisoformat(s.replace("Z", "+00:00")) for s in (start, finish))
    if a.tzinfo is None or b.tzinfo is None or b < a:
        raise ValueError("timestamps need timezones and chronological order")
    return (b - a).total_seconds()


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

