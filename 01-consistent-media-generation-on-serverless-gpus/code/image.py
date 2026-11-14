"""Generate images through an OpenAI-style image Endpoint or locally on a GPU (such as in a DevLab)."""
from __future__ import annotations

import base64
import json
import time
import urllib.error
from pathlib import Path

import common
from share_card import share_card

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "ImageGenerator",
    "cached_image",
    "image_comparison",
    "request_id",
    "share_card",
]

DEFAULT_MODEL = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
DEFAULT_REVISION = "ac0da2ff55fbe434795be0dce883042e4d49e2fc"


def _parse_size(size: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in size.lower().split("x", 1))
    except ValueError as exc:
        raise ValueError(f"size must look like 1024x1024, got {size!r}") from exc
    for side in (width, height):
        if not 256 <= side <= 2048 or side % 32:
            raise ValueError(
                f"size sides must be multiples of 32 between 256 and 2048, got {size!r}"
            )
    return width, height


class ImageGenerator:
    """Generate images via an OpenAI-style Endpoint (POST /v1/images/generations) or an in-process GPU pipeline."""

    def __init__(
        self,
        endpoint: str | None = None,
        token: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        pipe: object | None = None,
    ):
        if pipe is None and not endpoint:
            raise ValueError("Provide an Endpoint URL or use ImageGenerator.local()")
        self.endpoint = endpoint.rstrip("/") if endpoint else None
        self.token = token
        self.model = model
        self.model_revision: str | None = None
        self.variant: str | None = None
        self._pipe = pipe
        self.last_s: float | None = None
        self.load_s: float | None = None
        self.load_vram_gb: float | None = None
        self.peak_vram_gb: float | None = None

    @classmethod
    def local(
        cls,
        model: str = DEFAULT_MODEL,
        *,
        model_revision: str | None = DEFAULT_REVISION,
        variant: str = "fp16",
    ) -> ImageGenerator:
        """Load the Diffusers pipeline onto the local CUDA GPU once and return a ready ImageGenerator."""
        import torch

        from common import load_pipeline
        from models import SANA

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        pipe = load_pipeline(
            SANA.with_overrides(model=model, model_revision=model_revision, variant=variant)
        )
        torch.cuda.synchronize()
        load_s = time.perf_counter() - start
        load_vram_gb = torch.cuda.max_memory_allocated() / 1e9

        gen = cls(None, None, model, pipe=pipe)
        gen.model_revision = model_revision
        gen.variant = variant
        gen.load_s = load_s
        gen.load_vram_gb = load_vram_gb
        gen.peak_vram_gb = load_vram_gb
        print(f"Loaded {model} in {load_s:.1f} s | VRAM: {load_vram_gb:.2f} GB")
        return gen

    def generate_image(
        self,
        prompt: str,
        *,
        out: str | Path = "runs/first-image.png",
        seed: int | None = 7,
        size: str = "512x512",
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        negative_prompt: str = "",
        timeout: int = 300,
    ) -> Path:
        if self._pipe is not None:
            import torch

            width, height = _parse_size(size)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            rng = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
            image = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=rng,
            ).images[0]
            torch.cuda.synchronize()
            self.last_s = time.perf_counter() - start
            self.peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"Generated in {self.last_s:.1f} s | Peak VRAM: {self.peak_vram_gb:.2f} GB")
            path = Path(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG")
            return path.resolve()

        request_data = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
        }
        if seed is not None:  # no seed: the server picks a random one, so every call differs
            request_data["seed"] = seed
        start = time.perf_counter()
        try:
            payload = common.request(
                self.endpoint,
                self.token,
                "/v1/images/generations",
                request_data,
                timeout=timeout,
            )
        except urllib.error.HTTPError as exc:
            # Surface the server's message (e.g. 400 "this endpoint serves ...").
            detail = exc.read(500).decode(errors="replace") if exc.fp else ""
            raise RuntimeError(f"HTTP {exc.code} from /v1/images/generations: {detail}") from exc
        self.last_s = time.perf_counter() - start
        print(f"Generated in {self.last_s:.1f} s")
        png = base64.b64decode(payload["data"][0]["b64_json"], validate=True)
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Expected a PNG response")
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        return path.resolve()


def request_id(**request):
    """Short stable id of a generation request (model, prompt, seed, settings): same request, same id."""
    import hashlib

    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:8]


def cached_image(generator, folder, prompt, seed, *, name, **opts):
    """Generate once per model, prompt and seed. Re-running reuses the PNG if those are unchanged."""
    request = {"model": generator.model, "prompt": prompt, "seed": seed}
    path = Path(folder) / f"{name}-{request_id(**request)}.png"
    if not path.exists():
        generator.generate_image(prompt, out=path, seed=seed, **opts)
    return path.resolve()


def image_comparison(paths, height=320):
    """Build one horizontal row, caption each image with its file name, and return JPEG bytes for the notebook."""
    import io

    from PIL import Image, ImageDraw

    pad, caption_h = 12, 28
    tiles = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        tiles.append(img.resize((round(img.width * height / img.height), height)))
    width = sum(t.width for t in tiles) + pad * (len(tiles) + 1)
    sheet = Image.new("RGB", (width, height + caption_h + 2 * pad), "white")
    draw = ImageDraw.Draw(sheet)
    x = pad
    for path, tile in zip(paths, tiles):
        sheet.paste(tile, (x, pad))
        draw.text((x, pad + height + 6), Path(path).name, fill="black")
        x += tile.width + pad
    preview = io.BytesIO()
    sheet.save(preview, format="JPEG", quality=80)
    return preview.getvalue()
