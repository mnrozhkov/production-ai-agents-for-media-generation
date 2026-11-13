"""Generate images through an OpenAI-style image Endpoint or locally on a GPU (such as in a DevLab)."""
from __future__ import annotations

import base64
import json
import time
import urllib.error
from pathlib import Path

import common


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

        from generate import load_pipeline

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        pipe = load_pipeline(
            {"model": model, "model_revision": model_revision, "variant": variant}
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


# Course card palette: navy + soft white + electric lime, with pastel gradient accents.
_NAVY = (5, 43, 66)  # #052B42
_LIME = (224, 255, 79)  # #E0FF4F
_SOFT_WHITE = (245, 247, 248)  # #F5F7F8
_WHITE = (255, 255, 255)  # #FFFFFF
_MUTED = (196, 214, 224)  # brighter secondary text on navy
_BLUE = (61, 137, 242)  # #3D89F2
_ORCHID = (240, 178, 254)  # #F0B2FE
_PALE_PINK = (250, 208, 253)  # #FAD0FD
_MINT = (158, 230, 157)  # #9EE69D
_SHARE_HEADLINE = "I'm going to build\na Personalized\nMedia Factory"
_REGISTER_URL = "https://courses.datatalks.club/register/production-ai-agents-generative-media/"

FONT_DIR = Path(__file__).resolve().parent / "fonts"


def _font(size, *, bold=False):
    from PIL import ImageFont

    filename = "Inter-Bold.ttf" if bold else "Inter-Regular.ttf"
    path = FONT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Font file not found: {path.resolve()}")
    return ImageFont.truetype(str(path), size)


def _wrap(draw, text, font, width):
    lines, current = [], ""
    for word in text.split():
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _square(image, side):
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    return image.crop((left, top, left + crop, top + crop)).resize(
        (side, side), Image.Resampling.LANCZOS
    )


def _rounded(image, radius):
    from PIL import Image, ImageDraw

    image = image.convert("RGBA")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, image.size[0], image.size[1]), radius, fill=255)
    image.putalpha(mask)
    return image


def _rgb(color):
    """Accept an RGB tuple or a #RRGGBB string."""
    if isinstance(color, str):
        hex_color = color.removeprefix("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return tuple(color[:3])


def _lerp(a, b, t):
    a, b = _rgb(a), _rgb(b)
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _hgradient(size, left, right):
    """Horizontal RGB gradient of `size` (width, height)."""
    from PIL import Image

    width, height = size
    img = Image.new("RGB", size)
    px = img.load()
    for x in range(width):
        color = _lerp(left, right, x / max(width - 1, 1))
        for y in range(height):
            px[x, y] = color
    return img


def _media_tile(path, side):
    """Photo on a soft-white plate so it reads against the navy field."""
    from PIL import Image

    pad = 10
    radius = max(16, side // 22)
    plate = Image.new("RGB", (side + 2 * pad, side + 2 * pad), _SOFT_WHITE)
    plate = _rounded(plate, radius + pad)
    photo = _rounded(_square(Image.open(path), side), radius)
    plate.paste(photo, (pad, pad), photo)
    return plate


# def share_card(paths, headline=_SHARE_HEADLINE, *, out=None, model=DEFAULT_MODEL):
#     """Lay three images and a headline on one picture for a social post. Returns the PNG path."""
#     from PIL import Image, ImageDraw

#     paths = [Path(path) for path in paths]
#     if len(paths) != 3:
#         raise ValueError(f"share_card expects 3 images, got {len(paths)}")

#     width = 1080
#     margin, gap = 56, 16
#     pad = 6  # soft-white plate around each photo; keep the 2-up stack as tall as the large tile
#     large = 640
#     small = (large - gap) // 2  # two squares that stack to the large one's height
#     stack_gap = gap - 2 * pad
#     qr_side = 280  # lime modules, no plate; sits under the two small images

#     images = [_media_tile(paths[0], large), _media_tile(paths[1], small), _media_tile(paths[2], small)]

#     # Tall enough for a long headline; cropped to the footer once the type is measured.
#     card = Image.new("RGB", (width, 1800), _NAVY)
#     x0, y0 = margin - pad, margin - pad
#     card.paste(images[0], (x0, y0), images[0])
#     x1 = x0 + large + 2 * pad + gap
#     card.paste(images[1], (x1, y0), images[1])
#     card.paste(images[2], (x1, y0 + small + 2 * pad + stack_gap), images[2])

#     draw = ImageDraw.Draw(card)
#     text_x = margin
#     images_bottom = y0 + large + 2 * pad
#     # Banner: blue → pale pink under the photo strip.
#     banner = _hgradient((width - 2 * margin, 8), _BLUE, _PALE_PINK)
#     card.paste(banner, (margin, images_bottom + 24))
#     text_top = images_bottom + 52

#     eyebrow = _font(22, bold=False)
#     draw.text((text_x, text_top), "I'm building", font=eyebrow, fill=_LIME)
#     rule_y = text_top + 36
#     # Rule: lime → mint, then orchid tip for a second accent.
#     rule = _hgradient((96, 6), _LIME, _MINT)
#     tip = _hgradient((24, 6), _ORCHID, _PALE_PINK)
#     card.paste(rule, (text_x, rule_y))
#     card.paste(tip, (text_x + 96, rule_y))

#     # QR under the right column (the two small images), lime on navy, no plate.
#     col_w = small + 2 * pad
#     qr_gap = 28
#     qr_x = x1 + (col_w - qr_side) // 2
#     headline_width = qr_x - qr_gap - text_x  # title stops before the QR column
#     headline_top = rule_y + 28
#     headline_font, lines, leading = _fit_headline(draw, headline, headline_width, qr_side, max_size=64)
#     y = headline_top
#     for line in lines:
#         draw.text((text_x, y), line, font=headline_font, fill=_WHITE)
#         bbox = draw.textbbox((0, 0), line, font=headline_font)
#         y += (bbox[3] - bbox[1]) + leading

#     footer = _font(22, bold=False)
#     footer_lines = _wrap(
#         draw,
#         "Course: Production AI Agents for Generative Media",
#         footer,
#         headline_width,
#     )
#     footer_gap = 6
#     footer_y = y - leading + 28
#     for line in footer_lines:
#         draw.text((text_x, footer_y), line, font=footer, fill=_MUTED)
#         bbox = draw.textbbox((0, 0), line, font=footer)
#         footer_y += (bbox[3] - bbox[1]) + footer_gap
#     text_bottom = footer_y - footer_gap

#     # Under the two small images: same column, vertically centered below the photo strip.
#     height = max(text_bottom + margin, images_bottom + qr_side + margin)
#     if height % 2:
#         height += 1
#     qr_y = images_bottom + (height - images_bottom - qr_side) // 2
#     if qr_y % 2:
#         qr_y += 1

#     # Bottom edge accent: orchid → pale pink.
#     edge = _hgradient((width, 6), _ORCHID, _PALE_PINK)
#     card.paste(edge, (0, height - 6))

#     card = card.crop((0, 0, width, height))
#     card = card.resize((card.width // 2, card.height // 2), Image.Resampling.LANCZOS)
#     # Paste after the resize so modules stay sharp; light modules are navy (no plate).
#     card.paste(_qr(_REGISTER_URL, qr_side // 2), (qr_x // 2, qr_y // 2))

#     path = Path(out) if out else paths[0].with_name("share-card.png")
#     path.parent.mkdir(parents=True, exist_ok=True)
#     card.save(path, format="PNG")
#     return path.resolve()

def share_card(paths, headline=_SHARE_HEADLINE, *, out=None, model=DEFAULT_MODEL):
    """Lay three images and the original text on a branded social card."""
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont, ImageOps

    paths = [Path(path) for path in paths]
    if len(paths) != 3:
        raise ValueError(f"share_card expects 3 images, got {len(paths)}")

    # Course palette.
    navy = "#052B42"
    panel = "#173A4F"
    white = "#F5F7F8"
    lime = "#E0FF4F"
    muted = "#B7CBD6"
    blue = "#3D89F2"
    pink = "#FAD0FD"

    width = 1080
    margin = 48
    gap = 24
    content_width = width - 2 * margin

    # Both columns have the same total height.
    large = 648
    small = 312
    radius = 24

    def media_tile(path, side):
        with Image.open(path) as source:
            source = ImageOps.exif_transpose(source)
            rgb = ImageOps.fit(
                source.convert("RGB"),
                (side, side),
                method=Image.Resampling.LANCZOS,
            )

        mask = Image.new("L", (side, side), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, side - 1, side - 1),
            radius=radius,
            fill=255,
        )

        tile = rgb.convert("RGBA")
        tile.putalpha(mask)
        return tile

    def text_height(draw, text, font):
        box = draw.textbbox((0, 0), text, font=font, anchor="lt")
        return box[3] - box[1]

    # Measure everything before allocating the final canvas.
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    eyebrow_font = ImageFont.truetype(str(FONT_DIR / "Inter-Medium.ttf"), 24)
    headline_font, lines, leading = _fit_headline(
        probe,
        headline,
        content_width,
        270,
        max_size=76,
    )

    headline_heights = [
        text_height(probe, line, headline_font) for line in lines
    ]
    headline_height = (
        sum(headline_heights) + max(0, len(lines) - 1) * leading
    )

    images_bottom = margin + large
    eyebrow_y = images_bottom + 40
    headline_y = eyebrow_y + text_height(
        probe, "I'm building", eyebrow_font
    ) + 24

    divider_y = headline_y + headline_height + 32
    footer_y = divider_y + 28

    # Generate at final resolution and preserve the helper's quiet zone.
    qr_image = _qr(_REGISTER_URL, 176).convert("RGB")
    qr_x = width - margin - qr_image.width

    footer_font = _font(25, bold=False)
    footer_width = qr_x - margin - 40
    if footer_width <= 0:
        raise ValueError("QR image is too wide for the card layout")

    footer_lines = _wrap(
        probe,
        "Course: Production AI Agents for Generative Media",
        footer_font,
        footer_width,
    )
    footer_gap = 8
    footer_heights = [
        text_height(probe, line, footer_font) for line in footer_lines
    ]
    footer_text_height = (
        sum(footer_heights)
        + max(0, len(footer_lines) - 1) * footer_gap
    )

    footer_height = max(footer_text_height, qr_image.height)
    height = footer_y + footer_height + margin

    card = Image.new("RGB", (width, height), navy)
    draw = ImageDraw.Draw(card)

    # Image grid: one hero image and two supporting images.
    tiles = [
        media_tile(paths[0], large),
        media_tile(paths[1], small),
        media_tile(paths[2], small),
    ]
    right_x = margin + large + gap

    for tile, position in zip(
        tiles,
        [
            (margin, margin),
            (right_x, margin),
            (right_x, margin + small + gap),
        ],
    ):
        card.paste(tile, position, tile)

    # Original eyebrow text, with one restrained gradient accent.
    draw.text(
        (margin, eyebrow_y),
        "I'm building",
        font=eyebrow_font,
        fill=lime,
        anchor="lt",
    )
    eyebrow_box = draw.textbbox(
        (margin, eyebrow_y),
        "I'm building",
        font=eyebrow_font,
        anchor="lt",
    )
    accent = _hgradient((96, 5), blue, pink)
    accent_y = eyebrow_y + (
        eyebrow_box[3] - eyebrow_box[1] - accent.height
    ) // 2
    card.paste(accent, (eyebrow_box[2] + 24, accent_y))

    # Original headline, given the full available width.
    y = headline_y
    for line, line_height in zip(lines, headline_heights):
        color = lime if "Media Factory" in line else white
        draw.text(
            (margin, y),
            line,
            font=headline_font,
            fill=color,
            anchor="lt",
        )
        y += line_height + leading

    draw.line(
        (margin, divider_y, width - margin, divider_y),
        fill=panel,
        width=2,
    )

    # Original course attribution, vertically centered beside the QR.
    y = footer_y + (footer_height - footer_text_height) // 2
    for line, line_height in zip(footer_lines, footer_heights):
        draw.text(
            (margin, y),
            line,
            font=footer_font,
            fill=white,
            anchor="lt",
        )
        y += line_height + footer_gap

    qr_y = footer_y + (footer_height - qr_image.height) // 2
    card.paste(qr_image, (qr_x, qr_y))

    card = card.resize((card.width // 2, card.height // 2), Image.Resampling.LANCZOS)

    path = Path(out) if out else paths[0].with_name("share-card.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    card.save(path, format="PNG")
    return path.resolve()

def _qr(url, side):
    """QR code with no plate: lime modules on navy so it sits on the card field."""
    import io

    import segno
    from PIL import Image

    qr = segno.make(url, error="h")
    border = 2
    modules = qr.symbol_size(scale=1, border=border)[0]
    scale = max(1, side // modules)
    buf = io.BytesIO()
    # Light modules match the card so there is no background square.
    qr.save(buf, kind="png", scale=scale, border=border, dark=_LIME, light=_NAVY)
    img = Image.open(buf).convert("RGB")
    canvas = Image.new("RGB", (side, side), _NAVY)
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas


def _fit_headline(draw, headline, width, max_height, max_size=86):
    """Largest serif size whose wrapped lines fit the box under the photos."""
    for size in range(max_size, 34, -2):
        font = _font(size, bold=True)
        lines = []
        for part in headline.splitlines() or [""]:
            lines.extend(_wrap(draw, part, font, width))
        leading = max(6, size // 8)
        total = leading * (len(lines) - 1)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            total += bbox[3] - bbox[1]
        if total <= max_height:
            return font, lines, leading
    font = _font(36, bold=True)
    return font, _wrap(draw, " ".join(headline.split()), font, width), 6
