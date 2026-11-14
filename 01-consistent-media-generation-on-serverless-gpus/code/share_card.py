"""Build a branded social share card from three practice images."""
from __future__ import annotations

from pathlib import Path

# Course card palette: navy + soft white + electric lime, with pastel gradient accents.
_NAVY = "#052B42"
_PANEL = "#173A4F"
_WHITE = "#F5F7F8"
_LIME = "#E0FF4F"
_MUTED = "#B7CBD6"
_BLUE = "#3D89F2"
_PALE_PINK = "#FAD0FD"

_SHARE_HEADLINE = "I'm going to build\na Personalized\nMedia Factory"

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


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


def _fit_headline(draw, headline, width, max_height, max_size=86):
    """Largest bold size whose wrapped lines fit the box under the photos."""
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


def share_card(paths, headline=_SHARE_HEADLINE, *, out=None):
    """Lay three images and the headline on a branded social card."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    paths = [Path(path) for path in paths]
    if len(paths) != 3:
        raise ValueError(f"share_card expects 3 images, got {len(paths)}")

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
    headline_y = (
        eyebrow_y + text_height(probe, "I'm building", eyebrow_font) + 24
    )

    divider_y = headline_y + headline_height + 32
    footer_y = divider_y + 28

    footer_font = ImageFont.truetype(str(FONT_DIR / "Inter-Medium.ttf"), 38)
    footer_lines = _wrap(
        probe,
        "Course: Production AI Agents for Generative Media",
        footer_font,
        content_width,
    )
    footer_gap = 10
    footer_heights = [
        text_height(probe, line, footer_font) for line in footer_lines
    ]
    footer_text_height = (
        sum(footer_heights)
        + max(0, len(footer_lines) - 1) * footer_gap
    )

    height = footer_y + footer_text_height + margin

    card = Image.new("RGB", (width, height), _NAVY)
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

    # Eyebrow text, with one restrained gradient accent.
    draw.text(
        (margin, eyebrow_y),
        "I'm building",
        font=eyebrow_font,
        fill=_LIME,
        anchor="lt",
    )
    eyebrow_box = draw.textbbox(
        (margin, eyebrow_y),
        "I'm building",
        font=eyebrow_font,
        anchor="lt",
    )
    accent = _hgradient((96, 5), _BLUE, _PALE_PINK)
    accent_y = eyebrow_y + (
        eyebrow_box[3] - eyebrow_box[1] - accent.height
    ) // 2
    card.paste(accent, (eyebrow_box[2] + 24, accent_y))

    # Headline across the full available width.
    y = headline_y
    for line, line_height in zip(lines, headline_heights):
        draw.text(
            (margin, y),
            line,
            font=headline_font,
            fill=_WHITE,
            anchor="lt",
        )
        y += line_height + leading

    draw.line(
        (margin, divider_y, width - margin, divider_y),
        fill=_PANEL,
        width=2,
    )

    # Course attribution across the full width so it remains legible at small sizes.
    y = footer_y
    for line, line_height in zip(footer_lines, footer_heights):
        draw.text(
            (margin, y),
            line,
            font=footer_font,
            fill=_WHITE,
            anchor="lt",
        )
        y += line_height + footer_gap

    path = Path(out) if out else paths[0].with_name("share-card.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    card.save(path, format="PNG")
    return path.resolve()
