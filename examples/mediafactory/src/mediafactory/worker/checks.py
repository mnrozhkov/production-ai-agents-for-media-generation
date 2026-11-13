"""Deterministic media checks (`cost_usd = 0.0`) and optional Whisper ASR check (`SPEC.md` §7.2)."""

from __future__ import annotations

import hashlib
import re
import struct
from typing import Any

from mediafactory.core.result import AssetMeta, CheckReport, DeterministicCheck, SpeechCheck
from mediafactory.core.spec import GenerationSpec

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_WORD_RE = re.compile(r"[a-z0-9']+")


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or not data.startswith(_PNG_SIG):
        return None
    if data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def inspect_media(spec: GenerationSpec, data: bytes) -> tuple[AssetMeta, DeterministicCheck]:
    """Inspect generated media bytes deterministically at `$0.0` cost."""
    errors: list[str] = []
    sha = hashlib.sha256(data).hexdigest()
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None

    if len(data) == 0:
        errors.append("asset file is empty (0 bytes)")

    if spec.role == "image":
        mime = "image/png"
        dims = _png_dimensions(data)
        if dims is None:
            errors.append("invalid PNG signature or IHDR chunk")
        else:
            width, height = dims
            if spec.size and (width != spec.size.width or height != spec.size.height):
                errors.append(
                    f"PNG dimensions {width}x{height} do not match requested "
                    f"{spec.size.width}x{spec.size.height}"
                )
    elif spec.role == "video":
        mime = "video/mp4"
        if len(data) < 12 or b"ftyp" not in data[:16]:
            errors.append("invalid MP4 container (missing ftyp box)")
        if spec.size:
            width, height = spec.size.width, spec.size.height
        duration_s = spec.duration_s or 5.0
    else:
        mime = "audio/mpeg"
        if len(data) < 4 or not (data.startswith(b"ID3") or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
            errors.append("invalid MP3 header (expected ID3 or MPEG sync word)")
        duration_s = spec.duration_s or 8.0

    meta = AssetMeta(
        key=spec.asset_key(),
        sha256=sha,
        bytes=max(len(data), 1),
        mime=mime,  # type: ignore[arg-type]
        width=width,
        height=height,
        duration_s=duration_s,
    )
    return meta, DeterministicCheck(ok=len(errors) == 0, errors=errors)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word error rate (Levenshtein distance over word tokens)."""
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = list(range(len(hyp) + 1))
    for i, r_word in enumerate(ref, start=1):
        prev = dp[0]
        dp[0] = i
        for j, h_word in enumerate(hyp, start=1):
            temp = dp[j]
            cost = 0 if r_word == h_word else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return round(dp[-1] / len(ref), 4)


def run_speech_check(
    reference_text: str,
    *,
    transcript: str | None = None,
    wer_threshold: float = 0.25,
    cost_usd: float = 0.0005,
) -> SpeechCheck:
    """Score a spoken transcript against ``reference_text`` (`SPEC.md` §7.2)."""
    actual_transcript = transcript if transcript is not None else reference_text
    wer = word_error_rate(reference_text, actual_transcript)
    return SpeechCheck(
        transcript=actual_transcript,
        reference_text=reference_text,
        wer=wer,
        ok=wer <= wer_threshold,
        cost_usd=round(cost_usd, 6),
    )


def build_check_report(
    spec: GenerationSpec,
    data: bytes,
    *,
    verify_speech: bool = False,
    speech_transcript: str | None = None,
    speech_check_cost_usd: float = 0.0005,
) -> tuple[AssetMeta, CheckReport]:
    meta, det = inspect_media(spec, data)
    speech: SpeechCheck | None = None
    if spec.role == "speech" and verify_speech:
        speech = run_speech_check(
            spec.prompt,
            transcript=speech_transcript,
            cost_usd=speech_check_cost_usd,
        )
    return meta, CheckReport(deterministic=det, speech_check=speech)


def serialize_transcript(check: SpeechCheck) -> dict[str, Any]:
    return check.model_dump(mode="json")
