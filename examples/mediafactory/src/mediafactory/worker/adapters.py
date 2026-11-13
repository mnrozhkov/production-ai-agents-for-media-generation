"""Role adapters (`Chat`, `Images`, `Speech`, `Video`, `Judge`) and deterministic `FakeAdapter`."""

from __future__ import annotations

import hashlib
import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mediafactory.core.cost import token_cost_usd
from mediafactory.core.spec import GenerationSpec
from mediafactory.profile.application import ApplicationBundle, JudgeVerdict, RubricConfig


@dataclass(frozen=True)
class LoadedModel:
    """Represents a loaded pipeline in a worker process."""

    role: str
    model: str
    revision: str | None
    lora_sha256: str | None = None
    load_s: float = 0.01
    gpu_name: str = "NVIDIA L40S"
    peak_vram_gb: float = 10.5


ModelLoader = Callable[[tuple[str, str, str | None, str | None]], LoadedModel]
ModelRenderer = Callable[[GenerationSpec, LoadedModel], tuple[bytes, float, float]]


def make_fake_png(width: int = 1024, height: int = 1024, *, seed_tag: str = "") -> bytes:
    """Build a valid minimal PNG header+IHDR+IDAT+IEND with exact ``width`` and ``height``.

    Uses raw PNG chunk encoding so generating 200 1024x1024 fixtures in tests takes < 5 ms.
    """
    digest = hashlib.sha256(seed_tag.encode("utf-8")).digest()

    def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # Minimal valid compressed payload + deterministic seed_tag comment chunk
    idat_payload = zlib.compress(b"\x00" + digest[:3])
    text_payload = b"seed\x00" + digest.hex().encode("ascii")
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"tEXt", text_payload)
        + _chunk(b"IDAT", idat_payload)
        + _chunk(b"IEND", b"")
    )


def make_fake_mp3(tag: str = "") -> bytes:
    """Build a valid minimal MP3 with ID3v2 header and MPEG sync frame."""
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    id3_header = b"ID3\x04\x00\x00\x00\x00\x00\x20" + digest
    mpeg_frame = b"\xff\xfb\x90\x64" + digest + (b"\x00" * 64)
    return id3_header + mpeg_frame


def make_fake_mp4(tag: str = "") -> bytes:
    """Build a valid minimal MP4 container starting with an `ftyp` box and `mdat` payload."""
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    ftyp = b"\x00\x00\x00\x14ftypisom\x00\x00\x02\x00isom"
    mdat = struct.pack(">I", 8 + len(digest)) + b"mdat" + digest
    return ftyp + mdat


class FakeAdapter:
    """Deterministic scripted adapter for `--fake` CI and offline walkthroughs (`SPEC.md` §9)."""

    def __init__(self, *, fail_first_image_for_records: set[str] | None = None) -> None:
        self.fail_first_image_for_records = set(fail_first_image_for_records or set())
        self.load_counts: dict[tuple[str, str, str | None, str | None], int] = {}
        self.render_counts: dict[str, int] = {}

    def plan(
        self,
        bundle: ApplicationBundle,
        record: dict[str, Any],
        *,
        fix: str = "",
        price_in_per_1m_usd: float = 0.15,
        price_out_per_1m_usd: float = 0.60,
    ) -> tuple[dict[str, str], float, float]:
        """Generate structured copy validated against ``bundle.config.copy_schema``."""
        rendered_prompt = bundle.render_plan_prompt(record, fix=fix)
        _, style = bundle.config.style_for_record(record)
        rec_id = str(record["id"])
        subject = (
            record.get("wishlist")
            or record.get("product")
            or record.get("topic")
            or record.get("title")
            or record.get("name")
            or rec_id
        )
        recipient = record.get("name") or record.get("audience") or "friend"
        fix_clause = f" ({fix.strip()})" if fix.strip() else ""

        output: dict[str, str] = {}
        for field_name, field_spec in bundle.config.copy_schema.fields.items():
            if field_spec.type == "enum" and field_spec.values:
                output[field_name] = field_spec.values[0]
            elif field_name == "copy":
                output[field_name] = (
                    f"Hello {recipient}! Here is a special story featuring {subject}, "
                    f"crafted just for you{fix_clause}."
                )
            elif field_name == "speech_text":
                output[field_name] = (
                    f"Hello {recipient}! Here is a special story featuring {subject}, "
                    f"crafted just for you."
                )
            elif field_name == "image_prompt":
                lora_prefix = f"{style.lora.trigger_word} " if style.lora and style.lora.trigger_word else ""
                output[field_name] = (
                    f"{lora_prefix}{subject}, {style.visual}, avoid {style.negative}{fix_clause}"
                ).strip()
            elif field_name == "motion_prompt":
                output[field_name] = f"{subject} in motion: {style.motion or 'gentle camera drift'}"
            elif field_name == "music_tag":
                output[field_name] = style.music or "warm"
            else:
                output[field_name] = f"{field_name} for {subject} ({recipient}){fix_clause}"

        validated = bundle.config.copy_schema.validate_output(output)
        tokens_in = max(50, len(rendered_prompt.split()) * 2)
        tokens_out = max(40, sum(len(v.split()) for v in validated.values()) * 2)
        cost = token_cost_usd(
            tokens_in,
            tokens_out,
            price_in_per_1m_usd=price_in_per_1m_usd,
            price_out_per_1m_usd=price_out_per_1m_usd,
        )
        return validated, 0.02, cost

    def judge_image(
        self,
        *,
        record_id: str,
        attempt: int,
        image_bytes: bytes,
        rubric: RubricConfig,
        judge_model: str = "moonshotai/Kimi-K3",
        force_verdict: str | None = None,
    ) -> tuple[JudgeVerdict, float, float]:
        """Score an image against ``rubric`` (`PASS | RETRY | WEAK`)."""
        cost = token_cost_usd(350, 80, price_in_per_1m_usd=0.20, price_out_per_1m_usd=0.80)
        if force_verdict == "WEAK":
            return (
                JudgeVerdict(
                    verdict="WEAK",
                    score=20,
                    reasons=["Image completely misses the brief subject and cannot be salvaged."],
                    fix="",
                    rubric_version=rubric.rubric_version,
                    judge_model=judge_model,
                ),
                0.015,
                cost,
            )
        if force_verdict == "RETRY" or (
            record_id in self.fail_first_image_for_records and attempt == 1
        ):
            return (
                JudgeVerdict(
                    verdict="RETRY",
                    score=max(rubric.weak_score + 10, rubric.pass_score - 15),
                    reasons=["Hero object is partially occluded; remove background clutter."],
                    fix="center the hero object clearly and remove background text",
                    rubric_version=rubric.rubric_version,
                    judge_model=judge_model,
                ),
                0.015,
                cost,
            )
        return (
            JudgeVerdict(
                verdict="PASS",
                score=92,
                reasons=[f"Passed all {len(rubric.criteria)} rubric criteria."],
                fix="",
                rubric_version=rubric.rubric_version,
                judge_model=judge_model,
            ),
            0.015,
            cost,
        )

    def load_model(self, group_key: tuple[str, str, str | None, str | None]) -> LoadedModel:
        self.load_counts[group_key] = self.load_counts.get(group_key, 0) + 1
        role, model, revision, lora_sha = group_key
        vram = {"image": 10.9, "speech": 0.8, "audio": 11.8, "video": 27.9}.get(role, 8.0)
        load_s = {"image": 1.2, "speech": 0.4, "audio": 0.9, "video": 2.5}.get(role, 1.0)
        return LoadedModel(
            role=role,
            model=model,
            revision=revision,
            lora_sha256=lora_sha,
            load_s=load_s,
            gpu_name="NVIDIA L40S" if role != "video" else "NVIDIA H100",
            peak_vram_gb=vram,
        )

    def render_asset(
        self, spec: GenerationSpec, loaded: LoadedModel
    ) -> tuple[bytes, float, float]:
        self.render_counts[spec.role] = self.render_counts.get(spec.role, 0) + 1
        tag = spec.spec_hash()
        if spec.role == "image":
            w = spec.size.width if spec.size else 1024
            h = spec.size.height if spec.size else 1024
            return make_fake_png(w, h, seed_tag=tag), 4.4, loaded.peak_vram_gb
        if spec.role == "video":
            return make_fake_mp4(tag=tag), 45.0, loaded.peak_vram_gb
        return make_fake_mp3(tag=tag), 1.9, loaded.peak_vram_gb
