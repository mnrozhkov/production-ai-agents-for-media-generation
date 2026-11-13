"""FUSE-safe `staged_write`, local/in-memory storage, and content-addressed cache lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mediafactory.core.result import ResultReport
from mediafactory.core.spec import GenerationSpec


class StorageError(RuntimeError):
    """Storage read/write failure."""


def staged_write(dest: Path | str, data: bytes | str) -> Path:
    """Write ``data`` to ``dest`` atomically where possible and always verify byte length.

    Protects against FUSE mounts where ``os.replace`` can leave a zero-byte destination.
    """
    raw = data.encode("utf-8") if isinstance(data, str) else data
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(raw)
    try:
        tmp.replace(path)
    except OSError:
        pass

    landed = path.is_file() and path.stat().st_size == len(raw)
    if landed and not tmp.exists():
        return path

    path.write_bytes(raw)
    try:
        tmp.unlink()
    except OSError:
        pass
    if not (path.is_file() and path.stat().st_size == len(raw)):
        raise StorageError(
            f"could not write {path}: expected {len(raw)} bytes, found "
            f"{path.stat().st_size if path.exists() else 0}"
        )
    return path


class DiskStorage:
    """Local filesystem storage rooted at ``root`` (or a mounted Object Storage FUSE path)."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def resolve(self, key: str) -> Path:
        parts = [p for p in Path(key).parts if p not in ("", ".", "..")]
        return self.root.joinpath(*parts)

    def upload(self, key: str, data: bytes | str, *, content_type: str | None = None) -> Path:
        return staged_write(self.resolve(key), data)

    def exists(self, key: str) -> bool:
        return self.resolve(key).is_file()

    def download(self, key: str) -> bytes:
        path = self.resolve(key)
        if not path.is_file():
            raise StorageError(f"missing object key: {key}")
        return path.read_bytes()

    def list(self, prefix: str = "") -> list[str]:
        if not self.root.is_dir():
            return []
        keys: list[str] = []
        for item in sorted(self.root.rglob("*")):
            if not item.is_file() or item.name.startswith("."):
                continue
            rel = item.relative_to(self.root).as_posix()
            if not prefix or rel.startswith(prefix):
                keys.append(rel)
        return keys


class MemoryStorage:
    """In-memory storage backend for fast unit tests."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload(self, key: str, data: bytes | str, *, content_type: str | None = None) -> None:
        self.objects[key] = data.encode("utf-8") if isinstance(data, str) else bytes(data)

    def exists(self, key: str) -> bool:
        return key in self.objects

    def download(self, key: str) -> bytes:
        if key not in self.objects:
            raise StorageError(f"missing object key: {key}")
        return self.objects[key]

    def list(self, prefix: str = "") -> list[str]:
        return [k for k in sorted(self.objects) if k.startswith(prefix)]


class ContentCache:
    """Content-addressed cache indexed by ``spec_hash`` (`SPEC.md` §7.1, `FR-06`, `FR-09`)."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.index_dir = self.root / ".cache" / "by_spec_hash"

    def lookup(self, spec: GenerationSpec) -> tuple[ResultReport, bytes] | None:
        """Return ``(ResultReport, asset_bytes)`` if either target output or cache has ``spec_hash``."""
        digest = spec.spec_hash()
        target_result_path = self.root / spec.result_key()
        target_asset_path = self.root / spec.asset_key()

        if target_result_path.is_file() and target_asset_path.is_file():
            try:
                existing = ResultReport.from_file(target_result_path)
                if existing.status == "ok" and existing.spec_hash == digest:
                    return existing, target_asset_path.read_bytes()
            except Exception:
                pass

        cached_meta = self.index_dir / f"{digest}.json"
        cached_blob = self.index_dir / f"{digest}.bin"
        if cached_meta.is_file() and cached_blob.is_file():
            try:
                report = ResultReport.from_file(cached_meta)
                if report.status == "ok" and report.spec_hash == digest:
                    return report, cached_blob.read_bytes()
            except Exception:
                return None
        return None

    def store(self, report: ResultReport, asset_bytes: bytes) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        digest = report.spec_hash
        staged_write(self.index_dir / f"{digest}.bin", asset_bytes)
        staged_write(self.index_dir / f"{digest}.json", report.to_json())

    def materialize_hit(
        self, spec: GenerationSpec, cached_report: ResultReport, asset_bytes: bytes
    ) -> ResultReport:
        """Write cached asset and updated `result.json` (`cache_hit=True`, `marginal_compute_usd=0`)."""
        asset_path = self.root / spec.asset_key()
        result_path = self.root / spec.result_key()
        staged_write(asset_path, asset_bytes)

        data: dict[str, Any] = cached_report.model_dump(mode="json")
        data["cache_hit"] = True
        data["spec"] = spec.model_dump(mode="json")
        if data.get("asset"):
            data["asset"]["key"] = spec.asset_key()
        data["cost"]["marginal_compute_usd"] = 0.0
        hit_report = ResultReport.from_mapping(data)
        staged_write(result_path, hit_report.to_json())
        return hit_report
