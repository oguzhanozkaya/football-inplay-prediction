"""Validate local raw ESPN Soccer data and write source manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import fip.utils


@dataclass(frozen=True)
class DownloadRecord:
    """Manifest record for one local raw source group."""

    source_id: str
    title: str
    category: str
    source_type: str
    local_path: str
    status: str
    checked_at_utc: str
    file_count: int
    bytes: int
    sha256: str | None
    error: str | None = None


class DownloadError(RuntimeError):
    """Raised when required local raw data is missing."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_registry_records(
    sources: tuple[fip.utils.SourceDefinition, ...] = fip.utils.SOURCE_REGISTRY,
) -> list[dict[str, str]]:
    return [asdict(source) | {"raw_path": source.raw_path.as_posix()} for source in sources]


def write_source_registry(paths: fip.utils.ProjectPaths = fip.utils.DEFAULT_PATHS) -> Path:
    fip.utils.ensure_generated_directories(paths)
    registry_path = paths.raw_data / "source_registry.json"
    registry_path.write_text(json.dumps(source_registry_records(), indent=2), encoding="utf-8")
    return registry_path


def write_manifest(records: list[DownloadRecord], paths: fip.utils.ProjectPaths = fip.utils.DEFAULT_PATHS) -> Path:
    manifest_path = paths.raw_data / "source_manifest.json"
    manifest_path.write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")
    return manifest_path


def validate_source(source: fip.utils.SourceDefinition, paths: fip.utils.ProjectPaths) -> DownloadRecord:
    checked_at = datetime.now(UTC).isoformat(timespec="seconds")
    local_path = paths.raw_data / source.raw_path
    relative_path = local_path.relative_to(paths.root).as_posix()
    if not local_path.exists():
        return DownloadRecord(
            source.source_id,
            source.title,
            source.category,
            source.source_type,
            relative_path,
            "missing",
            checked_at,
            0,
            0,
            None,
            f"Missing required raw path: {relative_path}",
        )

    files = sorted(path for path in local_path.rglob("*.csv") if path.is_file())
    if local_path.is_dir() and not files:
        return DownloadRecord(
            source.source_id,
            source.title,
            source.category,
            source.source_type,
            relative_path,
            "missing",
            checked_at,
            0,
            0,
            None,
            f"No CSV files found under required raw path: {relative_path}",
        )

    bytes_total = sum(path.stat().st_size for path in files)
    first_hash = sha256_file(files[0]) if files else None
    return DownloadRecord(
        source.source_id,
        source.title,
        source.category,
        source.source_type,
        relative_path,
        "available",
        checked_at,
        len(files),
        bytes_total,
        first_hash,
        None,
    )


def download_sources(paths: fip.utils.ProjectPaths = fip.utils.DEFAULT_PATHS) -> list[DownloadRecord]:
    """Validate local raw data and write registry/manifest snapshots."""

    fip.utils.ensure_generated_directories(paths)
    registry_path = write_source_registry(paths)
    print(f"download: wrote source registry to {registry_path.relative_to(paths.root)}")
    records = [validate_source(source, paths) for source in fip.utils.SOURCE_REGISTRY]
    manifest_path = write_manifest(records, paths)
    print(f"download: wrote source manifest to {manifest_path.relative_to(paths.root)}")
    for record in records:
        print(
            f"download: source={record.source_id} status={record.status} files={record.file_count} bytes={record.bytes}"
        )
    failures = [
        record for record in records if record.status != "available" and record.source_id != "espn_player_stats"
    ]
    if failures:
        raise DownloadError("Missing required ESPN raw data: " + ", ".join(record.source_id for record in failures))
    return records


def main() -> None:
    download_sources()


if __name__ == "__main__":
    main()
