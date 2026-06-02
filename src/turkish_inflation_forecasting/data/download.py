"""Raw source download utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from turkish_inflation_forecasting.config import DEFAULT_PATHS, ProjectPaths, ensure_generated_directories
from turkish_inflation_forecasting.data.fx import (
    cbrt_fx_raw_path_for_date,
    cbrt_fx_url_for_date,
    iter_month_starts,
    month_end_candidates,
)
from turkish_inflation_forecasting.data.sources import SOURCE_REGISTRY, SourceDefinition, sources_by_category
from turkish_inflation_forecasting.data.text import extract_cbrt_text_links

USER_AGENT = "turkish-inflation-forecasting/0.1 (+https://github.com/oguzhanozkaya/turkish-inflation-forecasting)"


@dataclass(frozen=True)
class DownloadRecord:
    """Manifest record for one downloaded source."""

    source_id: str
    title: str
    category: str
    source_type: str
    url: str
    local_path: str
    status: str
    downloaded_at_utc: str
    sha256: str | None
    bytes: int
    error: str | None = None


class DownloadError(RuntimeError):
    """Raised when at least one required source cannot be downloaded."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_registry_records(sources: tuple[SourceDefinition, ...] = SOURCE_REGISTRY) -> list[dict[str, str]]:
    return [asdict(source) | {"raw_path": source.raw_path.as_posix()} for source in sources]


def write_source_registry(paths: ProjectPaths = DEFAULT_PATHS) -> Path:
    """Write the source registry snapshot used by the current download run."""

    ensure_generated_directories(paths)
    registry_path = paths.raw_data / "source_registry.json"
    registry_path.write_text(json.dumps(source_registry_records(), indent=2), encoding="utf-8")
    return registry_path


def write_manifest(records: list[DownloadRecord], paths: ProjectPaths = DEFAULT_PATHS) -> Path:
    """Write raw download manifest records."""

    manifest_path = paths.raw_data / "source_manifest.json"
    manifest = [asdict(record) for record in records]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def download_url(
    *,
    source_id: str,
    title: str,
    category: str,
    source_type: str,
    url: str,
    local_path: Path,
    paths: ProjectPaths,
    timeout_seconds: int,
) -> DownloadRecord:
    """Download one URL into a local raw path and return a manifest record."""

    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.now(UTC).isoformat(timespec="seconds")

    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as exc:
        return DownloadRecord(
            source_id=source_id,
            title=title,
            category=category,
            source_type=source_type,
            url=url,
            local_path=local_path.relative_to(paths.root).as_posix(),
            status="failed",
            downloaded_at_utc=downloaded_at,
            sha256=None,
            bytes=0,
            error=str(exc),
        )

    content = response.content
    local_path.write_bytes(content)
    return DownloadRecord(
        source_id=source_id,
        title=title,
        category=category,
        source_type=source_type,
        url=url,
        local_path=local_path.relative_to(paths.root).as_posix(),
        status="downloaded",
        downloaded_at_utc=downloaded_at,
        sha256=sha256_bytes(content),
        bytes=len(content),
        error=None,
    )


def download_source(
    source: SourceDefinition, paths: ProjectPaths = DEFAULT_PATHS, timeout_seconds: int = 60
) -> DownloadRecord:
    """Download one registered source into its registered raw path."""

    return download_url(
        source_id=source.source_id,
        title=source.title,
        category=source.category,
        source_type=source.source_type,
        url=source.url,
        local_path=paths.raw_data / source.raw_path,
        paths=paths,
        timeout_seconds=timeout_seconds,
    )


def download_cbrt_fx_month_end(
    source: SourceDefinition,
    paths: ProjectPaths = DEFAULT_PATHS,
    timeout_seconds: int = 60,
) -> list[DownloadRecord]:
    """Download one official CBRT FX XML snapshot for each completed month."""

    records = []
    for month_start in iter_month_starts():
        failed_candidates = []
        for effective_date in month_end_candidates(month_start):
            record = download_url(
                source_id=f"{source.source_id}_{month_start:%Y_%m}",
                title=f"{source.title} {month_start:%Y-%m}",
                category=source.category,
                source_type=source.source_type,
                url=cbrt_fx_url_for_date(effective_date),
                local_path=paths.raw_data / cbrt_fx_raw_path_for_date(source.raw_path, effective_date),
                paths=paths,
                timeout_seconds=timeout_seconds,
            )
            if record.status == "downloaded":
                records.append(record)
                break
            failed_candidates.append(record)
        else:
            records.append(failed_candidates[-1])
    return records


def download_registered_source(
    source: SourceDefinition,
    paths: ProjectPaths = DEFAULT_PATHS,
    timeout_seconds: int = 60,
) -> list[DownloadRecord]:
    """Download one registry source, including multi-file archive sources."""

    if source.source_type == "official_xml_month_end_archive":
        return download_cbrt_fx_month_end(source, paths, timeout_seconds)
    return [download_source(source, paths, timeout_seconds)]


def download_text_documents(paths: ProjectPaths = DEFAULT_PATHS, timeout_seconds: int = 60) -> list[DownloadRecord]:
    """Download official text document pages discovered from listing snapshots."""

    records = []
    for source in sources_by_category("text"):
        listing_path = paths.raw_data / source.raw_path
        if not listing_path.is_file():
            raise DownloadError(f"Missing text listing source before document download: {listing_path}")
        documents = extract_cbrt_text_links(listing_path.read_text(encoding="utf-8"), source)
        for document in documents.itertuples(index=False):
            records.append(
                download_url(
                    source_id=document.document_id,
                    title=document.title,
                    category="text",
                    source_type="official_html_document",
                    url=document.url,
                    local_path=paths.raw_data / document.raw_document_path,
                    paths=paths,
                    timeout_seconds=timeout_seconds,
                )
            )
    return records


def download_sources(
    paths: ProjectPaths = DEFAULT_PATHS,
    sources: tuple[SourceDefinition, ...] = SOURCE_REGISTRY,
    timeout_seconds: int = 60,
) -> list[DownloadRecord]:
    """Download all registered raw sources and write registry/manifest snapshots."""

    ensure_generated_directories(paths)
    write_source_registry(paths)
    records = []
    for source in sources:
        records.extend(download_registered_source(source, paths, timeout_seconds))
    failed = [record for record in records if record.status != "downloaded"]
    if not failed and any(source.category == "text" for source in sources):
        records.extend(download_text_documents(paths, timeout_seconds))
    write_manifest(records, paths)
    failed = [record for record in records if record.status != "downloaded"]
    if failed:
        failed_ids = ", ".join(record.source_id for record in failed)
        raise DownloadError(f"Failed to download required sources: {failed_ids}")
    return records
