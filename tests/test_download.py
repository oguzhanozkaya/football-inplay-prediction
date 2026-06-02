from pathlib import Path

from turkish_inflation_forecasting.config import build_paths
from turkish_inflation_forecasting.data.download import (
    DownloadRecord,
    download_text_documents,
    write_manifest,
    write_source_registry,
)
from turkish_inflation_forecasting.data.sources import sources_by_category


def test_write_source_registry_records_raw_paths(tmp_path: Path) -> None:
    paths = build_paths(tmp_path)

    registry_path = write_source_registry(paths)

    content = registry_path.read_text(encoding="utf-8")
    assert "cbrt_consumer_prices" in content
    assert "numeric/cbrt_consumer_prices.html" in content


def test_write_manifest_records_download_status(tmp_path: Path) -> None:
    paths = build_paths(tmp_path)
    paths.raw_data.mkdir(parents=True)
    record = DownloadRecord(
        source_id="source",
        title="Source",
        category="numeric",
        source_type="official_html",
        url="https://example.test",
        local_path="data/raw/source.html",
        status="downloaded",
        downloaded_at_utc="2026-01-01T00:00:00+00:00",
        sha256="abc",
        bytes=3,
    )

    manifest_path = write_manifest([record], paths)

    assert '"status": "downloaded"' in manifest_path.read_text(encoding="utf-8")


def test_download_text_documents_writes_document_pages(tmp_path: Path, monkeypatch) -> None:
    paths = build_paths(tmp_path)
    paths.raw_data.mkdir(parents=True)
    for source in sources_by_category("text"):
        raw_path = paths.raw_data / source.raw_path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        source_url_path = source.url.split("/MPC/")[-1].replace("%2B", "+")
        raw_path.write_text(
            f'<a href="/wps/wcm/connect/EN/TCMB+EN/MPC/{source_url_path}/ANO2026-17">Document (2026-17)</a>',
            encoding="utf-8",
        )

    class Response:
        content = b"<div id='tcmbMainContent'><div class='tcmb-content'>body</div></div>"

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr("turkish_inflation_forecasting.data.download.requests.get", lambda *_, **__: Response())

    records = download_text_documents(paths)

    assert len(records) == 2
    assert all(record.source_type == "official_html_document" for record in records)
    assert all((paths.root / record.local_path).is_file() for record in records)
