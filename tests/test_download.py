from pathlib import Path

import fip.download
import fip.utils


def test_write_source_registry_records_raw_paths(tmp_path: Path) -> None:
    paths = fip.utils.build_paths(tmp_path)

    registry_path = fip.download.write_source_registry(paths)

    content = registry_path.read_text(encoding="utf-8")
    assert "espn_plays" in content
    assert "plays_data" in content


def test_validate_source_records_local_status(tmp_path: Path) -> None:
    paths = fip.utils.build_paths(tmp_path)
    raw_path = paths.raw_data / "plays_data"
    raw_path.mkdir(parents=True)
    (raw_path / "plays_fixture.csv").write_text("eventId,text\n1,Goal\n", encoding="utf-8")

    record = fip.download.validate_source(fip.utils.source_by_id("espn_plays"), paths)

    assert record.status == "available"
    assert record.file_count == 1
    assert record.bytes > 0
    assert record.sha256 is not None


def test_write_manifest_records_validation_status(tmp_path: Path) -> None:
    paths = fip.utils.build_paths(tmp_path)
    paths.raw_data.mkdir(parents=True)
    record = fip.download.DownloadRecord(
        source_id="espn_plays",
        title="ESPN Plays",
        category="event",
        source_type="local_csv_directory",
        local_path="data/raw/plays_data",
        status="available",
        checked_at_utc="2026-01-01T00:00:00+00:00",
        file_count=1,
        bytes=3,
        sha256="abc",
    )

    manifest_path = fip.download.write_manifest([record], paths)

    assert '"status": "available"' in manifest_path.read_text(encoding="utf-8")
