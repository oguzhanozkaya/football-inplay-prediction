from pathlib import Path

from turkish_inflation_forecasting.config import build_paths
from turkish_inflation_forecasting.pipeline import run_download, run_pipeline


def test_download_stage_reports_success(tmp_path: Path, capsys, monkeypatch) -> None:
    paths = build_paths(tmp_path)

    def fake_download_sources(_) -> list[object]:
        paths.raw_data.mkdir(parents=True)
        (paths.raw_data / "source_manifest.json").write_text("[]", encoding="utf-8")
        return [object()]

    monkeypatch.setattr("turkish_inflation_forecasting.pipeline.download_sources", fake_download_sources)
    exit_code = run_download(paths)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert paths.raw_data.is_dir()
    assert "download" in captured.out
    assert "downloaded 1 raw sources" in captured.out


def test_run_entrypoint_reports_registered_pipeline(tmp_path: Path, capsys) -> None:
    paths = build_paths(tmp_path)

    exit_code = run_pipeline(paths)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert paths.reports.is_dir()
    assert "full forecasting pipeline" in captured.out
