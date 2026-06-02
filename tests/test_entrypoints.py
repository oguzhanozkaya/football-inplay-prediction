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


def test_run_entrypoint_runs_registered_pipeline(tmp_path: Path, capsys, monkeypatch) -> None:
    paths = build_paths(tmp_path)
    calls = []

    def fake_stage(_) -> int:
        calls.append("stage")
        paths.reports.mkdir(parents=True, exist_ok=True)
        return 0

    monkeypatch.setattr("turkish_inflation_forecasting.pipeline.run_download", fake_stage)
    monkeypatch.setattr("turkish_inflation_forecasting.pipeline.run_preprocess", fake_stage)
    monkeypatch.setattr("turkish_inflation_forecasting.pipeline.run_features", fake_stage)
    monkeypatch.setattr("turkish_inflation_forecasting.pipeline.run_train", fake_stage)
    monkeypatch.setattr("turkish_inflation_forecasting.pipeline.run_evaluate", fake_stage)
    monkeypatch.setattr("turkish_inflation_forecasting.pipeline.run_plots", fake_stage)

    exit_code = run_pipeline(paths)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert len(calls) == 6
    assert paths.reports.is_dir()
    assert "full forecasting pipeline completed" in captured.out
