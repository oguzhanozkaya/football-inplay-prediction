from pathlib import Path

import fip.utils


def test_paths_and_generated_directories(tmp_path: Path) -> None:
    paths = fip.utils.build_paths(tmp_path)

    directories = fip.utils.ensure_generated_directories(paths)

    assert paths.root == tmp_path.resolve()
    assert paths.raw_data == tmp_path.resolve() / "data" / "raw"
    assert paths.processed_data == tmp_path.resolve() / "data" / "processed"
    assert paths.reports == tmp_path.resolve() / "output" / "reports"
    assert directories == paths.generated_directories()
    assert all(directory.is_dir() for directory in directories)


def test_source_registry_and_window_constants() -> None:
    source_ids = [source.source_id for source in fip.utils.SOURCE_REGISTRY]

    assert len(source_ids) == len(set(source_ids))
    assert fip.utils.MAX_TOKENS_PER_WINDOW == 64
    assert fip.utils.window_count(45, 5) == 9
    assert fip.utils.source_by_id("espn_plays").category == "event"
    assert {source.source_id for source in fip.utils.sources_by_category("text")} == {"espn_commentary"}
