"""Pipeline stage entry functions.

The project exposes separate console entrypoints for each stage. These functions
hold the stage behavior so entrypoint modules stay small and testable.
"""

from __future__ import annotations

from turkish_inflation_forecasting.config import DEFAULT_PATHS, ProjectPaths, ensure_generated_directories
from turkish_inflation_forecasting.data.download import DownloadError, download_sources
from turkish_inflation_forecasting.data.preprocess import PreprocessError, preprocess_raw_sources


def _run_pending_stage(stage_name: str, next_step: str, paths: ProjectPaths = DEFAULT_PATHS) -> int:
    ensure_generated_directories(paths)
    print(f"{stage_name}: project directories are ready.")
    print(f"{stage_name}: {next_step}")
    return 0


def run_download(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    try:
        records = download_sources(paths)
    except DownloadError as exc:
        print(f"download: {exc}")
        return 1
    print(f"download: downloaded {len(records)} raw sources.")
    print(f"download: manifest written to {(paths.raw_data / 'source_manifest.json').relative_to(paths.root)}")
    return 0


def run_preprocess(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    try:
        result = preprocess_raw_sources(paths)
    except (PreprocessError, ValueError) as exc:
        print(f"preprocess: {exc}")
        return 1
    cpi_target_path = result.cpi_target_path.relative_to(paths.root)
    print(f"preprocess: wrote {result.cpi_target_rows} CPI target rows to {cpi_target_path}")
    print(
        "preprocess: wrote "
        f"{result.numeric_series_rows} numeric source rows to {result.numeric_series_path.relative_to(paths.root)}"
    )
    print(
        "preprocess: wrote "
        f"{result.monthly_numeric_rows} monthly numeric rows to {result.monthly_numeric_path.relative_to(paths.root)}"
    )
    print(
        "preprocess: wrote "
        f"{result.text_document_rows} text document rows to {result.text_documents_path.relative_to(paths.root)}"
    )
    return 0


def run_features(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    return _run_pending_stage("features", "numeric and text feature generation is not implemented yet.", paths)


def run_train(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    return _run_pending_stage("train", "baseline and deep model training is not implemented yet.", paths)


def run_evaluate(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    return _run_pending_stage("evaluate", "chronological evaluation is not implemented yet.", paths)


def run_plots(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    return _run_pending_stage("plots", "report figure generation is not implemented yet.", paths)


def run_pipeline(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    ensure_generated_directories(paths)
    print("run: project directories are ready.")
    print("run: full forecasting pipeline is registered but not implemented yet.")
    return 0
