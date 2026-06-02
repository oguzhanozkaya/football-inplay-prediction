"""Pipeline stage entry functions.

The project exposes separate console entrypoints for each stage. These functions
hold the stage behavior so entrypoint modules stay small and testable.
"""

from __future__ import annotations

from tif.config import DEFAULT_PATHS, ProjectPaths, ensure_generated_directories
from tif.data.download import DownloadError, download_sources
from tif.data.preprocess import PreprocessError, preprocess_raw_sources


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
    from tif.features.build import FeatureGenerationError, generate_features

    try:
        result = generate_features(paths)
    except (FeatureGenerationError, ValueError, FileNotFoundError) as exc:
        print(f"features: {exc}")
        return 1
    print(
        "features: wrote "
        f"{result.row_count} model rows with {result.numeric_feature_count} numeric features to "
        f"{result.dataset_path.relative_to(paths.root)}"
    )
    print(
        "features: wrote vocabulary with "
        f"{result.vocabulary_size} tokens to {result.vocabulary_path.relative_to(paths.root)}"
    )
    return 0


def run_train(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    from tif.training.train import TrainingError, train_models

    try:
        result = train_models(paths)
    except (TrainingError, ValueError, FileNotFoundError) as exc:
        print(f"train: {exc}")
        return 1
    print(f"train: trained {result.model_count} models")
    print(
        "train: wrote "
        f"{result.prediction_rows} prediction rows to {result.predictions_csv_path.relative_to(paths.root)}"
    )
    return 0


def run_evaluate(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    from tif.evaluation.metrics import EvaluationError, evaluate_predictions

    try:
        result = evaluate_predictions(paths)
    except (EvaluationError, ValueError, FileNotFoundError) as exc:
        print(f"evaluate: {exc}")
        return 1
    print(f"evaluate: wrote {result.metric_rows} metric rows to {result.metrics_json_path.relative_to(paths.root)}")
    print(f"evaluate: wrote markdown report to {result.metrics_markdown_path.relative_to(paths.root)}")
    return 0


def run_plots(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    from tif.visualization.plots import PlotError, generate_plots

    try:
        result = generate_plots(paths)
    except (PlotError, ValueError, FileNotFoundError) as exc:
        print(f"plots: {exc}")
        return 1
    print(f"plots: wrote {len(result.figure_paths)} figures")
    for figure_path in result.figure_paths:
        print(f"plots: wrote {figure_path.relative_to(paths.root)}")
    return 0


def run_pipeline(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    for stage_name, stage in (
        ("download", run_download),
        ("preprocess", run_preprocess),
        ("features", run_features),
        ("train", run_train),
        ("evaluate", run_evaluate),
        ("plots", run_plots),
    ):
        exit_code = stage(paths)
        if exit_code != 0:
            print(f"run: stopped at {stage_name}")
            return exit_code
    print("run: full forecasting pipeline completed.")
    return 0
