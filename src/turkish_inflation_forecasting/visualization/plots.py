"""Report plot generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

from turkish_inflation_forecasting.config import DEFAULT_PATHS, ProjectPaths, ensure_generated_directories

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class PlotResult:
    """Summary of generated plot artifacts."""

    figure_paths: tuple[Path, ...]


class PlotError(RuntimeError):
    """Raised when plot inputs are missing or invalid."""


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _best_model_from_metrics(metrics: pd.DataFrame) -> str:
    validation = metrics[metrics["split"] == "validation"]
    if validation.empty:
        validation = metrics[metrics["split"] == "test"]
    if validation.empty:
        validation = metrics
    return str(validation.sort_values("mae").iloc[0]["model_name"])


def generate_plots(paths: ProjectPaths = DEFAULT_PATHS) -> PlotResult:
    """Generate report figures from processed data, predictions, and metrics."""

    ensure_generated_directories(paths)
    cpi_path = paths.interim_data / "cpi_mom.parquet"
    dataset_path = paths.processed_data / "model_dataset.parquet"
    metadata_path = paths.processed_data / "feature_metadata.json"
    predictions_path = paths.predictions / "predictions.parquet"
    metrics_path = paths.reports / "metrics.json"
    missing = [
        path for path in (cpi_path, dataset_path, metadata_path, predictions_path, metrics_path) if not path.is_file()
    ]
    if missing:
        missing_paths = ", ".join(path.relative_to(paths.root).as_posix() for path in missing)
        raise PlotError(f"Missing plot inputs: {missing_paths}. Run `just evaluate` first.")

    figure_paths: list[Path] = []
    cpi = pd.read_parquet(cpi_path)
    dataset = pd.read_parquet(dataset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(predictions_path)
    metrics = pd.DataFrame(json.loads(metrics_path.read_text(encoding="utf-8")))

    plt.figure(figsize=(10, 4.5))
    plt.plot(cpi["target_month_start"], cpi["target_cpi_mom_percent"], linewidth=1.4)
    plt.title("Turkish CPI Month-over-Month Inflation")
    plt.xlabel("Month")
    plt.ylabel("CPI MoM (%)")
    path = paths.figures / "cpi_history.png"
    _save_current_figure(path)
    figure_paths.append(path)

    numeric_columns = list(metadata["numeric_feature_columns"])
    coverage = dataset[numeric_columns].notna().mean().sort_values().tail(25)
    plt.figure(figsize=(9, 6))
    coverage.plot(kind="barh")
    plt.title("Processed Feature Coverage")
    plt.xlabel("Non-missing share")
    path = paths.figures / "feature_coverage.png"
    _save_current_figure(path)
    figure_paths.append(path)

    best_model = _best_model_from_metrics(metrics)
    model_predictions = predictions[predictions["model_name"] == best_model].copy()
    plot_split = "test" if (model_predictions["split"] == "test").any() else model_predictions["split"].iloc[-1]
    model_predictions = model_predictions[model_predictions["split"] == plot_split].sort_values("target_month_start")
    plt.figure(figsize=(10, 4.5))
    plt.plot(model_predictions["target_month_start"], model_predictions["actual_cpi_mom_percent"], label="Actual")
    plt.plot(
        model_predictions["target_month_start"],
        model_predictions["prediction_cpi_mom_percent"],
        label=f"Prediction ({best_model})",
    )
    plt.title(f"Prediction vs Actual ({plot_split})")
    plt.xlabel("Target month")
    plt.ylabel("CPI MoM (%)")
    plt.legend()
    path = paths.figures / "predictions_vs_actual.png"
    _save_current_figure(path)
    figure_paths.append(path)

    residuals = model_predictions["prediction_cpi_mom_percent"] - model_predictions["actual_cpi_mom_percent"]
    plt.figure(figsize=(7, 4.5))
    plt.hist(residuals, bins=min(20, max(5, len(residuals))), edgecolor="black")
    plt.title(f"Residual Distribution ({best_model}, {plot_split})")
    plt.xlabel("Prediction error")
    plt.ylabel("Count")
    path = paths.figures / "residuals.png"
    _save_current_figure(path)
    figure_paths.append(path)

    comparison_split = "test" if (metrics["split"] == "test").any() else "validation"
    comparison = metrics[metrics["split"] == comparison_split].sort_values("mae")
    plt.figure(figsize=(9, 4.8))
    plt.bar(comparison["model_name"], comparison["mae"])
    plt.title(f"Model Comparison by MAE ({comparison_split})")
    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.xticks(rotation=35, ha="right")
    path = paths.figures / "model_comparison.png"
    _save_current_figure(path)
    figure_paths.append(path)

    return PlotResult(figure_paths=tuple(figure_paths))
