"""Evaluation metrics and report figures for forecast outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

import tif.utils

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class EvaluationResult:
    """Summary of generated evaluation artifacts."""

    metrics_json_path: Path
    metrics_markdown_path: Path
    metric_rows: int
    figure_paths: tuple[Path, ...]
    report_paths: tuple[Path, ...]


class EvaluationError(RuntimeError):
    """Raised when prediction artifacts are missing or invalid."""


def regression_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    """Compute deterministic regression metrics for one model/split group."""

    actual = group["actual_cpi_mom_percent"].to_numpy(dtype=float)
    prediction = group["prediction_cpi_mom_percent"].to_numpy(dtype=float)
    previous = group["previous_cpi_mom_percent"].to_numpy(dtype=float)
    error = prediction - actual
    actual_direction = np.sign(actual - previous)
    prediction_direction = np.sign(prediction - previous)
    metrics: dict[str, float | int] = {
        "row_count": int(len(group)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "direction_accuracy": float(np.mean(actual_direction == prediction_direction)),
    }
    if "cpi_mom_trailing_std_12" in group.columns:
        volatility = group["cpi_mom_trailing_std_12"].to_numpy(dtype=float)
        valid = np.isfinite(volatility) & (volatility > 1e-12)
        metrics["volatility_normalized_mae"] = (
            float(np.mean(np.abs(error[valid]) / volatility[valid])) if valid.any() else float("nan")
        )
    return metrics


def build_metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build metrics by model and split, including last-value baseline deltas."""

    rows = []
    group_columns = ["model_name", "model_type", "split"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        model_name, model_type, split = keys
        rows.append(
            {
                "model_name": model_name,
                "model_type": model_type,
                "split": split,
                **regression_metrics(group),
            }
        )
    metrics = pd.DataFrame(rows).sort_values(["split", "mae", "model_name"]).reset_index(drop=True)
    baseline_mae = metrics[metrics["model_name"] == "last_value"][["split", "mae"]].rename(
        columns={"mae": "last_value_mae"}
    )
    metrics = metrics.merge(baseline_mae, on="split", how="left")
    metrics["mae_improvement_vs_last_value"] = metrics["last_value_mae"] - metrics["mae"]
    return metrics.drop(columns=["last_value_mae"])


def _write_metrics_markdown(path: Path, metrics: pd.DataFrame) -> None:
    lines = ["# Evaluation Metrics", ""]
    lines.append("MAE is the main metric. Positive baseline delta means lower MAE than the last-value baseline.")
    lines.append("")
    has_volatility = "volatility_normalized_mae" in metrics.columns
    if has_volatility:
        lines.append(
            "| Split | Model | Type | Rows | MAE | RMSE | Direction Accuracy | "
            "Vol-Normalized MAE | MAE Delta vs Last Value |"
        )
        lines.append(
            "| ----- | ----- | ---- | ---- | --- | ---- | ------------------ | "
            "------------------ | ----------------------- |"
        )
    else:
        lines.append("| Split | Model | Type | Rows | MAE | RMSE | Direction Accuracy | MAE Delta vs Last Value |")
        lines.append("| ----- | ----- | ---- | ---- | --- | ---- | ------------------ | ----------------------- |")
    for row in metrics.itertuples(index=False):
        base = (
            f"| {row.split} | `{row.model_name}` | {row.model_type} | {row.row_count} | "
            f"{row.mae:.4f} | {row.rmse:.4f} | {row.direction_accuracy:.3f} | "
        )
        if has_volatility:
            lines.append(base + f"{row.volatility_normalized_mae:.4f} | {row.mae_improvement_vs_last_value:.4f} |")
        else:
            lines.append(base + f"{row.mae_improvement_vs_last_value:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _write_markdown(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _best_model_from_metrics(metrics: pd.DataFrame) -> str:
    validation = metrics[metrics["split"] == "validation"]
    if validation.empty:
        validation = metrics[metrics["split"] == "test"]
    if validation.empty:
        validation = metrics
    return str(validation.sort_values("mae").iloc[0]["model_name"])


def generate_plots(paths: tif.utils.ProjectPaths, predictions: pd.DataFrame, metrics: pd.DataFrame) -> tuple[Path, ...]:
    """Generate report figures from processed data, predictions, and metrics."""

    cpi_path = paths.processed_data / "cpi_mom.parquet"
    dataset_path = paths.processed_data / "model_dataset.parquet"
    metadata_path = paths.processed_data / "feature_metadata.json"
    missing = [path for path in (cpi_path, dataset_path, metadata_path) if not path.is_file()]
    if missing:
        missing_paths = ", ".join(path.relative_to(paths.root).as_posix() for path in missing)
        raise EvaluationError(f"Missing plot inputs: {missing_paths}. Run `just preprocess` first.")

    figure_paths: list[Path] = []
    cpi = pd.read_parquet(cpi_path)
    dataset = pd.read_parquet(dataset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

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

    plt.figure(figsize=(7, 5.5))
    plt.scatter(
        model_predictions["actual_cpi_mom_percent"],
        model_predictions["prediction_cpi_mom_percent"],
        alpha=0.75,
    )
    axis_min = float(
        min(model_predictions["actual_cpi_mom_percent"].min(), model_predictions["prediction_cpi_mom_percent"].min())
    )
    axis_max = float(
        max(model_predictions["actual_cpi_mom_percent"].max(), model_predictions["prediction_cpi_mom_percent"].max())
    )
    plt.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", color="black", linewidth=1)
    plt.title(f"Predicted vs Actual CPI MoM ({best_model}, {plot_split})")
    plt.xlabel("Actual CPI MoM (%)")
    plt.ylabel("Predicted CPI MoM (%)")
    path = paths.figures / "predicted_vs_actual_scatter.png"
    _save_current_figure(path)
    figure_paths.append(path)

    model_predictions["absolute_error"] = residuals.abs()
    model_predictions["rolling_mae_6"] = model_predictions["absolute_error"].rolling(6, min_periods=1).mean()
    plt.figure(figsize=(10, 4.5))
    plt.plot(model_predictions["target_month_start"], model_predictions["absolute_error"], label="Absolute error")
    plt.plot(model_predictions["target_month_start"], model_predictions["rolling_mae_6"], label="6-month rolling MAE")
    plt.title(f"Forecast Error Over Time ({best_model}, {plot_split})")
    plt.xlabel("Target month")
    plt.ylabel("Absolute error")
    plt.legend()
    path = paths.figures / "rolling_mae.png"
    _save_current_figure(path)
    figure_paths.append(path)

    plt.figure(figsize=(10, 4.5))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.plot(model_predictions["target_month_start"], residuals)
    plt.title(f"Residuals Over Time ({best_model}, {plot_split})")
    plt.xlabel("Target month")
    plt.ylabel("Prediction error")
    path = paths.figures / "residuals_over_time.png"
    _save_current_figure(path)
    figure_paths.append(path)

    if "cpi_mom_trailing_std_12" in model_predictions.columns:
        volatility = model_predictions["cpi_mom_trailing_std_12"].replace(0, np.nan)
        model_predictions["volatility_normalized_abs_error"] = model_predictions["absolute_error"] / volatility
        plt.figure(figsize=(10, 4.5))
        plt.plot(
            model_predictions["target_month_start"],
            model_predictions["volatility_normalized_abs_error"],
            label="Abs error / trailing std",
        )
        plt.title(f"Volatility-Normalized Error ({best_model}, {plot_split})")
        plt.xlabel("Target month")
        plt.ylabel("Normalized absolute error")
        plt.legend()
        path = paths.figures / "volatility_normalized_error.png"
        _save_current_figure(path)
        figure_paths.append(path)

        plt.figure(figsize=(10, 4.5))
        plt.plot(model_predictions["target_month_start"], model_predictions["actual_cpi_mom_percent"], label="Actual")
        plt.plot(
            model_predictions["target_month_start"],
            model_predictions["prediction_cpi_mom_percent"],
            label=f"Prediction ({best_model})",
        )
        plt.fill_between(
            model_predictions["target_month_start"],
            model_predictions["actual_cpi_mom_percent"] - model_predictions["cpi_mom_trailing_std_12"],
            model_predictions["actual_cpi_mom_percent"] + model_predictions["cpi_mom_trailing_std_12"],
            alpha=0.18,
            label="Actual +/- trailing std",
        )
        plt.title(f"Prediction vs Actual With Trailing Volatility ({plot_split})")
        plt.xlabel("Target month")
        plt.ylabel("CPI MoM (%)")
        plt.legend()
        path = paths.figures / "prediction_with_volatility_band.png"
        _save_current_figure(path)
        figure_paths.append(path)

    return tuple(figure_paths)


def write_summary_reports(
    paths: tif.utils.ProjectPaths,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[Path, ...]:
    """Write human-readable evaluation and data summary reports."""

    dataset = pd.read_parquet(paths.processed_data / "model_dataset.parquet")
    metadata = json.loads((paths.processed_data / "feature_metadata.json").read_text(encoding="utf-8"))
    text_documents = pd.read_parquet(paths.processed_data / "text_documents.parquet")
    best_model = _best_model_from_metrics(metrics)
    best_split = "test" if (predictions["split"] == "test").any() else "validation"
    best_predictions = predictions[
        (predictions["model_name"] == best_model) & (predictions["split"] == best_split)
    ].copy()
    best_predictions["absolute_error"] = (
        best_predictions["prediction_cpi_mom_percent"] - best_predictions["actual_cpi_mom_percent"]
    ).abs()
    report_paths: list[Path] = []

    split_counts = dataset["split"].value_counts().reindex(["train", "validation", "test"], fill_value=0)
    report_paths.append(
        _write_markdown(
            paths.reports / "split_summary.md",
            [
                "# Split Summary",
                "",
                "| Split | Rows | First Origin | Last Origin |",
                "| ----- | ---- | ------------ | ----------- |",
                *[
                    f"| {split} | {count} | "
                    f"{dataset.loc[dataset['split'] == split, 'forecast_origin_month'].min()} | "
                    f"{dataset.loc[dataset['split'] == split, 'forecast_origin_month'].max()} |"
                    for split, count in split_counts.items()
                ],
            ],
        )
    )

    numeric_columns = list(metadata["numeric_feature_columns"])
    vocabulary_size = len(json.loads((paths.processed_data / "text_vocabulary.json").read_text(encoding="utf-8")))
    first_origin = dataset["forecast_origin_month"].min()
    last_origin = dataset["forecast_origin_month"].max()
    report_paths.append(
        _write_markdown(
            paths.reports / "data_summary.md",
            [
                "# Data Summary",
                "",
                f"Rows: `{len(dataset)}`",
                f"Forecast origins: `{first_origin}` to `{last_origin}`",
                f"Target months: `{dataset['target_month'].min()}` to `{dataset['target_month'].max()}`",
                f"Numeric features: `{len(numeric_columns)}`",
                f"Vocabulary size: `{vocabulary_size}`",
                f"Text documents: `{len(text_documents)}`",
            ],
        )
    )

    best_metric_rows = metrics[metrics["model_name"] == best_model].sort_values("split")
    report_paths.append(
        _write_markdown(
            paths.reports / "best_model_summary.md",
            [
                "# Best Model Summary",
                "",
                f"Best model selected by validation MAE: `{best_model}`",
                "",
                "| Split | MAE | RMSE | Direction Accuracy |",
                "| ----- | --- | ---- | ------------------ |",
                *[
                    f"| {row.split} | {row.mae:.4f} | {row.rmse:.4f} | {row.direction_accuracy:.3f} |"
                    for row in best_metric_rows.itertuples(index=False)
                ],
            ],
        )
    )

    worst = best_predictions.sort_values("absolute_error", ascending=False).head(10)
    report_paths.append(
        _write_markdown(
            paths.reports / "error_examples.md",
            [
                "# Largest Forecast Errors",
                "",
                f"Model: `{best_model}`; split: `{best_split}`",
                "",
                "| Target Month | Actual | Prediction | Abs Error |",
                "| ------------ | ------ | ---------- | --------- |",
                *[
                    f"| {row.target_month} | {row.actual_cpi_mom_percent:.4f} | "
                    f"{row.prediction_cpi_mom_percent:.4f} | {row.absolute_error:.4f} |"
                    for row in worst.itertuples(index=False)
                ],
            ],
        )
    )

    coverage = dataset[numeric_columns].notna().mean().sort_values()
    report_paths.append(
        _write_markdown(
            paths.reports / "feature_summary.md",
            [
                "# Feature Summary",
                "",
                f"Numeric feature count: `{len(numeric_columns)}`",
                "",
                "| Lowest-Coverage Feature | Coverage |",
                "| ----------------------- | -------- |",
                *[f"| `{feature}` | {value:.3f} |" for feature, value in coverage.head(15).items()],
            ],
        )
    )

    report_paths.append(
        _write_markdown(
            paths.reports / "text_corpus_summary.md",
            [
                "# Text Corpus Summary",
                "",
                f"Documents: `{len(text_documents)}`",
                f"First publication: `{text_documents['published_at'].min()}`",
                f"Last publication: `{text_documents['published_at'].max()}`",
                f"Median body words: `{text_documents['body_word_count'].median():.0f}`",
                f"Mean body words: `{text_documents['body_word_count'].mean():.1f}`",
            ],
        )
    )

    return tuple(report_paths)


def evaluate_predictions(paths: tif.utils.ProjectPaths = tif.utils.DEFAULT_PATHS) -> EvaluationResult:
    """Evaluate generated model predictions and write report artifacts."""

    tif.utils.ensure_generated_directories(paths)
    print(
        "evaluate: starting "
        f"predictions_dir={paths.predictions.relative_to(paths.root)} "
        f"reports_dir={paths.reports.relative_to(paths.root)} "
        f"figures_dir={paths.figures.relative_to(paths.root)}"
    )
    predictions_path = paths.predictions / "predictions.parquet"
    if not predictions_path.is_file():
        raise EvaluationError(f"Missing predictions: {predictions_path}. Run `just train` first.")
    predictions = pd.read_parquet(predictions_path)
    required_columns = {
        "model_name",
        "model_type",
        "split",
        "actual_cpi_mom_percent",
        "prediction_cpi_mom_percent",
        "previous_cpi_mom_percent",
    }
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise EvaluationError(f"Predictions are missing required columns: {sorted(missing_columns)}")

    metrics = build_metrics_table(predictions)
    split_counts = predictions["split"].value_counts().reindex(["train", "validation", "test"], fill_value=0).to_dict()
    best_model = _best_model_from_metrics(metrics)
    print(
        "evaluate: predictions "
        f"rows={len(predictions)} models={predictions['model_name'].nunique()} "
        f"splits={split_counts} best_model={best_model}"
    )
    metrics_json_path = paths.reports / "metrics.json"
    metrics_markdown_path = paths.reports / "metrics.md"
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_path.write_text(json.dumps(metrics.to_dict(orient="records"), indent=2), encoding="utf-8")
    _write_metrics_markdown(metrics_markdown_path, metrics)
    report_paths = write_summary_reports(paths, predictions, metrics)
    figure_paths = generate_plots(paths, predictions, metrics)
    return EvaluationResult(
        metrics_json_path=metrics_json_path,
        metrics_markdown_path=metrics_markdown_path,
        metric_rows=len(metrics),
        figure_paths=figure_paths,
        report_paths=report_paths,
    )


def main() -> int:
    try:
        result = evaluate_predictions(tif.utils.DEFAULT_PATHS)
    except (EvaluationError, ValueError, FileNotFoundError) as exc:
        print(f"evaluate: {exc}")
        return 1
    print(
        "evaluate: wrote "
        f"{result.metric_rows} metric rows to {result.metrics_json_path.relative_to(tif.utils.DEFAULT_PATHS.root)}"
    )
    print(
        f"evaluate: wrote markdown report to {result.metrics_markdown_path.relative_to(tif.utils.DEFAULT_PATHS.root)}"
    )
    print(f"evaluate: wrote {len(result.figure_paths)} figures")
    for figure_path in result.figure_paths:
        print(f"evaluate: wrote {figure_path.relative_to(tif.utils.DEFAULT_PATHS.root)}")
    print(f"evaluate: wrote {len(result.report_paths)} summary reports")
    for report_path in result.report_paths:
        print(f"evaluate: wrote {report_path.relative_to(tif.utils.DEFAULT_PATHS.root)}")
    return 0
