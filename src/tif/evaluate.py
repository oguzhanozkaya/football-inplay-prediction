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

    return tuple(figure_paths)


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
    figure_paths = generate_plots(paths, predictions, metrics)
    return EvaluationResult(
        metrics_json_path=metrics_json_path,
        metrics_markdown_path=metrics_markdown_path,
        metric_rows=len(metrics),
        figure_paths=figure_paths,
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
    return 0
