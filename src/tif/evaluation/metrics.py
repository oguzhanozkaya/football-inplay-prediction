"""Evaluation metrics for forecast outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tif.config import DEFAULT_PATHS, ProjectPaths, ensure_generated_directories


@dataclass(frozen=True)
class EvaluationResult:
    """Summary of generated evaluation artifacts."""

    metrics_json_path: Path
    metrics_markdown_path: Path
    metric_rows: int


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
    return {
        "row_count": int(len(group)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "direction_accuracy": float(np.mean(actual_direction == prediction_direction)),
    }


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
    lines.append("| Split | Model | Type | Rows | MAE | RMSE | Direction Accuracy | MAE Delta vs Last Value |")
    lines.append("| ----- | ----- | ---- | ---- | --- | ---- | ------------------ | ----------------------- |")
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.split} | `{row.model_name}` | {row.model_type} | {row.row_count} | "
            f"{row.mae:.4f} | {row.rmse:.4f} | {row.direction_accuracy:.3f} | "
            f"{row.mae_improvement_vs_last_value:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_predictions(paths: ProjectPaths = DEFAULT_PATHS) -> EvaluationResult:
    """Evaluate generated model predictions and write report artifacts."""

    ensure_generated_directories(paths)
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
    metrics_json_path = paths.reports / "metrics.json"
    metrics_markdown_path = paths.reports / "metrics.md"
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_path.write_text(json.dumps(metrics.to_dict(orient="records"), indent=2), encoding="utf-8")
    _write_metrics_markdown(metrics_markdown_path, metrics)
    return EvaluationResult(
        metrics_json_path=metrics_json_path,
        metrics_markdown_path=metrics_markdown_path,
        metric_rows=len(metrics),
    )
