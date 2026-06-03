"""Evaluate football match outcome classification predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support

import fip.utils

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class EvaluationResult:
    metrics_json_path: Path
    metrics_markdown_path: Path
    metric_rows: int
    figure_paths: tuple[Path, ...]
    report_paths: tuple[Path, ...]


class EvaluationError(RuntimeError):
    """Raised when prediction artifacts are missing or invalid."""


def classification_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    actual = group["actual"].to_numpy(dtype=int)
    prediction = group["prediction"].to_numpy(dtype=int)
    probabilities = group[["prob_home", "prob_draw", "prob_away"]].to_numpy(dtype=float)
    probabilities = probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), 1e-12, None)
    return {
        "row_count": int(len(group)),
        "accuracy": float(accuracy_score(actual, prediction)),
        "macro_f1": float(f1_score(actual, prediction, average="macro", zero_division=0)),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1, 2])),
        "mean_confidence": float(group["confidence"].mean()),
    }


def build_metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in predictions.groupby("split", sort=True):
        rows.append({"model_name": "fusion_gru", "split": split, **classification_metrics(group)})
    return pd.DataFrame(rows).sort_values(["split"]).reset_index(drop=True)


def _write_metrics_markdown(path: Path, metrics: pd.DataFrame) -> None:
    lines = [
        "# Evaluation Metrics",
        "",
        "| Split | Rows | Accuracy | Macro F1 | Log Loss | Mean Confidence |",
        "| ----- | ---- | -------- | -------- | -------- | --------------- |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.row_count} | {row.accuracy:.4f} | {row.macro_f1:.4f} | "
            f"{row.log_loss:.4f} | {row.mean_confidence:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_current_figure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def generate_plots(paths: fip.utils.ProjectPaths, predictions: pd.DataFrame, metrics: pd.DataFrame) -> tuple[Path, ...]:
    figure_paths = []
    labels = list(fip.utils.LABELS)
    plot_split = "test" if (predictions["split"] == "test").any() else predictions["split"].iloc[-1]
    split_predictions = predictions[predictions["split"] == plot_split]

    matrix = confusion_matrix(split_predictions["actual"], split_predictions["prediction"], labels=[0, 1, 2])
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Confusion Matrix ({plot_split})")
    plt.xticks(range(3), labels)
    plt.yticks(range(3), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    figure_paths.append(_save_current_figure(paths.figures / "confusion_matrix.png"))

    class_counts = predictions.groupby(["split", "actual_label"]).size().unstack(fill_value=0)
    plt.figure(figsize=(8, 4.5))
    class_counts.reindex(columns=labels, fill_value=0).plot(kind="bar", ax=plt.gca())
    plt.title("Class Distribution by Split")
    plt.xlabel("Split")
    plt.ylabel("Matches")
    plt.xticks(rotation=0)
    figure_paths.append(_save_current_figure(paths.figures / "class_distribution.png"))

    plt.figure(figsize=(7, 4.5))
    plt.hist(split_predictions["confidence"], bins=20, edgecolor="black")
    plt.title(f"Prediction Confidence ({plot_split})")
    plt.xlabel("Maximum class probability")
    plt.ylabel("Matches")
    figure_paths.append(_save_current_figure(paths.figures / "prediction_confidence.png"))

    plt.figure(figsize=(7, 4.5))
    split_metrics = metrics.set_index("split")
    split_metrics[["accuracy", "macro_f1"]].plot(kind="bar", ax=plt.gca())
    plt.title("Classification Metrics by Split")
    plt.xlabel("Split")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    figure_paths.append(_save_current_figure(paths.figures / "metric_comparison.png"))
    return tuple(figure_paths)


def _write_classification_report(path: Path, predictions: pd.DataFrame) -> Path:
    lines = ["# Per-Class Report", ""]
    for split, group in predictions.groupby("split", sort=True):
        precision, recall, f1, support = precision_recall_fscore_support(
            group["actual"], group["prediction"], labels=[0, 1, 2], zero_division=0
        )
        lines.extend(
            [
                f"## {split.title()}",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "| ----- | --------- | ------ | -- | ------- |",
            ]
        )
        for index, label in enumerate(fip.utils.LABELS):
            lines.append(
                f"| {label} | {precision[index]:.4f} | {recall[index]:.4f} | {f1[index]:.4f} | {support[index]} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_error_examples(path: Path, predictions: pd.DataFrame) -> Path:
    errors = (
        predictions[predictions["actual"] != predictions["prediction"]]
        .sort_values("confidence", ascending=False)
        .head(25)
    )
    lines = [
        "# High-Confidence Errors",
        "",
        "| Split | Date | Event | League | Actual | Predicted | Confidence |",
        "| ----- | ---- | ----- | ------ | ------ | --------- | ---------- |",
    ]
    for row in errors.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.date} | {row.eventId} | {row.league} | {row.actual_label} | "
            f"{row.prediction_label} | {row.confidence:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluate_predictions(paths: fip.utils.ProjectPaths = fip.utils.DEFAULT_PATHS) -> EvaluationResult:
    fip.utils.ensure_generated_directories(paths)
    predictions_path = paths.predictions / "predictions.csv"
    if not predictions_path.is_file():
        raise EvaluationError("Missing predictions. Run `just train` first.")
    predictions = pd.read_csv(predictions_path)
    metrics = build_metrics_table(predictions)
    metrics_json = paths.reports / "metrics.json"
    metrics_md = paths.reports / "metrics.md"
    metrics.to_json(metrics_json, orient="records", indent=2)
    _write_metrics_markdown(metrics_md, metrics)
    figures = generate_plots(paths, predictions, metrics)
    reports = (
        _write_classification_report(paths.reports / "class_report.md", predictions),
        _write_error_examples(paths.reports / "error_examples.md", predictions),
    )
    print(f"evaluate: wrote metrics rows={len(metrics)} figures={len(figures)} reports={len(reports)}")
    return EvaluationResult(metrics_json, metrics_md, len(metrics), figures, reports)


def main() -> None:
    evaluate_predictions()


if __name__ == "__main__":
    main()
