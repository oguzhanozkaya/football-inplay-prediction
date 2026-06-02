"""Leakage-safe model feature generation."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tif.config import DEFAULT_PATHS, ProjectPaths, ensure_generated_directories

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
MARKET_PREFIXES = ("usd_try", "eur_try", "fx_basket", "brent_oil")
CPI_LAGS = (1, 2, 3, 6, 12)
MARKET_LAGS = (0, 1, 2, 3, 6, 12)
DELAYED_MACRO_LAGS = (2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)
TEXT_LOOKBACK_MONTHS = 12
MAX_VOCAB_SIZE = 5000
MAX_TEXT_TOKENS = 256


@dataclass(frozen=True)
class FeatureGenerationResult:
    """Summary of generated processed feature artifacts."""

    dataset_path: Path
    metadata_path: Path
    vocabulary_path: Path
    split_summary_path: Path
    row_count: int
    numeric_feature_count: int
    vocabulary_size: int


class FeatureGenerationError(RuntimeError):
    """Raised when required interim artifacts are missing or invalid."""


def tokenize(text: str) -> list[str]:
    """Tokenize project text with a small from-scratch word tokenizer."""

    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def build_vocabulary(texts: pd.Series, max_size: int = MAX_VOCAB_SIZE, min_frequency: int = 1) -> dict[str, int]:
    """Build a token-id vocabulary from training texts only."""

    counter: Counter[str] = Counter()
    for text in texts.fillna(""):
        counter.update(tokenize(str(text)))

    vocabulary = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, count in counter.most_common(max_size - len(vocabulary)):
        if count < min_frequency:
            continue
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int], max_tokens: int = MAX_TEXT_TOKENS) -> list[int]:
    """Convert text to a bounded sequence of token ids."""

    tokens = tokenize(text)
    return [vocabulary.get(token, UNK_ID) for token in tokens[:max_tokens]]


def assign_chronological_splits(
    frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> pd.DataFrame:
    """Assign deterministic chronological train, validation, and test labels."""

    if frame.empty:
        raise FeatureGenerationError("Cannot assign splits to an empty dataset")

    ordered = frame.sort_values("forecast_origin_month_start").reset_index(drop=True).copy()
    row_count = len(ordered)
    if row_count < 3:
        ordered["split"] = "train"
        return ordered

    train_end = max(1, int(row_count * train_fraction))
    validation_end = max(train_end + 1, int(row_count * (train_fraction + validation_fraction)))
    if validation_end >= row_count:
        validation_end = row_count - 1

    ordered["split"] = "test"
    ordered.loc[: train_end - 1, "split"] = "train"
    ordered.loc[train_end : validation_end - 1, "split"] = "validation"
    return ordered


def _month_offset(month_start: pd.Timestamp, months: int) -> pd.Timestamp:
    return (month_start - pd.DateOffset(months=months)).normalize()


def _safe_value(monthly: pd.DataFrame, month_start: pd.Timestamp, column: str) -> float:
    if month_start not in monthly.index or column not in monthly.columns:
        return float("nan")
    value = monthly.at[month_start, column]
    if pd.isna(value):
        return float("nan")
    return float(value)


def _window_values(monthly: pd.DataFrame, end_month: pd.Timestamp, column: str, window: int) -> list[float]:
    months = [_month_offset(end_month, offset) for offset in reversed(range(window))]
    return [_safe_value(monthly, month, column) for month in months]


def _add_lag_features(
    row: dict[str, object],
    feature_columns: list[str],
    monthly: pd.DataFrame,
    origin_month: pd.Timestamp,
    source_column: str,
    feature_prefix: str,
    lags: tuple[int, ...],
) -> None:
    for lag in lags:
        feature_name = f"{feature_prefix}_lag_{lag}"
        row[feature_name] = _safe_value(monthly, _month_offset(origin_month, lag), source_column)
        feature_columns.append(feature_name)


def _add_rolling_features(
    row: dict[str, object],
    feature_columns: list[str],
    monthly: pd.DataFrame,
    end_month: pd.Timestamp,
    source_column: str,
    feature_prefix: str,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> None:
    for window in windows:
        values = np.array(_window_values(monthly, end_month, source_column, window), dtype=float)
        mean_name = f"{feature_prefix}_rolling_mean_{window}"
        std_name = f"{feature_prefix}_rolling_std_{window}"
        if np.isnan(values).any():
            row[mean_name] = float("nan")
            row[std_name] = float("nan")
        else:
            row[mean_name] = float(values.mean())
            row[std_name] = float(values.std(ddof=0))
        feature_columns.extend([mean_name, std_name])


def _add_change_feature(
    row: dict[str, object], feature_columns: list[str], current_name: str, previous_name: str
) -> None:
    feature_name = f"{current_name}_change_1"
    current_value = row.get(current_name)
    previous_value = row.get(previous_name)
    if pd.isna(current_value) or pd.isna(previous_value) or abs(float(previous_value)) < 1e-12:
        row[feature_name] = float("nan")
    else:
        row[feature_name] = (float(current_value) - float(previous_value)) / abs(float(previous_value))
    feature_columns.append(feature_name)


def _text_window_for_origin(documents: pd.DataFrame, forecast_origin_month_start: pd.Timestamp) -> tuple[str, int, int]:
    cutoff_date = forecast_origin_month_start + pd.offsets.MonthEnd(0)
    window_start = cutoff_date - pd.DateOffset(months=TEXT_LOOKBACK_MONTHS)
    window = documents[(documents["published_at"] <= cutoff_date) & (documents["published_at"] > window_start)]
    window = window.sort_values(["published_at", "document_id"])
    text = "\n\n".join(window["body_text"].fillna("").astype(str).tolist())
    return text, len(window), len(tokenize(text))


def build_model_dataset(
    cpi_target: pd.DataFrame,
    monthly_numeric: pd.DataFrame,
    text_documents: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, int], dict[str, int]]:
    """Build the leakage-safe model dataset and metadata."""

    cpi_monthly = cpi_target[["target_month_start", "target_month", "target_cpi_mom_percent", "cpi_yoy_percent"]].copy()
    cpi_monthly = cpi_monthly.rename(columns={"target_month_start": "month_start"}).set_index("month_start")
    numeric_monthly = monthly_numeric.copy().set_index("month_start")
    numeric_source_columns = [column for column in numeric_monthly.columns if column != "month"]
    market_columns = [column for column in numeric_source_columns if column.startswith(MARKET_PREFIXES)]
    delayed_macro_columns = [column for column in numeric_source_columns if column not in market_columns]

    rows: list[dict[str, object]] = []
    numeric_feature_columns: list[str] = []
    for target in cpi_target.sort_values("forecast_origin_month_start").itertuples(index=False):
        origin_month = pd.Timestamp(target.forecast_origin_month_start).normalize()
        row: dict[str, object] = {
            "forecast_origin_month_start": origin_month,
            "forecast_origin_month": target.forecast_origin_month,
            "target_month_start": pd.Timestamp(target.target_month_start).normalize(),
            "target_month": target.target_month,
            "target_cpi_mom_percent": float(target.target_cpi_mom_percent),
        }

        feature_columns: list[str] = []
        _add_lag_features(
            row,
            feature_columns,
            cpi_monthly,
            origin_month,
            "target_cpi_mom_percent",
            "cpi_mom",
            CPI_LAGS,
        )
        _add_lag_features(row, feature_columns, cpi_monthly, origin_month, "cpi_yoy_percent", "cpi_yoy", CPI_LAGS)
        _add_rolling_features(
            row,
            feature_columns,
            cpi_monthly,
            _month_offset(origin_month, 1),
            "target_cpi_mom_percent",
            "cpi_mom",
        )

        for column in market_columns:
            _add_lag_features(row, feature_columns, numeric_monthly, origin_month, column, column, MARKET_LAGS)
            _add_rolling_features(row, feature_columns, numeric_monthly, origin_month, column, column, windows=(3, 6))
            _add_change_feature(row, feature_columns, f"{column}_lag_0", f"{column}_lag_1")

        for column in delayed_macro_columns:
            _add_lag_features(row, feature_columns, numeric_monthly, origin_month, column, column, DELAYED_MACRO_LAGS)
            _add_rolling_features(
                row, feature_columns, numeric_monthly, _month_offset(origin_month, 2), column, column, windows=(3, 6)
            )
            _add_change_feature(row, feature_columns, f"{column}_lag_2", f"{column}_lag_3")

        if not numeric_feature_columns:
            numeric_feature_columns = feature_columns.copy()
        rows.append(row)

    dataset = pd.DataFrame(rows)
    dataset = dataset.dropna(subset=["target_cpi_mom_percent", *numeric_feature_columns]).reset_index(drop=True)
    if dataset.empty:
        raise FeatureGenerationError("Feature generation produced no complete model rows")

    text_documents = text_documents.copy()
    text_documents["published_at"] = pd.to_datetime(text_documents["published_at"])
    text_windows = [
        _text_window_for_origin(text_documents, pd.Timestamp(origin))
        for origin in dataset["forecast_origin_month_start"]
    ]
    dataset["text_window"] = [window[0] for window in text_windows]
    dataset["text_document_count"] = [window[1] for window in text_windows]
    dataset["text_window_token_count"] = [window[2] for window in text_windows]
    numeric_feature_columns.extend(["text_document_count", "text_window_token_count"])

    dataset = assign_chronological_splits(dataset)
    vocabulary = build_vocabulary(dataset.loc[dataset["split"] == "train", "text_window"])
    dataset["text_token_ids"] = dataset["text_window"].map(lambda text: encode_text(str(text), vocabulary))

    split_summary = dataset["split"].value_counts().reindex(["train", "validation", "test"], fill_value=0).to_dict()
    metadata: dict[str, object] = {
        "target_column": "target_cpi_mom_percent",
        "split_column": "split",
        "numeric_feature_columns": numeric_feature_columns,
        "text_token_column": "text_token_ids",
        "text_window_column": "text_window",
        "max_text_tokens": MAX_TEXT_TOKENS,
        "text_lookback_months": TEXT_LOOKBACK_MONTHS,
        "market_columns": market_columns,
        "delayed_macro_columns": delayed_macro_columns,
        "split_summary": split_summary,
        "row_count": len(dataset),
    }
    return dataset, metadata, vocabulary, split_summary


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_features(paths: ProjectPaths = DEFAULT_PATHS) -> FeatureGenerationResult:
    """Read interim tables and write model-ready processed feature artifacts."""

    ensure_generated_directories(paths)
    cpi_path = paths.interim_data / "cpi_mom.parquet"
    monthly_numeric_path = paths.interim_data / "monthly_numeric.parquet"
    text_documents_path = paths.interim_data / "text_documents.parquet"
    missing = [path for path in (cpi_path, monthly_numeric_path, text_documents_path) if not path.is_file()]
    if missing:
        missing_paths = ", ".join(path.relative_to(paths.root).as_posix() for path in missing)
        raise FeatureGenerationError(f"Missing interim artifacts: {missing_paths}. Run `just preprocess` first.")

    dataset, metadata, vocabulary, split_summary = build_model_dataset(
        pd.read_parquet(cpi_path),
        pd.read_parquet(monthly_numeric_path),
        pd.read_parquet(text_documents_path),
    )

    dataset_path = paths.processed_data / "model_dataset.parquet"
    metadata_path = paths.processed_data / "feature_metadata.json"
    vocabulary_path = paths.processed_data / "text_vocabulary.json"
    split_summary_path = paths.processed_data / "split_summary.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(dataset_path, index=False)
    _write_json(metadata_path, metadata)
    _write_json(vocabulary_path, vocabulary)
    _write_json(split_summary_path, split_summary)
    return FeatureGenerationResult(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        vocabulary_path=vocabulary_path,
        split_summary_path=split_summary_path,
        row_count=len(dataset),
        numeric_feature_count=len(metadata["numeric_feature_columns"]),
        vocabulary_size=len(vocabulary),
    )
