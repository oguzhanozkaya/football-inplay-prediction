"""Single-script ESPN Soccer first-60-minute outcome prediction pipeline.

Run locally with `uv run python fig.py` or in Colab after installing the dependencies.
The script downloads the Kaggle dataset when `data/raw/` does not already contain it,
builds leakage-safe 5-minute numeric windows, trains one Temporal CNN classifier,
and writes predictions, metrics, reports, and figures under `output/`.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import subprocess
import sys
import zipfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

os.environ["MPLBACKEND"] = "Agg"
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"

KAGGLE_DATASET = "excel4soccer/espn-soccer-data"

# Pipeline and model hyperparameters. Edit these constants before running the
# script instead of passing environment variables.
SEED = 67
EPOCHS = 600
PATIENCE = 120
BATCH_SIZE = 64
LEARNING_RATE = 0.0000005
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_MIN_DELTA = 0.001
DEVICE = "cuda"
CUTOFF_MINUTE = 60
WINDOW_MINUTES = 5
TOP_PLAY_TYPES = 30
TOP_KEY_EVENT_TYPES = 24
TOP_FORMATIONS = 12
NUMERIC_PROJECTION_SIZE = 128
TEMPORAL_CHANNEL_COUNT = 128
TEMPORAL_KERNEL_SIZE = 3
TEMPORAL_BLOCK_COUNT = 2
FUSION_HIDDEN_SIZE = 256
MLP_HIDDEN_SIZE = 256
DROPOUT = 0.1
LABEL_SMOOTHING = 0.03
NORMALIZATION_GROUPS = 8
USE_CLASS_WEIGHTS = True
MODEL_TYPE = "tcn"
TEAM_FORM_WINDOW = 5
ELO_INITIAL_RATING = 1500.0
ELO_K_FACTOR = 20.0
ELO_HOME_ADVANTAGE = 60.0
DATALOADER_WORKERS = 4
MIXED_PRECISION = True
COMPILE_MODEL = False
MATCH_LIMIT = 0
USE_PREPROCESS_CACHE = True
FORCE_REPROCESS = False
CHECKPOINT_INTERVAL_EPOCHS = 25
PREPROCESS_CACHE_VERSION = 3

LABELS = ("home", "draw", "away")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
COMPLETED_STATUS_IDS = {28}


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    epochs: int = EPOCHS
    patience: int = PATIENCE
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    min_delta: float = EARLY_STOPPING_MIN_DELTA
    device: str = DEVICE
    cutoff_minute: int = CUTOFF_MINUTE
    window_minutes: int = WINDOW_MINUTES
    top_play_types: int = TOP_PLAY_TYPES
    top_key_event_types: int = TOP_KEY_EVENT_TYPES
    top_formations: int = TOP_FORMATIONS
    numeric_projection_size: int = NUMERIC_PROJECTION_SIZE
    temporal_channel_count: int = TEMPORAL_CHANNEL_COUNT
    temporal_kernel_size: int = TEMPORAL_KERNEL_SIZE
    temporal_block_count: int = TEMPORAL_BLOCK_COUNT
    fusion_hidden_size: int = FUSION_HIDDEN_SIZE
    mlp_hidden_size: int = MLP_HIDDEN_SIZE
    dropout: float = DROPOUT
    label_smoothing: float = LABEL_SMOOTHING
    normalization_groups: int = NORMALIZATION_GROUPS
    use_class_weights: bool = USE_CLASS_WEIGHTS
    model_type: str = MODEL_TYPE
    team_form_window: int = TEAM_FORM_WINDOW
    elo_initial_rating: float = ELO_INITIAL_RATING
    elo_k_factor: float = ELO_K_FACTOR
    elo_home_advantage: float = ELO_HOME_ADVANTAGE
    dataloader_workers: int = DATALOADER_WORKERS
    mixed_precision: bool = MIXED_PRECISION
    compile_model: bool = COMPILE_MODEL
    match_limit: int = MATCH_LIMIT
    use_preprocess_cache: bool = USE_PREPROCESS_CACHE
    force_reprocess: bool = FORCE_REPROCESS
    checkpoint_interval_epochs: int = CHECKPOINT_INTERVAL_EPOCHS


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def raw_dataset_present() -> bool:
    required = [
        RAW_DIR / "base_data" / "fixtures.csv",
        RAW_DIR / "base_data" / "leagues.csv",
        RAW_DIR / "plays_data",
        RAW_DIR / "keyEvents_data",
        RAW_DIR / "commentary_data",
        RAW_DIR / "lineup_data",
    ]
    return all(path.exists() for path in required)


def download_kaggle_dataset() -> None:
    """Download and unzip the Kaggle dataset if local raw files are absent."""

    if raw_dataset_present():
        print("download: raw ESPN dataset already present")
        return
    ensure_dirs()
    zip_path = RAW_DIR / "espn-soccer-data.zip"
    print(f"download: downloading Kaggle dataset {KAGGLE_DATASET}")
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(RAW_DIR),
        "--force",
    ]
    try:
        subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise RuntimeError(
            "Kaggle download failed. Install the kaggle package and provide credentials via "
            "~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY. In Colab, upload kaggle.json "
            "to /root/.kaggle/kaggle.json with chmod 600."
        ) from error
    candidates = sorted(RAW_DIR.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    archive = zip_path if zip_path.exists() else candidates[0]
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(RAW_DIR)
    print(f"download: extracted {archive.name}")
    if not raw_dataset_present():
        raise RuntimeError(f"Downloaded archive did not create expected ESPN directories under {RAW_DIR}")


def read_csv_group(directory: Path, columns: list[str] | None = None, prefix: str | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("*.csv")):
        if prefix is not None and not path.name.startswith(prefix):
            continue
        header = pd.read_csv(path, nrows=0)
        usecols = [column for column in (columns or header.columns.tolist()) if column in header.columns]
        if not usecols:
            continue
        frame = pd.read_csv(path, usecols=usecols)
        frame["raw_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*(columns or []), "raw_file"])
    return pd.concat(frames, ignore_index=True)


def clock_to_minute(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return numeric / 60.0 if numeric > 120 else numeric
    text = str(value).strip()
    stoppage = re.match(r"^(?P<minute>\d+)\s*'\s*\+\s*(?P<extra>\d+)", text)
    if stoppage:
        return float(stoppage.group("minute")) + float(stoppage.group("extra"))
    minute = re.search(r"(?P<minute>\d+)", text)
    return float(minute.group("minute")) if minute else None


def build_fixture_table(config: Config) -> pd.DataFrame:
    fixtures = pd.read_csv(RAW_DIR / "base_data" / "fixtures.csv")
    leagues = pd.read_csv(RAW_DIR / "base_data" / "leagues.csv")
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures = fixtures[fixtures["statusId"].isin(COMPLETED_STATUS_IDS)].copy()
    fixtures["homeTeamScore"] = pd.to_numeric(fixtures["homeTeamScore"], errors="coerce")
    fixtures["awayTeamScore"] = pd.to_numeric(fixtures["awayTeamScore"], errors="coerce")
    fixtures = fixtures.dropna(subset=["date", "homeTeamScore", "awayTeamScore"])
    fixtures["target_label"] = np.select(
        [fixtures["homeTeamScore"] > fixtures["awayTeamScore"], fixtures["homeTeamScore"] < fixtures["awayTeamScore"]],
        ["home", "away"],
        default="draw",
    )
    fixtures["target"] = fixtures["target_label"].map(LABEL_TO_ID).astype(int)
    fixtures = fixtures.merge(
        leagues[["seasonType", "leagueId", "year", "midsizeName", "leagueName"]].drop_duplicates(),
        on=["seasonType", "leagueId"],
        how="left",
    )
    fixtures["league_key"] = fixtures.apply(lambda row: f"{row.seasonType}-{row.leagueId}-{row.year}", axis=1)
    fixtures = fixtures.sort_values(["league_key", "date", "eventId"]).reset_index(drop=True)
    if config.match_limit > 0:
        fixtures = fixtures.head(config.match_limit).copy()
    return fixtures[
        [
            "seasonType",
            "leagueId",
            "year",
            "midsizeName",
            "leagueName",
            "league_key",
            "eventId",
            "date",
            "homeTeamId",
            "awayTeamId",
            "homeTeamScore",
            "awayTeamScore",
            "target_label",
            "target",
        ]
    ]


def assign_league_chronological_splits(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Split every league chronologically so leagues are represented homogeneously across sets."""

    frames = []
    for _league_key, group in fixtures.groupby("league_key", sort=False):
        group = group.sort_values(["date", "eventId"]).copy()
        count = len(group)
        if count >= 10:
            train_end = max(1, int(math.floor(count * 0.70)))
            validation_end = max(train_end + 1, int(math.floor(count * 0.85)))
            validation_end = min(validation_end, count - 1)
            labels = np.array(["test"] * count, dtype=object)
            labels[:train_end] = "train"
            labels[train_end:validation_end] = "validation"
        elif count >= 3:
            labels = np.array(["train"] * count, dtype=object)
            labels[-2] = "validation"
            labels[-1] = "test"
        else:
            labels = np.array(["train"] * count, dtype=object)
        group["split"] = labels
        frames.append(group)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "eventId"]).reset_index(drop=True)


def prepare_plays(event_ids: set[int], cutoff_minute: int) -> pd.DataFrame:
    columns = [
        "eventId",
        "typeId",
        "text",
        "shortText",
        "period",
        "clockValue",
        "clockDisplayValue",
        "teamId",
        "scoringPlay",
        "fieldpositionX",
        "fieldPositionX",
        "fieldPositionY",
    ]
    plays = read_csv_group(RAW_DIR / "plays_data", columns=columns, prefix="plays_")
    if plays.empty:
        return plays
    plays = plays[plays["eventId"].isin(event_ids)].copy()
    plays["minute"] = (
        plays["clockValue"].map(clock_to_minute).combine_first(plays["clockDisplayValue"].map(clock_to_minute))
    )
    plays = plays[plays["minute"].notna() & (plays["minute"] <= cutoff_minute)].copy()
    plays["text_combined"] = plays[["text", "shortText"]].fillna("").agg(" ".join, axis=1)
    return plays


def prepare_key_events(event_ids: set[int], cutoff_minute: int) -> pd.DataFrame:
    columns = [
        "eventId",
        "keyEventTypeId",
        "period",
        "clockValue",
        "clockDisplayValue",
        "scoringPlay",
        "keyEventText",
        "keyEventShortText",
        "teamId",
        "fieldPositionX",
        "fieldPositionY",
    ]
    events = read_csv_group(RAW_DIR / "keyEvents_data", columns=columns, prefix="keyEvents_")
    if events.empty:
        return events
    events = events[events["eventId"].isin(event_ids)].copy()
    events["minute"] = (
        events["clockValue"].map(clock_to_minute).combine_first(events["clockDisplayValue"].map(clock_to_minute))
    )
    events = events[events["minute"].notna() & (events["minute"] <= cutoff_minute)].copy()
    events["text_combined"] = events[["keyEventText", "keyEventShortText"]].fillna("").agg(" ".join, axis=1)
    return events


def prepare_commentary(event_ids: set[int], cutoff_minute: int) -> pd.DataFrame:
    commentary = read_csv_group(
        RAW_DIR / "commentary_data", columns=["eventId", "clockDisplayValue", "commentaryText"], prefix="commentary_"
    )
    if commentary.empty:
        return commentary
    commentary = commentary[commentary["eventId"].isin(event_ids)].copy()
    commentary["minute"] = commentary["clockDisplayValue"].map(clock_to_minute).fillna(0.0)
    commentary = commentary[commentary["minute"] <= cutoff_minute].copy()
    commentary["text_combined"] = commentary["commentaryText"].fillna("")
    return commentary


def prepare_lineups(event_ids: set[int]) -> pd.DataFrame:
    lineups = read_csv_group(
        RAW_DIR / "lineup_data", columns=["eventId", "homeAway", "formation", "starter"], prefix="lineup_"
    )
    return lineups[lineups["eventId"].isin(event_ids)].copy() if not lineups.empty else lineups


def top_values(series: pd.Series, limit: int) -> list[int]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    return values.value_counts().head(limit).index.tolist()


def add_event_aggregates(
    features: pd.DataFrame,
    events: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    prefix: str,
    type_column: str,
    type_values: list[int],
) -> None:
    if events.empty:
        return
    enriched = events.merge(fixtures[["eventId", "homeTeamId", "awayTeamId"]], on="eventId", how="left")
    features.loc[:, f"{prefix}_count"] = features[f"{prefix}_count"].add(
        enriched.groupby("eventId").size(), fill_value=0
    )

    team_id = pd.to_numeric(enriched["teamId"], errors="coerce")
    home_team_id = pd.to_numeric(enriched["homeTeamId"], errors="coerce")
    away_team_id = pd.to_numeric(enriched["awayTeamId"], errors="coerce")
    enriched["side"] = np.select([team_id == home_team_id, team_id == away_team_id], ["home", "away"], default="")
    side_counts = (
        enriched[enriched["side"].isin(["home", "away"])].groupby(["eventId", "side"]).size().unstack(fill_value=0)
    )
    for side in ("home", "away"):
        if side in side_counts:
            features.loc[:, f"{side}_{prefix}_count"] = features[f"{side}_{prefix}_count"].add(
                side_counts[side], fill_value=0
            )

    scoring = pd.to_numeric(enriched["scoringPlay"], errors="coerce").fillna(0).astype(float)
    scoring_rows = enriched[scoring == 1.0]
    if not scoring_rows.empty:
        scoring_counts = scoring_rows.groupby("eventId").size()
        features.loc[:, "scoring_play_count"] = features["scoring_play_count"].add(scoring_counts, fill_value=0)
        goal_counts = (
            scoring_rows[scoring_rows["side"].isin(["home", "away"])]
            .groupby(["eventId", "side"])
            .size()
            .unstack(fill_value=0)
        )
        for side in ("home", "away"):
            if side in goal_counts:
                features.loc[:, f"{side}_goals"] = features[f"{side}_goals"].add(goal_counts[side], fill_value=0)

    if type_values:
        typed = enriched[enriched[type_column].notna()].copy()
        typed[type_column] = pd.to_numeric(typed[type_column], errors="coerce")
        typed = typed[typed[type_column].isin(type_values)]
        if not typed.empty:
            type_counts = pd.crosstab(typed["eventId"], typed[type_column].astype(int))
            type_counts = type_counts.rename(columns={value: f"{prefix}_type_{value}" for value in type_counts.columns})
            for column in type_counts.columns:
                if column in features:
                    features.loc[:, column] = features[column].add(type_counts[column], fill_value=0)


def build_position_features(plays: pd.DataFrame, key_events: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for frame in (plays, key_events):
        if frame.empty:
            continue
        x_column = "fieldpositionX" if "fieldpositionX" in frame.columns else "fieldPositionX"
        if x_column not in frame.columns or "fieldPositionY" not in frame.columns:
            continue
        positions = frame[["eventId", x_column, "fieldPositionY"]].copy()
        positions = positions.rename(columns={x_column: "mean_field_x", "fieldPositionY": "mean_field_y"})
        positions[["mean_field_x", "mean_field_y"]] = positions[["mean_field_x", "mean_field_y"]].apply(
            pd.to_numeric, errors="coerce"
        )
        positions = positions.replace(0, np.nan).dropna(subset=["mean_field_x", "mean_field_y"])
        if not positions.empty:
            frames.append(positions)
    if not frames:
        return pd.DataFrame(columns=["mean_field_x", "mean_field_y"])
    return pd.concat(frames, ignore_index=True).groupby("eventId")[["mean_field_x", "mean_field_y"]].mean()


def build_lineup_feature_frame(lineups: pd.DataFrame, formations: list[str]) -> pd.DataFrame:
    if lineups.empty:
        return pd.DataFrame()
    prepared = lineups[lineups["homeAway"].isin(["home", "away"])].copy()
    if prepared.empty:
        return pd.DataFrame()
    prepared["starter"] = pd.to_numeric(prepared["starter"], errors="coerce").fillna(0.0)
    starter_counts = prepared.groupby(["eventId", "homeAway"])["starter"].sum().unstack().fillna(0.0)
    starter_counts = starter_counts.rename(columns={"home": "home_starter_count", "away": "away_starter_count"})

    pieces = [starter_counts]
    if formations:
        selected = prepared[prepared["formation"].astype(str).isin(formations)].copy()
        if not selected.empty:
            selected["feature"] = selected["homeAway"].astype(str) + "_formation_" + selected["formation"].astype(str)
            formation_flags = pd.crosstab(selected["eventId"], selected["feature"]).clip(upper=1).astype(float)
            pieces.append(formation_flags)
    return pd.concat(pieces, axis=1)


def _team_snapshot(stats: dict[str, object], prefix: str, config: Config) -> dict[str, float]:
    matches = float(stats["matches"])
    recent = list(stats["recent"])
    recent_count = float(len(recent))
    if recent:
        recent_points = float(np.mean([item["points"] for item in recent]))
        recent_goal_diff = float(np.mean([item["goal_diff"] for item in recent]))
        recent_goals_for = float(np.mean([item["goals_for"] for item in recent]))
        recent_goals_against = float(np.mean([item["goals_against"] for item in recent]))
        recent_win_rate = float(np.mean([item["win"] for item in recent]))
    else:
        recent_points = 0.0
        recent_goal_diff = 0.0
        recent_goals_for = 0.0
        recent_goals_against = 0.0
        recent_win_rate = 0.0
    return {
        f"{prefix}_matches_played_before": matches,
        f"{prefix}_recent_match_count": recent_count,
        f"{prefix}_season_points_per_match": float(stats["points"]) / max(matches, 1.0),
        f"{prefix}_season_goal_diff_per_match": float(stats["goal_diff"]) / max(matches, 1.0),
        f"{prefix}_season_goals_for_per_match": float(stats["goals_for"]) / max(matches, 1.0),
        f"{prefix}_season_goals_against_per_match": float(stats["goals_against"]) / max(matches, 1.0),
        f"{prefix}_season_win_rate": float(stats["wins"]) / max(matches, 1.0),
        f"{prefix}_recent_points_per_match": recent_points,
        f"{prefix}_recent_goal_diff_per_match": recent_goal_diff,
        f"{prefix}_recent_goals_for_per_match": recent_goals_for,
        f"{prefix}_recent_goals_against_per_match": recent_goals_against,
        f"{prefix}_recent_win_rate": recent_win_rate,
        f"{prefix}_elo_before": float(stats.get("elo", config.elo_initial_rating)),
    }


def build_team_strength_features(fixtures: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Build pre-match form and Elo features using only earlier fixtures."""

    def new_stats() -> dict[str, object]:
        return {
            "matches": 0.0,
            "points": 0.0,
            "goal_diff": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "wins": 0.0,
            "recent": deque(maxlen=config.team_form_window),
            "elo": config.elo_initial_rating,
        }

    stats_by_team: defaultdict[int, dict[str, object]] = defaultdict(new_stats)
    rows = []
    sorted_fixtures = fixtures.sort_values(["date", "eventId"])
    for _date, date_group in sorted_fixtures.groupby("date", sort=True):
        pending_updates = []
        for match in date_group.itertuples(index=False):
            home_team_id = int(match.homeTeamId)
            away_team_id = int(match.awayTeamId)
            home_stats = stats_by_team[home_team_id]
            away_stats = stats_by_team[away_team_id]
            home_features = _team_snapshot(home_stats, "home", config)
            away_features = _team_snapshot(away_stats, "away", config)
            row = {"eventId": int(match.eventId), **home_features, **away_features}
            for name in (
                "season_points_per_match",
                "season_goal_diff_per_match",
                "season_goals_for_per_match",
                "season_goals_against_per_match",
                "season_win_rate",
                "recent_points_per_match",
                "recent_goal_diff_per_match",
                "recent_goals_for_per_match",
                "recent_goals_against_per_match",
                "recent_win_rate",
                "matches_played_before",
                "recent_match_count",
            ):
                row[f"diff_{name}"] = row[f"home_{name}"] - row[f"away_{name}"]
            row["elo_diff"] = row["home_elo_before"] - row["away_elo_before"]
            rows.append(row)

            home_score = float(match.homeTeamScore)
            away_score = float(match.awayTeamScore)
            if home_score > away_score:
                home_points, away_points, home_result = 3.0, 0.0, 1.0
            elif home_score < away_score:
                home_points, away_points, home_result = 0.0, 3.0, 0.0
            else:
                home_points, away_points, home_result = 1.0, 1.0, 0.5
            home_expected = 1.0 / (
                1.0
                + 10.0 ** ((float(away_stats["elo"]) - (float(home_stats["elo"]) + config.elo_home_advantage)) / 400.0)
            )
            home_elo_delta = config.elo_k_factor * (home_result - home_expected)
            pending_updates.append((home_team_id, home_score, away_score, home_points, home_elo_delta))
            pending_updates.append((away_team_id, away_score, home_score, away_points, -home_elo_delta))
        for team_id, goals_for, goals_against, points, elo_delta in pending_updates:
            stats = stats_by_team[team_id]
            goal_diff = goals_for - goals_against
            win = 1.0 if points == 3.0 else 0.0
            stats["matches"] = float(stats["matches"]) + 1.0
            stats["points"] = float(stats["points"]) + points
            stats["goals_for"] = float(stats["goals_for"]) + goals_for
            stats["goals_against"] = float(stats["goals_against"]) + goals_against
            stats["goal_diff"] = float(stats["goal_diff"]) + goal_diff
            stats["wins"] = float(stats["wins"]) + win
            stats["elo"] = float(stats["elo"]) + elo_delta
            stats["recent"].append(
                {
                    "points": points,
                    "goal_diff": goal_diff,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "win": win,
                }
            )
    return pd.DataFrame(rows).set_index("eventId").sort_index()


def window_count(config: Config) -> int:
    return int(math.ceil(config.cutoff_minute / config.window_minutes))


def preprocess_cache_key(config: Config) -> dict[str, object]:
    """Settings that change the processed dataset shape or values."""

    return {
        "version": PREPROCESS_CACHE_VERSION,
        "cutoff_minute": config.cutoff_minute,
        "window_minutes": config.window_minutes,
        "top_play_types": config.top_play_types,
        "top_key_event_types": config.top_key_event_types,
        "top_formations": config.top_formations,
        "match_limit": config.match_limit,
        "team_form_window": config.team_form_window,
        "elo_initial_rating": config.elo_initial_rating,
        "elo_k_factor": config.elo_k_factor,
        "elo_home_advantage": config.elo_home_advantage,
    }


def load_cached_dataset(config: Config) -> tuple[pd.DataFrame, dict[str, object]] | None:
    dataset_path = PROCESSED_DIR / "model_dataset.parquet"
    metadata_path = PROCESSED_DIR / "feature_metadata.json"
    if config.force_reprocess or not config.use_preprocess_cache:
        return None
    if not dataset_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("preprocess_cache_key") != preprocess_cache_key(config):
        print("preprocess: cache miss because preprocessing settings changed")
        return None
    dataset = pd.read_parquet(dataset_path)
    if "numeric_sequence" not in dataset.columns:
        print("preprocess: cache miss because cached dataset is not numeric-window format")
        return None
    print(f"preprocess: loaded cached model_dataset rows={len(dataset):,}")
    return dataset, metadata


def load_or_build_dataset(config: Config) -> tuple[pd.DataFrame, dict[str, object]]:
    cached = load_cached_dataset(config)
    if cached is not None:
        return cached
    dataset, metadata = build_dataset(config)
    write_dataset_artifacts(dataset, metadata)
    return dataset, metadata


def event_window_filter(frame: pd.DataFrame, window_index: int, config: Config) -> pd.DataFrame:
    start_minute = window_index * config.window_minutes
    end_minute = min((window_index + 1) * config.window_minutes, config.cutoff_minute)
    lower_mask = frame["minute"] >= start_minute if window_index == 0 else frame["minute"] > start_minute
    return frame[lower_mask & (frame["minute"] <= end_minute)]


def add_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["score_diff"] = frame["home_goals"] - frame["away_goals"]
    frame["is_home_leading"] = (frame["score_diff"] > 0).astype(float)
    frame["is_draw"] = (frame["score_diff"] == 0).astype(float)
    frame["is_away_leading"] = (frame["score_diff"] < 0).astype(float)
    frame["play_diff"] = frame["home_play_count"] - frame["away_play_count"]
    frame["key_event_diff"] = frame["home_key_event_count"] - frame["away_key_event_count"]
    total_home_events = frame["home_play_count"] + frame["home_key_event_count"]
    total_away_events = frame["away_play_count"] + frame["away_key_event_count"]
    total_events = total_home_events + total_away_events
    frame["home_event_share"] = np.divide(
        total_home_events,
        total_events,
        out=np.full(len(frame), 0.5, dtype=float),
        where=total_events.to_numpy(dtype=float) > 0,
    )
    return frame


def window_feature_frame(
    fixtures: pd.DataFrame,
    plays: pd.DataFrame,
    key_events: pd.DataFrame,
    commentary: pd.DataFrame,
    play_types: list[int],
    key_types: list[int],
    count_features: list[str],
    window_index: int,
    config: Config,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        0.0,
        index=pd.Index(fixtures["eventId"].astype(int), name="eventId"),
        columns=[*count_features, "mean_field_x", "mean_field_y"],
    )
    window_plays = event_window_filter(plays, window_index, config) if not plays.empty else plays
    window_key_events = event_window_filter(key_events, window_index, config) if not key_events.empty else key_events
    window_commentary = event_window_filter(commentary, window_index, config) if not commentary.empty else commentary
    add_event_aggregates(frame, window_plays, fixtures, prefix="play", type_column="typeId", type_values=play_types)
    add_event_aggregates(
        frame, window_key_events, fixtures, prefix="key_event", type_column="keyEventTypeId", type_values=key_types
    )
    if not window_commentary.empty:
        frame.loc[:, "commentary_count"] = frame["commentary_count"].add(
            window_commentary.groupby("eventId").size(), fill_value=0
        )
    position_features = build_position_features(window_plays, window_key_events)
    for column in position_features.columns:
        if column in frame:
            frame.loc[:, column] = frame[column].add(position_features[column], fill_value=0)
    return add_state_features(frame.fillna(0.0))


def prefixed_frame(frame: pd.DataFrame, prefix: str, columns: list[str]) -> pd.DataFrame:
    return frame[columns].rename(columns={column: f"{prefix}_{column}" for column in columns})


def build_dataset(config: Config) -> tuple[pd.DataFrame, dict[str, object]]:
    print("preprocess: building fixtures with league-aware chronological splits")
    fixtures = assign_league_chronological_splits(build_fixture_table(config))
    event_ids = set(fixtures["eventId"].astype(int))
    print(f"preprocess: fixtures={len(fixtures):,} splits={fixtures['split'].value_counts().to_dict()}")

    print("preprocess: loading first-60-minute plays, key events, commentary, and safe lineups")
    plays = prepare_plays(event_ids, config.cutoff_minute)
    key_events = prepare_key_events(event_ids, config.cutoff_minute)
    commentary = prepare_commentary(event_ids, config.cutoff_minute)
    lineups = prepare_lineups(event_ids)
    print(
        "preprocess: rows "
        f"plays={len(plays):,} key_events={len(key_events):,} commentary={len(commentary):,} lineups={len(lineups):,}"
    )

    train_event_ids = set(fixtures.loc[fixtures["split"] == "train", "eventId"])
    play_types = top_values(plays[plays["eventId"].isin(train_event_ids)]["typeId"], config.top_play_types)
    key_types = top_values(
        key_events[key_events["eventId"].isin(train_event_ids)]["keyEventTypeId"], config.top_key_event_types
    )
    formations = (
        lineups[lineups["eventId"].isin(train_event_ids)]["formation"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(config.top_formations)
        .index.tolist()
        if not lineups.empty
        else []
    )
    count_features = [
        "play_count",
        "key_event_count",
        "commentary_count",
        "home_play_count",
        "away_play_count",
        "home_key_event_count",
        "away_key_event_count",
        "home_goals",
        "away_goals",
        "scoring_play_count",
        *[f"play_type_{value}" for value in play_types],
        *[f"key_event_type_{value}" for value in key_types],
    ]
    state_features = [
        *count_features,
        "score_diff",
        "is_home_leading",
        "is_draw",
        "is_away_leading",
        "play_diff",
        "key_event_diff",
        "home_event_share",
    ]
    window_only_features = ["mean_field_x", "mean_field_y"]
    team_strength_frame = build_team_strength_features(fixtures, config)
    team_strength_features = team_strength_frame.columns.tolist()
    static_features = [
        "home_starter_count",
        "away_starter_count",
        *[f"home_formation_{formation}" for formation in formations],
        *[f"away_formation_{formation}" for formation in formations],
        *team_strength_features,
    ]
    numeric_features = [
        *[f"window_{column}" for column in [*state_features, *window_only_features]],
        *[f"cumulative_{column}" for column in state_features],
        *static_features,
    ]
    print("preprocess: building 5-minute numeric windows with vectorized groupby operations")
    lineup_frame = build_lineup_feature_frame(lineups, formations)
    static_frame = pd.DataFrame(
        0.0, index=pd.Index(fixtures["eventId"].astype(int), name="eventId"), columns=static_features
    )
    for column in lineup_frame.columns:
        if column in static_frame:
            static_frame.loc[:, column] = static_frame[column].add(lineup_frame[column], fill_value=0)
    for column in team_strength_frame.columns:
        if column in static_frame:
            static_frame.loc[:, column] = static_frame[column].add(team_strength_frame[column], fill_value=0)
    static_frame = static_frame.fillna(0.0)

    cumulative_counts = pd.DataFrame(
        0.0, index=pd.Index(fixtures["eventId"].astype(int), name="eventId"), columns=count_features
    )
    sequence_frames = []
    for index in range(window_count(config)):
        current = window_feature_frame(
            fixtures, plays, key_events, commentary, play_types, key_types, count_features, index, config
        )
        cumulative_counts = cumulative_counts.add(current[count_features], fill_value=0.0)
        cumulative = add_state_features(cumulative_counts.fillna(0.0))
        combined = pd.concat(
            [
                prefixed_frame(current, "window", [*state_features, *window_only_features]),
                prefixed_frame(cumulative, "cumulative", state_features),
                static_frame,
            ],
            axis=1,
        )
        sequence_frames.append(combined[numeric_features].fillna(0.0))
    numeric_sequence = np.stack([frame.to_numpy(dtype=np.float32) for frame in sequence_frames], axis=1)

    dataset = fixtures[
        [
            "eventId",
            "date",
            "midsizeName",
            "league_key",
            "homeTeamId",
            "awayTeamId",
            "homeTeamScore",
            "awayTeamScore",
            "target_label",
            "target",
            "split",
        ]
    ].copy()
    dataset = dataset.rename(columns={"midsizeName": "league"})
    dataset["numeric_sequence"] = numeric_sequence.tolist()
    metadata = {
        "target_labels": list(LABELS),
        "cutoff_minute": config.cutoff_minute,
        "window_minutes": config.window_minutes,
        "window_count": window_count(config),
        "model": "numeric_window_tcn",
        "split_strategy": "chronological_within_each_league_key",
        "numeric_feature_columns": numeric_features,
        "play_type_ids": play_types,
        "key_event_type_ids": key_types,
        "formation_values": formations,
        "team_strength_feature_columns": team_strength_features,
        "preprocess_cache_key": preprocess_cache_key(config),
        "config": asdict(config),
    }
    return dataset, metadata


def write_dataset_artifacts(dataset: pd.DataFrame, metadata: dict[str, object]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(PROCESSED_DIR / "model_dataset.parquet", index=False)
    (PROCESSED_DIR / "feature_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    split_summary = dataset.groupby(["league", "split"]).size().unstack(fill_value=0).sort_index()
    split_summary.to_csv(PROCESSED_DIR / "league_split_summary.csv")
    class_summary = dataset.groupby("split")["target_label"].value_counts().unstack(fill_value=0)
    class_summary.to_csv(PROCESSED_DIR / "split_class_summary.csv")
    print(
        f"preprocess: wrote model_dataset rows={len(dataset):,} features={len(metadata['numeric_feature_columns']):,}"
    )


class MatchDataset(Dataset):
    def __init__(self, numeric: np.ndarray, target: np.ndarray) -> None:
        self.numeric = torch.as_tensor(numeric.copy(), dtype=torch.float32)
        self.target = torch.as_tensor(target.copy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"numeric": self.numeric[index], "target": self.target[index]}


class TemporalConvBlock(nn.Module):
    def __init__(self, channel_count: int, kernel_size: int, dropout: float, normalization_groups: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        group_count = normalization_groups if channel_count % normalization_groups == 0 else 1
        self.block = nn.Sequential(
            nn.Conv1d(channel_count, channel_count, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(group_count, channel_count),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channel_count, channel_count, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(group_count, channel_count),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class NumericWindowTCN(nn.Module):
    def __init__(self, numeric_input_size: int, config: Config) -> None:
        super().__init__()
        self.window_projection = nn.Sequential(
            nn.Linear(numeric_input_size, config.numeric_projection_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.channel_projection = nn.Conv1d(
            config.numeric_projection_size, config.temporal_channel_count, kernel_size=1
        )
        self.temporal_blocks = nn.Sequential(
            *[
                TemporalConvBlock(
                    config.temporal_channel_count,
                    config.temporal_kernel_size,
                    config.dropout,
                    config.normalization_groups,
                )
                for _ in range(config.temporal_block_count)
            ]
        )
        pooled_size = config.temporal_channel_count * 3
        self.head = nn.Sequential(
            nn.Linear(pooled_size, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_size, len(LABELS)),
        )

    def forward(self, numeric_sequence: torch.Tensor) -> torch.Tensor:
        projected = self.window_projection(numeric_sequence).transpose(1, 2)
        encoded = self.temporal_blocks(self.channel_projection(projected))
        max_pool = torch.amax(encoded, dim=2)
        mean_pool = torch.mean(encoded, dim=2)
        last_state = encoded[:, :, -1]
        return self.head(torch.cat([max_pool, mean_pool, last_state], dim=1))


class FlattenedNumericMLP(nn.Module):
    def __init__(self, window_count_value: int, numeric_input_size: int, config: Config) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_count_value * numeric_input_size, config.mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_size, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_size, len(LABELS)),
        )

    def forward(self, numeric_sequence: torch.Tensor) -> torch.Tensor:
        return self.head(numeric_sequence)


def build_model(window_count_value: int, numeric_input_size: int, config: Config) -> nn.Module:
    if config.model_type == "tcn":
        return NumericWindowTCN(numeric_input_size, config)
    if config.model_type == "mlp":
        return FlattenedNumericMLP(window_count_value, numeric_input_size, config)
    raise ValueError("MODEL_TYPE must be 'tcn' or 'mlp'")


def model_artifact_name(config: Config) -> str:
    return "numeric_tcn" if config.model_type == "tcn" else "flattened_mlp"


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def arrays_from_dataset(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    numeric = np.stack(
        dataset["numeric_sequence"].map(
            lambda values: np.stack([np.asarray(window, dtype=np.float32) for window in values])
        )
    ).astype(np.float32)
    target = dataset["target"].to_numpy(dtype=np.int64)
    split_indices = {
        split: np.flatnonzero(dataset["split"].to_numpy(dtype=str) == split).astype(np.int64)
        for split in ("train", "validation", "test")
    }
    train_numeric = numeric[split_indices["train"]].reshape(-1, numeric.shape[2])
    mean = train_numeric.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_numeric.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    numeric = ((numeric - mean) / std).astype(np.float32)
    return numeric, target, split_indices


def make_loader(
    tensor_dataset: MatchDataset, indices: np.ndarray, config: Config, *, shuffle: bool, device: torch.device
) -> DataLoader:
    workers = 0 if device.type == "cuda" and len(indices) < config.batch_size else config.dataloader_workers
    return DataLoader(
        Subset(tensor_dataset, indices.tolist()),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def class_weight_tensor(
    target: np.ndarray, train_indices: np.ndarray, device: torch.device, config: Config
) -> torch.Tensor | None:
    if not config.use_class_weights:
        return None
    counts = np.bincount(target[train_indices], minlength=len(LABELS)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts * len(LABELS), 1.0)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            numeric = batch["numeric"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(numeric)
                loss = criterion(logits, target)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(target)
            correct += int((logits.argmax(dim=1) == target).sum().detach().cpu())
            total += len(target)
    return total_loss / max(total, 1), correct / max(total, 1)


def predict(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    targets = []
    with torch.inference_mode():
        for batch in loader:
            numeric = batch["numeric"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(numeric)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(batch["target"].numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def save_epoch_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    metadata: dict[str, object],
    config: Config,
    history: list[dict[str, float | int]],
) -> None:
    if config.checkpoint_interval_epochs <= 0 or epoch % config.checkpoint_interval_epochs != 0:
        return
    checkpoint_path = MODELS_DIR / f"{model_artifact_name(config)}_epoch_{epoch:05d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metadata": metadata,
            "config": asdict(config),
            "history": history,
        },
        checkpoint_path,
    )
    write_training_history(pd.DataFrame(history))
    print(f"train: wrote checkpoint {checkpoint_path.relative_to(ROOT)}")


def train_model(dataset: pd.DataFrame, metadata: dict[str, object], config: Config) -> pd.DataFrame:
    set_seed(config.seed)
    device = select_device(config.device)
    numeric, target, split_indices = arrays_from_dataset(dataset)
    tensor_dataset = MatchDataset(numeric, target)
    loaders = {
        split: make_loader(tensor_dataset, indices, config, shuffle=(split == "train"), device=device)
        for split, indices in split_indices.items()
        if len(indices) > 0
    }
    model = build_model(numeric.shape[1], numeric.shape[2], config).to(device)
    if config.compile_model:
        model = torch.compile(model)
    class_weights = class_weight_tensor(target, split_indices["train"], device, config)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    split_counts = {split: int(len(indices)) for split, indices in split_indices.items()}
    print(
        "train: start "
        f"model={config.model_type} device={device} rows={len(dataset):,} "
        f"windows={numeric.shape[1]} features={numeric.shape[2]} "
        f"splits={split_counts}"
    )
    if class_weights is not None:
        print(f"train: class_weights={class_weights.detach().cpu().numpy().round(4).tolist()}")

    best_state = None
    best_validation_loss = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer, scaler, use_amp=use_amp
        )
        validation_loss, validation_accuracy = run_epoch(
            model, loaders["validation"], criterion, device, None, None, use_amp=use_amp
        )
        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        improved = validation_loss < best_validation_loss - config.min_delta
        if improved:
            best_validation_loss = validation_loss
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": current_lr,
                "stale_epochs": stale_epochs,
            }
        )
        print(
            f"train: epoch={epoch:04d} train_loss={train_loss:.6f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.6f} val_acc={validation_accuracy:.4f} lr={current_lr:.2e} stale={stale_epochs}"
        )
        save_epoch_checkpoint(epoch, model, optimizer, scheduler, metadata, config, history)
        if stale_epochs >= config.patience:
            print(f"train: early stopping at epoch={epoch} best_validation_loss={best_validation_loss:.6f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {"model_state": model.state_dict(), "metadata": metadata, "config": asdict(config)},
        MODELS_DIR / f"{model_artifact_name(config)}.pt",
    )
    write_training_history(pd.DataFrame(history))

    prediction_frames = []
    for split, indices in split_indices.items():
        if len(indices) == 0:
            continue
        loader = make_loader(tensor_dataset, indices, config, shuffle=False, device=device)
        probs, actual = predict(model, loader, device, use_amp=use_amp)
        split_rows = dataset.iloc[split_indices[split]].reset_index(drop=True).copy()
        split_rows["actual"] = actual
        split_rows["prediction"] = probs.argmax(axis=1)
        for index, label in enumerate(LABELS):
            split_rows[f"prob_{label}"] = probs[:, index]
        split_rows["confidence"] = probs.max(axis=1)
        split_rows["actual_label"] = split_rows["actual"].map(ID_TO_LABEL)
        split_rows["prediction_label"] = split_rows["prediction"].map(ID_TO_LABEL)
        prediction_frames.append(split_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    output_columns = [
        "split",
        "date",
        "eventId",
        "league",
        "actual",
        "actual_label",
        "prediction",
        "prediction_label",
        "prob_home",
        "prob_draw",
        "prob_away",
        "confidence",
    ]
    predictions[output_columns].to_csv(PREDICTIONS_DIR / "predictions.csv", index=False)
    predictions[output_columns].to_parquet(PREDICTIONS_DIR / "predictions.parquet", index=False)
    return predictions[output_columns]


def write_training_history(history: pd.DataFrame) -> None:
    history.to_csv(REPORTS_DIR / "training_history.csv", index=False)
    lines = [
        "# Training History",
        "",
        "| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy | LR | Stale |",
        "| ----- | ---------- | -------------- | --------------- | ------------------- | -- | ----- |",
    ]
    for row in history.itertuples(index=False):
        lines.append(
            f"| {row.epoch} | {row.train_loss:.4f} | {row.train_accuracy:.4f} | "
            f"{row.validation_loss:.4f} | {row.validation_accuracy:.4f} | "
            f"{row.learning_rate:.2e} | {row.stale_epochs} |"
        )
    (REPORTS_DIR / "training_history.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not history.empty:
        plt.figure(figsize=(7, 4.5))
        plt.plot(history["epoch"], history["train_loss"], label="train")
        plt.plot(history["epoch"], history["validation_loss"], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        save_figure(FIGURES_DIR / "training_loss.png")


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


def save_figure(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def evaluate(predictions: pd.DataFrame, model_name: str) -> None:
    metrics = pd.DataFrame(
        [
            {"model_name": model_name, "split": split, **classification_metrics(group)}
            for split, group in predictions.groupby("split", sort=True)
        ]
    )
    metrics.to_json(REPORTS_DIR / "metrics.json", orient="records", indent=2)
    metric_lines = [
        "# Evaluation Metrics",
        "",
        "| Split | Rows | Accuracy | Macro F1 | Log Loss | Mean Confidence |",
        "| ----- | ---- | -------- | -------- | -------- | --------------- |",
    ]
    for row in metrics.itertuples(index=False):
        metric_lines.append(
            f"| {row.split} | {row.row_count} | {row.accuracy:.4f} | {row.macro_f1:.4f} | "
            f"{row.log_loss:.4f} | {row.mean_confidence:.4f} |"
        )
    (REPORTS_DIR / "metrics.md").write_text("\n".join(metric_lines) + "\n", encoding="utf-8")

    class_lines = ["# Per-Class Report", ""]
    for split, group in predictions.groupby("split", sort=True):
        precision, recall, f1, support = precision_recall_fscore_support(
            group["actual"], group["prediction"], labels=[0, 1, 2], zero_division=0
        )
        class_lines.extend(
            [
                f"## {split.title()}",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "| ----- | --------- | ------ | -- | ------- |",
            ]
        )
        for index, label in enumerate(LABELS):
            class_lines.append(
                f"| {label} | {precision[index]:.4f} | {recall[index]:.4f} | {f1[index]:.4f} | {support[index]} |"
            )
        class_lines.append("")
    (REPORTS_DIR / "class_report.md").write_text("\n".join(class_lines), encoding="utf-8")

    errors = (
        predictions[predictions["actual"] != predictions["prediction"]]
        .sort_values("confidence", ascending=False)
        .head(25)
    )
    error_lines = [
        "# High-Confidence Errors",
        "",
        "| Split | Date | Event | League | Actual | Predicted | Confidence |",
        "| ----- | ---- | ----- | ------ | ------ | --------- | ---------- |",
    ]
    for row in errors.itertuples(index=False):
        error_lines.append(
            f"| {row.split} | {row.date} | {row.eventId} | {row.league} | {row.actual_label} | "
            f"{row.prediction_label} | {row.confidence:.4f} |"
        )
    (REPORTS_DIR / "error_examples.md").write_text("\n".join(error_lines) + "\n", encoding="utf-8")
    write_evaluation_figures(predictions, metrics)
    print(f"evaluate: wrote metrics rows={len(metrics)} predictions={len(predictions):,}")


def write_evaluation_figures(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    plot_split = "test" if (predictions["split"] == "test").any() else predictions["split"].iloc[-1]
    split_predictions = predictions[predictions["split"] == plot_split]
    matrix = confusion_matrix(split_predictions["actual"], split_predictions["prediction"], labels=[0, 1, 2])
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Confusion Matrix ({plot_split})")
    plt.xticks(range(3), LABELS)
    plt.yticks(range(3), LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    save_figure(FIGURES_DIR / "confusion_matrix.png")

    plt.figure(figsize=(8, 4.5))
    predictions.groupby(["split", "actual_label"]).size().unstack(fill_value=0).reindex(
        columns=LABELS, fill_value=0
    ).plot(kind="bar", ax=plt.gca())
    plt.title("Class Distribution by Split")
    plt.xlabel("Split")
    plt.ylabel("Matches")
    plt.xticks(rotation=0)
    save_figure(FIGURES_DIR / "class_distribution.png")

    plt.figure(figsize=(7, 4.5))
    plt.hist(split_predictions["confidence"], bins=20, edgecolor="black")
    plt.title(f"Prediction Confidence ({plot_split})")
    plt.xlabel("Maximum class probability")
    plt.ylabel("Matches")
    save_figure(FIGURES_DIR / "prediction_confidence.png")

    plt.figure(figsize=(7, 4.5))
    metrics.set_index("split")[["accuracy", "macro_f1"]].plot(kind="bar", ax=plt.gca())
    plt.title("Classification Metrics by Split")
    plt.xlabel("Split")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    save_figure(FIGURES_DIR / "metric_comparison.png")


def clean_old_package_artifacts() -> None:
    for path in (
        PROCESSED_DIR / "train_tensors.pt",
        PROCESSED_DIR / "text_vocabulary.json",
        MODELS_DIR / "fusion_gru.pt",
        MODELS_DIR / "textcnn_mlp.pt",
    ):
        if path.exists():
            path.unlink()


def run_pipeline(config: Config) -> None:
    ensure_dirs()
    clean_old_package_artifacts()
    download_kaggle_dataset()
    dataset, metadata = load_or_build_dataset(config)
    predictions = train_model(dataset, metadata, config)
    evaluate(predictions, model_artifact_name(config))
    print("done: outputs written under output/ and data/processed/")


def main() -> None:
    run_pipeline(Config())


if __name__ == "__main__":
    main()
