"""Preprocess ESPN Soccer raw tables into model-ready in-play artifacts."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import fip.utils

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
COMPLETED_STATUS_IDS = {28}
TOP_PLAY_TYPES = int(os.environ.get("FIP_TOP_PLAY_TYPES", "24"))
TOP_KEY_EVENT_TYPES = int(os.environ.get("FIP_TOP_KEY_EVENT_TYPES", "24"))
TOP_FORMATIONS = int(os.environ.get("FIP_TOP_FORMATIONS", "16"))


@dataclass(frozen=True)
class PreprocessResult:
    """Summary of generated processed artifacts."""

    fixtures_path: Path
    dataset_path: Path
    metadata_path: Path
    vocabulary_path: Path
    split_summary_path: Path
    fixture_rows: int
    model_rows: int
    numeric_feature_count: int
    vocabulary_size: int


def _csv_files(directory: Path, prefix: str | None = None) -> list[Path]:
    files = sorted(directory.glob("*.csv"))
    if prefix is not None:
        files = [path for path in files if path.name.startswith(prefix)]
    return files


def read_csv_group(directory: Path, *, columns: list[str] | None = None, prefix: str | None = None) -> pd.DataFrame:
    """Read a directory of CSV files into one frame, skipping absent requested columns."""

    frames = []
    for path in _csv_files(directory, prefix):
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
    """Parse ESPN clock display values such as `18'`, `45'+2'`, or seconds."""

    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return numeric / 60.0 if numeric > 120 else numeric
    text = str(value).strip()
    if not text:
        return None
    stoppage = re.match(r"^(?P<minute>\d+)\s*'\s*\+\s*(?P<extra>\d+)", text)
    if stoppage:
        return float(stoppage.group("minute")) + float(stoppage.group("extra"))
    minute = re.search(r"(?P<minute>\d+)", text)
    if minute:
        return float(minute.group("minute"))
    return None


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def build_vocabulary(text_windows: pd.Series, max_size: int = fip.utils.MAX_VOCAB_SIZE) -> dict[str, int]:
    """Build a train-split vocabulary from nested text windows."""

    counter: Counter[str] = Counter()
    for windows in text_windows:
        for text in windows:
            counter.update(tokenize(str(text)))
    vocabulary = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, _count in counter.most_common(max(max_size - len(vocabulary), 0)):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text_windows(
    text_windows: list[str], vocabulary: dict[str, int], max_tokens: int = fip.utils.MAX_TOKENS_PER_WINDOW
) -> list[list[int]]:
    encoded = []
    for text in text_windows:
        ids = [vocabulary.get(token, UNK_ID) for token in tokenize(str(text))[:max_tokens]]
        ids.extend([PAD_ID] * (max_tokens - len(ids)))
        encoded.append(ids)
    return encoded


def build_fixture_table(raw_data_path: Path) -> pd.DataFrame:
    """Build completed-match labels from fixture and league metadata."""

    fixtures = pd.read_csv(raw_data_path / "base_data" / "fixtures.csv")
    leagues = pd.read_csv(raw_data_path / "base_data" / "leagues.csv")
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures = fixtures[fixtures["statusId"].isin(COMPLETED_STATUS_IDS)].copy()
    fixtures = fixtures.dropna(subset=["date", "homeTeamScore", "awayTeamScore"])
    fixtures["homeTeamScore"] = pd.to_numeric(fixtures["homeTeamScore"], errors="coerce")
    fixtures["awayTeamScore"] = pd.to_numeric(fixtures["awayTeamScore"], errors="coerce")
    fixtures = fixtures.dropna(subset=["homeTeamScore", "awayTeamScore"])
    fixtures["target_label"] = np.select(
        [fixtures["homeTeamScore"] > fixtures["awayTeamScore"], fixtures["homeTeamScore"] < fixtures["awayTeamScore"]],
        ["home", "away"],
        default="draw",
    )
    fixtures["target"] = fixtures["target_label"].map(fip.utils.LABEL_TO_ID).astype(int)
    fixtures = fixtures.merge(
        leagues[["seasonType", "leagueId", "year", "midsizeName", "leagueName"]].drop_duplicates(),
        on=["seasonType", "leagueId"],
        how="left",
    )
    fixtures = fixtures.sort_values(["date", "eventId"]).reset_index(drop=True)
    return fixtures[
        [
            "seasonType",
            "leagueId",
            "year",
            "midsizeName",
            "leagueName",
            "eventId",
            "date",
            "venueId",
            "homeTeamId",
            "awayTeamId",
            "homeTeamScore",
            "awayTeamScore",
            "target_label",
            "target",
        ]
    ]


def assign_chronological_splits(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic chronological train, validation, and test splits."""

    fixtures = fixtures.sort_values(["date", "eventId"]).reset_index(drop=True).copy()
    row_count = len(fixtures)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    fixtures["split"] = "test"
    fixtures.loc[: max(train_end - 1, -1), "split"] = "train"
    fixtures.loc[train_end : max(validation_end - 1, train_end - 1), "split"] = "validation"
    return fixtures


def _window_index(minute: float, cutoff_minute: int, window_minutes: int) -> int | None:
    if minute < 0 or minute > cutoff_minute:
        return None
    index = int(minute // window_minutes)
    return min(index, fip.utils.window_count(cutoff_minute, window_minutes) - 1)


def _prepare_plays(raw_data_path: Path, event_ids: set[int], cutoff_minute: int) -> pd.DataFrame:
    columns = [
        "seasonType",
        "eventId",
        "typeId",
        "text",
        "shortText",
        "period",
        "clockValue",
        "clockDisplayValue",
        "teamId",
        "scoringPlay",
        "goalPositionX",
        "goalPositionY",
        "fieldpositionX",
        "fieldPositionY",
        "fieldPosition2X",
        "fieldPosition2Y",
    ]
    plays = read_csv_group(raw_data_path / "plays_data", columns=columns, prefix="plays_")
    if plays.empty:
        return plays
    plays = plays[plays["eventId"].isin(event_ids)].copy()
    plays["minute"] = plays["clockValue"].map(clock_to_minute)
    missing_minute = plays["minute"].isna()
    if missing_minute.any():
        plays.loc[missing_minute, "minute"] = plays.loc[missing_minute, "clockDisplayValue"].map(clock_to_minute)
    plays["period"] = pd.to_numeric(plays["period"], errors="coerce").fillna(1)
    plays = plays[(plays["period"] <= 1) & plays["minute"].notna() & (plays["minute"] <= cutoff_minute)].copy()
    plays["text_combined"] = plays[["text", "shortText"]].fillna("").agg(" ".join, axis=1)
    return plays


def _prepare_key_events(raw_data_path: Path, event_ids: set[int], cutoff_minute: int) -> pd.DataFrame:
    columns = [
        "seasonType",
        "eventId",
        "keyEventTypeId",
        "period",
        "clockValue",
        "clockDisplayValue",
        "scoringPlay",
        "keyEventText",
        "keyEventShortText",
        "teamId",
        "goalPositionX",
        "goalPositionY",
        "fieldPositionX",
        "fieldPositionY",
        "fieldPosition2X",
        "fieldPosition2Y",
    ]
    events = read_csv_group(raw_data_path / "keyEvents_data", columns=columns, prefix="keyEvents_")
    if events.empty:
        return events
    events = events[events["eventId"].isin(event_ids)].copy()
    events["minute"] = events["clockValue"].map(clock_to_minute)
    missing_minute = events["minute"].isna()
    if missing_minute.any():
        events.loc[missing_minute, "minute"] = events.loc[missing_minute, "clockDisplayValue"].map(clock_to_minute)
    events["period"] = pd.to_numeric(events["period"], errors="coerce").fillna(1)
    events = events[(events["period"] <= 1) & events["minute"].notna() & (events["minute"] <= cutoff_minute)].copy()
    events["text_combined"] = events[["keyEventText", "keyEventShortText"]].fillna("").agg(" ".join, axis=1)
    return events


def _prepare_commentary(raw_data_path: Path, event_ids: set[int], cutoff_minute: int) -> pd.DataFrame:
    columns = ["seasonType", "eventId", "clockDisplayValue", "commentaryText"]
    commentary = read_csv_group(raw_data_path / "commentary_data", columns=columns, prefix="commentary_")
    if commentary.empty:
        return commentary
    commentary = commentary[commentary["eventId"].isin(event_ids)].copy()
    commentary["minute"] = commentary["clockDisplayValue"].map(clock_to_minute).fillna(0.0)
    commentary = commentary[commentary["minute"] <= cutoff_minute].copy()
    commentary["text_combined"] = commentary["commentaryText"].fillna("")
    return commentary


def _prepare_lineups(raw_data_path: Path, event_ids: set[int]) -> pd.DataFrame:
    columns = ["eventId", "teamId", "homeAway", "formation", "starter", "position", "formationPlace"]
    lineups = read_csv_group(raw_data_path / "lineup_data", columns=columns, prefix="lineup_")
    if lineups.empty:
        return lineups
    return lineups[lineups["eventId"].isin(event_ids)].copy()


def _top_values(series: pd.Series, limit: int) -> list[int]:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    return values.value_counts().head(limit).index.tolist()


def _formation_values(lineups: pd.DataFrame, fixtures: pd.DataFrame) -> list[str]:
    if lineups.empty or "formation" not in lineups.columns:
        return []
    train_event_ids = set(fixtures.loc[fixtures["split"] == "train", "eventId"])
    return (
        lineups[lineups["eventId"].isin(train_event_ids)]["formation"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(TOP_FORMATIONS)
        .index.tolist()
    )


def _empty_text_windows(window_count: int) -> list[str]:
    return [""] * window_count


def _lineup_features(lineups: pd.DataFrame, formations: list[str]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = defaultdict(dict)
    if lineups.empty:
        return result
    for (event_id, home_away), group in lineups.groupby(["eventId", "homeAway"], dropna=True):
        prefix = "home" if str(home_away) == "home" else "away"
        result[int(event_id)][f"{prefix}_starter_count"] = float(
            pd.to_numeric(group["starter"], errors="coerce").fillna(0).sum()
        )
        for formation in formations:
            result[int(event_id)][f"{prefix}_formation_{formation}"] = float(
                (group["formation"].astype(str) == formation).any()
            )
    return result


def build_model_dataset(
    fixtures: pd.DataFrame,
    plays: pd.DataFrame,
    key_events: pd.DataFrame,
    commentary: pd.DataFrame,
    lineups: pd.DataFrame,
    *,
    cutoff_minute: int = fip.utils.CUTOFF_MINUTE,
    window_minutes: int = fip.utils.WINDOW_MINUTES,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build one row per match with text and numeric sequences through the cutoff minute."""

    count = fip.utils.window_count(cutoff_minute, window_minutes)
    train_event_ids = set(fixtures.loc[fixtures["split"] == "train", "eventId"])
    play_types = (
        _top_values(plays[plays["eventId"].isin(train_event_ids)]["typeId"], TOP_PLAY_TYPES) if not plays.empty else []
    )
    key_types = (
        _top_values(key_events[key_events["eventId"].isin(train_event_ids)]["keyEventTypeId"], TOP_KEY_EVENT_TYPES)
        if not key_events.empty
        else []
    )
    formations = _formation_values(lineups, fixtures)
    lineup_features = _lineup_features(lineups, formations)
    numeric_feature_names = [
        "play_count",
        "key_event_count",
        "commentary_count",
        "home_play_count",
        "away_play_count",
        "home_key_event_count",
        "away_key_event_count",
        "home_goals",
        "away_goals",
        "score_diff",
        "scoring_play_count",
        "mean_field_x",
        "mean_field_y",
        *[f"play_type_{value}" for value in play_types],
        *[f"key_event_type_{value}" for value in key_types],
        "home_starter_count",
        "away_starter_count",
        *[f"home_formation_{formation}" for formation in formations],
        *[f"away_formation_{formation}" for formation in formations],
    ]

    play_groups = {int(event_id): group for event_id, group in plays.groupby("eventId")} if not plays.empty else {}
    key_groups = (
        {int(event_id): group for event_id, group in key_events.groupby("eventId")} if not key_events.empty else {}
    )
    commentary_groups = (
        {int(event_id): group for event_id, group in commentary.groupby("eventId")} if not commentary.empty else {}
    )
    rows = []
    for fixture in fixtures.itertuples(index=False):
        event_id = int(fixture.eventId)
        text_windows = _empty_text_windows(count)
        numeric_windows = [{name: 0.0 for name in numeric_feature_names} for _ in range(count)]
        home_team_id = int(fixture.homeTeamId)
        away_team_id = int(fixture.awayTeamId)

        for frame, type_column, prefix in (
            (play_groups.get(event_id, pd.DataFrame()), "typeId", "play"),
            (key_groups.get(event_id, pd.DataFrame()), "keyEventTypeId", "key_event"),
        ):
            for event in frame.itertuples(index=False):
                index = _window_index(float(event.minute), cutoff_minute, window_minutes)
                if index is None:
                    continue
                for cumulative_index in range(index, count):
                    numeric_windows[cumulative_index][f"{prefix}_count"] += 1.0
                    team_id = int(event.teamId) if pd.notna(event.teamId) else -1
                    if team_id == home_team_id:
                        numeric_windows[cumulative_index][f"home_{prefix}_count"] += 1.0
                    elif team_id == away_team_id:
                        numeric_windows[cumulative_index][f"away_{prefix}_count"] += 1.0
                    if float(getattr(event, "scoringPlay", 0) or 0) == 1.0:
                        numeric_windows[cumulative_index]["scoring_play_count"] += 1.0
                        if team_id == home_team_id:
                            numeric_windows[cumulative_index]["home_goals"] += 1.0
                        elif team_id == away_team_id:
                            numeric_windows[cumulative_index]["away_goals"] += 1.0
                    event_type = getattr(event, type_column)
                    if pd.notna(event_type):
                        feature = f"{prefix}_type_{int(event_type)}"
                        if feature in numeric_windows[cumulative_index]:
                            numeric_windows[cumulative_index][feature] += 1.0
                text = str(getattr(event, "text_combined", "") or "")
                if text:
                    text_windows[index] = f"{text_windows[index]} {text}".strip()

        for event in commentary_groups.get(event_id, pd.DataFrame()).itertuples(index=False):
            index = _window_index(float(event.minute), cutoff_minute, window_minutes)
            if index is None:
                continue
            text_windows[index] = f"{text_windows[index]} {event.text_combined}".strip()
            for cumulative_index in range(index, count):
                numeric_windows[cumulative_index]["commentary_count"] += 1.0

        combined_positions = []
        for frame in (play_groups.get(event_id, pd.DataFrame()), key_groups.get(event_id, pd.DataFrame())):
            if frame.empty:
                continue
            x_column = "fieldpositionX" if "fieldpositionX" in frame.columns else "fieldPositionX"
            y_column = "fieldPositionY"
            positions = frame[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").replace(0, np.nan).dropna()
            if not positions.empty:
                combined_positions.append(positions.rename(columns={x_column: "x", y_column: "y"}))
        if combined_positions:
            positions = pd.concat(combined_positions, ignore_index=True)
            for window in numeric_windows:
                window["mean_field_x"] = float(positions["x"].mean())
                window["mean_field_y"] = float(positions["y"].mean())

        static_features = lineup_features.get(event_id, {})
        for window in numeric_windows:
            window.update(static_features)
            window["score_diff"] = window["home_goals"] - window["away_goals"]

        rows.append(
            {
                "eventId": event_id,
                "date": fixture.date,
                "league": fixture.midsizeName,
                "homeTeamId": home_team_id,
                "awayTeamId": away_team_id,
                "homeTeamScore": float(fixture.homeTeamScore),
                "awayTeamScore": float(fixture.awayTeamScore),
                "target_label": fixture.target_label,
                "target": int(fixture.target),
                "split": fixture.split,
                "text_windows": text_windows,
                "numeric_sequence": [
                    [float(window.get(name, 0.0)) for name in numeric_feature_names] for window in numeric_windows
                ],
            }
        )

    dataset = pd.DataFrame(rows)
    vocabulary = build_vocabulary(dataset.loc[dataset["split"] == "train", "text_windows"])
    dataset["token_windows"] = dataset["text_windows"].map(lambda windows: encode_text_windows(windows, vocabulary))
    metadata = {
        "target_labels": list(fip.utils.LABELS),
        "cutoff_minute": cutoff_minute,
        "window_minutes": window_minutes,
        "window_count": count,
        "numeric_feature_columns": numeric_feature_names,
        "play_type_ids": play_types,
        "key_event_type_ids": key_types,
        "formation_values": formations,
        "vocabulary_size": len(vocabulary),
        "max_tokens_per_window": fip.utils.MAX_TOKENS_PER_WINDOW,
    }
    metadata["vocabulary"] = vocabulary
    return dataset, metadata


def preprocess_raw_sources(paths: fip.utils.ProjectPaths = fip.utils.DEFAULT_PATHS) -> PreprocessResult:
    fip.utils.ensure_generated_directories(paths)
    raw_relative = paths.raw_data.relative_to(paths.root)
    processed_relative = paths.processed_data.relative_to(paths.root)
    print(f"preprocess: raw_dir={raw_relative} processed_dir={processed_relative}")
    fixtures = assign_chronological_splits(build_fixture_table(paths.raw_data))
    event_ids = set(fixtures["eventId"].astype(int))
    print(f"preprocess: completed fixtures={len(fixtures)}")
    plays = _prepare_plays(paths.raw_data, event_ids, fip.utils.CUTOFF_MINUTE)
    key_events = _prepare_key_events(paths.raw_data, event_ids, fip.utils.CUTOFF_MINUTE)
    commentary = _prepare_commentary(paths.raw_data, event_ids, fip.utils.CUTOFF_MINUTE)
    lineups = _prepare_lineups(paths.raw_data, event_ids)
    print(
        "preprocess: rows "
        f"plays={len(plays)} key_events={len(key_events)} commentary={len(commentary)} lineups={len(lineups)}"
    )
    dataset, metadata = build_model_dataset(fixtures, plays, key_events, commentary, lineups)
    vocabulary = metadata.pop("vocabulary")

    fixtures_path = paths.processed_data / "fixtures.parquet"
    dataset_path = paths.processed_data / "model_dataset.parquet"
    metadata_path = paths.processed_data / "feature_metadata.json"
    vocabulary_path = paths.processed_data / "text_vocabulary.json"
    split_summary_path = paths.processed_data / "split_summary.json"
    fixtures.to_parquet(fixtures_path, index=False)
    dataset.to_parquet(dataset_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    vocabulary_path.write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    split_summary = (
        dataset.groupby("split")["target_label"].value_counts().unstack(fill_value=0).to_dict(orient="index")
    )
    split_summary_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    print(
        f"preprocess: wrote dataset rows={len(dataset)} numeric_features={len(metadata['numeric_feature_columns'])} "
        f"vocabulary={len(vocabulary)}"
    )
    return PreprocessResult(
        fixtures_path,
        dataset_path,
        metadata_path,
        vocabulary_path,
        split_summary_path,
        len(fixtures),
        len(dataset),
        len(metadata["numeric_feature_columns"]),
        len(vocabulary),
    )


def main() -> None:
    preprocess_raw_sources()


if __name__ == "__main__":
    main()
