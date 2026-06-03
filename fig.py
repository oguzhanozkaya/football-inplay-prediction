"""Single-script ESPN Soccer first-half outcome prediction pipeline.

Run locally with `uv run python fig.py` or in Colab after installing the dependencies.
The script downloads the Kaggle dataset when `data/raw/` does not already contain it,
builds leakage-safe first-half features, trains one TextCNN + numeric MLP classifier,
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
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

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

KAGGLE_DATASET = os.environ.get("FIP_KAGGLE_DATASET", "excel4soccer/espn-soccer-data")
LABELS = ("home", "draw", "away")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
COMPLETED_STATUS_IDS = {28}
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class Config:
    seed: int = int(os.environ.get("FIP_SEED", "67"))
    epochs: int = int(os.environ.get("FIP_EPOCHS", "1200"))
    patience: int = int(os.environ.get("FIP_PATIENCE", "50"))
    batch_size: int = int(os.environ.get("FIP_BATCH_SIZE", "512"))
    learning_rate: float = float(os.environ.get("FIP_LEARNING_RATE", "0.0001"))
    weight_decay: float = float(os.environ.get("FIP_WEIGHT_DECAY", "0.0001"))
    min_delta: float = float(os.environ.get("FIP_EARLY_STOPPING_MIN_DELTA", "0.001"))
    device: str = os.environ.get("FIP_DEVICE", "cuda")
    cutoff_minute: int = int(os.environ.get("FIP_CUTOFF_MINUTE", "45"))
    max_tokens: int = int(os.environ.get("FIP_MAX_TOKENS", "256"))
    max_vocab_size: int = int(os.environ.get("FIP_MAX_VOCAB_SIZE", "6000"))
    top_play_types: int = int(os.environ.get("FIP_TOP_PLAY_TYPES", "30"))
    top_key_event_types: int = int(os.environ.get("FIP_TOP_KEY_EVENT_TYPES", "24"))
    top_formations: int = int(os.environ.get("FIP_TOP_FORMATIONS", "12"))
    text_embedding_dim: int = int(os.environ.get("FIP_TEXT_EMBEDDING_DIM", "64"))
    text_channel_count: int = int(os.environ.get("FIP_TEXT_CHANNEL_COUNT", "48"))
    text_dropout: float = float(os.environ.get("FIP_TEXT_DROPOUT", "0.25"))
    numeric_hidden_size: int = int(os.environ.get("FIP_NUMERIC_HIDDEN_SIZE", "128"))
    fusion_hidden_size: int = int(os.environ.get("FIP_FUSION_HIDDEN_SIZE", "128"))
    dropout: float = float(os.environ.get("FIP_DROPOUT", "0.25"))
    dataloader_workers: int = int(os.environ.get("FIP_DATALOADER_WORKERS", "4"))
    mixed_precision: bool = os.environ.get("FIP_MIXED_PRECISION", "true").lower() in {"1", "true", "yes", "on"}
    compile_model: bool = os.environ.get("FIP_COMPILE_MODEL", "false").lower() in {"1", "true", "yes", "on"}
    match_limit: int = int(os.environ.get("FIP_MATCH_LIMIT", "0"))

    @property
    def text_kernel_sizes(self) -> tuple[int, ...]:
        raw = os.environ.get("FIP_TEXT_KERNEL_SIZES", "3,4,5")
        return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


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


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


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
    plays["period"] = pd.to_numeric(plays["period"], errors="coerce").fillna(1)
    plays = plays[(plays["period"] <= 1) & plays["minute"].notna() & (plays["minute"] <= cutoff_minute)].copy()
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
    events["period"] = pd.to_numeric(events["period"], errors="coerce").fillna(1)
    events = events[(events["period"] <= 1) & events["minute"].notna() & (events["minute"] <= cutoff_minute)].copy()
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


def lineup_features(lineups: pd.DataFrame, formations: list[str]) -> dict[int, dict[str, float]]:
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


def build_vocabulary(texts: pd.Series, max_size: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(str(text)))
    vocabulary = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, _count in counter.most_common(max(max_size - len(vocabulary), 0)):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int], max_tokens: int) -> list[int]:
    token_ids = [vocabulary.get(token, UNK_ID) for token in tokenize(str(text))[:max_tokens]]
    token_ids.extend([PAD_ID] * (max_tokens - len(token_ids)))
    return token_ids


def build_dataset(config: Config) -> tuple[pd.DataFrame, dict[str, object], dict[str, int]]:
    print("preprocess: building fixtures with league-aware chronological splits")
    fixtures = assign_league_chronological_splits(build_fixture_table(config))
    event_ids = set(fixtures["eventId"].astype(int))
    print(f"preprocess: fixtures={len(fixtures):,} splits={fixtures['split'].value_counts().to_dict()}")

    print("preprocess: loading first-half plays, key events, commentary, and safe lineups")
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
    numeric_features = [
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
    lineups_by_event = lineup_features(lineups, formations)

    rows = []
    for fixture in fixtures.itertuples(index=False):
        event_id = int(fixture.eventId)
        home_team_id = int(fixture.homeTeamId)
        away_team_id = int(fixture.awayTeamId)
        features = {name: 0.0 for name in numeric_features}
        texts = []

        for frame, type_column, prefix in (
            (play_groups.get(event_id, pd.DataFrame()), "typeId", "play"),
            (key_groups.get(event_id, pd.DataFrame()), "keyEventTypeId", "key_event"),
        ):
            for event in frame.itertuples(index=False):
                features[f"{prefix}_count"] += 1.0
                team_id = int(event.teamId) if pd.notna(event.teamId) else -1
                if team_id == home_team_id:
                    features[f"home_{prefix}_count"] += 1.0
                elif team_id == away_team_id:
                    features[f"away_{prefix}_count"] += 1.0
                if float(getattr(event, "scoringPlay", 0) or 0) == 1.0:
                    features["scoring_play_count"] += 1.0
                    if team_id == home_team_id:
                        features["home_goals"] += 1.0
                    elif team_id == away_team_id:
                        features["away_goals"] += 1.0
                event_type = getattr(event, type_column)
                if pd.notna(event_type):
                    name = f"{prefix}_type_{int(event_type)}"
                    if name in features:
                        features[name] += 1.0
                text = str(getattr(event, "text_combined", "") or "")
                if text:
                    texts.append(text)

        for event in commentary_groups.get(event_id, pd.DataFrame()).itertuples(index=False):
            features["commentary_count"] += 1.0
            text = str(event.text_combined or "")
            if text:
                texts.append(text)

        positions = []
        for frame in (play_groups.get(event_id, pd.DataFrame()), key_groups.get(event_id, pd.DataFrame())):
            if frame.empty:
                continue
            x_column = "fieldpositionX" if "fieldpositionX" in frame.columns else "fieldPositionX"
            if x_column in frame.columns and "fieldPositionY" in frame.columns:
                selected = (
                    frame[[x_column, "fieldPositionY"]]
                    .apply(pd.to_numeric, errors="coerce")
                    .replace(0, np.nan)
                    .dropna()
                )
                if not selected.empty:
                    positions.append(selected.rename(columns={x_column: "x", "fieldPositionY": "y"}))
        if positions:
            merged = pd.concat(positions, ignore_index=True)
            features["mean_field_x"] = float(merged["x"].mean())
            features["mean_field_y"] = float(merged["y"].mean())
        features.update(lineups_by_event.get(event_id, {}))
        features["score_diff"] = features["home_goals"] - features["away_goals"]
        rows.append(
            {
                "eventId": event_id,
                "date": fixture.date,
                "league": fixture.midsizeName,
                "league_key": fixture.league_key,
                "homeTeamId": home_team_id,
                "awayTeamId": away_team_id,
                "homeTeamScore": float(fixture.homeTeamScore),
                "awayTeamScore": float(fixture.awayTeamScore),
                "target_label": fixture.target_label,
                "target": int(fixture.target),
                "split": fixture.split,
                "text": " ".join(texts),
                "numeric_features": [float(features.get(name, 0.0)) for name in numeric_features],
            }
        )

    dataset = pd.DataFrame(rows)
    vocabulary = build_vocabulary(dataset.loc[dataset["split"] == "train", "text"], config.max_vocab_size)
    dataset["token_ids"] = dataset["text"].map(lambda text: encode_text(text, vocabulary, config.max_tokens))
    metadata = {
        "target_labels": list(LABELS),
        "cutoff_minute": config.cutoff_minute,
        "model": "first_half_textcnn_numeric_mlp",
        "split_strategy": "chronological_within_each_league_key",
        "numeric_feature_columns": numeric_features,
        "play_type_ids": play_types,
        "key_event_type_ids": key_types,
        "formation_values": formations,
        "vocabulary_size": len(vocabulary),
        "max_tokens": config.max_tokens,
        "config": asdict(config),
    }
    return dataset, metadata, vocabulary


def write_dataset_artifacts(dataset: pd.DataFrame, metadata: dict[str, object], vocabulary: dict[str, int]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(PROCESSED_DIR / "model_dataset.parquet", index=False)
    (PROCESSED_DIR / "feature_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (PROCESSED_DIR / "text_vocabulary.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    split_summary = dataset.groupby(["league", "split"]).size().unstack(fill_value=0).sort_index()
    split_summary.to_csv(PROCESSED_DIR / "league_split_summary.csv")
    class_summary = dataset.groupby("split")["target_label"].value_counts().unstack(fill_value=0)
    class_summary.to_csv(PROCESSED_DIR / "split_class_summary.csv")
    print(
        f"preprocess: wrote model_dataset rows={len(dataset):,} features={len(metadata['numeric_feature_columns']):,}"
    )


class MatchDataset(Dataset):
    def __init__(self, numeric: np.ndarray, tokens: np.ndarray, target: np.ndarray) -> None:
        self.numeric = torch.as_tensor(numeric.copy(), dtype=torch.float32)
        self.tokens = torch.as_tensor(tokens.copy(), dtype=torch.long)
        self.target = torch.as_tensor(target.copy(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"numeric": self.numeric[index], "tokens": self.tokens[index], "target": self.target[index]}


class TextCNNEncoder(nn.Module):
    def __init__(self, vocabulary_size: int, config: Config) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config.text_embedding_dim, padding_idx=PAD_ID)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(config.text_embedding_dim, config.text_channel_count, kernel_size=size)
            for size in config.text_kernel_sizes
        )
        self.dropout = nn.Dropout(config.text_dropout)
        self.output_size = config.text_channel_count * len(config.text_kernel_sizes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for convolution in self.convolutions:
            activation = torch.relu(convolution(embedded))
            pooled.append(torch.max(activation, dim=2).values)
        return self.dropout(torch.cat(pooled, dim=1))


class FirstHalfClassifier(nn.Module):
    def __init__(self, numeric_input_size: int, vocabulary_size: int, config: Config) -> None:
        super().__init__()
        self.text_encoder = TextCNNEncoder(vocabulary_size, config)
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_input_size, config.numeric_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.numeric_hidden_size, config.numeric_hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.text_encoder.output_size + config.numeric_hidden_size, config.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_size, len(LABELS)),
        )

    def forward(self, numeric: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.numeric_encoder(numeric), self.text_encoder(tokens)], dim=1))


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


def arrays_from_dataset(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    numeric = np.stack(dataset["numeric_features"].map(lambda values: np.asarray(values, dtype=np.float32))).astype(
        np.float32
    )
    tokens = np.stack(dataset["token_ids"].map(lambda values: np.asarray(values, dtype=np.int64))).astype(np.int64)
    target = dataset["target"].to_numpy(dtype=np.int64)
    split_indices = {
        split: np.flatnonzero(dataset["split"].to_numpy(dtype=str) == split).astype(np.int64)
        for split in ("train", "validation", "test")
    }
    mean = numeric[split_indices["train"]].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = numeric[split_indices["train"]].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    numeric = ((numeric - mean) / std).astype(np.float32)
    return numeric, tokens, target, split_indices


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
            tokens = batch["tokens"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(numeric, tokens)
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
            tokens = batch["tokens"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(numeric, tokens)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(batch["target"].numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def train_model(dataset: pd.DataFrame, metadata: dict[str, object], config: Config) -> pd.DataFrame:
    set_seed(config.seed)
    device = select_device(config.device)
    numeric, tokens, target, split_indices = arrays_from_dataset(dataset)
    tensor_dataset = MatchDataset(numeric, tokens, target)
    loaders = {
        split: make_loader(tensor_dataset, indices, config, shuffle=(split == "train"), device=device)
        for split, indices in split_indices.items()
        if len(indices) > 0
    }
    model = FirstHalfClassifier(numeric.shape[1], int(metadata["vocabulary_size"]), config).to(device)
    if config.compile_model:
        model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    split_counts = {split: int(len(indices)) for split, indices in split_indices.items()}
    print(
        "train: start "
        f"device={device} rows={len(dataset):,} features={numeric.shape[1]} vocab={metadata['vocabulary_size']} "
        f"splits={split_counts}"
    )

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
        if stale_epochs >= config.patience:
            print(f"train: early stopping at epoch={epoch} best_validation_loss={best_validation_loss:.6f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {"model_state": model.state_dict(), "metadata": metadata, "config": asdict(config)},
        MODELS_DIR / "textcnn_mlp.pt",
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


def evaluate(predictions: pd.DataFrame) -> None:
    metrics = pd.DataFrame(
        [
            {"model_name": "textcnn_mlp", "split": split, **classification_metrics(group)}
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
    old_tensor = PROCESSED_DIR / "train_tensors.pt"
    if old_tensor.exists():
        old_tensor.unlink()
    old_model = MODELS_DIR / "fusion_gru.pt"
    if old_model.exists():
        old_model.unlink()


def main() -> None:
    config = Config()
    ensure_dirs()
    clean_old_package_artifacts()
    download_kaggle_dataset()
    dataset, metadata, vocabulary = build_dataset(config)
    write_dataset_artifacts(dataset, metadata, vocabulary)
    predictions = train_model(dataset, metadata, config)
    evaluate(predictions)
    print("done: outputs written under output/ and data/processed/")


if __name__ == "__main__":
    main()
