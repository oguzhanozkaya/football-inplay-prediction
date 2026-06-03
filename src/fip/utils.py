"""Project paths and shared configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CUTOFF_MINUTE = int(os.environ.get("FIP_CUTOFF_MINUTE", "45"))
WINDOW_MINUTES = int(os.environ.get("FIP_WINDOW_MINUTES", "5"))
MAX_TOKENS_PER_WINDOW = int(os.environ.get("FIP_MAX_TOKENS_PER_WINDOW", "64"))
MAX_VOCAB_SIZE = int(os.environ.get("FIP_MAX_VOCAB_SIZE", "20000"))
LABELS = ("home", "draw", "away")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used by pipeline stages."""

    root: Path
    data: Path
    raw_data: Path
    processed_data: Path
    output: Path
    figures: Path
    models: Path
    predictions: Path
    reports: Path

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        root = root.resolve()
        data = root / "data"
        output = root / "output"
        return cls(
            root=root,
            data=data,
            raw_data=data / "raw",
            processed_data=data / "processed",
            output=output,
            figures=output / "figures",
            models=output / "models",
            predictions=output / "predictions",
            reports=output / "reports",
        )

    def generated_directories(self) -> tuple[Path, ...]:
        return (self.raw_data, self.processed_data, self.figures, self.models, self.predictions, self.reports)


def build_paths(root: Path | None = None) -> ProjectPaths:
    """Build project paths for the repository or a test root."""

    return ProjectPaths.from_root(PROJECT_ROOT if root is None else root)


DEFAULT_PATHS = build_paths()


def ensure_generated_directories(paths: ProjectPaths = DEFAULT_PATHS) -> tuple[Path, ...]:
    """Create generated data and output directories if they are missing."""

    directories = paths.generated_directories()
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


@dataclass(frozen=True)
class SourceDefinition:
    """A local ESPN Soccer dataset source group under `data/raw/`."""

    source_id: str
    title: str
    category: str
    source_type: str
    raw_path: Path
    notes: str


SOURCE_REGISTRY: tuple[SourceDefinition, ...] = (
    SourceDefinition(
        "espn_base_data",
        "ESPN Soccer Base Data",
        "metadata",
        "local_csv_directory",
        Path("base_data"),
        "Fixtures, teams, leagues, venues, status labels, standings, and full-match team statistics.",
    ),
    SourceDefinition(
        "espn_commentary",
        "ESPN Soccer Commentary",
        "text",
        "local_csv_directory",
        Path("commentary_data"),
        "Minute-by-minute commentary text grouped by season and league.",
    ),
    SourceDefinition(
        "espn_plays",
        "ESPN Soccer Plays",
        "event",
        "local_csv_directory",
        Path("plays_data"),
        "Play-by-play event text, event types, clocks, scoring flags, teams, and pitch coordinates.",
    ),
    SourceDefinition(
        "espn_key_events",
        "ESPN Soccer Key Events",
        "event",
        "local_csv_directory",
        Path("keyEvents_data"),
        "Key soccer events with event types, text, clocks, teams, athletes, and pitch coordinates.",
    ),
    SourceDefinition(
        "espn_lineups",
        "ESPN Soccer Lineups",
        "prematch",
        "local_csv_directory",
        Path("lineup_data"),
        "Formations, starters, positions, and substitutions. Winner fields are excluded during preprocessing.",
    ),
    SourceDefinition(
        "espn_player_stats",
        "ESPN Soccer Player Stats",
        "future_optional",
        "local_csv_directory",
        Path("playerStats_data"),
        "Season player aggregates; not used in the first model because scrape-time aggregation can leak future "
        "results.",
    ),
)


def sources_by_category(category: str) -> tuple[SourceDefinition, ...]:
    """Return sources for one registry category."""

    return tuple(source for source in SOURCE_REGISTRY if source.category == category)


def source_by_id(source_id: str) -> SourceDefinition:
    """Return one source definition by id."""

    for source in SOURCE_REGISTRY:
        if source.source_id == source_id:
            return source
    raise KeyError(f"Unknown source id: {source_id}")


def window_count(cutoff_minute: int = CUTOFF_MINUTE, window_minutes: int = WINDOW_MINUTES) -> int:
    """Return the number of fixed windows from kickoff through the cutoff minute."""

    if cutoff_minute <= 0 or window_minutes <= 0 or cutoff_minute % window_minutes != 0:
        raise ValueError("FIP_CUTOFF_MINUTE must be a positive multiple of FIP_WINDOW_MINUTES")
    return cutoff_minute // window_minutes
