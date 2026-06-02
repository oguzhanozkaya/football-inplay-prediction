"""Project paths and shared configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used by pipeline stages."""

    root: Path
    data: Path
    raw_data: Path
    interim_data: Path
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
            interim_data=data / "interim",
            processed_data=data / "processed",
            output=output,
            figures=output / "figures",
            models=output / "models",
            predictions=output / "predictions",
            reports=output / "reports",
        )

    def generated_directories(self) -> tuple[Path, ...]:
        return (
            self.raw_data,
            self.interim_data,
            self.processed_data,
            self.figures,
            self.models,
            self.predictions,
            self.reports,
        )


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
