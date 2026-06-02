"""Preprocess raw data into interim artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from turkish_inflation_forecasting.config import DEFAULT_PATHS, ProjectPaths, ensure_generated_directories
from turkish_inflation_forecasting.data.cpi import preprocess_cpi_target
from turkish_inflation_forecasting.data.numeric import preprocess_numeric_sources
from turkish_inflation_forecasting.data.sources import CBRT_CONSUMER_PRICES, SourceDefinition, sources_by_category
from turkish_inflation_forecasting.data.text import preprocess_text_documents


@dataclass(frozen=True)
class PreprocessResult:
    """Summary of generated interim artifacts."""

    cpi_target_path: Path
    cpi_target_rows: int
    numeric_series_path: Path
    numeric_series_rows: int
    monthly_numeric_path: Path
    monthly_numeric_rows: int
    text_documents_path: Path
    text_document_rows: int


class PreprocessError(RuntimeError):
    """Raised when required raw inputs are missing or invalid."""


def raw_source_exists(paths: ProjectPaths, source: SourceDefinition) -> bool:
    """Return whether a registered raw source has been downloaded."""

    raw_path = paths.raw_data / source.raw_path
    if source.source_type == "official_xml_month_end_archive":
        return raw_path.is_dir() and any(raw_path.glob("*/*.xml"))
    return raw_path.is_file()


def preprocess_raw_sources(paths: ProjectPaths = DEFAULT_PATHS) -> PreprocessResult:
    """Convert downloaded raw sources into initial interim tables."""

    ensure_generated_directories(paths)
    cpi_raw_path = paths.raw_data / CBRT_CONSUMER_PRICES.raw_path
    if not cpi_raw_path.is_file():
        raise PreprocessError(f"Missing raw CPI source: {cpi_raw_path}. Run `just download` first.")

    numeric_sources = tuple(source for source in sources_by_category("numeric") if source != CBRT_CONSUMER_PRICES)
    missing_numeric_sources = [source for source in numeric_sources if not raw_source_exists(paths, source)]
    if missing_numeric_sources:
        missing_ids = ", ".join(source.source_id for source in missing_numeric_sources)
        raise PreprocessError(f"Missing raw numeric sources: {missing_ids}. Run `just download` first.")

    text_sources = sources_by_category("text")
    missing_text_sources = [source for source in text_sources if not (paths.raw_data / source.raw_path).is_file()]
    if missing_text_sources:
        missing_ids = ", ".join(source.source_id for source in missing_text_sources)
        raise PreprocessError(f"Missing raw text sources: {missing_ids}. Run `just download` first.")

    cpi_target_path = paths.interim_data / "cpi_mom.parquet"
    numeric_series_path = paths.interim_data / "numeric_series.parquet"
    monthly_numeric_path = paths.interim_data / "monthly_numeric.parquet"
    text_documents_path = paths.interim_data / "text_documents.parquet"
    cpi_target = preprocess_cpi_target(cpi_raw_path, cpi_target_path)
    numeric_series, monthly_numeric = preprocess_numeric_sources(
        paths.raw_data, numeric_series_path, monthly_numeric_path
    )
    text_documents = preprocess_text_documents(paths.raw_data, text_documents_path, text_sources)
    return PreprocessResult(
        cpi_target_path=cpi_target_path,
        cpi_target_rows=len(cpi_target),
        numeric_series_path=numeric_series_path,
        numeric_series_rows=len(numeric_series),
        monthly_numeric_path=monthly_numeric_path,
        monthly_numeric_rows=len(monthly_numeric),
        text_documents_path=text_documents_path,
        text_document_rows=len(text_documents),
    )
