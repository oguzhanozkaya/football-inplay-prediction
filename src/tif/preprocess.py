"""Preprocess raw data into interim artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
from bs4 import BeautifulSoup

import tif.utils

CPI_ROW_PATTERN = re.compile(r"(?P<month>\d{2})-(?P<year>\d{4})\s+(?P<yoy>-?\d+(?:\.\d+)?)\s+(?P<mom>-?\d+(?:\.\d+)?)")


def parse_cbrt_consumer_prices_html(html: str) -> pd.DataFrame:
    """Parse CBRT CPI year-to-year and month-to-month rates from HTML."""

    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    normalized = text.replace("\u00a0", " ").replace("−", "-").replace("\\-", "-")
    rows = []
    for match in CPI_ROW_PATTERN.finditer(normalized):
        year = int(match.group("year"))
        month = int(match.group("month"))
        target_month_start = pd.Timestamp(year=year, month=month, day=1)
        rows.append(
            {
                "target_month_start": target_month_start,
                "target_month": target_month_start.strftime("%Y-%m"),
                "cpi_yoy_percent": float(match.group("yoy")),
                "cpi_mom_percent": float(match.group("mom")),
                "source_id": "cbrt_consumer_prices",
            }
        )

    if not rows:
        raise ValueError("Could not parse any CPI rows from CBRT consumer prices HTML")

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["target_month"])
        .sort_values("target_month_start")
        .reset_index(drop=True)
    )


def build_cpi_target_table(cpi_rates: pd.DataFrame) -> pd.DataFrame:
    """Build one-month-ahead CPI MoM target rows.

    Each row represents a forecast made at the end of `forecast_origin_month`
    for the CPI MoM value in `target_month`.
    """

    required_columns = {"target_month_start", "target_month", "cpi_mom_percent", "source_id"}
    missing_columns = required_columns.difference(cpi_rates.columns)
    if missing_columns:
        raise ValueError(f"Missing CPI columns: {sorted(missing_columns)}")

    target = cpi_rates.copy().sort_values("target_month_start").reset_index(drop=True)
    target["forecast_origin_month_start"] = target["target_month_start"] - pd.DateOffset(months=1)
    target["forecast_origin_month"] = target["forecast_origin_month_start"].dt.strftime("%Y-%m")
    target = target.rename(columns={"cpi_mom_percent": "target_cpi_mom_percent"})
    columns = [
        "forecast_origin_month_start",
        "forecast_origin_month",
        "target_month_start",
        "target_month",
        "target_cpi_mom_percent",
        "cpi_yoy_percent",
        "source_id",
    ]
    return target[columns]


def preprocess_cpi_target(raw_html_path: Path, output_path: Path) -> pd.DataFrame:
    """Read raw CBRT CPI HTML and write the CPI MoM target table."""

    html = raw_html_path.read_text(encoding="utf-8")
    target = build_cpi_target_table(parse_cbrt_consumer_prices_html(html))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target.to_parquet(output_path, index=False)
    return target


def parse_fred_csv(content: str, source_id: str) -> pd.DataFrame:
    """Parse one public FRED CSV into the project's long numeric schema."""

    fred_column, series_id, frequency = tif.utils.FRED_SERIES[source_id]
    frame = pd.read_csv(StringIO(content), na_values=[".", ""])
    frame = frame.rename(columns={"observation_date": "date", fred_column: "value"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    frame["month_start"] = frame["date"].dt.to_period("M").dt.to_timestamp()
    frame["series_id"] = series_id
    frame["source_id"] = source_id
    frame["frequency"] = frequency
    return frame[["date", "month_start", "series_id", "value", "source_id", "frequency"]].reset_index(drop=True)


def parse_cbrt_fx_archive(raw_data_path: Path) -> pd.DataFrame:
    """Parse downloaded CBRT month-end FX XML files into long series rows."""

    xml_paths = sorted((raw_data_path / tif.utils.CBRT_FX_MONTH_END.raw_path).glob("*/*.xml"))
    if not xml_paths:
        raise FileNotFoundError(
            f"No CBRT FX XML files found under {raw_data_path / tif.utils.CBRT_FX_MONTH_END.raw_path}"
        )

    fx_rates = pd.concat([parse_cbrt_fx_xml(path.read_bytes()) for path in xml_paths], ignore_index=True)
    rows = []
    for rate in fx_rates.itertuples(index=False):
        rows.append(
            {
                "date": rate.date,
                "month_start": rate.month_start,
                "series_id": f"{rate.currency.lower()}_try_fx_selling_month_end",
                "value": rate.forex_selling,
                "source_id": rate.source_id,
                "frequency": "monthly",
            }
        )
    frame = pd.DataFrame(rows)
    basket = (
        frame.pivot(index="month_start", columns="series_id", values="value")
        .assign(
            fx_basket_try_month_end=lambda data: (
                (data["usd_try_fx_selling_month_end"] + data["eur_try_fx_selling_month_end"]) / 2
            )
        )["fx_basket_try_month_end"]
        .reset_index()
    )
    basket = basket.merge(fx_rates.groupby("month_start", as_index=False)["date"].max(), on="month_start", how="left")
    basket["series_id"] = "fx_basket_try_month_end"
    basket["source_id"] = tif.utils.CBRT_FX_MONTH_END.source_id
    basket["frequency"] = "monthly"
    basket = basket.rename(columns={"fx_basket_try_month_end": "value"})
    return pd.concat([frame, basket[frame.columns]], ignore_index=True).sort_values(["month_start", "series_id"])


def build_numeric_series(raw_data_path: Path) -> pd.DataFrame:
    """Build a long table of normalized numeric source observations."""

    frames = [parse_cbrt_fx_archive(raw_data_path)]
    for source in (
        tif.utils.FRED_BRENT_OIL,
        tif.utils.FRED_TURKEY_INDUSTRIAL_PRODUCTION,
        tif.utils.FRED_TURKEY_UNEMPLOYMENT_RATE,
    ):
        frames.append(parse_fred_csv((raw_data_path / source.raw_path).read_text(encoding="utf-8"), source.source_id))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "series_id"]).reset_index(drop=True)


def build_monthly_numeric(numeric_series: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized numeric observations to one monthly feature table."""

    monthly_frames = []
    monthly_series = numeric_series[numeric_series["frequency"] == "monthly"]
    monthly_frames.append(
        monthly_series.pivot_table(index="month_start", columns="series_id", values="value", aggfunc="last")
    )

    brent = numeric_series[numeric_series["series_id"] == "brent_oil_usd"].sort_values("date")
    brent_monthly = brent.groupby("month_start")["value"].agg(
        brent_oil_usd_month_avg="mean",
        brent_oil_usd_month_end="last",
    )
    monthly_frames.append(brent_monthly)

    monthly = pd.concat(monthly_frames, axis=1).sort_index().reset_index()
    monthly["month"] = monthly["month_start"].dt.strftime("%Y-%m")
    return monthly[
        ["month_start", "month", *[column for column in monthly.columns if column not in {"month_start", "month"}]]
    ]


def preprocess_numeric_sources(
    raw_data_path: Path, numeric_series_path: Path, monthly_numeric_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write long numeric observations and monthly aligned numeric features."""

    numeric_series = build_numeric_series(raw_data_path)
    monthly_numeric = build_monthly_numeric(numeric_series)
    numeric_series_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_numeric_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_series.to_parquet(numeric_series_path, index=False)
    monthly_numeric.to_parquet(monthly_numeric_path, index=False)
    return numeric_series, monthly_numeric


def extract_cbrt_document_body_text(html: str) -> str:
    """Extract clean body text from an official CBRT document page."""

    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#tcmbMainContent .tcmb-content") or soup.select_one("#tcmbMainContent")
    if content is None:
        raise ValueError("Could not locate CBRT document body content")

    for element in content.select("script, style"):
        element.decompose()

    lines = []
    for line in content.get_text("\n", strip=True).splitlines():
        line = " ".join(line.split())
        if line:
            lines.append(line)

    body_text = "\n".join(lines)
    if not body_text:
        raise ValueError("CBRT document body content is empty")
    return body_text


def preprocess_text_documents(
    raw_data_path: Path,
    output_path: Path,
    sources: tuple[tif.utils.SourceDefinition, ...],
) -> pd.DataFrame:
    """Build text document metadata and body text from downloaded official pages."""

    frames = []
    for source in sources:
        html = (raw_data_path / source.raw_path).read_text(encoding="utf-8")
        frames.append(tif.utils.extract_cbrt_text_links(html, source))
    documents = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["document_id"])

    body_texts = []
    published_dates = []
    for document in documents.itertuples(index=False):
        raw_path = raw_data_path / document.raw_document_path
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw text document: {raw_path}. Run `just download` first.")
        body_text = extract_cbrt_document_body_text(raw_path.read_text(encoding="utf-8"))
        body_texts.append(body_text)
        published_dates.append(tif.utils.published_at_from_body_text(body_text))

    documents = documents.copy()
    documents["body_text"] = body_texts
    missing_dates = documents["published_at"].isna()
    documents.loc[missing_dates, "published_at"] = pd.Series(published_dates, index=documents.index)[missing_dates]
    documents["body_char_count"] = documents["body_text"].str.len()
    documents["body_word_count"] = documents["body_text"].str.split().str.len()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents.to_parquet(output_path, index=False)
    return documents


def parse_cbrt_fx_xml(content: bytes) -> pd.DataFrame:
    """Parse USD and EUR rates from one official CBRT exchange-rate XML file."""

    root = ElementTree.fromstring(content)
    if tarih := root.attrib.get("Tarih"):
        effective_date = pd.to_datetime(tarih, format="%d.%m.%Y").normalize()
    else:
        effective_date = pd.to_datetime(root.attrib["Date"], format="%m/%d/%Y").normalize()
    rows = []
    for currency in root.findall("Currency"):
        code = currency.attrib.get("CurrencyCode") or currency.attrib.get("Kod")
        if code not in {"USD", "EUR"}:
            continue
        unit = float(currency.findtext("Unit") or 1)
        forex_buying = float(currency.findtext("ForexBuying") or "nan") / unit
        forex_selling = float(currency.findtext("ForexSelling") or "nan") / unit
        rows.append(
            {
                "date": effective_date,
                "month_start": effective_date.to_period("M").to_timestamp(),
                "currency": code,
                "forex_buying": forex_buying,
                "forex_selling": forex_selling,
                "source_id": "cbrt_fx_month_end",
            }
        )
    if len(rows) != 2:
        raise ValueError("CBRT FX XML does not contain both USD and EUR rates")
    return pd.DataFrame(rows)


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


def raw_source_exists(paths: tif.utils.ProjectPaths, source: tif.utils.SourceDefinition) -> bool:
    """Return whether a registered raw source has been downloaded."""

    raw_path = paths.raw_data / source.raw_path
    if source.source_type == "official_xml_month_end_archive":
        return raw_path.is_dir() and any(raw_path.glob("*/*.xml"))
    return raw_path.is_file()


def preprocess_raw_sources(paths: tif.utils.ProjectPaths = tif.utils.DEFAULT_PATHS) -> PreprocessResult:
    """Convert downloaded raw sources into initial interim tables."""

    tif.utils.ensure_generated_directories(paths)
    cpi_raw_path = paths.raw_data / tif.utils.CBRT_CONSUMER_PRICES.raw_path
    if not cpi_raw_path.is_file():
        raise PreprocessError(f"Missing raw CPI source: {cpi_raw_path}. Run `just download` first.")

    numeric_sources = tuple(
        source for source in tif.utils.sources_by_category("numeric") if source != tif.utils.CBRT_CONSUMER_PRICES
    )
    missing_numeric_sources = [source for source in numeric_sources if not raw_source_exists(paths, source)]
    if missing_numeric_sources:
        missing_ids = ", ".join(source.source_id for source in missing_numeric_sources)
        raise PreprocessError(f"Missing raw numeric sources: {missing_ids}. Run `just download` first.")

    text_sources = tif.utils.sources_by_category("text")
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


def main() -> int:
    try:
        result = preprocess_raw_sources(tif.utils.DEFAULT_PATHS)
    except (PreprocessError, ValueError) as exc:
        print(f"preprocess: {exc}")
        return 1
    cpi_target_path = result.cpi_target_path.relative_to(tif.utils.DEFAULT_PATHS.root)
    print(f"preprocess: wrote {result.cpi_target_rows} CPI target rows to {cpi_target_path}")
    print(
        "preprocess: wrote "
        f"{result.numeric_series_rows} numeric source rows to "
        f"{result.numeric_series_path.relative_to(tif.utils.DEFAULT_PATHS.root)}"
    )
    print(
        "preprocess: wrote "
        f"{result.monthly_numeric_rows} monthly numeric rows to "
        f"{result.monthly_numeric_path.relative_to(tif.utils.DEFAULT_PATHS.root)}"
    )
    print(
        "preprocess: wrote "
        f"{result.text_document_rows} text document rows to "
        f"{result.text_documents_path.relative_to(tif.utils.DEFAULT_PATHS.root)}"
    )
    return 0
