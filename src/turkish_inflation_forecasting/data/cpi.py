"""CPI target preprocessing."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

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
