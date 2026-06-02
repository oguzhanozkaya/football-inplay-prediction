"""CBRT public exchange-rate archive helpers."""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

CBRT_FX_ARCHIVE_BASE_URL = "https://www.tcmb.gov.tr/kurlar"
FX_START_MONTH = date(2005, 1, 1)


def latest_completed_month(today: date | None = None) -> date:
    """Return the first day of the latest completed calendar month."""

    today = datetime.now(UTC).date() if today is None else today
    first_day_this_month = date(today.year, today.month, 1)
    latest_month_end = first_day_this_month - timedelta(days=1)
    return date(latest_month_end.year, latest_month_end.month, 1)


def iter_month_starts(start_month: date = FX_START_MONTH, end_month: date | None = None) -> list[date]:
    """Return month starts from `start_month` through `end_month`, inclusive."""

    end_month = latest_completed_month() if end_month is None else end_month
    month = date(start_month.year, start_month.month, 1)
    end_month = date(end_month.year, end_month.month, 1)
    months = []
    while month <= end_month:
        months.append(month)
        year = month.year + (month.month // 12)
        next_month = 1 if month.month == 12 else month.month + 1
        month = date(year, next_month, 1)
    return months


def month_end_candidates(month_start: date, fallback_days: int = 14) -> list[date]:
    """Return candidate archive dates from month-end backward."""

    last_day = calendar.monthrange(month_start.year, month_start.month)[1]
    month_end = date(month_start.year, month_start.month, last_day)
    return [month_end - timedelta(days=offset) for offset in range(fallback_days + 1)]


def cbrt_fx_url_for_date(effective_date: date) -> str:
    """Build the public CBRT XML URL for one archive date."""

    return f"{CBRT_FX_ARCHIVE_BASE_URL}/{effective_date:%Y%m}/{effective_date:%d%m%Y}.xml"


def cbrt_fx_raw_path_for_date(base_path: Path, effective_date: date) -> Path:
    """Build the local raw XML path for one archive date."""

    return base_path / f"{effective_date:%Y%m}" / f"{effective_date:%d%m%Y}.xml"


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
