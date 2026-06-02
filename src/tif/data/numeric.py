"""Numeric macro-financial preprocessing."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

from tif.data.fx import parse_cbrt_fx_xml
from tif.data.sources import (
    CBRT_FX_MONTH_END,
    FRED_BRENT_OIL,
    FRED_TURKEY_INDUSTRIAL_PRODUCTION,
    FRED_TURKEY_UNEMPLOYMENT_RATE,
)

FRED_SERIES = {
    FRED_BRENT_OIL.source_id: ("DCOILBRENTEU", "brent_oil_usd", "daily"),
    FRED_TURKEY_INDUSTRIAL_PRODUCTION.source_id: (
        "TURPRINTO01GYSAM",
        "turkey_industrial_production_yoy_sa",
        "monthly",
    ),
    FRED_TURKEY_UNEMPLOYMENT_RATE.source_id: ("LRHUTTTTTRM156S", "turkey_unemployment_rate_sa", "monthly"),
}


def parse_fred_csv(content: str, source_id: str) -> pd.DataFrame:
    """Parse one public FRED CSV into the project's long numeric schema."""

    fred_column, series_id, frequency = FRED_SERIES[source_id]
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

    xml_paths = sorted((raw_data_path / CBRT_FX_MONTH_END.raw_path).glob("*/*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No CBRT FX XML files found under {raw_data_path / CBRT_FX_MONTH_END.raw_path}")

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
    basket["source_id"] = CBRT_FX_MONTH_END.source_id
    basket["frequency"] = "monthly"
    basket = basket.rename(columns={"fx_basket_try_month_end": "value"})
    return pd.concat([frame, basket[frame.columns]], ignore_index=True).sort_values(["month_start", "series_id"])


def build_numeric_series(raw_data_path: Path) -> pd.DataFrame:
    """Build a long table of normalized numeric source observations."""

    frames = [parse_cbrt_fx_archive(raw_data_path)]
    for source in (FRED_BRENT_OIL, FRED_TURKEY_INDUSTRIAL_PRODUCTION, FRED_TURKEY_UNEMPLOYMENT_RATE):
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
