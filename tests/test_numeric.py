import pandas as pd

from turkish_inflation_forecasting.data.fx import parse_cbrt_fx_xml
from turkish_inflation_forecasting.data.numeric import build_monthly_numeric, parse_fred_csv
from turkish_inflation_forecasting.data.sources import FRED_BRENT_OIL


def test_parse_cbrt_fx_xml_extracts_usd_and_eur_rates() -> None:
    xml = b"""
    <Tarih_Date Tarih="30.04.2024" Date="30.04.2024">
      <Currency CrossOrder="0" Kod="USD" CurrencyCode="USD">
        <Unit>1</Unit>
        <ForexBuying>32.3000</ForexBuying>
        <ForexSelling>32.4000</ForexSelling>
      </Currency>
      <Currency CrossOrder="1" Kod="EUR" CurrencyCode="EUR">
        <Unit>1</Unit>
        <ForexBuying>34.7000</ForexBuying>
        <ForexSelling>34.9000</ForexSelling>
      </Currency>
    </Tarih_Date>
    """

    rates = parse_cbrt_fx_xml(xml)

    assert set(rates["currency"]) == {"USD", "EUR"}
    assert rates["date"].eq(pd.Timestamp("2024-04-30")).all()
    assert rates["month_start"].eq(pd.Timestamp("2024-04-01")).all()
    assert rates.set_index("currency").loc["USD", "forex_selling"] == 32.4


def test_parse_fred_csv_normalizes_daily_source_rows() -> None:
    csv = "observation_date,DCOILBRENTEU\n2024-04-29,88.4\n2024-04-30,.\n2024-05-01,83.1\n"

    rows = parse_fred_csv(csv, FRED_BRENT_OIL.source_id)

    assert rows["series_id"].unique().tolist() == ["brent_oil_usd"]
    assert rows["frequency"].unique().tolist() == ["daily"]
    assert rows["date"].tolist() == [pd.Timestamp("2024-04-29"), pd.Timestamp("2024-05-01")]
    assert rows["value"].tolist() == [88.4, 83.1]


def test_build_monthly_numeric_aggregates_daily_and_monthly_series() -> None:
    numeric_series = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-04-29"),
                "month_start": pd.Timestamp("2024-04-01"),
                "series_id": "brent_oil_usd",
                "value": 80.0,
                "source_id": "fred_brent_oil",
                "frequency": "daily",
            },
            {
                "date": pd.Timestamp("2024-04-30"),
                "month_start": pd.Timestamp("2024-04-01"),
                "series_id": "brent_oil_usd",
                "value": 90.0,
                "source_id": "fred_brent_oil",
                "frequency": "daily",
            },
            {
                "date": pd.Timestamp("2024-04-30"),
                "month_start": pd.Timestamp("2024-04-01"),
                "series_id": "usd_try_fx_selling_month_end",
                "value": 32.0,
                "source_id": "cbrt_fx_month_end",
                "frequency": "monthly",
            },
        ]
    )

    monthly = build_monthly_numeric(numeric_series)

    assert monthly.loc[0, "month"] == "2024-04"
    assert monthly.loc[0, "brent_oil_usd_month_avg"] == 85.0
    assert monthly.loc[0, "brent_oil_usd_month_end"] == 90.0
    assert monthly.loc[0, "usd_try_fx_selling_month_end"] == 32.0
