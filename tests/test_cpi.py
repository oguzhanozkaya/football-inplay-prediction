import pandas as pd

from turkish_inflation_forecasting.data.cpi import build_cpi_target_table, parse_cbrt_consumer_prices_html


def test_parse_cbrt_consumer_prices_html_extracts_mom_rows() -> None:
    html = """
    <h1>Consumer Price Index</h1>
    <table>
      <tr><th>CPI (Year to Year % Changes)</th><th>CPI (Month to Month % Changes)</th></tr>
      <tr><td>02-2026</td><td>31.53</td><td>2.96</td></tr>
      <tr><td>01-2026</td><td>30.65</td><td>4.84</td></tr>
    </table>
    """

    result = parse_cbrt_consumer_prices_html(html)

    assert result["target_month"].tolist() == ["2026-01", "2026-02"]
    assert result["cpi_mom_percent"].tolist() == [4.84, 2.96]


def test_build_cpi_target_table_aligns_forecast_origin_to_previous_month() -> None:
    cpi_rates = pd.DataFrame(
        {
            "target_month_start": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")],
            "target_month": ["2026-01", "2026-02"],
            "cpi_yoy_percent": [30.65, 31.53],
            "cpi_mom_percent": [4.84, 2.96],
            "source_id": ["cbrt_consumer_prices", "cbrt_consumer_prices"],
        }
    )

    target = build_cpi_target_table(cpi_rates)

    assert target["forecast_origin_month"].tolist() == ["2025-12", "2026-01"]
    assert target["target_month"].tolist() == ["2026-01", "2026-02"]
    assert target["target_cpi_mom_percent"].tolist() == [4.84, 2.96]
