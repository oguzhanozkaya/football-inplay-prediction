from pathlib import Path

import pandas as pd

import tif.preprocess
import tif.utils


def write_required_numeric_sources(paths) -> None:
    fx_path = paths.raw_data / tif.utils.CBRT_FX_MONTH_END.raw_path / "202604" / "30042026.xml"
    fx_path.parent.mkdir(parents=True, exist_ok=True)
    fx_path.write_text(
        """
        <Tarih_Date Tarih="30.04.2026" Date="30.04.2026">
          <Currency Kod="USD" CurrencyCode="USD">
            <Unit>1</Unit><ForexBuying>39.0000</ForexBuying><ForexSelling>39.1000</ForexSelling>
          </Currency>
          <Currency Kod="EUR" CurrencyCode="EUR">
            <Unit>1</Unit><ForexBuying>43.0000</ForexBuying><ForexSelling>43.2000</ForexSelling>
          </Currency>
        </Tarih_Date>
        """,
        encoding="utf-8",
    )
    fred_sources = {
        tif.utils.FRED_BRENT_OIL: "observation_date,DCOILBRENTEU\n2026-04-29,83.0\n2026-04-30,84.0\n",
        tif.utils.FRED_TURKEY_INDUSTRIAL_PRODUCTION: "observation_date,TURPRINTO01GYSAM\n2026-04-01,2.5\n",
        tif.utils.FRED_TURKEY_UNEMPLOYMENT_RATE: "observation_date,LRHUTTTTTRM156S\n2026-04-01,8.6\n",
    }
    for source, content in fred_sources.items():
        raw_path = paths.raw_data / source.raw_path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(content, encoding="utf-8")


def test_parse_cpi_fred_fx_and_monthly_numeric() -> None:
    cpi = tif.preprocess.parse_cbrt_consumer_prices_html("02-2026 31.53 2.96 01-2026 30.65 4.84")
    target = tif.preprocess.build_cpi_target_table(cpi)
    fred = tif.preprocess.parse_fred_csv(
        "observation_date,DCOILBRENTEU\n2024-04-29,88.4\n2024-04-30,.\n2024-05-01,83.1\n",
        tif.utils.FRED_BRENT_OIL.source_id,
    )
    fx = tif.preprocess.parse_cbrt_fx_xml(
        b"""
        <Tarih_Date Tarih="30.04.2024" Date="30.04.2024">
          <Currency Kod="USD" CurrencyCode="USD"><Unit>1</Unit><ForexSelling>32.4000</ForexSelling></Currency>
          <Currency Kod="EUR" CurrencyCode="EUR"><Unit>1</Unit><ForexSelling>34.9000</ForexSelling></Currency>
        </Tarih_Date>
        """
    )
    monthly = tif.preprocess.build_monthly_numeric(fred)

    assert target["forecast_origin_month"].tolist() == ["2025-12", "2026-01"]
    assert fred["date"].tolist() == [pd.Timestamp("2024-04-29"), pd.Timestamp("2024-05-01")]
    assert set(fx["currency"]) == {"USD", "EUR"}
    assert monthly.loc[0, "brent_oil_usd_month_avg"] == 88.4


def test_preprocess_raw_sources_writes_initial_interim_tables(tmp_path: Path) -> None:
    paths = tif.utils.build_paths(tmp_path)
    paths.raw_data.mkdir(parents=True)
    cpi_path = paths.raw_data / tif.utils.CBRT_CONSUMER_PRICES.raw_path
    cpi_path.parent.mkdir(parents=True)
    cpi_path.write_text("02-2026 31.53 2.96 01-2026 30.65 4.84", encoding="utf-8")
    write_required_numeric_sources(paths)

    for source in tif.utils.sources_by_category("text"):
        raw_path = paths.raw_data / source.raw_path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        source_url_path = source.url.split("/MPC/")[-1].replace("%2B", "+")
        raw_path.write_text(
            f'<a href="/wps/wcm/connect/EN/TCMB+EN/MPC/{source_url_path}/ANO2026-17">Document (2026-17)</a>',
            encoding="utf-8",
        )
        document_path = paths.raw_data / tif.utils.raw_document_path(f"{source.source_id}_ano2026-17")
        document_path.parent.mkdir(parents=True, exist_ok=True)
        document_path.write_text(
            """
            <div id="tcmbMainContent">
              <div class="tcmb-content type-prg">
                <p>April 22, 2026</p>
                <p>The tight monetary policy stance will strengthen the disinflation process.</p>
              </div>
            </div>
            """,
            encoding="utf-8",
        )

    result = tif.preprocess.preprocess_raw_sources(paths)

    assert result.cpi_target_rows == 2
    assert result.numeric_series_rows == 7
    assert result.monthly_numeric_rows == 1
    assert result.text_document_rows == 2
    assert result.cpi_target_path.is_file()
    assert result.numeric_series_path.is_file()
    assert result.monthly_numeric_path.is_file()
    assert result.text_documents_path.is_file()

    text_documents = pd.read_parquet(result.text_documents_path)
    assert text_documents["body_text"].str.contains("disinflation process").all()
    assert text_documents["published_at"].notna().all()
