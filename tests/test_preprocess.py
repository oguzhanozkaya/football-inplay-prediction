from pathlib import Path

import pandas as pd

from turkish_inflation_forecasting.config import build_paths
from turkish_inflation_forecasting.data.preprocess import preprocess_raw_sources
from turkish_inflation_forecasting.data.sources import (
    CBRT_CONSUMER_PRICES,
    CBRT_FX_MONTH_END,
    FRED_BRENT_OIL,
    FRED_TURKEY_INDUSTRIAL_PRODUCTION,
    FRED_TURKEY_UNEMPLOYMENT_RATE,
    sources_by_category,
)
from turkish_inflation_forecasting.data.text import raw_document_path


def write_required_numeric_sources(paths) -> None:
    fx_path = paths.raw_data / CBRT_FX_MONTH_END.raw_path / "202604" / "30042026.xml"
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
        FRED_BRENT_OIL: "observation_date,DCOILBRENTEU\n2026-04-29,83.0\n2026-04-30,84.0\n",
        FRED_TURKEY_INDUSTRIAL_PRODUCTION: "observation_date,TURPRINTO01GYSAM\n2026-04-01,2.5\n",
        FRED_TURKEY_UNEMPLOYMENT_RATE: "observation_date,LRHUTTTTTRM156S\n2026-04-01,8.6\n",
    }
    for source, content in fred_sources.items():
        raw_path = paths.raw_data / source.raw_path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(content, encoding="utf-8")


def test_preprocess_raw_sources_writes_initial_interim_tables(tmp_path: Path) -> None:
    paths = build_paths(tmp_path)
    paths.raw_data.mkdir(parents=True)
    cpi_path = paths.raw_data / CBRT_CONSUMER_PRICES.raw_path
    cpi_path.parent.mkdir(parents=True)
    cpi_path.write_text("02-2026 31.53 2.96 01-2026 30.65 4.84", encoding="utf-8")
    write_required_numeric_sources(paths)

    for source in sources_by_category("text"):
        raw_path = paths.raw_data / source.raw_path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        source_url_path = source.url.split("/MPC/")[-1].replace("%2B", "+")
        raw_path.write_text(
            f'<a href="/wps/wcm/connect/EN/TCMB+EN/MPC/{source_url_path}/ANO2026-17">Document (2026-17)</a>',
            encoding="utf-8",
        )
        document_path = paths.raw_data / raw_document_path(f"{source.source_id}_ano2026-17")
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

    result = preprocess_raw_sources(paths)

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
