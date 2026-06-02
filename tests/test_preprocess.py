from pathlib import Path

import pandas as pd

from turkish_inflation_forecasting.config import build_paths
from turkish_inflation_forecasting.data.preprocess import preprocess_raw_sources
from turkish_inflation_forecasting.data.sources import CBRT_CONSUMER_PRICES, sources_by_category
from turkish_inflation_forecasting.data.text import raw_document_path


def test_preprocess_raw_sources_writes_initial_interim_tables(tmp_path: Path) -> None:
    paths = build_paths(tmp_path)
    paths.raw_data.mkdir(parents=True)
    cpi_path = paths.raw_data / CBRT_CONSUMER_PRICES.raw_path
    cpi_path.parent.mkdir(parents=True)
    cpi_path.write_text("02-2026 31.53 2.96 01-2026 30.65 4.84", encoding="utf-8")

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
    assert result.text_document_rows == 2
    assert result.cpi_target_path.is_file()
    assert result.text_documents_path.is_file()

    text_documents = pd.read_parquet(result.text_documents_path)
    assert text_documents["body_text"].str.contains("disinflation process").all()
    assert text_documents["published_at"].notna().all()
