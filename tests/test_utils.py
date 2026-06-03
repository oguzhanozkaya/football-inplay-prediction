from datetime import date
from pathlib import Path

import pandas as pd

import tif.utils


def test_paths_and_generated_directories(tmp_path: Path) -> None:
    paths = tif.utils.build_paths(tmp_path)

    directories = tif.utils.ensure_generated_directories(paths)

    assert paths.root == tmp_path.resolve()
    assert paths.raw_data == tmp_path.resolve() / "data" / "raw"
    assert paths.reports == tmp_path.resolve() / "output" / "reports"
    assert directories == paths.generated_directories()
    assert all(directory.is_dir() for directory in directories)


def test_source_registry_helpers_and_shared_constants() -> None:
    source_ids = [source.source_id for source in tif.utils.SOURCE_REGISTRY]

    assert len(source_ids) == len(set(source_ids))
    assert tif.utils.MAX_TOKENS == 256
    assert {source.source_id for source in tif.utils.sources_by_category("numeric")} == {
        "cbrt_consumer_prices",
        "cbrt_fx_month_end",
        "fred_brent_oil",
        "fred_turkey_industrial_production",
        "fred_turkey_unemployment_rate",
    }
    assert tif.utils.source_by_id("cbrt_consumer_prices").category == "numeric"


def test_cbrt_text_and_fx_helpers() -> None:
    html = """
    <a href="/wps/wcm/connect/EN/TCMB+EN/MPC/MPC+Meeting+Decisions/ANO2026-17">
      Press Release on Interest Rates (2026-17)
    </a>
    <a href="/wps/wcm/connect/EN/TCMB+EN/MPC/MPC+Meeting+Decisions/ANO2017-46">
      Press Release on Interest Rates - 14/12/2017, (2017-46)
    </a>
    """

    documents = tif.utils.extract_cbrt_text_links(html, tif.utils.CBRT_MPC_DECISIONS)

    assert documents["document_id"].tolist() == [
        "cbrt_mpc_decisions_ano2017-46",
        "cbrt_mpc_decisions_ano2026-17",
    ]
    assert documents.loc[0, "published_at"] == pd.Timestamp("2017-12-14")
    assert pd.isna(documents.loc[1, "published_at"])
    assert tif.utils.iter_month_starts(date(2026, 1, 1), date(2026, 3, 1)) == [
        date(2026, 1, 1),
        date(2026, 2, 1),
        date(2026, 3, 1),
    ]
    assert tif.utils.cbrt_fx_url_for_date(date(2026, 4, 30)).endswith("/202604/30042026.xml")
