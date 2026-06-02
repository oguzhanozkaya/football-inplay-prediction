import pandas as pd

from turkish_inflation_forecasting.data.sources import CBRT_MPC_DECISIONS
from turkish_inflation_forecasting.data.text import extract_cbrt_document_body_text, extract_cbrt_text_links


def test_extract_cbrt_text_links_keeps_official_announcement_links() -> None:
    html = """
    <a href="/wps/wcm/connect/EN/TCMB+EN/MPC/MPC+Meeting+Decisions/ANO2026-17">
      Press Release on Interest Rates (2026-17)
    </a>
    <a href="/wps/wcm/connect/EN/TCMB+EN/MPC/MPC+Meeting+Decisions/ANO2017-46">
      Press Release on Interest Rates - 14/12/2017, (2017-46)
    </a>
    <a href="/wps/wcm/connect/EN/TCMB+EN/Main+Menu/About+the+Bank">About</a>
    """

    documents = extract_cbrt_text_links(html, CBRT_MPC_DECISIONS)

    assert documents["document_id"].tolist() == [
        "cbrt_mpc_decisions_ano2017-46",
        "cbrt_mpc_decisions_ano2026-17",
    ]
    assert documents.loc[0, "published_at"] == pd.Timestamp("2017-12-14")
    assert pd.isna(documents.loc[1, "published_at"])
    assert documents.loc[1, "raw_document_path"] == "text/documents/cbrt_mpc_decisions_ano2026-17.html"


def test_extract_cbrt_document_body_text_uses_main_content() -> None:
    html = """
    <nav>Navigation text that should be ignored</nav>
    <div id="tcmbMainContent">
      <div class="tcmb-content type-prg">
        <p>No: 2026-17</p>
        <p>April 22, 2026</p>
        <h2>Press Release on Interest Rates</h2>
        <p>The underlying trend of inflation declined in March.</p>
      </div>
      <a class="pdf" href="document.pdf">PDF link that should be ignored</a>
    </div>
    """

    body_text = extract_cbrt_document_body_text(html)

    assert "Navigation text" not in body_text
    assert "PDF link" not in body_text
    assert "April 22, 2026" in body_text
    assert "underlying trend of inflation" in body_text
