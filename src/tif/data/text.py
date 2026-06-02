"""Official text metadata preprocessing."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup

from tif.data.sources import SourceDefinition

CBRT_BASE_URL = "https://www.tcmb.gov.tr"
ANNOUNCEMENT_CODE_PATTERN = re.compile(r"ANO(?P<year>\d{4})-(?P<number>\d{2})")
DATE_IN_TITLE_PATTERN = re.compile(r"(?P<day>\d{1,2})[/.](?P<month>\d{1,2})[/.](?P<year>\d{4})")
ENGLISH_DATE_PATTERN = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December) "
    r"(?P<day>\d{1,2}), (?P<year>\d{4})\b"
)
DAY_FIRST_ENGLISH_DATE_PATTERN = re.compile(
    r"\b(?P<day>\d{1,2}) "
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December) "
    r"(?P<year>\d{4})\b"
)
MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def _normalize_url(href: str) -> str:
    return urljoin(CBRT_BASE_URL, href)


def _document_id(source_id: str, url: str) -> str:
    match = ANNOUNCEMENT_CODE_PATTERN.search(url)
    if match:
        return f"{source_id}_{match.group(0).lower()}"
    path = urlparse(url).path.rstrip("/").split("/")[-1]
    return f"{source_id}_{path.lower()}"


def raw_document_path(document_id: str) -> Path:
    """Return the deterministic raw HTML path for one official text document."""

    return Path("text/documents") / f"{document_id}.html"


def _published_at_from_title(title: str) -> pd.Timestamp | pd.NaT:
    match = DATE_IN_TITLE_PATTERN.search(title)
    if not match:
        return pd.NaT
    return pd.Timestamp(year=int(match.group("year")), month=int(match.group("month")), day=int(match.group("day")))


def _published_at_from_body_text(text: str) -> pd.Timestamp | pd.NaT:
    match = ENGLISH_DATE_PATTERN.search(text) or DAY_FIRST_ENGLISH_DATE_PATTERN.search(text)
    if not match:
        return pd.NaT
    return pd.Timestamp(
        year=int(match.group("year")),
        month=MONTHS[match.group("month")],
        day=int(match.group("day")),
    )


def extract_cbrt_document_body_text(html: str) -> str:
    """Extract clean body text from an official CBRT document page."""

    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#tcmbMainContent .tcmb-content") or soup.select_one("#tcmbMainContent")
    if content is None:
        raise ValueError("Could not locate CBRT document body content")

    for element in content.select("script, style"):
        element.decompose()

    lines = []
    for line in content.get_text("\n", strip=True).splitlines():
        line = " ".join(line.split())
        if line:
            lines.append(line)

    body_text = "\n".join(lines)
    if not body_text:
        raise ValueError("CBRT document body content is empty")
    return body_text


def extract_cbrt_text_links(html: str, source: SourceDefinition) -> pd.DataFrame:
    """Extract official CBRT text document links from a listing page."""

    soup = BeautifulSoup(html, "html.parser")
    rows = []
    expected_path = source.url.split("/EN/TCMB%2BEN/")[-1].replace("%2B", "+")

    for anchor in soup.find_all("a", href=True):
        title = " ".join(anchor.get_text(" ", strip=True).split())
        if not title:
            continue
        url = _normalize_url(anchor["href"])
        if expected_path not in url or not ANNOUNCEMENT_CODE_PATTERN.search(url):
            continue
        document_id = _document_id(source.source_id, url)
        rows.append(
            {
                "document_id": document_id,
                "source_id": source.source_id,
                "source_type": source.source_type,
                "title": title,
                "url": url,
                "published_at": _published_at_from_title(title),
                "raw_listing_path": source.raw_path.as_posix(),
                "raw_document_path": raw_document_path(document_id).as_posix(),
            }
        )

    if not rows:
        raise ValueError(f"Could not extract text links from {source.source_id}")

    documents = pd.DataFrame(rows).drop_duplicates(subset=["document_id"]).reset_index(drop=True)
    return documents.sort_values(["published_at", "document_id"], na_position="last").reset_index(drop=True)


def preprocess_text_documents(
    raw_data_path: Path, output_path: Path, sources: tuple[SourceDefinition, ...]
) -> pd.DataFrame:
    """Build text document metadata and body text from downloaded official pages."""

    frames = []
    for source in sources:
        html = (raw_data_path / source.raw_path).read_text(encoding="utf-8")
        frames.append(extract_cbrt_text_links(html, source))
    documents = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["document_id"])

    body_texts = []
    published_dates = []
    for document in documents.itertuples(index=False):
        raw_path = raw_data_path / document.raw_document_path
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw text document: {raw_path}. Run `just download` first.")
        body_text = extract_cbrt_document_body_text(raw_path.read_text(encoding="utf-8"))
        body_texts.append(body_text)
        published_dates.append(_published_at_from_body_text(body_text))

    documents = documents.copy()
    documents["body_text"] = body_texts
    missing_dates = documents["published_at"].isna()
    documents.loc[missing_dates, "published_at"] = pd.Series(published_dates, index=documents.index)[missing_dates]
    documents["body_char_count"] = documents["body_text"].str.len()
    documents["body_word_count"] = documents["body_text"].str.split().str.len()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    documents.to_parquet(output_path, index=False)
    return documents
