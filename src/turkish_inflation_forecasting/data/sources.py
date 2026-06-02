"""Reproducible source registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceDefinition:
    """A source that can be downloaded into `data/raw/`."""

    source_id: str
    title: str
    category: str
    source_type: str
    url: str
    raw_path: Path
    notes: str


CBRT_CONSUMER_PRICES = SourceDefinition(
    source_id="cbrt_consumer_prices",
    title="CBRT Consumer Prices",
    category="numeric",
    source_type="official_html",
    url="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Statistics/Inflation+Data/Consumer+Prices",
    raw_path=Path("numeric/cbrt_consumer_prices.html"),
    notes="Official CBRT page listing TURKSTAT CPI year-to-year and month-to-month rates.",
)

CBRT_MPC_DECISIONS = SourceDefinition(
    source_id="cbrt_mpc_decisions",
    title="CBRT MPC Meeting Decisions",
    category="text",
    source_type="official_html_listing",
    url="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB%2BEN/MPC/MPC%2BMeeting%2BDecisions",
    raw_path=Path("text/cbrt_mpc_decisions.html"),
    notes="Official listing page for interest-rate press releases and MPC decisions.",
)

CBRT_MPC_SUMMARIES = SourceDefinition(
    source_id="cbrt_mpc_summaries",
    title="CBRT MPC Meeting Summaries",
    category="text",
    source_type="official_html_listing",
    url="https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB%2BEN/MPC/MPC%2BMeeting%2BSummaries",
    raw_path=Path("text/cbrt_mpc_summaries.html"),
    notes="Official listing page for MPC meeting summaries.",
)

SOURCE_REGISTRY: tuple[SourceDefinition, ...] = (
    CBRT_CONSUMER_PRICES,
    CBRT_MPC_DECISIONS,
    CBRT_MPC_SUMMARIES,
)


def sources_by_category(category: str) -> tuple[SourceDefinition, ...]:
    """Return sources for one registry category."""

    return tuple(source for source in SOURCE_REGISTRY if source.category == category)


def source_by_id(source_id: str) -> SourceDefinition:
    """Return one source definition by id."""

    for source in SOURCE_REGISTRY:
        if source.source_id == source_id:
            return source
    raise KeyError(f"Unknown source id: {source_id}")
