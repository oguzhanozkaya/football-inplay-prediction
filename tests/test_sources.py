from turkish_inflation_forecasting.data.sources import SOURCE_REGISTRY, source_by_id, sources_by_category


def test_source_registry_ids_are_unique() -> None:
    source_ids = [source.source_id for source in SOURCE_REGISTRY]

    assert len(source_ids) == len(set(source_ids))


def test_source_registry_contains_initial_numeric_and_text_sources() -> None:
    assert source_by_id("cbrt_consumer_prices").category == "numeric"
    assert {source.source_id for source in sources_by_category("text")} == {
        "cbrt_mpc_decisions",
        "cbrt_mpc_summaries",
    }
