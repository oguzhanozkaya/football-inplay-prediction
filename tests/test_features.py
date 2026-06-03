import pandas as pd

import tif.features


def test_build_model_dataset_uses_chronological_splits_and_safe_cpi_lags() -> None:
    target_months = pd.date_range("2020-01-01", "2022-12-01", freq="MS")
    cpi_target = pd.DataFrame(
        {
            "target_month_start": target_months,
            "target_month": target_months.strftime("%Y-%m"),
            "forecast_origin_month_start": target_months - pd.DateOffset(months=1),
            "forecast_origin_month": (target_months - pd.DateOffset(months=1)).strftime("%Y-%m"),
            "target_cpi_mom_percent": [float(index) for index in range(len(target_months))],
            "cpi_yoy_percent": [20.0 + index for index in range(len(target_months))],
            "source_id": "fixture",
        }
    )
    numeric_months = pd.date_range("2019-01-01", "2022-12-01", freq="MS")
    monthly_numeric = pd.DataFrame(
        {
            "month_start": numeric_months,
            "month": numeric_months.strftime("%Y-%m"),
            "usd_try_fx_selling_month_end": range(100, 100 + len(numeric_months)),
            "eur_try_fx_selling_month_end": range(200, 200 + len(numeric_months)),
            "fx_basket_try_month_end": range(150, 150 + len(numeric_months)),
            "brent_oil_usd_month_avg": range(50, 50 + len(numeric_months)),
            "brent_oil_usd_month_end": range(60, 60 + len(numeric_months)),
            "turkey_industrial_production_yoy_sa": range(10, 10 + len(numeric_months)),
            "turkey_unemployment_rate_sa": range(5, 5 + len(numeric_months)),
        }
    )
    text_documents = pd.DataFrame(
        {
            "document_id": ["doc_1", "doc_2"],
            "published_at": [pd.Timestamp("2021-01-15"), pd.Timestamp("2022-01-15")],
            "body_text": ["Inflation pressure remains high.", "Disinflation requires tight policy."],
        }
    )

    dataset, metadata, vocabulary, split_summary = tif.features.build_model_dataset(
        cpi_target, monthly_numeric, text_documents
    )

    assert dataset["target_month_start"].gt(dataset["forecast_origin_month_start"]).all()
    assert dataset[metadata["numeric_feature_columns"]].notna().all().all()
    assert set(split_summary) == {"train", "validation", "test"}
    assert (
        dataset[dataset["split"] == "train"]["forecast_origin_month_start"].max()
        < dataset[dataset["split"] == "validation"]["forecast_origin_month_start"].min()
    )
    row = dataset[dataset["forecast_origin_month"] == "2021-06"].iloc[0]
    expected_lag = cpi_target.set_index("target_month").loc["2021-05", "target_cpi_mom_percent"]
    assert row["cpi_mom_lag_1"] == expected_lag
    assert "inflation" in vocabulary
