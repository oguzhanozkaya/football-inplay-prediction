import pandas as pd
import pytest

import fip.evaluate


def test_build_metrics_table_computes_classification_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "split": ["test", "test", "test"],
            "actual": [0, 1, 2],
            "prediction": [0, 2, 2],
            "prob_home": [0.8, 0.2, 0.1],
            "prob_draw": [0.1, 0.3, 0.1],
            "prob_away": [0.1, 0.5, 0.8],
            "confidence": [0.8, 0.5, 0.8],
        }
    )

    metrics = fip.evaluate.build_metrics_table(predictions)

    row = metrics.iloc[0]
    assert row["accuracy"] == pytest.approx(2 / 3)
    assert row["row_count"] == 3
    assert row["log_loss"] > 0


def test_classification_metrics_handles_missing_predicted_class() -> None:
    predictions = pd.DataFrame(
        {
            "actual": [0, 1, 2],
            "prediction": [0, 0, 0],
            "prob_home": [0.8, 0.8, 0.8],
            "prob_draw": [0.1, 0.1, 0.1],
            "prob_away": [0.1, 0.1, 0.1],
            "confidence": [0.8, 0.8, 0.8],
        }
    )

    metrics = fip.evaluate.classification_metrics(predictions)

    assert metrics["macro_f1"] >= 0
