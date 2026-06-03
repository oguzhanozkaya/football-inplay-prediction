import pandas as pd
import pytest

import tif.evaluate


def test_build_metrics_table_adds_baseline_delta() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["last_value", "last_value", "ridge", "ridge"],
            "model_type": ["baseline", "baseline", "classical", "classical"],
            "split": ["test", "test", "test", "test"],
            "actual_cpi_mom_percent": [2.0, 3.0, 2.0, 3.0],
            "previous_cpi_mom_percent": [1.0, 4.0, 1.0, 4.0],
            "prediction_cpi_mom_percent": [1.0, 4.0, 2.2, 2.8],
        }
    )

    metrics = tif.evaluate.build_metrics_table(predictions)

    ridge = metrics.set_index("model_name").loc["ridge"]
    assert ridge["mae"] == pytest.approx(0.2)
    assert ridge["mae_improvement_vs_last_value"] == pytest.approx(0.8)
