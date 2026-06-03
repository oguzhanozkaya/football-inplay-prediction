import pandas as pd

import tif.plots


def test_best_model_from_metrics_prefers_validation_mae() -> None:
    metrics = pd.DataFrame(
        {
            "split": ["validation", "validation", "test"],
            "model_name": ["slow", "fast", "test_best"],
            "mae": [2.0, 1.0, 0.1],
        }
    )

    assert tif.plots._best_model_from_metrics(metrics) == "fast"
