# Turkish Inflation Forecasting with Numeric and Text Signals

## Abstract

This project forecasts next-month Turkish CPI month-over-month inflation using a reproducible deep learning pipeline. Each forecast is made at the end of month `t` for CPI MoM in month `t + 1`. The pipeline combines macro-financial features with official central-bank text documents while enforcing chronological splits and feature cutoff rules. Models include simple baselines, classical machine-learning baselines, raw PyTorch numeric and text models, and a numeric-text fusion model.

Final result placeholders in this article should be filled after the extended training run.

## Problem Definition

The target is CPI MoM inflation. CPI YoY is used only as a lagged contextual feature, not as the forecast target. For every row, the model observes only data available by the forecast origin month.

## Data

Numeric data is downloaded from public reproducible sources:

- CBRT Consumer Prices page for CPI MoM and CPI YoY.
- CBRT public exchange-rate XML archive for month-end USD/TRY and EUR/TRY.
- FRED public CSV files for Brent crude oil, Turkish industrial production growth, and Turkish unemployment.

Text data comes from official CBRT MPC meeting decisions and meeting summaries. Documents are included in a forecast row only when their publication date is no later than the end of the forecast origin month.

## Feature Engineering

The processed dataset is written by `just preprocess` to `data/processed/model_dataset.parquet`.

Feature availability rules are conservative:

| Feature Group                          | Rule                                            |
| -------------------------------------- | ----------------------------------------------- |
| CPI history                            | Use month `t - 1` and earlier                   |
| FX and Brent                           | Use month `t` and earlier                       |
| Industrial production and unemployment | Use month `t - 2` and earlier                   |
| Text documents                         | Use documents published by the end of month `t` |

The text tokenizer is trained from scratch on the training split only. No pretrained language model, pretrained embedding, or external model API is used.

## Models

The implemented model set is:

| Model         | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| Last value    | Forecasts next CPI MoM as the latest available CPI MoM         |
| Rolling mean  | Uses recent CPI MoM average                                    |
| Ridge         | Linear numeric-feature baseline                                |
| Random Forest | Nonlinear classical numeric baseline                           |
| Numeric MLP   | Raw PyTorch tabular numeric model                              |
| Numeric GRU   | Raw PyTorch lag-structured numeric sequence model              |
| TextCNN       | Raw PyTorch text model with embeddings trained from scratch    |
| Fusion MLP    | Combines numeric MLP representation and TextCNN representation |

## Evaluation

The split is chronological. The test period is never used for fitting scalers, tokenizers, or model parameters.

Metrics:

- MAE is the primary metric.
- RMSE captures larger forecast misses.
- Direction accuracy measures whether the model predicts the direction of CPI MoM movement relative to the previous available CPI MoM.
- MAE delta vs last-value baseline reports improvement over the simplest benchmark.

## Results

Replace this section after the extended training run.

| Model         | Validation MAE | Test MAE | Test RMSE | Test Direction Accuracy |
| ------------- | -------------- | -------- | --------- | ----------------------- |
| Last value    | TODO           | TODO     | TODO      | TODO                    |
| Rolling mean  | TODO           | TODO     | TODO      | TODO                    |
| Ridge         | TODO           | TODO     | TODO      | TODO                    |
| Random Forest | TODO           | TODO     | TODO      | TODO                    |
| Numeric MLP   | TODO           | TODO     | TODO      | TODO                    |
| Numeric GRU   | TODO           | TODO     | TODO      | TODO                    |
| TextCNN       | TODO           | TODO     | TODO      | TODO                    |
| Fusion MLP    | TODO           | TODO     | TODO      | TODO                    |

Generated figures after `just plots`:

- `output/figures/cpi_history.png`
- `output/figures/feature_coverage.png`
- `output/figures/predictions_vs_actual.png`
- `output/figures/residuals.png`
- `output/figures/model_comparison.png`

## Reproducibility

Run the full pipeline:

```bash
just sync
just run
```

For a longer final training run:

```bash
TIF_EPOCHS=200 TIF_PATIENCE=20 just train
just evaluate
just plots
```

## Limitations

The current text corpus is limited to official CBRT MPC documents. Broader news data is not included. Some macro indicators have publication delays, so this implementation uses conservative lag rules. Final claims about model superiority should be made only after the long training run and test-period report are reviewed.
