# Football In-Play Prediction with Numeric Event Windows

## Abstract

This project predicts the final outcome of a football match at minute 60 using a reproducible deep learning pipeline. The target is a three-class result: home win, draw, or away win. The model converts ESPN soccer event streams into 5-minute numeric windows while enforcing a strict cutoff: no play, key event, commentary, or unsafe lineup information after minute 60 can enter the input.

Final result placeholders in this article should be filled after the extended training run.

## Problem Definition

Each row represents one completed match. The model observes information available through minute 60 and predicts the final result. This creates an in-play forecasting task rather than a pre-match prediction task.

## Data

Raw data comes from the ESPN Soccer dataset. `fig.py` downloads it from Kaggle when the expected `data/raw/` directories are missing:

- `base_data/fixtures.csv` for dates, teams, final scores, status, and labels.
- `plays_data/*.csv` for play-by-play events, clocks, text, scoring flags, teams, and coordinates.
- `keyEvents_data/*.csv` for important match events and event text.
- `commentary_data/*.csv` for minute-by-minute commentary.
- `lineup_data/*.csv` for safe formation and starter metadata.

Full-match team statistics, scrape-time standings, and season player aggregates are excluded from the first model to avoid leakage.

## Feature Engineering

The processed dataset is written by `python fig.py` to `data/processed/model_dataset.parquet`.

Feature availability rules are conservative:

| Source     | Rule                                                                                         |
| ---------- | -------------------------------------------------------------------------------------------- |
| Fixtures   | Final scores are used only for labels.                                                       |
| Plays      | Use rows with parsed clock at or before minute 60.                                           |
| Key events | Use rows with parsed clock at or before minute 60.                                           |
| Commentary | Use rows with parsed clock at or before minute 60; missing clocks are treated as early text. |
| Lineups    | Use formation and starter metadata; exclude winner fields and post-cutoff substitutions.     |

Each match is represented as 12 five-minute windows from `0-5` through `55-60`. Windows contain current event counts, cumulative match state, score-state flags, score and event differentials, coordinate summaries, event-type counts, commentary counts, safe lineup features, and leakage-safe pre-match team-strength features. Text is not used as an active model input.

## Model

Only one architecture is trained: a numeric Temporal CNN classifier.

For each match:

- a linear layer projects every numeric window into a learned representation;
- residual 1D convolution blocks model temporal patterns across the 12 windows;
- max pooling, mean pooling, and the last window state are concatenated for classification.

No GRU, LSTM, TextCNN, or text embedding branch is used.

## Evaluation

The split is chronological within each league-season key. Every league with enough matches contributes early matches to train, later matches to validation, and latest matches to test. The test period is not used for fitting scalers, model parameters, or early stopping.

Metrics:

- Accuracy measures overall correct predictions.
- Macro F1 measures class-balanced quality.
- Log loss measures probability quality.
- Per-class precision, recall, and F1 show class-specific behavior.
- Confusion matrix shows error structure.

## Results

Replace this section after the extended training run.

| Split      | Accuracy | Macro F1 | Log Loss |
| ---------- | -------- | -------- | -------- |
| Validation | TODO     | TODO     | TODO     |
| Test       | TODO     | TODO     | TODO     |

Generated figures after `just run`:

- `output/figures/confusion_matrix.png`
- `output/figures/class_distribution.png`
- `output/figures/prediction_confidence.png`
- `output/figures/metric_comparison.png`
- `output/figures/training_loss.png`

## Reproducibility

Run the full pipeline:

```bash
just sync
just run
```

Short CPU smoke run:

```bash
just smoke
```

## Limitations

The current first-pass features use conservative event counts and coarse coordinate summaries. ESPN event semantics can vary across competitions, and clocks are normalized conservatively. Future work should add lagged pre-match team strength features, richer event taxonomy mappings, and calibration analysis after the base model is stable.
