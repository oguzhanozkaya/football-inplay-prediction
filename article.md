# Football In-Play Prediction with Text and Event Sequences

## Abstract

This project predicts the final outcome of a football match at minute 45 using a reproducible deep learning pipeline. The target is a three-class result: home win, draw, or away win. The model combines ESPN soccer commentary text with numerical event-stream features while enforcing a strict cutoff: no play, key event, commentary, or unsafe lineup information after minute 45 can enter the input.

Final result placeholders in this article should be filled after the extended training run.

## Problem Definition

Each row represents one completed match. The model observes information available through minute 45 and predicts the final result. This creates an in-play forecasting task rather than a pre-match prediction task.

## Data

Raw data comes from the local ESPN Soccer dataset under `data/raw/`:

- `base_data/fixtures.csv` for dates, teams, final scores, status, and labels.
- `plays_data/*.csv` for play-by-play events, clocks, text, scoring flags, teams, and coordinates.
- `keyEvents_data/*.csv` for important match events and event text.
- `commentary_data/*.csv` for minute-by-minute commentary.
- `lineup_data/*.csv` for safe formation and starter metadata.

Full-match team statistics, scrape-time standings, and season player aggregates are excluded from the first model to avoid leakage.

## Feature Engineering

The processed dataset is written by `just preprocess` to `data/processed/model_dataset.parquet`.

Feature availability rules are conservative:

| Source     | Rule                                                                                         |
| ---------- | -------------------------------------------------------------------------------------------- |
| Fixtures   | Final scores are used only for labels.                                                       |
| Plays      | Use first-half rows with parsed clock at or before minute 45.                                |
| Key events | Use first-half rows with parsed clock at or before minute 45.                                |
| Commentary | Use rows with parsed clock at or before minute 45; missing clocks are treated as early text. |
| Lineups    | Use formation and starter metadata; exclude winner fields and post-cutoff substitutions.     |

Each match is sliced into configured windows through minute 45. The command default uses `0-15`, `15-30`, and `30-45`; setting `FIP_WINDOW_MINUTES=5` creates 9 finer-grained windows. The text tokenizer is trained from scratch on the training split only. No pretrained language model, pretrained embedding, or external model API is used.

## Model

Only one architecture is trained: `FusionGRUClassifier`.

For each configured match-time window:

- a TextCNN encodes tokenized commentary and event text;
- a numeric projection encodes event counts, score state, key-event counts, coordinates, and safe lineup features;
- the two vectors are concatenated and projected into a fused window representation.

A GRU reads the fused window vectors. The final hidden state is passed to a three-class classifier for home/draw/away logits.

## Evaluation

The split is chronological. The test period is not used for fitting vocabulary, scalers, model parameters, or early stopping.

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

Generated figures after `just evaluate`:

- `output/figures/confusion_matrix.png`
- `output/figures/class_distribution.png`
- `output/figures/prediction_confidence.png`
- `output/figures/metric_comparison.png`
- `output/figures/training_loss_fusion_gru.png`

## Reproducibility

Run the full pipeline:

```bash
just sync
just run
```

Short CPU smoke run:

```bash
FIP_DEVICE=cpu FIP_EPOCHS=1 FIP_PATIENCE=1 just train
just evaluate
```

## Limitations

The current first-pass features use conservative event counts and coarse coordinate summaries. ESPN event semantics can vary across competitions, and clocks are normalized conservatively. Future work should add lagged pre-match team strength features, richer event taxonomy mappings, and calibration analysis after the base model is stable.
