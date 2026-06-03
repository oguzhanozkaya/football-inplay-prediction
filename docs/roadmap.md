---
description: Tasks, priorities, known bugs, and the project roadmap.
---

# Roadmap

## Status Overview

The repository is implemented as a single-script football in-play prediction pipeline. The current pipeline downloads or validates ESPN Soccer data, builds leakage-safe 5-minute numeric windows through minute 60, trains one numeric Temporal CNN classifier, and evaluates home/draw/away classification outputs.

| Area                             | Status      |
| -------------------------------- | ----------- |
| Documentation structure          | Implemented |
| Single root script               | Implemented |
| Command workflow                 | Implemented |
| Kaggle raw-data download         | Implemented |
| Raw data validation              | Implemented |
| Minute-60 preprocessing          | Implemented |
| League-aware chronological split | Implemented |
| Single numeric Temporal CNN      | Implemented |
| Evaluation reports               | Implemented |
| Article draft                    | Implemented |

## Active Tasks

| Priority | Task                             | Exit Criteria                                                                   |
| -------- | -------------------------------- | ------------------------------------------------------------------------------- |
| High     | Run extended training            | `just run` completes on the selected device                                     |
| High     | Regenerate reports and figures   | `output/reports/` and `output/figures/` contain final run artifacts             |
| High     | Fill article result placeholders | Final accuracy, macro F1, figures, and interpretation are added to `article.md` |

## Delivered Capabilities

- `fig.py` runs the complete pipeline from download through evaluation.
- `just run` executes `uv run python fig.py`.
- `just smoke` runs a short CPU end-to-end pipeline through a direct `Config` override.
- The script downloads the Kaggle ESPN Soccer dataset when `data/raw/` is missing required directories.
- Preprocessing writes `model_dataset.parquet`, metadata, and split summaries.
- Preprocessing slices plays, key events, and commentary through minute 60 only.
- Preprocessing adds leakage-safe rolling team form and Elo-like pre-match strength features.
- Splits are assigned chronologically inside each league-season key.
- Full-match team statistics, standings snapshots, and scrape-time player aggregates are excluded from first-model inputs.
- The model has no GRU, LSTM, TextCNN, or text embedding branch.
- Training writes a PyTorch checkpoint, predictions, training history, and a training-loss figure.
- Evaluation writes JSON/Markdown classification metrics, a per-class report, high-confidence error examples, and evaluation figures.

## Limitations

- ESPN clocks and event semantics are normalized conservatively; some commentary rows with missing clocks are treated as pre-match or early text.
- Team strength features from standings and player aggregates are not used until explicit lagging avoids leakage.
- The current first-pass features use event counts and coarse spatial summaries, not a full soccer event ontology.
- Final reported results depend on the selected training run and must be regenerated after preprocessing.

## Future Work

- Add lagged pre-match team form and standings features with strict date cutoffs.
- Add richer event taxonomy mappings from `keyEventDescription.csv`.
- Add learning-rate warmup, cosine decay, and class weighting experiments.
- Add calibration analysis and probability reliability reports.
- Add league- or season-specific validation slices for robustness analysis.
