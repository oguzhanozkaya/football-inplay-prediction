---
description: Tasks, priorities, known bugs, and the project roadmap.
---

# Roadmap

## Status Overview

The repository is migrated to football in-play prediction. The current pipeline validates local ESPN Soccer data, builds leakage-safe minute-45 match sequences, trains one hybrid fusion GRU classifier, and evaluates home/draw/away classification outputs.

| Area                             | Status      |
| -------------------------------- | ----------- |
| Documentation structure          | Implemented |
| Python package structure         | Implemented |
| Command workflow                 | Implemented |
| Raw data validation              | Implemented |
| Minute-45 sequence preprocessing | Implemented |
| Single fusion model              | Implemented |
| Evaluation reports               | Implemented |
| Article draft                    | Implemented |

## Active Tasks

| Priority | Task                             | Exit Criteria                                                                   |
| -------- | -------------------------------- | ------------------------------------------------------------------------------- |
| High     | Run extended training            | `FIP_EPOCHS=80 just train` completes on the selected device                     |
| High     | Regenerate reports and figures   | `just evaluate` reflects final trained model predictions                        |
| High     | Fill article result placeholders | Final accuracy, macro F1, figures, and interpretation are added to `article.md` |

## Delivered Capabilities

- Flat package exists under `src/fip/`.
- Console entrypoints are registered as `fip-download`, `fip-preprocess`, `fip-train`, and `fip-evaluate`.
- `just` exposes pipeline, quality, test, and documentation commands.
- `just download` validates local ESPN raw data directories and writes raw source manifests.
- `just preprocess` writes completed fixtures, minute-45 model sequences, feature metadata, vocabulary, and split summaries.
- Preprocessing slices plays, key events, and commentary to 5-minute windows through minute 45.
- Preprocessing builds the text vocabulary from the train split only.
- Full-match team statistics, standings snapshots, and scrape-time player aggregates are excluded from first-model inputs.
- `just train` trains only the single fusion GRU classifier.
- `just train` writes a PyTorch checkpoint, predictions, training history, and a training-loss figure.
- `just evaluate` writes JSON/Markdown classification metrics, a per-class report, high-confidence error examples, and evaluation figures.
- Tests cover source registry integrity, manifest writing, minute cutoff behavior, sequence shapes, model forward shape, config parsing, and metrics.

## Limitations

- ESPN clocks and event semantics are normalized conservatively; some commentary rows with missing clocks are treated as pre-match or early text.
- Team strength features from standings and player aggregates are not used until explicit lagging avoids leakage.
- The current first-pass features use event counts and coarse spatial summaries, not a full soccer event ontology.
- Final reported results depend on the selected training run and must be regenerated after preprocessing.

## Future Work

- Add lagged pre-match team form and standings features with strict date cutoffs.
- Add richer event taxonomy mappings from `keyEventDescription.csv`.
- Add next-goal prediction as a second task after the final-outcome classifier is stable.
- Add calibration analysis and probability reliability reports.
- Add league- or season-specific validation slices for robustness analysis.
