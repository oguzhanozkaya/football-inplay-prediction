# Specification

## Project Scope

This project predicts the final result of a football match from in-play information available at minute 45. The target is a three-class outcome: home win, draw, or away win.

The project combines ESPN soccer commentary text with numerical event streams. Text captures match flow descriptions. Numerical inputs capture score state, event counts, key events, pitch coordinates, and safe lineup information.

## Forecast Target

| Field          | Definition                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------ |
| Target         | Final match outcome: `home`, `draw`, or `away`                                             |
| Forecast time  | Minute 45                                                                                  |
| Forecast rule  | Use only events, key events, commentary, and safe lineup data available through minute 45. |
| Split strategy | Chronological train, validation, and test periods                                          |
| Main metric    | Accuracy and macro F1                                                                      |

## Inputs

The raw dataset is the local Kaggle ESPN Soccer dataset placed under `data/raw/`.

| Directory                 | Purpose                                                                             | Used In First Model                     |
| ------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------- |
| `base_data/fixtures.csv`  | Match dates, teams, status, final scores, and labels                                | Yes                                     |
| `base_data/leagues.csv`   | League and season metadata                                                          | Yes                                     |
| `plays_data/*.csv`        | Play-by-play event text, clocks, teams, scoring flags, event types, and coordinates | Yes                                     |
| `keyEvents_data/*.csv`    | Important events with clocks, teams, text, event types, and coordinates             | Yes                                     |
| `commentary_data/*.csv`   | Minute-by-minute commentary text                                                    | Yes                                     |
| `lineup_data/*.csv`       | Formations, starters, positions, and substitution metadata                          | Safe pre-match fields only              |
| `playerStats_data/*.csv`  | Season player aggregates                                                            | No, reserved for future lagged features |
| `base_data/teamStats.csv` | Full-match team statistics                                                          | No, excluded to prevent leakage         |
| `base_data/standings.csv` | Scrape-time standings snapshots                                                     | No, reserved for future lagged features |

## Data Contract

Each observation in `data/processed/model_dataset.parquet` represents one completed match sampled at minute 45.

| Field Group      | Required Content                                                                      |
| ---------------- | ------------------------------------------------------------------------------------- |
| Time keys        | Match date, split, league, event id                                                   |
| Target           | Final result label and numeric class id                                               |
| Text sequence    | One tokenized text window per 5-minute interval from 0-45                             |
| Numeric sequence | One numeric feature vector per 5-minute interval from 0-45                            |
| Leakage rule     | No play, key event, commentary, or substitution after minute 45 may enter model input |

The default sequence has 9 windows: `0-5`, `5-10`, `10-15`, `15-20`, `20-25`, `25-30`, `30-35`, `35-40`, and `40-45`.

## Outputs

| Output              | Format             | Directory             |
| ------------------- | ------------------ | --------------------- |
| Raw source manifest | JSON               | `data/raw/`           |
| Model-ready dataset | Parquet            | `data/processed/`     |
| Model checkpoint    | PyTorch checkpoint | `output/models/`      |
| Predictions         | CSV and Parquet    | `output/predictions/` |
| Metrics and reports | JSON and Markdown  | `output/reports/`     |
| Figures             | PNG                | `output/figures/`     |

Current artifacts:

| Artifact                               | Command                       | Purpose                                                              |
| -------------------------------------- | ----------------------------- | -------------------------------------------------------------------- |
| `data/raw/source_registry.json`        | `just download`               | Snapshot of expected ESPN source groups                              |
| `data/raw/source_manifest.json`        | `just download`               | Local raw file availability, byte counts, and hashes                 |
| `data/processed/fixtures.parquet`      | `just preprocess`             | Completed fixtures with chronological splits and labels              |
| `data/processed/model_dataset.parquet` | `just preprocess`             | Minute-45 text and numeric sequences                                 |
| `data/processed/feature_metadata.json` | `just preprocess`             | Feature names, window settings, class labels, and tokenizer settings |
| `data/processed/text_vocabulary.json`  | `just preprocess`             | Train-split vocabulary built from project text                       |
| `output/models/fusion_gru.pt`          | `just train`                  | Single hybrid classifier checkpoint                                  |
| `output/predictions/predictions.*`     | `just train`                  | Home/draw/away probabilities and predictions                         |
| `output/reports/training_history.*`    | `just train`                  | Per-epoch loss and accuracy history                                  |
| `output/reports/metrics.*`             | `just evaluate`               | Accuracy, macro F1, log loss, and confidence metrics                 |
| `output/reports/class_report.md`       | `just evaluate`               | Per-class precision, recall, F1, and support                         |
| `output/reports/error_examples.md`     | `just evaluate`               | Highest-confidence wrong predictions                                 |
| `output/figures/*.png`                 | `just train`, `just evaluate` | Training and evaluation plots                                        |

## Usage

### Prerequisites

| Tool   | Requirement                    |
| ------ | ------------------------------ |
| uv     | Python environment management  |
| Python | Project-pinned version         |
| just   | Standardized project commands  |
| Nix    | Reproducible development shell |

### Run

Place the downloaded ESPN Soccer dataset directories under `data/raw/`, then run:

```bash
just sync
just run
```

Stage-level commands:

```bash
just download
just preprocess
just train
just evaluate
```

Short CPU smoke run:

```bash
FIP_DEVICE=cpu FIP_EPOCHS=1 FIP_PATIENCE=1 just train
```

## Configuration

| Variable                       | Default   | Description                                |
| ------------------------------ | --------- | ------------------------------------------ |
| `FIP_SEED`                     | `447`     | Random seed for deterministic setup.       |
| `FIP_EPOCHS`                   | `80`      | Maximum training epochs.                   |
| `FIP_PATIENCE`                 | `12`      | Early-stopping patience.                   |
| `FIP_BATCH_SIZE`               | `64`      | Mini-batch size.                           |
| `FIP_LEARNING_RATE`            | `0.0001`  | Adam learning rate.                        |
| `FIP_WEIGHT_DECAY`             | `0.0`     | Adam L2 weight decay.                      |
| `FIP_EARLY_STOPPING_MIN_DELTA` | `0.00001` | Minimum validation-loss improvement.       |
| `FIP_DEVICE`                   | `cuda`    | Training device: `cuda`, `cpu`, or `auto`. |
| `FIP_CUTOFF_MINUTE`            | `45`      | Last match minute allowed in model inputs. |
| `FIP_WINDOW_MINUTES`           | `5`       | Window size for match sequence steps.      |
| `FIP_MAX_TOKENS_PER_WINDOW`    | `64`      | Maximum token ids per time window.         |
| `FIP_MAX_VOCAB_SIZE`           | `20000`   | Maximum train-split vocabulary size.       |
| `FIP_TOP_PLAY_TYPES`           | `24`      | Number of play type count features.        |
| `FIP_TOP_KEY_EVENT_TYPES`      | `24`      | Number of key-event type count features.   |
| `FIP_TOP_FORMATIONS`           | `16`      | Number of formation indicators.            |
| `FIP_TEXT_EMBEDDING_DIM`       | `64`      | Random text embedding dimension.           |
| `FIP_TEXT_CHANNEL_COUNT`       | `48`      | TextCNN channels per kernel size.          |
| `FIP_TEXT_KERNEL_SIZES`        | `3,4,5`   | TextCNN kernel sizes.                      |
| `FIP_TEXT_DROPOUT`             | `0.20`    | Text branch dropout.                       |
| `FIP_NUMERIC_PROJECTION_SIZE`  | `64`      | Numeric projection width per window.       |
| `FIP_FUSION_HIDDEN_SIZE`       | `128`     | Per-window fusion projection width.        |
| `FIP_GRU_HIDDEN_SIZE`          | `128`     | GRU hidden state size.                     |
| `FIP_DROPOUT`                  | `0.20`    | Fusion and classifier dropout.             |

## Success Criteria

The project is complete when the repository can reproducibly validate local raw ESPN data, build leakage-safe minute-45 match sequences, train the single fusion GRU classifier, evaluate chronological validation and test periods, and generate reports and figures.
