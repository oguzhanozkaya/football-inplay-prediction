# Specification

## Project Scope

This project predicts the final result of a football match from in-play information available at minute 45. The target is a three-class outcome: home win, draw, or away win.

The pipeline is a single root script, `fig.py`. It downloads the Kaggle ESPN Soccer dataset when local raw files are missing, builds leakage-safe first-half features, trains one raw-PyTorch classifier, and writes evaluation artifacts.

## Forecast Target

| Field          | Definition                                                                                |
| -------------- | ----------------------------------------------------------------------------------------- |
| Target         | Final match outcome: `home`, `draw`, or `away`                                            |
| Forecast time  | Minute 45                                                                                 |
| Forecast rule  | Use only plays, key events, commentary, and safe lineup data available through minute 45. |
| Split strategy | Chronological train, validation, and test periods within each league-season key           |
| Main metric    | Accuracy and macro F1                                                                     |

## Inputs

The source dataset is [ESPN Soccer Data](https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data). `fig.py` uses Kaggle credentials to download it into `data/raw/` if the expected directories are absent.

| Directory                 | Purpose                                                                             | Used                                     |
| ------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------- |
| `base_data/fixtures.csv`  | Match dates, teams, status, final scores, and labels                                | Yes                                      |
| `base_data/leagues.csv`   | League and season metadata                                                          | Yes                                      |
| `plays_data/*.csv`        | Play-by-play event text, clocks, teams, scoring flags, event types, and coordinates | Yes                                      |
| `keyEvents_data/*.csv`    | Important events with clocks, teams, text, event types, and coordinates             | Yes                                      |
| `commentary_data/*.csv`   | Minute-by-minute commentary text                                                    | Yes                                      |
| `lineup_data/*.csv`       | Formations, starters, positions, and substitution metadata                          | Safe pre-match fields only               |
| `playerStats_data/*.csv`  | Season player aggregates                                                            | No, excluded until lagged features exist |
| `base_data/teamStats.csv` | Full-match team statistics                                                          | No, excluded to prevent leakage          |
| `base_data/standings.csv` | Scrape-time standings snapshots                                                     | No, excluded until lagged features exist |

## Data Contract

Each observation in `data/processed/model_dataset.parquet` represents one completed match sampled at minute 45.

| Field Group  | Required Content                                                                      |
| ------------ | ------------------------------------------------------------------------------------- |
| Time keys    | Match date, split, league, league-season key, event id                                |
| Target       | Final result label and numeric class id                                               |
| Text         | One tokenized first-half text sequence per match                                      |
| Numeric      | One first-half numeric feature vector per match                                       |
| Leakage rule | No play, key event, commentary, or unsafe lineup data after minute 45 may enter input |

## Outputs

| Output              | Format             | Directory             |
| ------------------- | ------------------ | --------------------- |
| Model-ready dataset | Parquet            | `data/processed/`     |
| Split summaries     | CSV                | `data/processed/`     |
| Model checkpoint    | PyTorch checkpoint | `output/models/`      |
| Predictions         | CSV and Parquet    | `output/predictions/` |
| Metrics and reports | JSON and Markdown  | `output/reports/`     |
| Figures             | PNG                | `output/figures/`     |

## Usage

Local run:

```bash
just sync
just run
```

Short CPU smoke run:

```bash
just smoke
```

Direct script run:

```bash
uv run python fig.py
```

## Kaggle Credentials

Local options:

- Place `kaggle.json` at `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`.
- Or export `KAGGLE_USERNAME` and `KAGGLE_KEY` in the shell.

Colab options:

- Upload `kaggle.json` to `/root/.kaggle/kaggle.json` and run `chmod 600 /root/.kaggle/kaggle.json`.
- Or store `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab secrets and assign them to environment variables before running `fig.py`.

Never commit `kaggle.json` or credential values.

## Configuration

| Variable                       | Default  | Description                                      |
| ------------------------------ | -------- | ------------------------------------------------ |
| `FIP_SEED`                     | `67`     | Random seed.                                     |
| `FIP_EPOCHS`                   | `1200`   | Maximum training epochs.                         |
| `FIP_PATIENCE`                 | `50`     | Early-stopping patience.                         |
| `FIP_BATCH_SIZE`               | `512`    | Mini-batch size.                                 |
| `FIP_LEARNING_RATE`            | `0.0001` | AdamW learning rate.                             |
| `FIP_WEIGHT_DECAY`             | `0.0001` | AdamW weight decay.                              |
| `FIP_EARLY_STOPPING_MIN_DELTA` | `0.001`  | Minimum validation-loss improvement.             |
| `FIP_DEVICE`                   | `cuda`   | Training device: `cuda`, `cpu`, or `auto`.       |
| `FIP_CUTOFF_MINUTE`            | `45`     | Last match minute allowed in model inputs.       |
| `FIP_MAX_TOKENS`               | `256`    | Maximum first-half text tokens per match.        |
| `FIP_MAX_VOCAB_SIZE`           | `6000`   | Maximum train-split vocabulary size.             |
| `FIP_TEXT_EMBEDDING_DIM`       | `64`     | Text embedding dimension.                        |
| `FIP_TEXT_CHANNEL_COUNT`       | `48`     | TextCNN channels per kernel size.                |
| `FIP_TEXT_KERNEL_SIZES`        | `3,4,5`  | TextCNN kernel sizes.                            |
| `FIP_NUMERIC_HIDDEN_SIZE`      | `128`    | Numeric MLP hidden width.                        |
| `FIP_FUSION_HIDDEN_SIZE`       | `128`    | Final fusion classifier width.                   |
| `FIP_DROPOUT`                  | `0.25`   | Numeric and fusion dropout.                      |
| `FIP_TEXT_DROPOUT`             | `0.25`   | Text branch dropout.                             |
| `FIP_DATALOADER_WORKERS`       | `4`      | DataLoader worker count.                         |
| `FIP_MIXED_PRECISION`          | `true`   | Use CUDA automatic mixed precision.              |
| `FIP_COMPILE_MODEL`            | `false`  | Compile the PyTorch model before training.       |
| `FIP_MATCH_LIMIT`              | `0`      | Optional smoke/debug match limit; `0` means all. |

## Success Criteria

The project is complete when `just run` can download or reuse local ESPN data, build leakage-safe first-half match features, train the single TextCNN plus numeric MLP classifier, evaluate chronological league-aware splits, and generate report-ready outputs.
