# Specification

## Project Scope

This project predicts the final result of a football match from in-play information available through minute 60. The target is a three-class outcome: home win, draw, or away win.

The pipeline is a single root script, `fig.py`. It downloads the Kaggle ESPN Soccer dataset when local raw files are missing, builds leakage-safe 5-minute numeric windows through minute 60, trains one raw-PyTorch Temporal CNN classifier, and writes evaluation artifacts.

## Forecast Target

| Field          | Definition                                                                                |
| -------------- | ----------------------------------------------------------------------------------------- |
| Target         | Final match outcome: `home`, `draw`, or `away`                                            |
| Forecast time  | Minute 60                                                                                 |
| Forecast rule  | Use only plays, key events, commentary, and safe lineup data available through minute 60. |
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

Each observation in `data/processed/model_dataset.parquet` represents one completed match sampled at minute 60.

| Field Group  | Required Content                                                                      |
| ------------ | ------------------------------------------------------------------------------------- |
| Time keys    | Match date, split, league, league-season key, event id                                |
| Target       | Final result label and numeric class id                                               |
| Numeric      | One 5-minute numeric feature sequence per match, from `0-5` through `55-60`, including safe pre-match team strength features |
| Leakage rule | No play, key event, commentary, or unsafe lineup data after minute 60 may enter input |

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

Pipeline and model settings are Python constants near the top of `fig.py`. Edit those constants before running the script.

| Constant                   | Default     | Description                                      |
| -------------------------- | ----------- | ------------------------------------------------ |
| `SEED`                     | `67`        | Random seed.                                     |
| `EPOCHS`                   | `300`       | Maximum training epochs.                         |
| `PATIENCE`                 | `30`        | Early-stopping patience.                         |
| `BATCH_SIZE`               | `1024`      | Mini-batch size.                                 |
| `LEARNING_RATE`            | `0.001`     | AdamW learning rate.                             |
| `WEIGHT_DECAY`             | `0.001`     | AdamW weight decay.                              |
| `EARLY_STOPPING_MIN_DELTA` | `0.001`     | Minimum validation-loss improvement.             |
| `DEVICE`                   | `cuda`      | Training device: `cuda`, `cpu`, or `auto`.       |
| `CUTOFF_MINUTE`            | `60`        | Last match minute allowed in model inputs.       |
| `WINDOW_MINUTES`           | `5`         | Match-time window size.                          |
| `NUMERIC_PROJECTION_SIZE`  | `128`       | Per-window numeric projection width.             |
| `TEMPORAL_CHANNEL_COUNT`   | `128`       | Temporal CNN channel count.                      |
| `TEMPORAL_KERNEL_SIZE`     | `3`         | Temporal convolution kernel size.                |
| `TEMPORAL_BLOCK_COUNT`     | `2`         | Residual temporal convolution block count.       |
| `FUSION_HIDDEN_SIZE`       | `128`       | Final fusion classifier width.                   |
| `MLP_HIDDEN_SIZE`          | `256`       | Optional flattened MLP diagnostic hidden width.  |
| `DROPOUT`                  | `0.15`      | Numeric and fusion dropout.                      |
| `LABEL_SMOOTHING`          | `0.05`      | Cross-entropy label smoothing.                   |
| `NORMALIZATION_GROUPS`     | `8`         | GroupNorm group count for temporal blocks.       |
| `USE_CLASS_WEIGHTS`        | `True`      | Weight classes inversely to train frequencies.   |
| `MODEL_TYPE`               | `tcn`       | Active model: `tcn` or diagnostic `mlp`.         |
| `TEAM_FORM_WINDOW`         | `5`         | Recent-match window for rolling team form.       |
| `ELO_INITIAL_RATING`       | `1500.0`    | Initial team Elo rating.                         |
| `ELO_K_FACTOR`             | `20.0`      | Elo update step size.                            |
| `ELO_HOME_ADVANTAGE`       | `60.0`      | Home advantage added in Elo expectation.         |
| `DATALOADER_WORKERS`       | `4`         | DataLoader worker count.                         |
| `MIXED_PRECISION`          | `True`      | Use CUDA automatic mixed precision.              |
| `COMPILE_MODEL`            | `False`     | Compile the PyTorch model before training.       |
| `MATCH_LIMIT`              | `0`         | Optional smoke/debug match limit; `0` means all. |
| `USE_PREPROCESS_CACHE`     | `True`      | Reuse matching `model_dataset.parquet` cache.    |
| `FORCE_REPROCESS`          | `False`     | Ignore cache and rebuild processed data.         |
| `CHECKPOINT_INTERVAL_EPOCHS` | `25`      | Save model checkpoint/history every N epochs; `0` disables periodic checkpoints. |

## Success Criteria

The project is complete when `just run` can download or reuse local ESPN data, build leakage-safe 5-minute numeric windows, train the single numeric Temporal CNN classifier, evaluate chronological league-aware splits, and generate report-ready outputs.
