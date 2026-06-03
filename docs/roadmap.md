---
description: Tasks, priorities, known bugs, and the project roadmap.
---

# Roadmap

## Status Overview

The repository now contains the full command pipeline. The remaining project work is to run a longer training job, review the generated metrics and figures, and replace result placeholders in the article.

| Area                     | Status                          |
| ------------------------ | ------------------------------- |
| Documentation structure  | In progress                     |
| Python package structure | Foundation implemented          |
| Command workflow         | Foundation implemented          |
| Data download pipeline   | Macro foundation implemented    |
| Text collection pipeline | Metadata foundation implemented |
| Feature generation       | Implemented                     |
| Baselines                | Implemented                     |
| Deep learning models     | Implemented                     |
| Evaluation reports       | Implemented                     |
| Article draft            | Template implemented            |

## Active Tasks

| Priority | Task                             | Exit Criteria                                                    |
| -------- | -------------------------------- | ---------------------------------------------------------------- |
| High     | Run extended training            | `TIF_EPOCHS=200 TIF_PATIENCE=20 just train` completes            |
| High     | Regenerate reports and figures   | `just evaluate` reflects final trained models                    |
| High     | Fill article result placeholders | Final MAE, figures, and interpretation are added to `article.md` |

## Milestones

### Phase 1: Project Foundation

| Task                                           | Output                                                               |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| Document architecture and repository structure | Updated `docs/` pages                                                |
| Create package layout                          | `src/tif/`                                                           |
| Configure stage entrypoints                    | Stage-level commands callable through `uv run` and `just`            |
| Add tests directory and fixtures policy        | Deterministic tests can be added without committing full datasets    |
| Update ignore rules                            | Generated `data/` and `output/` artifacts stay out of source control |

### Phase 2: Data Pipeline

| Task                                     | Output                                                      |
| ---------------------------------------- | ----------------------------------------------------------- |
| Implement source definitions             | Reproducible numeric and text source registry               |
| Download CPI and macro-financial data    | Raw files under `data/raw/`                                 |
| Download or scrape official text sources | Raw text and metadata under `data/raw/`                     |
| Preprocess raw sources                   | Cleaned tables and model-ready data under `data/processed/` |
| Align monthly observations               | One row per forecast origin and target month                |

### Phase 3: Feature Engineering

| Task                              | Output                                    |
| --------------------------------- | ----------------------------------------- |
| Build CPI MoM target              | Leakage-safe target column                |
| Generate lag and rolling features | Numeric feature table                     |
| Train project tokenizer           | Vocabulary built only from project corpus |
| Convert text into model inputs    | Token IDs or monthly text windows         |
| Build model-ready dataset         | Parquet files under `data/processed/`     |

### Phase 4: Modeling

| Task                                  | Output                                                    |
| ------------------------------------- | --------------------------------------------------------- |
| Implement naive and rolling baselines | Baseline forecasts and metrics                            |
| Implement classical baselines         | Ridge and random forest comparisons                       |
| Implement numeric deep models         | MLP, 1D CNN, GRU or LSTM experiments                      |
| Implement text encoder                | TextCNN, BiGRU, or small Transformer trained from scratch |
| Implement fusion model                | Combined numeric and text forecast model                  |

### Phase 5: Evaluation and Reporting

| Task                           | Output                                                        |
| ------------------------------ | ------------------------------------------------------------- |
| Add chronological validation   | Train, validation, and test splits                            |
| Add rolling-origin backtesting | Forecast history across multiple origins                      |
| Generate metrics               | JSON and Markdown reports                                     |
| Generate figures               | CPI, features, predictions, residuals, and model comparisons  |
| Add interpretability outputs   | Feature importance, ablations, and text contribution analysis |

### Phase 6: Final Article and Polish

| Task                                   | Output                                            |
| -------------------------------------- | ------------------------------------------------- |
| Write article draft in `docs/index.md` | Reviewable project article                        |
| Add reproducibility instructions       | Clear clone, sync, run, and evaluate workflow     |
| Summarize limitations                  | Honest explanation of data and model constraints  |
| Prepare final GitHub presentation      | Clean README, docs, figures, and command workflow |

## Delivered Capabilities

- Initial documentation scaffold exists.
- `uv`, `just`, and Zensical are already present in the repository template.
- Python package foundation exists under `src/tif/`.
- Stage-specific console entrypoints are registered through `pyproject.toml`.
- `just` exposes pipeline, quality, test, and documentation commands.
- Generated `data/` and `output/` directory placeholders are tracked while generated contents are ignored.
- Initial smoke tests cover path setup and stage entrypoint behavior.
- Source registry exists for official CBRT CPI, CBRT FX, public FRED macro-financial series, and MPC text listing pages.
- `just download` writes raw official source snapshots, official CBRT MPC document pages, and a source manifest.
- `just download` writes official CBRT month-end FX XML snapshots and public FRED macro-financial CSV snapshots.
- `just preprocess` writes cleaned source tables, a leakage-safe processed dataset, chronological splits, tokenizer vocabulary, and feature metadata under `data/processed/`.
- `data/processed/numeric_series.parquet` includes normalized FX, Brent oil, industrial production, and unemployment observations.
- `data/processed/monthly_numeric.parquet` includes the monthly numeric feature base with FX, Brent oil, industrial production, and unemployment columns.
- `data/processed/text_documents.parquet` includes document metadata, publication dates, clean body text, and body length counts.
- `just train` trains last-value and rolling baselines, Ridge, Random Forest, numeric MLP, numeric GRU, TextCNN, and fusion models.
- `just evaluate` writes JSON/Markdown metrics, split/data/best-model/feature/text summaries, largest-error examples, and CPI history, feature coverage, prediction, residual, scatter, rolling-error, volatility-normalized, and model-comparison figures.
- Tests cover CPI target alignment, source registry integrity, manifest writing, official text metadata extraction, body extraction, FX parsing, FRED CSV parsing, monthly numeric aggregation, chronological feature construction, and evaluation metrics.

## Limitations

- Macro-financial indicators beyond the initial FX, Brent oil, industrial production, and unemployment sources still need expansion.
- Text sources are currently limited to official CBRT MPC decisions and summaries; broader news sources are not implemented yet.
- Final reported results depend on the extended training run selected for the final article.

## Future Work

- Add self-supervised text pretraining on the project corpus if it improves the text branch.
- Compare numeric-only, text-only, and fusion models.
- Add ablation studies for text source categories and numeric feature groups.
- Add a lightweight forecast command that loads the latest generated model and writes a next-month forecast.
- Add deployment-oriented documentation after the pipeline is stable.
