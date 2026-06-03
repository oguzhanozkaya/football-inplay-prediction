---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Map

The repository is command-first and centered on one root script.

```
/
├── data/
│   ├── raw/                    # Local ESPN Soccer dataset directories
│   └── processed/              # Model-ready dataset, metadata, vocabulary, split summaries
├── docs/                       # Documentation source
├── output/
│   ├── figures/                # Training and evaluation plots
│   ├── models/                 # PyTorch checkpoints
│   ├── predictions/            # Forecast outputs
│   └── reports/                # Metrics and generated summaries
├── fig.py                      # Download, preprocess, train, and evaluate pipeline
├── article.md                  # Article draft using generated outputs
├── justfile                    # Project command wrapper
├── pyproject.toml              # Python dependency config
├── uv.lock                     # Python dependency lock
└── zensical.toml               # Website configuration
```

Generated data and model artifacts live outside source-controlled code.

## Script Responsibilities

| Part of `fig.py`     | Responsibility                                                                                 |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| Download             | Reuse local ESPN raw data or download it from Kaggle with configured credentials               |
| Fixture preparation  | Build completed-match labels and chronological train/validation/test splits within each league |
| Event preparation    | Filter plays, key events, and commentary to information available through minute 45            |
| Feature construction | Build one first-half text field and one first-half numeric vector per match                    |
| Training             | Train the raw-PyTorch TextCNN plus numeric MLP classifier                                      |
| Evaluation           | Write predictions, metrics, per-class reports, error examples, and figures                     |

## Data Directories

| Path              | Purpose                                      | Git Policy |
| ----------------- | -------------------------------------------- | ---------- |
| `data/raw/`       | Local Kaggle ESPN Soccer dataset directories | ignored    |
| `data/processed/` | Leakage-safe model dataset and metadata      | ignored    |

The script writes `model_dataset.parquet`, `feature_metadata.json`, `text_vocabulary.json`, `league_split_summary.csv`, and `split_class_summary.csv` under `data/processed/`.

## Output Directories

| Path                  | Purpose                                                        | Git Policy         |
| --------------------- | -------------------------------------------------------------- | ------------------ |
| `output/models/`      | Trained PyTorch checkpoint                                     | ignored            |
| `output/predictions/` | Prediction CSV and Parquet outputs                             | ignored            |
| `output/reports/`     | Metrics, class reports, training summaries, and error examples | ignored            |
| `output/figures/`     | Training and evaluation plots                                  | ignored by default |

## Command Interface

| Recipe       | Command                    | Responsibility                           |
| ------------ | -------------------------- | ---------------------------------------- |
| `just sync`  | `uv sync`                  | Install or update the Python environment |
| `just run`   | `uv run python fig.py`     | Run the complete pipeline                |
| `just smoke` | constrained `fig.py` run   | Run a short CPU smoke pipeline           |
| `just check` | formatting and lint checks | Verify source and documentation style    |
| `just fix`   | format and lint fixes      | Apply automated formatting fixes         |
| `just ci`    | `just check`               | Run the current verification gate        |

### Deployment

Documentation is deployed through `.github/workflows/docs.yml`. On pushes to `main`, GitHub Pages builds the site with Zensical and publishes the generated `site/` directory.
