---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Map

The repository is command-first and package-based. Generated data and model artifacts live outside source-controlled code.

```
/
├── data/
│   ├── raw/                    # Local ESPN Soccer dataset directories
│   └── processed/              # Model-ready fixtures, sequences, metadata, vocabulary
├── docs/                       # Documentation source
├── output/
│   ├── figures/                # Training and evaluation plots
│   ├── models/                 # PyTorch checkpoints
│   ├── predictions/            # Forecast outputs
│   └── reports/                # Metrics and generated summaries
├── src/
│   └── fip/
│       ├── __init__.py         # Package marker
│       ├── utils.py            # Shared paths, source registry, constants
│       ├── download.py         # Local ESPN raw-data validation and manifests
│       ├── preprocess.py       # Minute-45 sequence construction
│       ├── train.py            # Single fusion GRU classifier
│       └── evaluate.py         # Classification metrics, reports, and figures
├── tests/                      # One test file per source file
├── justfile                    # Project command wrapper
├── pyproject.toml              # Python package config
├── uv.lock                     # Python dependency lock
└── zensical.toml               # Website configuration
```

The package is intentionally flat. Cross-stage code lives in `fip.utils`; stage modules should import it as a module:

```python
import fip.utils

max_tokens = fip.utils.MAX_TOKENS_PER_WINDOW
```

## Python Package (`src/fip/`)

| Path            | Responsibility                                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------------------------- |
| `utils.py`      | Project paths, source registry, class labels, window constants                                                 |
| `download.py`   | Validate local ESPN raw directories and write raw manifests                                                    |
| `preprocess.py` | Build completed fixture labels, chronological splits, minute-45 text/numeric windows, vocabulary, and metadata |
| `train.py`      | Define and train the single TextCNN plus numeric projection plus GRU classifier                                |
| `evaluate.py`   | Compute classification metrics, Markdown reports, and static figures                                           |

Tests mirror this source layout. Each source file has one corresponding test file under `tests/`.

## Data Directories

| Path              | Purpose                                      | Git Policy |
| ----------------- | -------------------------------------------- | ---------- |
| `data/raw/`       | Local Kaggle ESPN Soccer dataset directories | ignored    |
| `data/processed/` | Leakage-safe model datasets and metadata     | ignored    |
| `tests/`          | Deterministic unit tests for source files    | committed  |

The preprocess stage writes `fixtures.parquet`, `model_dataset.parquet`, `feature_metadata.json`, `text_vocabulary.json`, and `split_summary.json` under `data/processed/`.

## Output Directories

| Path                  | Purpose                                                        | Git Policy         |
| --------------------- | -------------------------------------------------------------- | ------------------ |
| `output/models/`      | Trained PyTorch checkpoint                                     | ignored            |
| `output/predictions/` | Prediction CSV and Parquet outputs                             | ignored            |
| `output/reports/`     | Metrics, class reports, training summaries, and error examples | ignored            |
| `output/figures/`     | Training and evaluation plots                                  | ignored by default |

## Documentation (`docs/`)

| Path              | Purpose                                          |
| ----------------- | ------------------------------------------------ |
| `index.md`        | Documentation home                               |
| `spec.md`         | Usage guide and product specifications           |
| `development.md`  | Development rules and workflow                   |
| `structure.md`    | Repository structure and tooling reference       |
| `architecture.md` | System architecture, data flow, and model design |
| `roadmap.md`      | Plan, status, and roadmap                        |

## Command Interface

Stage-level command mapping:

| Recipe            | Console entrypoint               | Responsibility                                |
| ----------------- | -------------------------------- | --------------------------------------------- |
| `just sync`       | `uv sync`                        | Install or update the Python environment      |
| `just download`   | `fip-download`                   | Validate local ESPN raw data                  |
| `just preprocess` | `fip-preprocess`                 | Build model-ready minute-45 sequences         |
| `just train`      | `fip-train`                      | Train the single hybrid classifier            |
| `just evaluate`   | `fip-evaluate`                   | Evaluate predictions and generate figures     |
| `just run`        | stage recipe chain               | Run download, preprocess, train, and evaluate |
| `just check`      | formatting and lint commands     | Run formatting and lint checks                |
| `just fix`        | formatting and lint fix commands | Apply automated formatting and lint fixes     |
| `just ci`         | `just check` and `just test`     | Run the full verification gate                |

### Deployment

Documentation is deployed through `.github/workflows/docs.yml`. On pushes to `main`, GitHub Pages builds the site with Zensical and publishes the generated `site/` directory.
