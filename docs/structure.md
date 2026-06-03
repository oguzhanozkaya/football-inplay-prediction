---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Map

The target repository structure is command-first and package-based. Generated data and model artifacts live outside source-controlled code.

```
/
├── .github/                    # GitHub workflows (currently docs deployment)
├── data/                       # Reproducible local datasets, ignored by git
│   ├── raw/                    # Downloaded or scraped source files
│   ├── interim/                # Cleaned intermediate files
│   └── processed/              # Model-ready datasets
├── docs/                       # Documentation source
├── output/                     # Generated reports, figures, predictions, and checkpoints
│   ├── figures/                # Article and report plots
│   ├── models/                 # PyTorch checkpoints
│   ├── predictions/            # Forecast outputs
│   └── reports/                # Metrics and generated summaries
├── src/                        # Python source tree
│   └── tif/
│       ├── __init__.py         # Package marker
│       ├── utils.py            # Shared paths, source registry, constants, helper functions
│       ├── download.py         # Download entrypoint and raw source manifest writing
│       ├── preprocess.py       # Preprocess entrypoint and interim table construction
│       ├── features.py         # Feature entrypoint and processed dataset construction
│       ├── train.py            # Training entrypoint, baselines, and PyTorch models
│       ├── evaluate.py         # Evaluation entrypoint and metrics reports
│       └── plots.py            # Plot entrypoint and report figure generation
├── tests/                      # One test file per source file
├── .gitattributes              # Git attributes
├── .gitignore                  # Git ignore rules
├── README.md                   # Project readme
├── flake.nix                   # Reproducible development shell
├── justfile                    # Project command wrapper
├── pyproject.toml              # Python package config
├── uv.lock                     # Python dependency lock
└── zensical.toml               # Website configuration
```

The package is intentionally flat: each pipeline stage has one importable source file and one console entrypoint. Cross-stage code lives in `tif.utils`; stage modules should import it as a module:

```python
import tif.utils

max_tokens = tif.utils.MAX_TOKENS
```

## Python Package (`src/tif/`)

| Path            | Responsibility                                                                  |
| --------------- | ------------------------------------------------------------------------------- |
| `utils.py`      | Project paths, source registry, shared constants, CBRT/FRED helper functions    |
| `download.py`   | Raw source downloads, CBRT FX archive downloads, text document downloads        |
| `preprocess.py` | CPI target parsing, numeric normalization, text body extraction, interim writes |
| `features.py`   | Leakage-safe lag, rolling, text-window, tokenizer, and split feature generation |
| `train.py`      | Baselines, classical models, PyTorch model definitions, training, predictions   |
| `evaluate.py`   | MAE, RMSE, direction accuracy, and baseline delta reports                       |
| `plots.py`      | Static report figures for CPI history, predictions, residuals, and comparison   |

Tests mirror this source layout. Each source file has one corresponding test file under `tests/`, for example `src/tif/preprocess.py` is covered by `tests/test_preprocess.py`.

## Data Directories

| Path              | Purpose                                              | Git Policy |
| ----------------- | ---------------------------------------------------- | ---------- |
| `data/raw/`       | Source-native downloaded files and scraped documents | ignored    |
| `data/interim/`   | Cleaned source tables before final modeling joins    | ignored    |
| `data/processed/` | Leakage-safe model-ready datasets                    | ignored    |
| `tests/`          | Deterministic unit tests for source files            | committed  |

Full datasets should not be committed. The repository should commit the code and source definitions needed to reproduce them.

The download stage writes `source_registry.json`, `source_manifest.json`, official CBRT CPI HTML, official CBRT FX XML snapshots, public FRED CSV snapshots, official CBRT MPC listing HTML, and official CBRT MPC document HTML pages under `data/raw/`. The preprocess stage writes `cpi_mom.parquet`, `numeric_series.parquet`, `monthly_numeric.parquet`, and `text_documents.parquet` under `data/interim/`. The features stage writes `model_dataset.parquet`, `feature_metadata.json`, `text_vocabulary.json`, and `split_summary.json` under `data/processed/`.

## Output Directories

| Path                  | Purpose                                             | Git Policy         |
| --------------------- | --------------------------------------------------- | ------------------ |
| `output/models/`      | Trained PyTorch checkpoints and model summaries     | ignored            |
| `output/predictions/` | Forecast CSV and Parquet outputs                    | ignored            |
| `output/reports/`     | Metrics, comparison tables, and generated summaries | ignored            |
| `output/figures/`     | Generated plots for reports and article drafts      | ignored by default |

Selected small figures may be copied into documentation assets only when they are intentionally referenced by the project report.

## Documentation (`docs/`)

| Path              | Purpose                                                 |
| ----------------- | ------------------------------------------------------- |
| `_assets`         | Documentation assets, extra.css, extra.js, and logo.svg |
| `index.md`        | Documentation home                                      |
| `spec.md`         | Usage guide and product specifications                  |
| `development.md`  | Development rules and workflow                          |
| `structure.md`    | Repo Map, project structure, and deployment             |
| `architecture.md` | System architecture, data flow, and model design        |
| `roadmap.md`      | Plan, status, and roadmap                               |

## Command Interface

Project commands are standardized through `justfile`. Recipes call stage-specific console entrypoints through `uv run` so development, automation, and documentation use the same commands.

Stage-level command mapping:

| Recipe            | Console entrypoint               | Responsibility                                  |
| ----------------- | -------------------------------- | ----------------------------------------------- |
| `just sync`       | `uv sync`                        | Install or update the Python environment        |
| `just download`   | `tif-download`                   | Download numeric data and text sources          |
| `just preprocess` | `tif-preprocess`                 | Clean raw sources and build interim tables      |
| `just features`   | `tif-features`                   | Build model-ready numeric and text features     |
| `just train`      | `tif-train`                      | Train baselines and deep learning models        |
| `just evaluate`   | `tif-evaluate`                   | Evaluate models on chronological splits         |
| `just plots`      | `tif-plots`                      | Generate figures for reports and article drafts |
| `just run`        | stage recipe chain               | Run the complete pipeline                       |
| `just check`      | formatting and lint commands     | Run formatting and lint checks                  |
| `just fix`        | formatting and lint fix commands | Apply automated formatting and lint fixes       |
| `just ci`         | `just check` and `just test`     | Run the full verification gate                  |

### Deployment

Documentation is deployed through `.github/workflows/docs.yml`. On pushes to `main`, GitHub
Pages builds the site with Zensical and publishes the generated `site/` directory.
