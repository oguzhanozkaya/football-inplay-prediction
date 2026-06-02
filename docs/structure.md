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
│   └── turkish_inflation_forecasting/
│       ├── config.py           # Paths and shared configuration
│       ├── pipeline.py         # Shared stage functions
│       ├── entrypoints/        # Stage-specific console entrypoints
│       ├── data/               # Downloads, scraping, preprocessing, alignment
│       ├── features/           # Numeric and text feature generation
│       ├── models/             # Baselines and PyTorch model definitions
│       ├── training/           # Training loops, checkpoints, splits
│       ├── evaluation/         # Metrics and backtesting
│       ├── visualization/      # Plot generation
│       └── utils/              # Shared utilities
├── tests/                      # Tests and small fixtures
├── .gitattributes              # Git attributes
├── .gitignore                  # Git ignore rules
├── README.md                   # Project readme
├── flake.nix                   # Reproducible development shell
├── justfile                    # Project command wrapper
├── pyproject.toml              # Python package config
├── uv.lock                     # Python dependency lock
└── zensical.toml               # Website configuration
```

Some directories may receive additional modules as implementation reaches the corresponding stage. The layout above is the target structure the codebase should converge to.

## Python Package (`src/turkish_inflation_forecasting/`)

The project uses a named package under `src/` instead of placing importable modules directly in `src/`. This keeps imports stable in tests, console entrypoints, and future packaging metadata.

| Path             | Responsibility                                                                         |
| ---------------- | -------------------------------------------------------------------------------------- |
| `config.py`      | Defines project paths, default file names, and shared constants                        |
| `pipeline.py`    | Contains shared stage behavior used by the console entrypoints                         |
| `entrypoints/`   | Defines one console entrypoint per stage without subcommand flags                      |
| `data/`          | Downloads source files, scrapes text, normalizes data, and aligns monthly observations |
| `features/`      | Builds lagged, rolling, transformed, tokenized, and windowed features                  |
| `models/`        | Contains baselines, numeric sequence models, text encoders, and fusion models          |
| `training/`      | Contains PyTorch training loops, early stopping, checkpoints, and chronological splits |
| `evaluation/`    | Calculates metrics, backtests models, and compares baselines                           |
| `visualization/` | Produces plots used by reports, documentation, and the article                         |
| `utils/`         | Shared logging, seed control, serialization, and small helpers                         |

The package name is `turkish_inflation_forecasting`. The import path looks like:

```python
from turkish_inflation_forecasting.data import preprocess
```

Current data modules:

| Path                 | Responsibility                                                                       |
| -------------------- | ------------------------------------------------------------------------------------ |
| `data/sources.py`    | Source registry for official numeric and text sources                                |
| `data/download.py`   | Raw HTML downloads, document page downloads, registry snapshot, and manifest writing |
| `data/cpi.py`        | CBRT Consumer Prices parsing and CPI MoM target construction                         |
| `data/text.py`       | CBRT text listing metadata and document body extraction                              |
| `data/preprocess.py` | Stage coordinator that writes initial interim tables                                 |

## Data Directories

| Path              | Purpose                                              | Git Policy |
| ----------------- | ---------------------------------------------------- | ---------- |
| `data/raw/`       | Source-native downloaded files and scraped documents | ignored    |
| `data/interim/`   | Cleaned source tables before final modeling joins    | ignored    |
| `data/processed/` | Leakage-safe model-ready datasets                    | ignored    |
| `tests/fixtures/` | Tiny deterministic files used by tests               | committed  |

Full datasets should not be committed. The repository should commit the code and source definitions needed to reproduce them.

The initial download stage writes `source_registry.json`, `source_manifest.json`, official CBRT CPI HTML, official CBRT MPC listing HTML, and official CBRT MPC document HTML pages under `data/raw/`. The initial preprocess stage writes `cpi_mom.parquet` and `text_documents.parquet` under `data/interim/`.

## Output Directories

| Path                  | Purpose                                             | Git Policy         |
| --------------------- | --------------------------------------------------- | ------------------ |
| `output/models/`      | Trained PyTorch checkpoints                         | ignored            |
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
| `just download`   | `turkish-inflation-download`     | Download numeric data and text sources          |
| `just preprocess` | `turkish-inflation-preprocess`   | Clean raw sources and build interim tables      |
| `just features`   | `turkish-inflation-features`     | Build model-ready numeric and text features     |
| `just train`      | `turkish-inflation-train`        | Train baselines and deep learning models        |
| `just evaluate`   | `turkish-inflation-evaluate`     | Evaluate models on chronological splits         |
| `just plots`      | `turkish-inflation-plots`        | Generate figures for reports and article drafts |
| `just run`        | `turkish-inflation-run`          | Run the complete pipeline                       |
| `just check`      | formatting and lint commands     | Run formatting and lint checks                  |
| `just fix`        | formatting and lint fix commands | Apply automated formatting and lint fixes       |
| `just ci`         | `just check` and `just test`     | Run the full verification gate                  |

### Deployment

Documentation is deployed through `.github/workflows/docs.yml`. On pushes to `main`, GitHub
Pages builds the site with Zensical and publishes the generated `site/` directory.
