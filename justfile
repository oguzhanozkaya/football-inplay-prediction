# Project recipes
# Usage: just <recipe>
# Run `just --list` to see all available recipes.

# Default environment variables
export FIP_SEED := env("FIP_SEED", "67")
export FIP_EPOCHS := env("FIP_EPOCHS", "1200")
export FIP_PATIENCE := env("FIP_PATIENCE", "300")
export FIP_BATCH_SIZE := env("FIP_BATCH_SIZE", "512")
export FIP_LEARNING_RATE := env("FIP_LEARNING_RATE", "0.00005")
export FIP_WEIGHT_DECAY := env("FIP_WEIGHT_DECAY", "0.0001")
export FIP_EARLY_STOPPING_MIN_DELTA := env("FIP_EARLY_STOPPING_MIN_DELTA", "0.001")
export FIP_DEVICE := env("FIP_DEVICE", "cuda")
export FIP_CUTOFF_MINUTE := env("FIP_CUTOFF_MINUTE", "45")
export FIP_WINDOW_MINUTES := env("FIP_WINDOW_MINUTES", "3")
export FIP_MAX_TOKENS_PER_WINDOW := env("FIP_MAX_TOKENS_PER_WINDOW", "256")
export FIP_MAX_VOCAB_SIZE := env("FIP_MAX_VOCAB_SIZE", "120")
export FIP_TOP_PLAY_TYPES := env("FIP_TOP_PLAY_TYPES", "30")
export FIP_TOP_KEY_EVENT_TYPES := env("FIP_TOP_KEY_EVENT_TYPES", "24")
export FIP_TOP_FORMATIONS := env("FIP_TOP_FORMATIONS", "12")
export FIP_TEXT_EMBEDDING_DIM := env("FIP_TEXT_EMBEDDING_DIM", "16")
export FIP_TEXT_CHANNEL_COUNT := env("FIP_TEXT_CHANNEL_COUNT", "12")
export FIP_TEXT_KERNEL_SIZES := env("FIP_TEXT_KERNEL_SIZES", "3,4,5")
export FIP_TEXT_DROPOUT := env("FIP_TEXT_DROPOUT", "0.12")
export FIP_NUMERIC_PROJECTION_SIZE := env("FIP_NUMERIC_PROJECTION_SIZE", "64")
export FIP_FUSION_HIDDEN_SIZE := env("FIP_FUSION_HIDDEN_SIZE", "128")
export FIP_GRU_HIDDEN_SIZE := env("FIP_GRU_HIDDEN_SIZE", "128")
export FIP_DROPOUT := env("FIP_DROPOUT", "0.06")
export FIP_DATALOADER_WORKERS := env("FIP_DATALOADER_WORKERS", "16")
export FIP_CACHE_TENSORS_ON_DEVICE := env("FIP_CACHE_TENSORS_ON_DEVICE", "true")
export FIP_MIXED_PRECISION := env("FIP_MIXED_PRECISION", "true")
export FIP_COMPILE_MODEL := env("FIP_COMPILE_MODEL", "false")

# Sync python environment
[group('dev')]
sync:
  uv sync

# Run the full pipeline
[group('run')]
run: download preprocess train evaluate

# Validate local ESPN raw data
[group('run')]
download:
  uv run fip-download

# Clean raw source files and build processed model data
[group('run')]
preprocess:
  uv run fip-preprocess

# Train the single text-numeric fusion classifier
[group('run')]
[default]
train:
  uv run fip-train

# Evaluate the classifier on chronological splits and generate figures
[group('run')]
evaluate:
  uv run fip-evaluate


# Fix: format and lint
[group('qual')]
fix:
  bunx prettier --log-level=warn --write .
  uv run ruff format .
  uv run ruff check . --fix

# Check code: format and lint
[group('qual')]
check:
  bunx prettier --log-level warn --check .
  uv run ruff format . --check
  uv run ruff check .

# Run tests
[group('qual')]
test:
  uv run pytest

# Full check + test gate (github ci runs this command)
[group('qual')]
ci: check test


# Remove build artifacts
[group('clean')]
clean:
  uv run ruff clean

# Remove all output and generated artifacts
[group('clean')]
clean-outputs:
  rm -rf output/figures/*
  rm -rf output/models/*
  rm -rf output/predictions/*
  rm -rf output/reports/*


# Clean and start docs website at localhost
[group('docs')]
docs:
  rm -rf .site/
  uv run --only-group docs zensical serve

# Build web page with clean cache
[group('docs')]
docs-build:
  uv run --only-group docs zensical build --clean
