# Project recipes
# Usage: just <recipe>

export FIP_SEED := env("FIP_SEED", "67")
export FIP_EPOCHS := env("FIP_EPOCHS", "1200")
export FIP_PATIENCE := env("FIP_PATIENCE", "50")
export FIP_BATCH_SIZE := env("FIP_BATCH_SIZE", "512")
export FIP_LEARNING_RATE := env("FIP_LEARNING_RATE", "0.0001")
export FIP_WEIGHT_DECAY := env("FIP_WEIGHT_DECAY", "0.0001")
export FIP_EARLY_STOPPING_MIN_DELTA := env("FIP_EARLY_STOPPING_MIN_DELTA", "0.001")
export FIP_DEVICE := env("FIP_DEVICE", "cuda")
export FIP_CUTOFF_MINUTE := env("FIP_CUTOFF_MINUTE", "45")
export FIP_MAX_TOKENS := env("FIP_MAX_TOKENS", "256")
export FIP_MAX_VOCAB_SIZE := env("FIP_MAX_VOCAB_SIZE", "6000")
export FIP_TOP_PLAY_TYPES := env("FIP_TOP_PLAY_TYPES", "30")
export FIP_TOP_KEY_EVENT_TYPES := env("FIP_TOP_KEY_EVENT_TYPES", "24")
export FIP_TOP_FORMATIONS := env("FIP_TOP_FORMATIONS", "12")
export FIP_TEXT_EMBEDDING_DIM := env("FIP_TEXT_EMBEDDING_DIM", "64")
export FIP_TEXT_CHANNEL_COUNT := env("FIP_TEXT_CHANNEL_COUNT", "48")
export FIP_TEXT_KERNEL_SIZES := env("FIP_TEXT_KERNEL_SIZES", "3,4,5")
export FIP_TEXT_DROPOUT := env("FIP_TEXT_DROPOUT", "0.25")
export FIP_NUMERIC_HIDDEN_SIZE := env("FIP_NUMERIC_HIDDEN_SIZE", "128")
export FIP_FUSION_HIDDEN_SIZE := env("FIP_FUSION_HIDDEN_SIZE", "128")
export FIP_DROPOUT := env("FIP_DROPOUT", "0.25")
export FIP_DATALOADER_WORKERS := env("FIP_DATALOADER_WORKERS", "4")
export FIP_MIXED_PRECISION := env("FIP_MIXED_PRECISION", "true")
export FIP_COMPILE_MODEL := env("FIP_COMPILE_MODEL", "false")
export FIP_MATCH_LIMIT := env("FIP_MATCH_LIMIT", "0")

[group('dev')]
sync:
  uv sync

[group('run')]
[default]
run:
  uv run python fig.py

[group('run')]
smoke:
  FIP_DEVICE=cpu FIP_EPOCHS=1 FIP_PATIENCE=1 FIP_DATALOADER_WORKERS=0 FIP_MIXED_PRECISION=false FIP_MATCH_LIMIT=200 uv run python fig.py

[group('qual')]
fix:
  bunx prettier --log-level=warn --write .
  uv run ruff format .
  uv run ruff check . --fix

[group('qual')]
check:
  bunx prettier --log-level warn --check .
  uv run ruff format . --check
  uv run ruff check .

[group('qual')]
ci: check

[group('clean')]
clean:
  uv run ruff clean

[group('clean')]
clean-outputs:
  rm -rf output/figures/*
  rm -rf output/models/*
  rm -rf output/predictions/*
  rm -rf output/reports/*

[group('docs')]
docs:
  rm -rf .site/
  uv run --only-group docs zensical serve

[group('docs')]
docs-build:
  uv run --only-group docs zensical build --clean
