# Turkish Inflation Forecasting - Project Commands
# Usage: just <recipe>
# Run `just --list` to see all available recipes.


# Sync python environment
[group('dev')]
sync:
  uv sync

# Run the main entry point
[default]
[group('run')]
run:
  uv run main


# Fix: format and lint
[group('qual')]
fix:
  bunx prettier --log-level=warn --write .
  uv run ruff format .
  uv run ruff check . --fix

# Check code: format and lint
[group('qual')]
check:
  uv run ruff format . --check
  uv run ruff check .


# Remove build artifacts
[group('clean')]
clean:
  uv run ruff clean

# Remove all output and generated artifacts
[group('clean')]
clean-outputs:
  rm -rf output/


# Clean and start docs website at localhost
[group('docs')]
docs:
  rm -rf site/
  uv run --only-group docs zensical serve

# Build web page with clean cache
[group('docs')]
docs-build:
  uv run --only-group docs zensical build --clean
