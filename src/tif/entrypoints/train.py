"""Training stage entrypoint."""

from tif.pipeline import run_train


def main() -> int:
    return run_train()
