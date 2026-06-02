"""Preprocess stage entrypoint."""

from tif.pipeline import run_preprocess


def main() -> int:
    return run_preprocess()
