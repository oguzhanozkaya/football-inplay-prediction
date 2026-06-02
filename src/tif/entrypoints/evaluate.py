"""Evaluation stage entrypoint."""

from tif.pipeline import run_evaluate


def main() -> int:
    return run_evaluate()
