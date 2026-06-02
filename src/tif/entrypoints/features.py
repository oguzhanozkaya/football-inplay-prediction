"""Feature generation stage entrypoint."""

from tif.pipeline import run_features


def main() -> int:
    return run_features()
