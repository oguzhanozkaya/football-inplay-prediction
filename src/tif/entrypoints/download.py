"""Download stage entrypoint."""

from tif.pipeline import run_download


def main() -> int:
    return run_download()
