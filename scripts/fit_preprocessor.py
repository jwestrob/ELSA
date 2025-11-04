"""Entry point for fitting the preprocessing pipeline."""

from __future__ import annotations

import sys

from operon_embed import cli


def main() -> int:
    return cli.main(["fit-preproc", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
