"""Wrapper script to train the optional linear metric."""

from __future__ import annotations

from operon_embed import cli


def main() -> int:
    return cli.main(["train-metric"])


if __name__ == "__main__":
    raise SystemExit(main())
