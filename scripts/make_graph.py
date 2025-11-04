"""Wrapper script to construct the kNN graph."""

from __future__ import annotations

from operon_embed import cli


def main() -> int:
    return cli.main(["graph"])


if __name__ == "__main__":
    raise SystemExit(main())
