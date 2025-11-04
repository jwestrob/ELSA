"""Wrapper script to cluster the operon graph."""

from __future__ import annotations

from operon_embed import cli


def main() -> int:
    return cli.main(["cluster"])


if __name__ == "__main__":
    raise SystemExit(main())
