"""Wrapper script to build the retrieval index."""

from __future__ import annotations

from operon_embed import cli


def main() -> int:
    return cli.main(["build-index"])


if __name__ == "__main__":
    raise SystemExit(main())
