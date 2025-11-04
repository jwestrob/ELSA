"""Wrapper script to run Sinkhorn re-ranking."""

from __future__ import annotations

from operon_embed import cli


def main() -> int:
    return cli.main(["rerank"])


if __name__ == "__main__":
    raise SystemExit(main())
