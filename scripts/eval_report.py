"""Wrapper script to compute evaluation reports."""

from __future__ import annotations

from operon_embed import cli


def main() -> int:
    return cli.main(["eval"])


if __name__ == "__main__":
    raise SystemExit(main())
