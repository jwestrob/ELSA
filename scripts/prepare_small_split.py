"""Prepare a minimal real-data split for tests.

This script will be implemented in later milestones. For now it simply
validates that the expected environment variables are available and
emits guidance when they are not.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    data_root = os.getenv("OPERON_TEST_DATA")
    if not data_root:
        sys.stderr.write("OPERON_TEST_DATA is not set; nothing to prepare.\n")
        return 1
    split_dir = os.getenv("OPERON_SMALL_SPLIT_DIR")
    if not split_dir:
        sys.stderr.write("OPERON_SMALL_SPLIT_DIR is not set; nothing to prepare.\n")
        return 1
    sys.stderr.write(
        "prepare_small_split is a placeholder; populate the output directory manually for now.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
