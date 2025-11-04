"""Pytest configuration for the operon embedding project."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Iterator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def operon_test_data() -> Iterator[Path]:
    """Provide the root path to the real dataset used by tests.

    Tests skip automatically when the ``OPERON_TEST_DATA`` environment variable
    is not configured or points at a missing location, ensuring that we never
    fabricate synthetic fixtures.
    """

    data_root = os.getenv("OPERON_TEST_DATA")
    if not data_root:
        pytest.skip(
            "OPERON_TEST_DATA is not set; real dataset required for this test suite",
            allow_module_level=True,
        )
    path = Path(data_root).expanduser()
    if not path.exists():
        pytest.skip(
            f"OPERON_TEST_DATA path does not exist: {path}",
            allow_module_level=True,
        )
    yield path
