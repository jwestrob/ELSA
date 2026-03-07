"""Utility helpers shared across modules."""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator


@contextlib.contextmanager
def section(name: str) -> Iterator[None]:
    """Context manager that logs entry and exit of a pipeline section."""
    logging.getLogger(__name__).info("Starting %s", name)
    try:
        yield
    finally:
        logging.getLogger(__name__).info("Finished %s", name)
