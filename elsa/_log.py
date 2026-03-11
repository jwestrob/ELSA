"""Timestamped pipeline logging for ELSA status output."""
import sys
from datetime import datetime


def tlog(msg: str):
    """Print timestamped message to stderr."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{ts} {msg}", file=sys.stderr, flush=True)
