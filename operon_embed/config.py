"""Configuration helpers for the operon embedding pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore[import-untyped]


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file and expand environment variables."""
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _expand_env_vars(data)


def dump_config(config: Dict[str, Any], path: str | Path) -> None:
    """Persist configuration to disk."""
    with Path(path).expanduser().resolve().open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _expand_env_vars(node: Any) -> Any:
    """Recursively expand environment variables in config values."""

    if isinstance(node, dict):
        return {key: _expand_env_vars(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_expand_env_vars(value) for value in node]
    if isinstance(node, str):
        return os.path.expandvars(node)
    return node
