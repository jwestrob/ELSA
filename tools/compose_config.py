#!/usr/bin/env python3
"""
Compose a full ELSA config by overlaying a partial YAML (override) onto a base YAML.
This preserves all non-overridden sections (embedding, shingling, chaining, etc.).

Usage:
  python tools/compose_config.py --base elsa.config.yaml --override configs/elsa_cluster_control.yaml --out configs/_composed_control.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def deep_merge(dst, src):
    if not isinstance(dst, dict) or not isinstance(src, dict):
        return src
    out = dict(dst)
    for k, v in src.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='elsa.config.yaml')
    p.add_argument('--override', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    if yaml is None:
        print('PyYAML required. Install with: pip install pyyaml', file=sys.stderr)
        sys.exit(2)

    base = Path(args.base)
    over = Path(args.override)
    outp = Path(args.out)
    with open(base, 'r') as f:
        base_cfg = yaml.safe_load(f) or {}
    with open(over, 'r') as f:
        ovr = yaml.safe_load(f) or {}
    merged = deep_merge(base_cfg, ovr)
    with open(outp, 'w') as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    print(f'Wrote composed config: {outp}')


if __name__ == '__main__':
    main()

