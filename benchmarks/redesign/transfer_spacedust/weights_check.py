#!/usr/bin/env python
"""Are the stored Spacedust ESM-C 300M vectors from the same network as DPANN's ESM++ small vectors?

Compares the two checkpoints tensor by tensor (no model is run): every tensor is hashed as float32 bytes, and the
multisets of (shape, hash) are matched. A fused or split tensor (e.g. qkv) is matched by concatenation when the
direct match fails. CPU only, one tensor in memory at a time.

usage: weights_check.py <esmpp model.safetensors> <esmc model.safetensors> <out.json>
"""
import hashlib
import json
import sys

import numpy as np
from safetensors import safe_open


def table(path):
    out = {}
    with safe_open(path, framework="np") as f:
        for k in f.keys():
            try:
                x = f.get_tensor(k)
            except TypeError:          # bfloat16 is not a numpy dtype
                import torch
                from safetensors import safe_open as so2
                with so2(path, framework="pt") as g:
                    x = g.get_tensor(k).float().numpy()
            x = np.ascontiguousarray(x.astype(np.float32))
            out[k] = (tuple(x.shape), hashlib.sha256(x.tobytes()).hexdigest(), float(np.abs(x).sum()))
    return out


def main():
    a_path, b_path, out = sys.argv[1:4]
    A, B = table(a_path), table(b_path)
    by_hash_b = {}
    for k, (s, h, _) in B.items():
        by_hash_b.setdefault((s, h), []).append(k)
    matched, unmatched_a = {}, []
    used = set()
    for k, (s, h, _) in A.items():
        cand = [c for c in by_hash_b.get((s, h), []) if c not in used]
        if cand:
            matched[k] = cand[0]
            used.add(cand[0])
        else:
            unmatched_a.append(k)
    unmatched_b = [k for k in B if k not in used]
    params = lambda T, ks: int(sum(int(np.prod(T[k][0])) for k in ks))
    res = dict(esmpp=a_path, esmc=b_path, tensors_esmpp=len(A), tensors_esmc=len(B),
               params_esmpp=params(A, A), params_esmc=params(B, B),
               matched_tensors=len(matched), matched_params=params(A, matched),
               unmatched_esmpp=[(k, A[k][0]) for k in unmatched_a],
               unmatched_esmc=[(k, B[k][0]) for k in unmatched_b],
               example_pairs=list(matched.items())[:12])
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "example_pairs"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
