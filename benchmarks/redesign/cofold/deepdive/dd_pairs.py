"""Per-interface metrics recomputed from returned structures (shared helper, local).

pair_metrics(fold_dir, fold_id) -> {pair 'AB': dict(pair_iptm, contacts, iface_pae, iface_plddt)}
  pair_iptm  max(M[i][j], M[j][i]) of pair_chains_iptm (screen-2 definition)
  contacts   residue pairs between the two chains with CB-CB <= 8 A (CA for Gly), from the PDB
  iface_pae  mean of (PAE(a,b) + PAE(b,a)) / 2 over the contact pairs (PAE blocks saved by biohub_cofold.py)
  iface_plddt  mean pLDDT (B-factor) of residues of either chain that take part in a contact
Pass rule (screen-2 predeclared, per replicate): pair_iptm >= 0.75, contacts >= 10, iface_pae <= 5 A,
iface_plddt >= 70.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=None)
def _metrics(fold_dir):
    out = {}
    for line in open(Path(fold_dir) / "metrics.jsonl"):
        m = json.loads(line)
        out[m["fold_id"]] = m
    return out


def read_pdb(p):
    res = {}
    for line in open(p):
        if not line.startswith("ATOM"):
            continue
        ch, rn, an, name = line[21], int(line[22:26]), line[12:16].strip(), line[17:20]
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        b = float(line[60:66])
        r = res.setdefault((ch, rn), dict(name=name, ca=None, cb=None, b=b))
        if an == "CA":
            r["ca"] = xyz
        if an == "CB":
            r["cb"] = xyz
    chains = {}
    for (ch, rn), r in sorted(res.items(), key=lambda x: (x[0][0], x[0][1])):
        rep = r["cb"] if (r["name"] != "GLY" and r["cb"] is not None) else r["ca"]
        chains.setdefault(ch, {"xyz": [], "b": []})
        chains[ch]["xyz"].append(rep)
        chains[ch]["b"].append(r["b"])
    return {c: (np.array(v["xyz"]), np.array(v["b"])) for c, v in chains.items()}


def pair_metrics(fold_dir, fold_id):
    m = _metrics(str(fold_dir))[fold_id]
    ids = [c for c in "ABCDEFGH"][:len(m["lengths"])]
    M = m.get("pair_chains_iptm")
    S = read_pdb(Path(fold_dir) / "pdb" / f"{fold_id}.pdb")
    P = np.load(Path(fold_dir) / "pae" / f"{fold_id}.npz") if (Path(fold_dir) / "pae" / f"{fold_id}.npz").exists() else None
    out = {}
    for i, a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            b = ids[j]
            xa, ba = S[a]
            xb, bb = S[b]
            d = np.sqrt(((xa[:, None, :] - xb[None, :, :]) ** 2).sum(-1))
            c = d <= 8.0
            r = dict(pair_iptm=float(max(M[i][j], M[j][i])) if M else None, contacts=int(c.sum()))
            if c.any() and P is not None:
                pab, pba = P[f"{a}{b}"].astype(float), P[f"{b}{a}"].astype(float)
                r["iface_pae"] = float(((pab + pba.T) / 2)[c].mean())
                r["iface_plddt"] = float(np.concatenate([ba[c.any(1)], bb[c.any(0)]]).mean())
            else:
                r["iface_pae"], r["iface_plddt"] = None, None
            r["pass"] = bool(r["pair_iptm"] is not None and r["pair_iptm"] >= 0.75 and r["contacts"] >= 10
                             and r["iface_pae"] is not None and r["iface_pae"] <= 5 and r["iface_plddt"] >= 70)
            out[a + b] = r
    return out, m
