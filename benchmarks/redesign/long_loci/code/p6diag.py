#!/usr/bin/env python
"""part 6 diagnosis (descriptive, after scoring): where B gains and loses ribosomal runs (DPANN, both bases, frozen theta).
For runs resolved by the base and not by B (and the reverse): does B's best locus span a bridged junction, its precision
and extension; verdict-1 runs of the audit repaired by B. Login node, one thread."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
LL = RD / "long_loci"
sys.path.insert(0, str(RD / "ribosomal/code"))
import ribosomal_check as RC
runs = pd.read_parquet(RD / "ribosomal/gold/runs.parquet")
runs = runs[(runs.run_set == "g2") & (runs.n_marked >= 16)].reset_index(drop=True)
cases = pd.read_csv(RD / "ribo_audit/work2/cases2.tsv", sep="\t")
out = {}
for base, nm, th, aname in (("P", "L4b", "0.2", "WA_L4b"), ("Z", "L4bp", "0.02", "Z_L4bp")):
    J = pd.read_parquet(LL / f"p6/bridges_dpann_{base}.parquet")
    br = np.sort(J.k[J.bridge].to_numpy())
    res = {}
    sc = {}
    for v in ("base", "B", "S", "SB"):
        df = pd.read_parquet(LL / f"loci/p6_dpann_{base}{v}/{nm}_t{th}.parquet")
        a, b = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
        o = np.argsort(a, kind="stable"); a, b = a[o], b[o]
        L = dict(a=a, b=b, mod=np.full(len(a), -1, np.int64), maxlen=int((b - a + 1).max()), n=len(a))
        s = RC.score(L, runs.gid_start.to_numpy(), runs.gid_end.to_numpy())
        bj = s.module  # unused
        sc[v] = (s, a, b)
    sb, ab, bb = sc["base"]
    sB, aB, bB = sc["B"]
    lost = np.flatnonzero(sb.resolved.to_numpy() & ~sB.resolved.to_numpy())
    gained = np.flatnonzero(~sb.resolved.to_numpy() & sB.resolved.to_numpy())
    def best_iv(s_, a_, b_, i):
        rs, re = runs.gid_start.iat[i], runs.gid_end.iat[i]
        idx = np.flatnonzero((a_ <= re) & (b_ >= rs))
        ov = np.minimum(b_[idx], re) - np.maximum(a_[idx], rs) + 1
        ln = b_[idx] - a_[idx] + 1
        k = np.lexsort((ln, -ov))[0]
        return int(a_[idx][k]), int(b_[idx][k])
    rows = []
    for i in lost:
        x0, x1 = best_iv(sB, aB, bB, i)
        spans_bridge = bool(((br >= x0) & (br < x1)).any())
        rows.append(dict(run=int(i), spans_bridge=spans_bridge, prec=float(sB.precision.iat[i]), best_frac=float(sB.best_frac.iat[i]),
                         len=x1 - x0 + 1, run_len=int(runs.gid_end.iat[i] - runs.gid_start.iat[i] + 1),
                         base_best_frac=float(sb.best_frac.iat[i]), base_prec=float(sb.precision.iat[i])))
    R = pd.DataFrame(rows)
    v1 = cases[(cases.output == aname) & (cases.verdict == 1)]
    rid = dict(zip(runs.run_id, range(len(runs))))
    v1i = [rid[r] for r in v1.run_id if r in rid]
    res = dict(lost=int(len(lost)), gained=int(len(gained)),
               lost_best_spans_bridge=int(R.spans_bridge.sum()) if len(R) else 0,
               lost_best_frac_ge_08=int((R.best_frac >= 0.8).sum()) if len(R) else 0,
               lost_prec_median=float(R.prec.median()) if len(R) else None,
               lost_base_best_frac_lt_1=int((R.base_best_frac < 1).sum()) if len(R) else 0,
               verdict1_runs=len(v1i), verdict1_resolved_base=int(sb.resolved.to_numpy()[v1i].sum()),
               verdict1_resolved_B=int(sB.resolved.to_numpy()[v1i].sum()),
               verdict1_resolved_S=int(sc["S"][0].resolved.to_numpy()[v1i].sum()),
               verdict1_resolved_SB=int(sc["SB"][0].resolved.to_numpy()[v1i].sum()))
    for v in ("S", "SB"):
        s_ = sc[v][0]
        res[f"{v}_gained"] = int((~sb.resolved.to_numpy() & s_.resolved.to_numpy()).sum())
        res[f"{v}_lost"] = int((sb.resolved.to_numpy() & ~s_.resolved.to_numpy()).sum())
    out[base] = res
    print(base, res, flush=True)
(LL / "p6/diag_B.json").write_text(json.dumps(out, indent=1) + "\n")
