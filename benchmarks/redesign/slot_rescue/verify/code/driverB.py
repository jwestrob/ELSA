#!/usr/bin/env python3
"""Check B driver (Mac, detached; predeclared_verify.json): fold the sample with ESMFold2 through the Biohub batch API
(fold_biohub.py), copy the models to biotite, run US-align there as a single-threaded serial loop on the login node, then
write the pair tables and verify/STATUS_B (DONE or FAILED)."""
import json, re, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parents[1]
B = H / "B"
STATUS = H / "STATUS_B"
FOLD = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/scripts/fold_biohub.py"
SSH = "jwestrob@biotite.berkeley.edu"
VD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue/verify"
PYR = "/home/jwestrob/miniconda3/envs/elsa/bin/python"
def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), *a, flush=True)
def run(cmd, **kw):
    log("RUN", " ".join(cmd) if isinstance(cmd, list) else cmd)
    r = subprocess.run(cmd, **kw)
    log("rc", r.returncode)
    return r.returncode
def share(x):
    x = list(x)
    return (round(float(np.mean(x)), 3), len(x)) if x else (None, 0)
try:
    STATUS.unlink(missing_ok=True)
    t0 = time.time()
    with open(B / "fold.log", "a") as fl:
        rc_fold = run([sys.executable, FOLD, str(B / "folds"), str(B / "fold_input.faa")], stdout=fl, stderr=subprocess.STDOUT)
    if not (B / "folds/summary.tsv").exists():
        raise RuntimeError(f"no summary.tsv after folding (rc {rc_fold})")
    t_fold = round(time.time() - t0)
    assert run(["ssh", "-o", "BatchMode=yes", SSH, f"mkdir -p {VD}/B/folds"]) == 0
    assert run(f"scp -q {B}/folds/*.pdb {B}/folds/summary.tsv {SSH}:{VD}/B/folds/", shell=True) == 0
    assert run(["scp", "-q", str(H / "data/B_pairs.tsv"), f"{SSH}:{VD}/data/B_pairs.tsv"]) == 0
    t1 = time.time()
    assert run(["ssh", "-o", "BatchMode=yes", SSH, f"cd {VD} && OMP_NUM_THREADS=1 {PYR} code/usalign_pairs.py data/B_pairs.tsv B/folds B/usalign_pairs.tsv"]) == 0
    t_cmp = round(time.time() - t1)
    assert run(["scp", "-q", f"{SSH}:{VD}/B/usalign_pairs.tsv", str(B / "usalign_pairs.tsv")]) == 0
    U = pd.read_csv(B / "usalign_pairs.tsv", sep="\t")
    sm = pd.read_csv(B / "folds/summary.tsv", sep="\t")
    nm = lambda p: re.sub(r"[^A-Za-z0-9_.-]", "_", p)[:100]
    pl = dict(zip(sm.name, sm.protein_plddt))
    U["plddt_protein"] = U.protein_id.map(lambda p: pl.get(nm(p)))
    U["plddt_partner"] = U.partner.map(lambda p: pl.get(nm(p)))
    U["both_plddt_ge70"] = (U.plddt_protein >= 70) & (U.plddt_partner >= 70)
    U["tm1_ge05"] = U.TM1 >= 0.5
    U["tm2_ge05"] = U.TM2 >= 0.5
    S = pd.read_csv(H / "data/B_sample.tsv", sep="\t")
    U = U.merge(S[["protein_id", "label_name", "votes_for", "vote_share", "confidence_bin", "length", "partner_length"]], on="protein_id", how="left")
    U.to_csv(B / "checkB_pairs.tsv", sep="\t", index=False)
    ok = U[U.status == "ok"]
    rows = []
    for st in ["S1", "S2", "S3", "all"]:
        g = ok if st == "all" else ok[ok.stratum == st]
        for an, gg in (("primary_both_plddt_ge70", g[g.both_plddt_ge70]), ("secondary_all_pairs", g)):
            t, n_ = gg[gg.kind == "true"], gg[gg.kind == "null"]
            st1, nt = share(t.tm1_ge05); sn1, nn = share(n_.tm1_ge05)
            st2, _ = share(t.tm2_ge05); sn2, _ = share(n_.tm2_ge05)
            rows.append(dict(stratum=st, analysis=an, true_pairs=nt, true_tm1_ge05=st1, null_pairs=nn, null_tm1_ge05=sn1,
                             true_tm2_ge05=st2, null_tm2_ge05=sn2,
                             true_median_tm1=round(float(t.TM1.median()), 3) if nt else None,
                             null_median_tm1=round(float(n_.TM1.median()), 3) if nn else None))
    R = pd.DataFrame(rows)
    R.to_csv(B / "checkB_summary.tsv", sep="\t", index=False)
    tr = U[U.kind == "true"]
    top = tr.label.value_counts().head(20).index
    lt = []
    for lab in top:
        g = tr[(tr.label == lab) & (tr.status == "ok")]
        lt.append(dict(label=lab, name=g.label_name.iloc[0] if len(g) else "", pairs=len(g), both_plddt_ge70=int(g.both_plddt_ge70.sum()),
                       tm1_ge05_all=int(g.tm1_ge05.sum()), tm1_ge05_primary=int((g.tm1_ge05 & g.both_plddt_ge70).sum()),
                       median_tm1=round(float(g.TM1.median()), 3) if len(g) else None))
    pd.DataFrame(lt).to_csv(B / "checkB_top20_labels.tsv", sep="\t", index=False)
    run(f"scp -q {B}/checkB_pairs.tsv {B}/checkB_summary.tsv {B}/checkB_top20_labels.tsv {SSH}:{VD}/B/", shell=True)
    folds = len(list((B / "folds").glob("*.pdb")))
    STATUS.write_text(f"DONE {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} folds {folds} (fold rc {rc_fold}, {t_fold} s) "
                      f"pairs ok {len(ok)} of {len(U)} (US-align {t_cmp} s) checkB_pairs.tsv checkB_summary.tsv checkB_top20_labels.tsv\n")
    log(STATUS.read_text().strip())
except Exception as e:
    STATUS.write_text(f"FAILED {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {type(e).__name__}: {str(e)[:300]}\n")
    traceback.print_exc()
    log(STATUS.read_text().strip())
