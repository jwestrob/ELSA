"""POST HOC scoring of frozen lut outputs on the Spicy-module truth (module_instances.parquet from x05_modules.py).

The lut collection was already used once as an untouched test (long_loci frozen run, 2026-11-14/19); these numbers
evaluate frozen output on a truth set defined afterwards. Nothing is rebuilt; loci are read as written.

Definitions, all from the frozen protocol (FROZEN.json 'scoring'), read-only scorers imported:
  recovered / resolved   ribosomal_check.score on the instance span [first member gid, last member gid]
                         (best = locus of largest overlap, ties shorter; recovered best_frac >= .5;
                         resolved best_frac >= .8 and overlap / locus length >= .8)
  F1, control, excess    score_dpann_loci.score on the member genes (shadow package, gold/instances.parquet = these
                         instances; best locus by F1; 200 rigid translations of the member pattern on its contig,
                         seed sha256('17:' + instance_id); excess = F1 - mean control F1)
  chance recovered / resolved   the same 200 translation origins applied to the span interval, scored by
                         ribosomal_check.score; reported as the mean count over draws
Outputs: score_modules.tsv (one row per module x output), score_per_instance.parquet, score_modules.json.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
FZ = RD / "long_loci/frozen"
PKG = RD / "slice_lut/testset/package_k8"
sys.path.insert(0, str(RD / "ws_dpann/package/scorer"))
sys.path.insert(0, str(RD / "ribosomal/code"))
import score_dpann_loci as S  # noqa: E402
import ribosomal_check as RC  # noqa: E402

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
OUTPUTS = {
    "SECONDARY_embedding_list_tree_Z_L4bp_t0.02": FZ / "loci/lut_Z/L4bp_t0.02.parquet",
    "PRIMARY_family_bond_tree_WA_L4b_t0.2": FZ / "loci/lut_WA/L4b_t0.2.parquet",
    "engine_DH": FZ / "loci/lut_engine/DH.parquet",
    "synton_faithful": FZ / "synton/runs/faithful/post/loci.parquet",
}
t0 = time.time()
log = lambda *a: print(f"{time.time() - t0:7.1f}s", *a, flush=True)

genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome", "contig", "contig_code", "position_index"])
contigs = pd.read_parquet(PKG / "universe/contigs.parquet").sort_values("contig_code")
first_gid = np.zeros(int(contigs.contig_code.max()) + 1, np.int64)
first_gid[contigs.contig_code.to_numpy()] = contigs.first_gid.to_numpy()
ngenes = np.zeros_like(first_gid)
ngenes[contigs.contig_code.to_numpy()] = contigs.n_genes.to_numpy()
gen_arr, con_arr, pos = genes.genome.to_numpy(), genes.contig.to_numpy(), genes.position_index.to_numpy()

I = pd.read_parquet("module_instances.parquet")
mem = I.members.map(lambda v: np.asarray(v, np.int64))
I["member_positions"] = [(m - first_gid[c]).tolist() for m, c in zip(mem, I.contig_code)]
I["contig"] = con_arr[I.first_gid.to_numpy()]
I["start_idx"] = pos[I.first_gid.to_numpy()]
I["end_idx"] = pos[I.last_gid.to_numpy()]
I["caller"] = "spicy_module"
I["system_type"] = I.module
I["n_members_resolved"] = I.n_members
I["resolved"] = True
I["order_trusted"] = True
shadow = Path("shadow")
(shadow / "gold").mkdir(parents=True, exist_ok=True)
if not (shadow / "universe").exists():
    (shadow / "universe").symlink_to(PKG / "universe")
I[["instance_id", "caller", "system_type", "genome", "contig", "contig_code", "start_idx", "end_idx", "span_genes",
   "member_positions", "n_members_resolved", "resolved", "order_trusted"]].to_parquet(shadow / "gold/instances.parquet", index=False)

# translation origins for the span-based chance expectation (same seed scheme as score_dpann_loci)
qs0 = I.first_gid.to_numpy(np.int64)
qe0 = I.last_gid.to_numpy(np.int64)
NCTL = 200
QS = np.zeros((len(I), NCTL), np.int64)
for k, (iid, cc, w) in enumerate(zip(I.instance_id, I.contig_code, I.span_genes - 1)):
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"17:{iid}".encode()).digest()[:8], "big"))
    QS[k] = first_gid[cc] + rng.integers(0, max(ngenes[cc] - w, 1), size=NCTL)   # = the F1 control's origins
QE = QS + (qe0 - qs0)[:, None]


def load_loci(label, path):
    df = pd.read_parquet(path)
    if label.startswith("synton"):
        s = first_gid[df.contig_code.to_numpy(np.int64)] + df.start.to_numpy(np.int64)
        e = first_gid[df.contig_code.to_numpy(np.int64)] + df.end.to_numpy(np.int64)
    else:
        s, e = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
    k = e > s
    se = pd.DataFrame(dict(s=s[k], e=e[k])).drop_duplicates().sort_values(["s", "e"]).reset_index(drop=True)
    return se.s.to_numpy(np.int64), se.e.to_numpy(np.int64)


rows, pers = [], []
for label, path in OUTPUTS.items():
    if not path.exists():
        log("absent", label)
        continue
    s, e = load_loci(label, path)
    ls = dict(a=s, b=e, mod=np.full(len(s), -1, np.int64), maxlen=int((e - s + 1).max()), n=int(len(s)))
    rc = RC.score(ls, qs0, qe0)
    ctl = RC.score(ls, QS.ravel(), QE.ravel())
    crec = ctl.recovered.to_numpy().reshape(len(I), NCTL)
    cres = ctl.resolved.to_numpy().reshape(len(I), NCTL)
    df = pd.DataFrame(dict(genome=gen_arr[s], contig=con_arr[s], start=pos[s], end=pos[e], module_id=np.full(len(s), -1)))
    per, _, _ = S.score(shadow, df, controls=NCTL, seed=17, threads=THREADS, min_members=2)
    per["excess"] = per.f1 - per.control_mean_f1
    per = per.set_index("instance_id").reindex(I.instance_id)
    p = pd.DataFrame(dict(instance_id=I.instance_id, module=I.module, output=label, recovered=rc.recovered.to_numpy(),
                          resolved=rc.resolved.to_numpy(), best_frac=rc.best_frac.to_numpy(), precision=rc.precision.to_numpy(),
                          best_len=rc.best_len.to_numpy(), chance_recovered=crec.mean(1), chance_resolved=cres.mean(1),
                          f1=per.f1.to_numpy(), control_f1=per.control_mean_f1.to_numpy(), excess=per.excess.to_numpy()))
    pers.append(p)
    for m, g in p.groupby("module", sort=False):
        rng = np.random.default_rng(17)
        idx = rng.integers(0, len(g), size=(10000, len(g)))
        ex = g.excess.to_numpy()
        rows.append(dict(module=m, output=label, instances=len(g), recovered=int(g.recovered.sum()),
                         resolved=int(g.resolved.sum()), chance_recovered=round(float(g.chance_recovered.sum()), 1),
                         chance_resolved=round(float(g.chance_resolved.sum()), 1),
                         f1=round(float(g.f1.mean()), 3), control_f1=round(float(g.control_f1.mean()), 3),
                         excess=round(float(ex.mean()), 3), excess_lo=round(float(np.quantile(ex[idx].mean(1), .025)), 3),
                         excess_hi=round(float(np.quantile(ex[idx].mean(1), .975)), 3),
                         median_best_len_recovered=float(g.best_len[g.recovered].median()) if g.recovered.any() else None))
    log(label, "done")
R = pd.DataFrame(rows)
R.to_csv("score_modules.tsv", sep="\t", index=False)
pd.concat(pers).to_parquet("score_per_instance.parquet", index=False)
json.dump(dict(outputs={k: str(v) for k, v in OUTPUTS.items()}, controls=NCTL, post_hoc=True,
               seconds=round(time.time() - t0, 1)), open("score_modules.json", "w"), indent=1)
print(R.to_string())
