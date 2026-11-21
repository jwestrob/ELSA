#!/usr/bin/env python3
"""Profile-HMM sensitivity check for missing lut components inside called loci (workstream slice_lut).

usage: 09_hmm_refine.py stage RESULTS_DIR PROTEINS_FAA UNIVERSE_DIR WORK_DIR THREADS
  stage export : write per-component training FASTA (reverse-best genes of main loci, <= 400 per component, seed 17) and
                 the window FASTA (every gene within 10 genes of a main or loose locus)
  stage score  : read WORK_DIR/hits.domtbl (hmmsearch of the component HMMs vs the window FASTA), assign each window gene
                 its best HMM at full-sequence E <= 1e-5, and recount the components per locus
                 -> RESULTS_DIR/hmm_refine_loci.tsv, RESULTS_DIR/hmm_refine_summary.tsv
"""
import json, os, sys
import numpy as np, pandas as pd, duckdb

stage, R, FAA, U, W, T = sys.argv[1:7]
os.makedirs(W, exist_ok=True)
L = pd.read_csv(f"{R}/loci.tsv", sep="\t")
G = pd.read_parquet(f"{U}/genes.parquet", columns=["gid", "contig_code", "position_index", "genome", "contig"])
LAB = {"META1_1778": "A", "META1_1779": "B", "META1_1780": "C", "META1_1781": "D", "META1_1782": "E", "META1_1783": "F",
       "META1_1784": "G", "META1_1785": "H", "META1_1786": "M", "META1_1787": "I"}
key = G.set_index(["genome", "contig"]).index
cc = dict(zip(zip(G.genome, G.contig), G.contig_code))
L["contig_code"] = [cc[(a, b)] for a, b in zip(L.genome, L.contig)]
sel = L[L.definition.isin(["main", "loose"])]
win = []
for r in sel.itertuples():
    win.append((r.locus_id, r.contig_code, r.start_pos - 10, r.end_pos + 10))
Wd = pd.DataFrame(win, columns=["locus_id", "contig_code", "lo", "hi"])
con = duckdb.connect(); con.execute(f"SET threads={T}")
con.register("Wd", Wd); con.register("G", G)
wg = con.execute("""SELECT Wd.locus_id, G.gid, G.position_index FROM Wd JOIN G ON G.contig_code = Wd.contig_code
                    AND G.position_index BETWEEN Wd.lo AND Wd.hi""").df()
if stage == "export":
    lg = pd.read_csv(f"{R}/locus_genes.tsv", sep="\t", usecols=["locus_id", "definition", "gid", "role", "best_component", "reverse_best"])
    tr = lg[(lg.definition == "main") & (lg.role == "component") & lg.reverse_best.astype(bool)]
    need = set(wg.gid) | set(tr.gid)
    seqs = {}
    with open(FAA) as fh:
        g = None
        for line in fh:
            if line[0] == ">":
                g = int(line[1:])
            elif g in need:
                seqs[g] = line.strip()
    rng = np.random.default_rng(17)
    counts = {}
    for comp, lab in LAB.items():
        gi = sorted(set(tr[tr.best_component == comp].gid))
        uniq = {}
        for x in gi:
            uniq.setdefault(seqs[x], x)
        gi = sorted(uniq.values())
        if len(gi) > 400:
            gi = sorted(rng.choice(gi, 400, replace=False).tolist())
        counts[lab] = len(gi)
        with open(f"{W}/train_{lab}.faa", "w") as f:
            for x in gi:
                f.write(f">{x}\n{seqs[x]}\n")
    with open(f"{W}/window.faa", "w") as f:
        for x in sorted(set(wg.gid)):
            f.write(f">{x}\n{seqs[x]}\n")
    json.dump(dict(train=counts, window_genes=int(wg.gid.nunique()), loci=int(len(sel))), open(f"{W}/export.json", "w"), indent=1)
    print(json.dumps(counts), wg.gid.nunique(), flush=True)
else:
    rows = []
    for line in open(f"{W}/hits.domtbl"):
        if line.startswith("#"):
            continue
        f = line.split()
        rows.append((int(f[0]), f[3], float(f[6]), float(f[7])))
    H = pd.DataFrame(rows, columns=["gid", "hmm", "evalue", "score"]).drop_duplicates(["gid", "hmm"])
    H = H[H.evalue <= 1e-5].sort_values(["gid", "score"], ascending=[True, False]).drop_duplicates("gid")
    hm = dict(zip(H.gid, H.hmm))
    lg = pd.read_csv(f"{R}/locus_genes.tsv", sep="\t", usecols=["locus_id", "gid", "role", "best_component", "reverse_best", "rel_pos"])
    out = []
    for r in sel.itertuples():
        x = lg[(lg.locus_id == r.locus_id)]
        inside = x[x.role != "flank"]
        before = set(LAB[c] for c in inside[inside.role == "component"].best_component.dropna() if c in LAB)
        # HMM calls on any gene of the locus span or within 3 genes outside it
        near = wg[(wg.locus_id == r.locus_id) & (wg.position_index >= r.start_pos - 3) & (wg.position_index <= r.end_pos + 3)]
        after_h = set(hm[g] for g in near.gid if g in hm)
        after = before | after_h
        out.append(dict(locus_id=r.locus_id, definition=r.definition, genome=r.genome, family=r.family, genus=r.genus,
                        before="".join(sorted(before)), hmm_calls="".join(sorted(after_h)), after="".join(sorted(after)),
                        added="".join(sorted(after - before))))
    O = pd.DataFrame(out)
    O.to_csv(f"{R}/hmm_refine_loci.tsv", sep="\t", index=False)
    s = []
    for d in ["main", "loose"]:
        o = O[O.definition == d]
        for lab in LAB.values():
            s.append(dict(definition=d, component=lab, loci=len(o), with_before=int(o.before.str.contains(lab).sum()),
                          with_after=int(o.after.str.contains(lab).sum()), added=int(o.added.str.contains(lab).sum())))
    pd.DataFrame(s).to_csv(f"{R}/hmm_refine_summary.tsv", sep="\t", index=False)
    print(pd.DataFrame(s).to_string(), flush=True)
