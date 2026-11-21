#!/usr/bin/env python
"""Frame-challenge test (lut): is each assayed unannotated candidate homologous to the label-carrying partners its function
came from? Under 'divergent family member' the candidate aligns to its partner over most of its length at clear (> 25%)
identity; under 'non-homologous gene in the same neighborhood slot' (positional replacement, or an unrelated neighbor
picked up by the slot rule) the alignment is short and near background identity.
Candidates: unannotated (not named) population genes folded in the fixed assay (readout_full own pair) and in the extended
tier (data/ext/readout_ext own pair). Partner = rep_partner_gid (best-ranked capped carrier holding the function KO,
genes_function.parquet). Partner sequences fetched from the public lut FASTA on biotite (r4_pools.py fetch).
Alignment: Biopython local Smith-Waterman, BLOSUM62, gap open -10 / extend -0.5; identity over aligned columns; coverage =
aligned candidate residues / candidate length. Background: the same alignment of each candidate to a length-matched
negative-pool protein (its negative-B partner).
usage: X1_partner_homology.py lut   writes data/partner_homology.tsv; prints the comparison by fold match."""
import json, subprocess, sys
from pathlib import Path
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices
from scipy.stats import mannwhitneyu

C = sys.argv[1]
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CFG = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
FD = H / "folds"
RS = CFG["remote"].replace("RD/", RD + "/")
B, PY = "jwestrob@biotite.berkeley.edu", "/home/jwestrob/miniconda3/envs/elsa/bin/python"


def read_faa(p):
    d, n = {}, None
    for l in open(p):
        if l.startswith(">"):
            n = l[1:].split()[0]; d[n] = []
        else:
            d[n].append(l.strip())
    return {k: "".join(v) for k, v in d.items()}


pop = pd.read_parquet(H / "data/population.parquet")
GF = pd.read_parquet(H / "data/genes_function.parquet", columns=["gid", "rep_partner_gid", "ko_top_n", "n_carriers_ko2"])
fx = pd.read_csv(H / "data/readout_full.tsv", sep="\t", dtype={"protein_id": str})
fx = fx[(fx.role == "candidate") & (fx.pair == "own")].assign(assay="fixed")
ex = pd.read_csv(H / "data/ext/readout_ext.tsv", sep="\t", dtype={"protein_id": str})
ex = ex[(ex.role == "candidate") & (ex.pair == "own")].assign(assay="extended")
X = pd.concat([fx[["protein_id", "func", "assay", "TM_ref", "fold_match", "PASS", "mean_plddt"]],
               ex[["protein_id", "func", "assay", "TM_ref", "fold_match", "PASS", "mean_plddt"]]], ignore_index=True)
X["gid"] = X.protein_id.astype(int)
X = X.merge(pop[["gid", "named", "tier", "support"]], on="gid").merge(GF, on="gid")
X = X[~X.named].copy()
sel_f = pd.read_csv(H / "data/selection_full.tsv", sep="\t", dtype={"protein_id": str, "matched_to": str})
sel_e = pd.read_csv(H / "data/ext/selection_ext.tsv", sep="\t", dtype={"protein_id": str, "matched_to": str})
negb = pd.concat([sel_f[sel_f.role == "negB"], sel_e[sel_e.role == "negB"]]).drop_duplicates("matched_to").set_index("matched_to").protein_id
X["negB"] = X.protein_id.map(negb)
ids = sorted(set(X.rep_partner_gid.astype(int)) - {-1})
(H / "data/x1_partner_ids.txt").write_text("\n".join(map(str, ids)) + "\n")
subprocess.run(["rsync", "-a", str(H / "data/x1_partner_ids.txt"), f"{B}:{RS}/data/"], check=True)
subprocess.run(["ssh", B, f"cd {RS} && {PY} code/r4_pools.py fetch --collection {C} --ids {RS}/data/x1_partner_ids.txt --out {RS}/data/x1_partners.faa"],
               check=True, capture_output=True)
subprocess.run(["rsync", "-a", f"{B}:{RS}/data/x1_partners.faa", str(FD / "seqs") + "/"], check=True)
SEQ = {**read_faa(FD / "seqs/candidates.faa"), **read_faa(FD / "seqs/negpool.faa"), **read_faa(FD / "seqs/x1_partners.faa")}
al = Align.PairwiseAligner(mode="local", open_gap_score=-10, extend_gap_score=-0.5)
al.substitution_matrix = substitution_matrices.load("BLOSUM62")


def stats(a, b):
    if not a or not b:
        return (None, None, None)
    x = al.align(a, b)[0]
    n = idn = 0
    for (s1, e1), (s2, e2) in zip(*x.aligned):
        for k in range(e1 - s1):
            n += 1; idn += a[s1 + k] == b[s2 + k]
    return (round(idn / max(1, n), 3), round(n / len(a), 3), round(x.score, 1))


rows = []
for r in X.itertuples():
    c = SEQ.get(r.protein_id, "")
    p = SEQ.get(str(int(r.rep_partner_gid)), "")
    q = SEQ.get(str(r.negB), "") if isinstance(r.negB, str) else ""
    i1, c1, s1 = stats(c, p)
    i0, c0, s0 = stats(c, q)
    rows.append(dict(protein_id=r.protein_id, func=r.func, assay=r.assay, tier=r.tier, support=r.support, TM_ref=r.TM_ref,
                     fold_match=r.fold_match, PASS=r.PASS, mean_plddt=round(r.mean_plddt, 1), partner=int(r.rep_partner_gid),
                     cand_len=len(c), partner_len=len(p), partner_identity=i1, partner_coverage=c1, partner_score=s1,
                     negB=r.negB, negB_identity=i0, negB_coverage=c0, negB_score=s0))
T = pd.DataFrame(rows)
T["homolog"] = (T.partner_identity >= 0.25) & (T.partner_coverage >= 0.5)
T["background_like"] = T.partner_score <= T.negB_score.fillna(0) * 1.5 + 20
T.to_csv(H / "data/partner_homology.tsv", sep="\t", index=False)
pd.set_option("display.width", 200)
print(T.groupby(["assay", "fold_match"]).agg(n=("protein_id", "size"), homolog=("homolog", "sum"), background_like=("background_like", "sum"),
      ident_med=("partner_identity", "median"), cov_med=("partner_coverage", "median"), score_med=("partner_score", "median"),
      negB_score_med=("negB_score", "median"), plddt_med=("mean_plddt", "median")).to_string())
print("all:", len(T), "homolog", int(T.homolog.sum()), "background-like", int(T.background_like.sum()))
a, b = T[T.fold_match].partner_score.dropna(), T[~T.fold_match].partner_score.dropna()
print("partner score fold-match vs not: median", a.median(), b.median(), "Mann-Whitney p", mannwhitneyu(a, b, alternative="greater").pvalue)
print("candidate-partner vs candidate-negB score, all: median", T.partner_score.median(), T.negB_score.median(),
      "paired share partner > negB", round((T.partner_score > T.negB_score).mean(), 3))
print(T.groupby("func").agg(n=("protein_id", "size"), homolog=("homolog", "sum"), fold=("fold_match", "sum"), ident=("partner_identity", "median"),
                            cov=("partner_coverage", "median")).sort_values("n", ascending=False).head(25).to_string())
