#!/usr/bin/env python3
"""Architecture screen, step 2: per-gene region map for every labeled gene.

Sources with coordinates (all observations):
  - Aksha weak hits of the 1,812 label models (check A raw output, --evalue 10 --domE 10; data/checkA_label_hits_raw.tsv)
  - CDD best hits per region (Sharur protein_resemblance, rpsblast E <= 0.01, best model per region)
Per gene: label envelope (union of proposed-label domains with i-E <= 10), model coverage, regions of other families
outside the label envelope, composition (ILVFMAWC, Kyte-Doolittle TM segments, SEG-like low complexity, exact
internal repeats). Writes data/regions.parquet (one row per region) and data/genes2.parquet."""
import duckdb, pandas as pd, numpy as np, os, math
from collections import Counter
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
G = pd.read_parquet("data/genes.parquet")
M = pd.read_csv("data/checkA_model_map.tsv", sep="\t")
PL = pd.read_csv("data/pfam_model_len.tsv", sep="\t", header=None, names=["model", "acc", "leng"])
PL["label"] = PL.acc.str.split(".").str[0]
KL = pd.read_csv("data/kofam_label_model_len.tsv", sep="\t")
mlen = dict(zip(PL.model, PL.leng)); mlen.update(dict(zip(KL.model, KL.leng)))
m2l = dict(zip(M.model, M.label))
H = pd.read_csv("data/checkA_label_hits_raw.tsv", sep="\t")
H["label"] = H.hmm_name.map(m2l)
H["mlen"] = H.hmm_name.map(mlen)
H = H[H.i_evalue <= 10]
lab_of = dict(zip(G.protein_id, G.acc))
H["is_proposed"] = H.label.values == H.sequence_id.map(lab_of).values
# proposed-label envelope per gene
P = H[H.is_proposed].groupby("sequence_id").agg(lab_from=("env_from", "min"), lab_to=("env_to", "max"), lab_best_ie=("i_evalue", "min"),
        lab_ndom=("env_from", "size"), lab_hmm_from=("hmm_from", "min"), lab_hmm_to=("hmm_to", "max"), lab_mlen=("mlen", "first"))
P["lab_hmm_cov"] = (P.lab_hmm_to - P.lab_hmm_from + 1) / P.lab_mlen
G = G.merge(P, left_on="protein_id", right_index=True, how="left")
# CDD regions
con = duckdb.connect(); con.execute("SET threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.register("ids", G[["protein_id"]])
CDD = con.execute("""SELECT r.protein_id, r.accession, r."name" AS fam, r.description, r.evalue, r.score, r.start_aa, r.end_aa
                     FROM d.protein_resemblance r JOIN ids USING (protein_id)""").df()
CDD["source"] = "cdd"
O = H[~H.is_proposed][["sequence_id", "label", "hmm_name", "i_evalue", "dom_bitscore", "env_from", "env_to", "hmm_from", "hmm_to", "mlen"]].rename(
    columns={"sequence_id": "protein_id", "hmm_name": "fam", "i_evalue": "evalue", "dom_bitscore": "score", "env_from": "start_aa", "env_to": "end_aa"})
O["accession"] = O.label; O["source"] = "label_model"; O["description"] = ""
Lp = H[H.is_proposed][["sequence_id", "label", "hmm_name", "i_evalue", "dom_bitscore", "env_from", "env_to", "hmm_from", "hmm_to", "mlen"]].rename(
    columns={"sequence_id": "protein_id", "hmm_name": "fam", "i_evalue": "evalue", "dom_bitscore": "score", "env_from": "start_aa", "env_to": "end_aa"})
Lp["accession"] = Lp.label; Lp["source"] = "proposed_label"; Lp["description"] = ""
R = pd.concat([Lp, O, CDD], ignore_index=True)
R = R.merge(G[["protein_id", "acc", "lab_from", "lab_to", "len"]], on="protein_id")
ov = (np.minimum(R.end_aa, R.lab_to) - np.maximum(R.start_aa, R.lab_from) + 1).clip(lower=0).fillna(0)
R["overlap_label_frac"] = ov / (R.end_aa - R.start_aa + 1)
# CDD pfam-derived model of the proposed label
R["cdd_is_label"] = (R.source == "cdd") & (R.accession.str.replace("pfam", "PF", regex=False).str.zfill(7) == R.acc)
R.to_parquet("data/regions.parquet", index=False)

# composition
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
def comp(seq):
    s = seq.rstrip("*"); n = len(s)
    # TM segments: KD window 19 >= 1.6, merged
    v = np.array([KD.get(c, 0.0) for c in s])
    tm = np.zeros(n, bool)
    if n >= 19:
        w = np.convolve(v, np.ones(19) / 19, "valid")
        for i in np.flatnonzero(w >= 1.6): tm[i:i + 19] = True
    segs = int(np.sum(np.diff(np.r_[0, tm.astype(int)]) == 1))
    # SEG-like low complexity: window 12, Shannon entropy < 2.2 bits
    lc = np.zeros(n, bool)
    for i in range(0, max(n - 11, 0)):
        c = Counter(s[i:i + 12]); e = -sum(k / 12 * math.log2(k / 12) for k in c.values())
        if e < 2.2: lc[i:i + 12] = True
    # exact internal repeats: residues covered by 8-mers that occur >= 2 times
    km = Counter(s[i:i + 8] for i in range(n - 7)); rp = np.zeros(n, bool)
    for i in range(n - 7):
        if km[s[i:i + 8]] > 1: rp[i:i + 8] = True
    return segs, tm.mean() if n else 0, lc.mean() if n else 0, rp.mean() if n else 0
X = np.array([comp(s) for s in G.seq])
G["tm_segs"], G["tm_frac"], G["lowc_frac"], G["exrep_frac"] = X[:, 0].astype(int), X[:, 1], X[:, 2], X[:, 3]
# max copies of one family among regions (repeat proxy)
cp = R.groupby(["protein_id", "fam"]).size().groupby("protein_id").max()
G["max_fam_copies"] = G.protein_id.map(cp).fillna(0).astype(int)
G.drop(columns=["seq"]).to_parquet("data/genes2.parquet", index=False)
print("genes", len(G), "with proposed-label envelope", int(G.lab_from.notna().sum()), "regions", len(R), R.source.value_counts().to_dict())
