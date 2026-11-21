#!/usr/bin/env python3
"""Question 2/3 (rung 3): split each long Thy1-like protein into its N- and C-terminal halves, align the halves with
the 78 Swiss-Prot reviewed ThyX sequences and with annotated DPANN Thy1 members (one-domain and two-domain, halves of
the latter split at their Pfam domain boundary), and read the residues at the T. maritima ThyX (UniProt Q9WYT0)
positions annotated as FAD-binding, dUMP-binding, active site, ThyX motif or mutagenesis-sensitive.
Split rule for lead genes: midpoint between the N-half Thy1 envelope end and the C-half envelope start when both halves
hit; otherwise C-hit ali_from - hmm_from - 10 (start of the C domain extrapolated from the model coordinate).
Writes data/msa/halves.faa, data/msa/halves_aln.faa (mafft L-INS-i), data/residues_by_half.tsv; prints summaries.
usage: python code/05_halves_msa.py"""
import os, random, subprocess, duckdb, pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
os.makedirs("data/msa", exist_ok=True)
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")

def readfa(fn):
    s, k = {}, None
    for l in open(fn):
        l = l.strip()
        if l.startswith(">"):
            k = l[1:].split()[0]; s[k] = ""
        elif k:
            s[k] += l
    return s

lead = readfa("data/lead1.faa")
L = pd.read_csv("data/lead_genes.tsv", sep="\t").query("lab=='Thy1'")
H = pd.read_csv("data/aksha_picked/user_hmms_hits_df.tsv", sep="\t").query("hmm_name=='Thy1'")
pieces = []  # (name, group, protein, start, end)
for r in L.itertuples():
    s = lead[r.protein_id]; n = len(s)
    h = H[H.sequence_id == r.protein_id].sort_values("env_from")
    if n < 420:
        pieces.append((f"{r.protein_id}|lead_short", "lead_short", r.protein_id, 1, n)); continue
    hN = h[h.env_to <= n * 0.6]; hC = h[h.env_from >= n * 0.45]
    if len(hN) and len(hC):
        cut = int((hN.env_to.max() + hC.env_from.min()) / 2)
    elif len(hC):
        c = hC.iloc[-1]; cut = int(c.ali_from - c.hmm_from - 10)
    else:
        cut = int(n * 0.54)
    cut = max(150, min(cut, n - 150))
    pieces.append((f"{r.protein_id}|lead_N", "lead_N", r.protein_id, 1, cut))
    pieces.append((f"{r.protein_id}|lead_C", "lead_C", r.protein_id, cut + 1, n))

M = pd.read_csv("data/annotated_members.tsv", sep="\t").query("source=='pfam' and nm=='Thy1'")
g = M.groupby("protein_id").agg(n=("start_aa", "size"), len=("len", "first"), s=("start_aa", list), e=("end_aa", list), tax=("taxonomy", "first")).reset_index()
random.seed(20261121)
two = g[(g.n == 2) & g.tax.str.contains("Nanobdellota")].sample(40, random_state=1)
two = pd.concat([two, g[(g.n == 2) & ~g.tax.str.contains("Nanobdellota")].sample(20, random_state=1)])
one = g[(g.n == 1) & (g.len.between(200, 330))].sample(60, random_state=1)
ids = tuple(set(two.protein_id) | set(one.protein_id))
seqs = dict(con.execute(f"SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN {ids}").fetchall())
for r in two.itertuples():
    s, e = sorted(r.s), sorted(r.e)
    cut = int((e[0] + s[1]) / 2)
    pieces.append((f"{r.protein_id}|ann2_N", "ann2_N", r.protein_id, 1, cut))
    pieces.append((f"{r.protein_id}|ann2_C", "ann2_C", r.protein_id, cut + 1, r.len))
for r in one.itertuples():
    pieces.append((f"{r.protein_id}|ann1", "ann1", r.protein_id, 1, r.len))
ref = readfa("data/ref/thyx_reviewed.faa")
with open("data/msa/halves.faa", "w") as f:
    for k, v in ref.items():
        acc = k.split("|")[1]
        f.write(f">{acc}|ref\n{v}\n")
    for name, grp, pid, a, b in pieces:
        s = (lead.get(pid) or seqs[pid]).rstrip("*")
        f.write(f">{name}:{a}-{b}\n{s[a-1:b]}\n")
P = pd.DataFrame(pieces, columns=["name", "group", "protein_id", "start", "end"])
P.to_csv("data/msa/pieces.tsv", sep="\t", index=False)
print(P.group.value_counts())
if not os.path.exists("data/msa/halves_aln.faa"):
    with open("data/msa/halves_aln.faa", "w") as out:
        subprocess.run(["mafft", "--localpair", "--maxiterate", "100", "--thread", "4", "--quiet", "data/msa/halves.faa"], stdout=out, check=True)
A = readfa("data/msa/halves_aln.faa")
tm = A["Q9WYT0|ref"]
col_of, k = {}, 0
for i, ch in enumerate(tm):
    if ch != "-":
        k += 1; col_of[k] = i
KEY = {53: "H", 55: "T", 75: "Q", 78: "R", 79: "H", 80: "R", 86: "E", 88: "S", 90: "R", 144: "E", 147: "R", 163: "N", 165: "R", 169: "N", 174: "R"}
for p, aa in KEY.items():
    assert tm[col_of[p]].upper() == aa, (p, tm[col_of[p]])
rows = []
for name, s in A.items():
    grp = name.split("|")[1].split(":")[0]
    rec = {"name": name, "group": grp}
    for p in KEY:
        rec[f"{KEY[p]}{p}"] = s[col_of[p]].upper()
    rows.append(rec)
R = pd.DataFrame(rows)
kc = [f"{KEY[p]}{p}" for p in KEY]
refR = R[R.group == "ref"]
cons = {c: (refR[c] == c[0]).mean() for c in kc}
print("conservation of the T. maritima residue among 78 reviewed ThyX:", {c: round(v, 2) for c, v in cons.items()})
core = [c for c in kc if cons[c] >= 0.85]
print("core residues (>= 85% identical in reviewed ThyX):", core)
R["core_match"] = R[core].apply(lambda x: sum(x[c] == c[0] for c in core), axis=1)
R["n_core"] = len(core)
R.to_csv("data/residues_by_half.tsv", sep="\t", index=False)
print(R.groupby("group").core_match.describe())
for c in core:
    print(c, {g_: round((R[R.group == g_][c] == c[0]).mean(), 2) for g_ in ["ref", "ann1", "ann2_N", "ann2_C", "lead_N", "lead_C", "lead_short"]})
print(R[R.group.str.startswith("lead")].sort_values("name")[["name"] + core + ["core_match"]].to_string(index=False))
