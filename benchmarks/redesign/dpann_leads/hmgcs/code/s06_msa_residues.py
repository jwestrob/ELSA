#!/usr/bin/env python3
"""MSA (mafft L-INS-i) of lead representatives, GA-annotated members of the same Micrarchaeota family, canonical
DPANN HMGCS and curated UniProt HMGCS / HMG-ACP synthase records; report the residue each sequence carries at the
active-site and substrate-binding positions annotated in the curated records (UniProt FT ACT_SITE / BINDING), and the
insertions of the lead relative to the references."""
import re, subprocess, sys, duckdb, pandas as pd
from collections import OrderedDict
W = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/hmgcs"
REF = f"{W}/ref"
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
lead_reps = ["OJT94780.1", "MCL5680306.1", "JAJZNR010000018.1_8"]          # one per species among the 17
fam_ga = ["DAIDLN010000011.1_2", "DAHXDE010000048.1_1", "DAHXET010000043.1_3", "DAHXQX010000030.1_19", "JAKAON010000015.1_14", "DAHXCN010000011.1_16"]
canon = ["CAISET010000016.1_25", "AJF61242.1", "MEW6295566.1"]
refs = ["A0A384E143", "Q58941", "Q9HI87", "Q6L233", "D4GWR6", "Q9FD71", "Q01581", "P40830", "Q6DNE9"]
seqs = OrderedDict()
df = con.execute("SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN (SELECT unnest(?))", [lead_reps + fam_ga + canon]).df().set_index("protein_id").sequence
for p in lead_reps + fam_ga + canon:
    seqs[p] = df[p].rstrip("*")
for a in refs:
    s = "".join(l.strip() for l in open(f"{REF}/{a}.faa") if not l.startswith(">"))
    seqs[a] = s
with open(f"{W}/msa_in.faa", "w") as fh:
    for k, v in seqs.items():
        fh.write(f">{k}\n{v}\n")
out = subprocess.run(["mafft", "--localpair", "--maxiterate", "1000", "--thread", "4", "--quiet", f"{W}/msa_in.faa"], capture_output=True, text=True).stdout
aln = OrderedDict(); cur = None
for l in out.splitlines():
    if l.startswith(">"):
        cur = l[1:].split()[0]; aln[cur] = ""
    else:
        aln[cur] += l.strip()
with open(f"{W}/msa_hmgcs.aln.faa", "w") as fh:
    for k, v in aln.items():
        fh.write(f">{k}\n{v}\n")
def col_of(seqid, resnum):            # alignment column for residue number (1-based) of seqid
    n = 0
    for i, c in enumerate(aln[seqid]):
        if c != "-":
            n += 1
            if n == resnum:
                return i
def res_at(seqid, col):
    c = aln[seqid][col]
    if c == "-":
        return "-"
    return f"{c}{len(aln[seqid][:col+1].replace('-', ''))}"
# curated sites from UniProt flat files
def sites(acc):
    txt = open(f"{REF}/{acc}.txt").read().splitlines()
    out = []
    for i, l in enumerate(txt):
        m = re.match(r"FT   (ACT_SITE|BINDING)\s+(\d+)(?:\.\.(\d+))?", l)
        if m:
            lig = ""
            for j in range(i + 1, min(i + 6, len(txt))):
                if txt[j].startswith("FT   ") and not txt[j].startswith("FT    "):
                    break
                mm = re.search(r'/(?:ligand|note)="([^"]+)', txt[j])
                if mm and not lig:
                    lig = mm.group(1)
            a, b = int(m.group(2)), int(m.group(3) or m.group(2))
            for r in range(a, b + 1):
                out.append((m.group(1), r, lig))
    return out
rows = []
for acc in ["A0A384E143", "Q9FD71", "Q01581", "P40830"]:
    for kind, r, lig in sites(acc):
        col = col_of(acc, r)
        rec = dict(ref=acc, kind=kind, ref_res=res_at(acc, col), what=lig[:45])
        for k in aln:
            rec[k] = res_at(k, col)
        rows.append(rec)
T = pd.DataFrame(rows).drop_duplicates(subset=["ref", "kind", "ref_res"])
pd.set_option("display.width", 400); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200)
show = ["ref", "kind", "ref_res", "what"] + lead_reps + fam_ga[:3] + ["CAISET010000016.1_25", "A0A384E143", "Q9FD71", "Q01581", "P40830", "Q6DNE9"]
print(T[show].to_string(index=False))
T.to_csv(f"{W}/catalytic_residue_table.tsv", sep="\t", index=False)
# identity of lead reps to each reference over aligned (non-gap both) columns
print("\n# pairwise identity over columns aligned in both (and aligned length)")
def ident(a, b):
    m = n = 0
    for x, y in zip(aln[a], aln[b]):
        if x != "-" and y != "-":
            n += 1; m += (x == y)
    return round(100 * m / n, 1), n
idt = pd.DataFrame({r: {q: "%s%% / %d" % ident(q, r) for q in lead_reps + fam_ga[:2] + canon[:1]} for r in refs + canon[:1]})
print(idt.T.to_string())
# insertion map for lead rep OJT94780.1 relative to the archaeal reference A0A384E143
print("\n# OJT94780.1 segments aligned to gaps in A0A384E143 (length >= 8)")
q, r = "OJT94780.1", "A0A384E143"
segs = []; st = None; qn = 0
for x, y in zip(aln[q], aln[r]):
    if x != "-":
        qn += 1
    if x != "-" and y == "-":
        if st is None: st = qn
        en = qn
    elif x != "-" and st is not None:
        segs.append((st, en)); st = None
if st is not None: segs.append((st, en))
print([(a, b, b - a + 1) for a, b in segs if b - a + 1 >= 8])
